//! Authenticated OpenAI-compatible gateway (successor to the Cursor adapter).
//!
//! Every route requires a per-client key (`mmk_<id>_<secret>`, only the
//! SHA-256 of the secret is stored) and a peer IP inside `allow_cidrs`.
//! Keys carry a `profile`: "openai" is a byte-faithful passthrough (Grok CLI
//! and friends); "cursor" applies the Cursor Responses→Chat translation and
//! model-alias rewrites this module was originally written for.
//!
//! Every request emits one structured audit line (tracing, JSON): ts, key
//! id+name, peer IP, path, model, status, token usage when present, duration.
//! Never prompt or response text.

use crate::api::SharedState;
use crate::auth;
use crate::config::{resolve_bind, sanitize, GatewayKey, ModelKind};
use crate::docker;
use axum::body::Body;
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{header, Method, StatusCode};
use axum::middleware::{from_fn_with_state, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{Json, Router};
use futures_util::{stream, StreamExt};
use ipnet::IpNet;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Semaphore;
use tower_http::cors::CorsLayer;

/// Max request body accepted by the gateway.
const BODY_CAP: usize = 32 * 1024 * 1024;

#[derive(Default)]
struct FunnelSnap {
    url: Option<String>,
    error: Option<String>,
}

static FUNNEL: Mutex<FunnelSnap> = Mutex::new(FunnelSnap {
    url: None,
    error: None,
});

/// Per-key concurrency limiter, shared across the gateway's requests.
#[derive(Default)]
pub struct GatewayShared {
    pub sems: Mutex<HashMap<String, Arc<Semaphore>>>,
}

impl GatewayShared {
    fn permit_for(&self, key_id: &str, max: u32) -> Option<tokio::sync::OwnedSemaphorePermit> {
        let sem = {
            let Ok(mut m) = self.sems.lock() else {
                return None;
            };
            m.entry(key_id.to_string())
                .or_insert_with(|| Arc::new(Semaphore::new(max as usize)))
                .clone()
        };
        sem.try_acquire_owned().ok()
    }
}

/// Request context attached by the auth middleware for the audit line.
#[derive(Clone)]
struct GwCtx {
    key_id: String,
    key_name: String,
    profile: String,
    peer: IpAddr,
}

pub fn advertised_id(alias: &str, context: u64) -> String {
    let base = alias.split('[').next().unwrap_or(alias).trim();
    let ctx = if context == 0 { 262144 } else { context };
    format!("{base}[context={}]", context_tag(ctx))
}

fn context_tag(ctx: u64) -> String {
    if ctx >= 1_000_000 {
        let m = ((ctx as f64) / 1_000_000.0).round() as u64;
        format!("{m}m")
    } else if ctx >= 1000 {
        let k = ((ctx as f64) / 1000.0).round() as u64;
        format!("{k}k")
    } else {
        ctx.to_string()
    }
}

/// Dashboard snapshot for the "Gateway" bar. Never includes key material.
pub fn snapshot(enabled: bool, port: u16, bind: &str, alias: &str, keys: &[GatewayKey], context: u64) -> Value {
    let f = FUNNEL.lock().ok();
    let ctx = if context == 0 { 262144 } else { context };
    let active = keys.iter().filter(|k| !k.revoked).count();
    json!({
        "enabled": enabled,
        "port": port,
        "bind": bind,
        "model": advertised_id(alias, ctx),
        "model_base": alias.split('[').next().unwrap_or(alias),
        "context": ctx,
        "keys": active,
        "funnel_url": f.as_ref().and_then(|s| s.url.clone()),
        "funnel_error": f.as_ref().and_then(|s| s.error.clone()),
        "base_url": format!("http://{bind}:{port}/v1"),
    })
}

pub fn serve(state: SharedState) {
    tokio::spawn(async move {
        let (bind, port) = {
            let cfg = state.config.lock().await;
            (cfg.gateway.bind.clone(), cfg.gateway.port)
        };
        // Always bind: requests 404 while disabled, so the dashboard toggle
        // takes effect without a restart.
        let ip = match resolve_bind(&bind) {
            Ok(ip) => ip,
            Err(e) => {
                tracing::error!("gateway bind {bind:?}: {e}");
                return;
            }
        };
        let addr = SocketAddr::from((ip, port));
        let app = Router::new()
            .route("/v1/models", get(models))
            .route("/models", get(models))
            .route("/cursor/models", get(models))
            .route("/cursor/v1/models", get(models))
            .route("/openai/v1/models", get(models))
            .route("/v1/chat/completions", post(proxy))
            .route("/v1/completions", post(proxy))
            .route("/v1/responses", post(proxy))
            .route("/v1/embeddings", post(proxy))
            .fallback(any(proxy))
            .layer(from_fn_with_state(state.clone(), gw_auth))
            .layer(CorsLayer::very_permissive())
            .with_state(state);
        tracing::info!("gateway listening on http://{addr}");
        let listener = match tokio::net::TcpListener::bind(addr).await {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("gateway bind {addr} failed: {e}");
                return;
            }
        };
        if let Err(e) = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        {
            tracing::error!("gateway exited: {e}");
        }
    });
}

#[derive(Deserialize)]
pub struct GatewayToggle {
    pub enabled: bool,
}

pub async fn set_enabled(state: SharedState, enabled: bool) -> Result<Value, String> {
    let (port, bind, alias, keys) = {
        let mut cfg = state.config.lock().await;
        cfg.gateway.enabled = enabled;
        cfg.save().map_err(|e| e.to_string())?;
        (
            cfg.gateway.port,
            cfg.gateway.bind.clone(),
            cfg.gateway.model_alias.clone(),
            cfg.gateway.keys.clone(),
        )
    };
    if let Ok(mut g) = FUNNEL.lock() {
        *g = FunnelSnap {
            url: guess_funnel_url(),
            error: None,
        };
    }
    let ctx = upstream(&state).await.map(|(_, _, c)| c as u64).unwrap_or(262144);
    Ok(snapshot(enabled, port, &bind, &alias, &keys, ctx))
}

fn guess_funnel_url() -> Option<String> {
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname").ok()?;
    let host = host.trim();
    if host.is_empty() {
        return None;
    }
    Some(format!("https://{host}.tail1ed6b9.ts.net"))
}

fn openai_err(msg: &str, code: StatusCode) -> Response {
    (
        code,
        Json(json!({"error": {"message": msg, "type": "invalid_request_error", "code": null}})),
    )
        .into_response()
}

// ---------- auth middleware: CIDR → key → concurrency ----------

fn cidrs_ok(cidrs: &[String]) -> Vec<IpNet> {
    cidrs.iter().filter_map(|c| c.parse().ok()).collect()
}

async fn gw_auth(
    State(state): State<SharedState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    mut req: Request,
    next: Next,
) -> Response {
    let peer_ip = peer.ip();
    let (cidrs, keys, max_conc, enabled) = {
        let cfg = state.config.lock().await;
        (
            cfg.gateway.allow_cidrs.clone(),
            cfg.gateway.keys.clone(),
            cfg.gateway.max_concurrency,
            cfg.gateway.enabled,
        )
    };
    if !enabled {
        return openai_err("gateway is off in Model Manager", StatusCode::NOT_FOUND);
    }
    let allowed = cidrs_ok(&cidrs);
    if !allowed.iter().any(|n| n.contains(&peer_ip)) {
        tracing::warn!("gateway: 403 peer {peer_ip} outside allow_cidrs");
        return openai_err("peer not allowed", StatusCode::FORBIDDEN);
    }
    let bearer = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
        .unwrap_or("")
        .to_string();
    let Some((id, secret)) = auth::parse_gateway_key(&bearer) else {
        tracing::warn!("gateway: 401 malformed key from {peer_ip}");
        return openai_err("invalid api key", StatusCode::UNAUTHORIZED);
    };
    let hash = auth::sha256_hex(secret.as_bytes());
    let key = keys
        .iter()
        .find(|k| k.id == id && !k.revoked && auth::ct_eq(&k.hash, &hash));
    let Some(key) = key else {
        tracing::warn!("gateway: 401 unknown/revoked key id={id} from {peer_ip}");
        return openai_err("invalid api key", StatusCode::UNAUTHORIZED);
    };
    let Some(permit) = state.gateway.permit_for(&key.id, max_conc) else {
        tracing::warn!("gateway: 429 concurrency cap for key {}", key.id);
        return openai_err("too many concurrent requests", StatusCode::TOO_MANY_REQUESTS);
    };
    let ctx = GwCtx {
        key_id: key.id.clone(),
        key_name: key.name.clone(),
        profile: key.profile.clone(),
        peer: peer_ip,
    };
    req.extensions_mut().insert(ctx);
    // The handler takes the permit out of this shared slot and holds it in
    // the response stream, so the slot is freed when the stream ends.
    req.extensions_mut()
        .insert(PermitSlot(Arc::new(Mutex::new(Some(permit)))));
    next.run(req).await
}

/// Carries the per-key concurrency permit out of the middleware.
#[derive(Clone)]
struct PermitSlot(Arc<Mutex<Option<tokio::sync::OwnedSemaphorePermit>>>);

// ---------- handlers ----------

async fn models(State(state): State<SharedState>, req: Request) -> Response {
    let ctx = req.extensions().get::<GwCtx>().cloned();
    let Some(ctx) = ctx else {
        return openai_err("invalid api key", StatusCode::UNAUTHORIZED);
    };
    let alias = { state.config.lock().await.gateway.model_alias.clone() };
    let (served, upctx) = match upstream(&state).await {
        Some((name, _, c)) => (name, c as u64),
        None => (alias.clone(), 262144),
    };
    let tagged = advertised_id(&alias, upctx);
    let mut ids = vec![tagged];
    let base = alias.split('[').next().unwrap_or(&alias).to_string();
    if !ids.iter().any(|x| x == &base) {
        ids.push(base);
    }
    if !ids.iter().any(|x| x == &served) {
        ids.push(served);
    }
    let data: Vec<Value> = ids.into_iter().map(|id| model_card(&id, upctx)).collect();
    audit(&ctx, req.uri().path(), "-", 200, None, Instant::now());
    Json(json!({"object": "list", "data": data})).into_response()
}

fn model_card(id: &str, context: u64) -> Value {
    let ctx = if context == 0 { 262144 } else { context };
    json!({
        "id": id,
        "object": "model",
        "owned_by": "model-manager",
        "context_length": ctx,
        "max_model_len": ctx,
        "max_tokens": ctx,
        "context_window": ctx,
        "meta": { "n_ctx_train": ctx, "max_model_len": ctx }
    })
}

async fn proxy(State(state): State<SharedState>, req: Request) -> Response {
    let start = Instant::now();
    let ctx = match req.extensions().get::<GwCtx>().cloned() {
        Some(c) => c,
        None => return openai_err("invalid api key", StatusCode::UNAUTHORIZED),
    };
    let permit = req
        .extensions()
        .get::<PermitSlot>()
        .and_then(|s| s.0.lock().ok().and_then(|mut g| g.take()));
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    if method == Method::GET && path.ends_with("/models") {
        return models(State(state), req).await;
    }
    if method != Method::POST {
        return openai_err(
            &format!("gateway: no handler for {method} {path}"),
            StatusCode::NOT_FOUND,
        );
    }

    let Some((served, port, _ctx)) = upstream(&state).await else {
        audit(&ctx, &path, "-", 503, None, start);
        return openai_err(
            "No LLM is loaded in Model Manager. Load one, then retry.",
            StatusCode::SERVICE_UNAVAILABLE,
        );
    };

    let raw = match axum::body::to_bytes(req.into_body(), BODY_CAP).await {
        Ok(b) => b,
        Err(e) => return openai_err(&format!("read body: {e}"), StatusCode::BAD_REQUEST),
    };
    let mut body: Value = match serde_json::from_slice(&raw) {
        Ok(v) => v,
        Err(_) => {
            return openai_err("request body is not JSON", StatusCode::BAD_REQUEST);
        }
    };
    let req_model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("-")
        .to_string();
    let cursor = ctx.profile == "cursor";

    // Upstream path: cursor profile always lands on /v1/chat/completions;
    // openai passes path and body through verbatim.
    let (up_path, up_body) = if cursor {
        body = to_chat_completions(body);
        body["model"] = json!(served);
        if body.get("messages").is_none() {
            return openai_err(
                "request has neither messages nor a convertible input[] payload",
                StatusCode::BAD_REQUEST,
            );
        }
        (
            "/v1/chat/completions".to_string(),
            serde_json::to_vec(&body).unwrap_or_default(),
        )
    } else {
        (path.clone(), raw.to_vec())
    };

    let url = format!("http://127.0.0.1:{port}{up_path}");
    let send = state
        .http
        .post(&url)
        .header("content-type", "application/json")
        .body(up_body)
        .send()
        .await;
    let resp = match send {
        Ok(r) => r,
        Err(e) => {
            audit(&ctx, &path, &req_model, 502, None, start);
            return openai_err(
                &format!("upstream {served} (:{port}) failed: {e}"),
                StatusCode::BAD_GATEWAY,
            );
        }
    };
    let status = resp.status();
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .cloned()
        .unwrap_or_else(|| header::HeaderValue::from_static("application/json"));

    // Scan response chunks for a `usage` object (SSE `data:` lines or a plain
    // JSON body) so the audit line can carry token counts.
    let usage = Arc::new(Mutex::new(Option::<(u64, u64)>::None));
    let usage2 = usage.clone();
    let mut tail = String::new();
    let scanned = resp.bytes_stream().map(move |chunk| {
        if let Ok(b) = &chunk {
            tail.push_str(&String::from_utf8_lossy(b));
            if tail.len() > 256 * 1024 {
                let drop = tail.len() - 256 * 1024;
                tail.drain(..drop);
            }
            if let Some(u) = extract_usage(&tail) {
                if let Ok(mut g) = usage2.lock() {
                    *g = Some(u);
                }
            }
        }
        chunk.map_err(|e| std::io::Error::other(e))
    });

    let mut builder = Response::builder().status(status.as_u16());
    builder = builder.header(header::CONTENT_TYPE, ct);
    if cursor {
        builder = builder.header("x-ratelimit-limit", "1000000");
        builder = builder.header("x-ratelimit-remaining", "999999");
        builder = builder.header("x-ratelimit-reset", "0");
    }
    // The permit is released when the stream is fully consumed.
    let body_stream = audited_stream(
        scanned,
        ctx,
        path,
        req_model,
        status.as_u16(),
        start,
        usage,
        permit,
    );
    match builder.body(Body::from_stream(body_stream)) {
        Ok(r) => r,
        Err(e) => openai_err(&format!("stream: {e}"), StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// Pull `usage.{prompt,completion}_tokens` (or responses-style names) out of
/// a trailing SSE `data:` line or a JSON body.
fn extract_usage(tail: &str) -> Option<(u64, u64)> {
    // SSE: try each "data:" line, last first; plain JSON: the whole tail.
    let mut lines: Vec<&str> = tail.lines().collect();
    lines.reverse();
    for line in &lines {
        let payload = line
            .trim()
            .strip_prefix("data:")
            .map(str::trim)
            .unwrap_or_else(|| line.trim());
        let Ok(v) = serde_json::from_str::<Value>(payload) else {
            continue;
        };
        if let Some(found) = usage_from(&v) {
            return Some(found);
        }
    }
    serde_json::from_str::<Value>(tail.trim())
        .ok()
        .and_then(|v| usage_from(&v))
}

fn usage_from(v: &Value) -> Option<(u64, u64)> {
    let u = v.get("usage")?;
    let pt = u
        .get("prompt_tokens")
        .or_else(|| u.get("input_tokens"))
        .and_then(|x| x.as_u64());
    let ct = u
        .get("completion_tokens")
        .or_else(|| u.get("output_tokens"))
        .and_then(|x| x.as_u64());
    if pt.is_none() && ct.is_none() {
        return None;
    }
    Some((pt.unwrap_or(0), ct.unwrap_or(0)))
}

fn audit(
    ctx: &GwCtx,
    path: &str,
    model: &str,
    status: u16,
    usage: Option<(u64, u64)>,
    start: Instant,
) {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let (pt, ct) = usage.unwrap_or((0, 0));
    tracing::info!(target: "gateway_audit",
        "{{\"ts\":{ts},\"key_id\":\"{}\",\"key_name\":\"{}\",\"peer\":\"{}\",\"path\":\"{path}\",\"model\":\"{model}\",\"status\":{status},\"prompt_tokens\":{pt},\"completion_tokens\":{ct},\"duration_ms\":{}}}",
        ctx.key_id,
        ctx.key_name,
        ctx.peer,
        start.elapsed().as_millis()
    );
}

/// Append a trailing empty chunk that fires the audit line at stream end.
#[allow(clippy::too_many_arguments)]
fn audited_stream<S>(
    s: S,
    ctx: GwCtx,
    path: String,
    model: String,
    status: u16,
    start: Instant,
    usage: Arc<Mutex<Option<(u64, u64)>>>,
    _permit: Option<tokio::sync::OwnedSemaphorePermit>,
) -> impl futures_util::Stream<Item = Result<bytes::Bytes, std::io::Error>>
where
    S: futures_util::Stream<Item = Result<bytes::Bytes, std::io::Error>>,
{
    let _permit = _permit;
    let mut audited = false;
    s.chain(stream::iter(std::iter::once(Ok::<bytes::Bytes, std::io::Error>(
        bytes::Bytes::new(),
    ))))
    .map(move |item| {
        if matches!(&item, Ok(b) if b.is_empty()) && !audited {
            audited = true;
            let u = usage.lock().ok().and_then(|g| *g);
            audit(&ctx, &path, &model, status, u, start);
        }
        item
    })
}

async fn upstream(state: &SharedState) -> Option<(String, u16, u32)> {
    let cfg = state.config.lock().await;
    let statuses = docker::list_managed(&state.docker).await.ok()?;
    let client = &state.http;
    for def in &cfg.models {
        if def.kind != ModelKind::Llm {
            continue;
        }
        let cname = sanitize(&def.name);
        let running = statuses.iter().any(|s| s.model_name == cname && s.running);
        if !running {
            continue;
        }
        let url = format!("http://127.0.0.1:{}/v1/models", def.host_port);
        let ok = client
            .get(&url)
            .timeout(Duration::from_secs(2))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false);
        if ok {
            return Some((def.name.clone(), def.host_port, def.context));
        }
    }
    None
}

// ---------- Cursor profile: Responses-API → Chat Completions ----------

fn to_chat_completions(mut v: Value) -> Value {
    if v.get("messages").is_none() {
        if let Some(input) = v.get("input").cloned() {
            v["messages"] = input_to_messages(input);
        }
    }
    if let Some(obj) = v.as_object_mut() {
        obj.remove("input");
    }
    if let Some(tools) = v.get("tools").cloned() {
        v["tools"] = convert_tools(tools);
    }
    if let Some(reasoning) = v.get("reasoning").cloned() {
        let effort = reasoning
            .get("effort")
            .and_then(|x| x.as_str())
            .unwrap_or("");
        if !effort.is_empty() && effort != "none" {
            v["reasoning_effort"] = json!(effort);
            v["chat_template_kwargs"] = json!({"enable_thinking": true});
        } else {
            v["chat_template_kwargs"] = json!({"enable_thinking": false});
        }
        if let Some(obj) = v.as_object_mut() {
            obj.remove("reasoning");
        }
    }
    if let Some(max) = v.get("max_output_tokens").cloned() {
        v["max_tokens"] = max;
    }
    if let Some(obj) = v.as_object_mut() {
        for k in [
            "store",
            "include",
            "previous_response_id",
            "truncation",
            "prompt_cache_retention",
            "max_output_tokens",
            "text",
            "prompt_cache_key",
            "safety_identifier",
        ] {
            obj.remove(k);
        }
    }
    if let Some(msgs) = v.get_mut("messages") {
        *msgs = normalize_messages(msgs.take());
    }
    v
}

fn input_to_messages(input: Value) -> Value {
    match input {
        Value::String(s) => json!([{"role": "user", "content": s}]),
        Value::Array(items) => {
            let msgs: Vec<Value> = items.into_iter().filter_map(item_to_message).collect();
            Value::Array(msgs)
        }
        other => json!([{"role": "user", "content": other.to_string()}]),
    }
}

fn item_to_message(item: Value) -> Option<Value> {
    if item.get("role").is_some() {
        return Some(item);
    }
    let ty = item.get("type").and_then(|x| x.as_str()).unwrap_or("");
    match ty {
        "message" | "" => Some(json!({
            "role": item.get("role").and_then(|x| x.as_str()).unwrap_or("user"),
            "content": item.get("content").cloned().unwrap_or(json!("")),
        })),
        "function_call" | "custom_tool_call" => Some(json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": item.get("call_id").or_else(|| item.get("id")).cloned().unwrap_or(json!("call")),
                "type": "function",
                "function": {
                    "name": item.get("name").cloned().unwrap_or(json!("tool")),
                    "arguments": item.get("arguments").cloned().unwrap_or(json!("{}")),
                }
            }]
        })),
        "function_call_output" | "custom_tool_call_output" => Some(json!({
            "role": "tool",
            "tool_call_id": item.get("call_id").or_else(|| item.get("id")).cloned().unwrap_or(json!("call")),
            "content": item.get("output").cloned().unwrap_or(json!("")),
        })),
        _ => None,
    }
}

fn convert_tools(tools: Value) -> Value {
    let Value::Array(arr) = tools else {
        return json!([]);
    };
    let out: Vec<Value> = arr
        .into_iter()
        .filter_map(|t| {
            let ty = t.get("type").and_then(|x| x.as_str()).unwrap_or("function");
            if ty == "custom" || ty == "web_search" || ty == "web_search_preview" {
                return None;
            }
            if t.get("function").is_some() {
                return Some(t);
            }
            let name = t.get("name")?.clone();
            Some(json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": t.get("description").cloned().unwrap_or(json!("")),
                    "parameters": t.get("parameters").cloned().unwrap_or(json!({"type":"object"})),
                }
            }))
        })
        .collect();
    Value::Array(out)
}

fn normalize_messages(msgs: Value) -> Value {
    let Value::Array(arr) = msgs else {
        return msgs;
    };
    let out: Vec<Value> = arr
        .into_iter()
        .map(|m| {
            if m.get("role").and_then(|x| x.as_str()) == Some("user") {
                if let Some(Value::Array(parts)) = m.get("content").cloned() {
                    if parts.iter().any(|p| p.get("type").and_then(|t| t.as_str()) == Some("tool_result"))
                    {
                        let text: String = parts
                            .iter()
                            .filter_map(|p| {
                                if p.get("type").and_then(|t| t.as_str()) == Some("tool_result") {
                                    p.get("content").map(|c| match c {
                                        Value::String(s) => s.clone(),
                                        Value::Array(a) => a
                                            .iter()
                                            .filter_map(|x| x.get("text").and_then(|t| t.as_str()))
                                            .collect::<Vec<_>>()
                                            .join("\n"),
                                        other => other.to_string(),
                                    })
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        let id = parts
                            .iter()
                            .find_map(|p| p.get("tool_use_id").cloned())
                            .unwrap_or(json!("tool"));
                        return json!({"role":"tool","tool_call_id": id, "content": text});
                    }
                }
            }
            m
        })
        .collect();
    Value::Array(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, GatewayKey};
    use axum::http::Request as HReq;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use tower::ServiceExt;

    fn test_state(cidrs: Vec<String>, keys: Vec<GatewayKey>) -> SharedState {
        let mut cfg = Config::default();
        cfg.gateway.enabled = true;
        cfg.gateway.allow_cidrs = cidrs;
        cfg.gateway.keys = keys;
        cfg.gateway.model_alias = "spark".into();
        Arc::new(crate::api::AppState {
            config: tokio::sync::Mutex::new(cfg),
            docker: crate::docker::connect().expect("docker client"),
            sys: tokio::sync::Mutex::new(sysinfo::System::new()),
            http: reqwest::Client::new(),
            loading: tokio::sync::Mutex::new(HashMap::new()),
            auth: crate::auth::AuthState::default(),
            gateway: Arc::new(GatewayShared::default()),
        })
    }

    fn gw_router(state: SharedState) -> Router {
        Router::new()
            .route("/v1/models", get(models))
            .route("/v1/chat/completions", post(proxy))
            .layer(from_fn_with_state(state.clone(), gw_auth))
            .with_state(state)
    }

    fn req(uri: &str, key: Option<&str>, peer: IpAddr) -> HReq<Body> {
        let mut b = HReq::builder().uri(uri).method(Method::GET);
        if let Some(k) = key {
            b = b.header(header::AUTHORIZATION, format!("Bearer {k}"));
        }
        let mut r = b.body(Body::empty()).unwrap();
        r.extensions_mut()
            .insert(ConnectInfo(SocketAddr::from((peer, 1234))));
        r
    }

    fn make_key(name: &str, revoked: bool) -> (GatewayKey, String) {
        let (id, secret, hash) = auth::gen_gateway_key();
        (
            GatewayKey {
                id,
                name: name.into(),
                hash,
                created_at: 0,
                revoked,
                profile: "openai".into(),
            },
            secret,
        )
    }

    #[tokio::test]
    async fn missing_and_fallback_keys_401() {
        let (k, _sec) = make_key("a", false);
        let st = test_state(vec!["127.0.0.0/8".into()], vec![k]);
        let app = gw_router(st);
        for key in [None, Some("local"), Some("sk-local"), Some("sk-mm-deadbeef"), Some("raw-token")] {
            let r = app
                .clone()
                .oneshot(req("/v1/models", key, Ipv4Addr::LOCALHOST.into()))
                .await
                .unwrap();
            assert_eq!(r.status(), StatusCode::UNAUTHORIZED, "key={key:?}");
        }
    }

    #[tokio::test]
    async fn good_key_200_and_revoked_401() {
        let (k, sec) = make_key("a", false);
        let st = test_state(vec!["127.0.0.0/8".into()], vec![k]);
        let r = gw_router(st.clone())
            .oneshot(req("/v1/models", Some(&sec), Ipv4Addr::LOCALHOST.into()))
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::OK);

        // Revoked key rejected.
        let (k2, sec2) = make_key("b", true);
        let st2 = test_state(vec!["127.0.0.0/8".into()], vec![k2]);
        let r = gw_router(st2)
            .oneshot(req("/v1/models", Some(&sec2), Ipv4Addr::LOCALHOST.into()))
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn disallowed_cidr_403() {
        let (k, sec) = make_key("a", false);
        let st = test_state(vec!["127.0.0.0/8".into()], vec![k]);
        // Peer 8.8.8.8 is outside the allowlist → 403 even with a valid key.
        let r = gw_router(st)
            .oneshot(req("/v1/models", Some(&sec), "8.8.8.8".parse().unwrap()))
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn concurrency_cap_429() {
        let (k, sec) = make_key("a", false);
        let st = test_state(vec!["127.0.0.0/8".into()], vec![k]);
        // Exhaust the semaphore for this key id, then a request → 429.
        {
            let cfg = st.config.lock().await;
            let _ = cfg;
        }
        let max = { st.config.lock().await.gateway.max_concurrency };
        let mut held = Vec::new();
        for _ in 0..max {
            held.push(st.gateway.permit_for("x", max).unwrap());
        }
        // different key id has its own bucket; exhaust our key's bucket:
        let kid = { st.config.lock().await.gateway.keys[0].id.clone() };
        let mut held2 = Vec::new();
        for _ in 0..max {
            if let Some(p) = st.gateway.permit_for(&kid, max) {
                held2.push(p);
            }
        }
        assert!(st.gateway.permit_for(&kid, max).is_none());
        let r = gw_router(st.clone())
            .oneshot(req("/v1/models", Some(&sec), Ipv4Addr::LOCALHOST.into()))
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::TOO_MANY_REQUESTS);
        drop(held);
        drop(held2);
    }

    #[test]
    fn cursor_translation_applies() {
        let v = json!({
            "input": [{"type":"message","role":"user","content":"hi"}],
            "max_output_tokens": 50,
            "reasoning": {"effort":"high"},
            "store": true,
        });
        let out = to_chat_completions(v);
        assert!(out.get("messages").is_some());
        assert_eq!(out["max_tokens"], 50);
        assert_eq!(out["reasoning_effort"], "high");
        assert!(out.get("input").is_none());
        assert!(out.get("store").is_none());
        assert_eq!(out["chat_template_kwargs"]["enable_thinking"], true);
    }

    #[test]
    fn usage_extraction() {
        let sse = "data: {\"choices\":[]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":34}}\n\ndata: [DONE]\n";
        assert_eq!(extract_usage(sse), Some((12, 34)));
        let plain = "{\"usage\":{\"input_tokens\":5,\"output_tokens\":7}}";
        assert_eq!(extract_usage(plain), Some((5, 7)));
        assert_eq!(extract_usage("data: [DONE]"), None);
    }

    #[tokio::test]
    async fn sse_stream_is_byte_faithful() {
        use futures_util::stream;
        let chunks: Vec<Result<bytes::Bytes, std::io::Error>> = vec![
            Ok(bytes::Bytes::from_static(b"data: one\n\n")),
            Ok(bytes::Bytes::from_static(b"data: two\n\n")),
            Ok(bytes::Bytes::from_static(b"data: [DONE]\n\n")),
        ];
        let s = stream::iter(chunks);
        let ctx = GwCtx {
            key_id: "k".into(),
            key_name: "n".into(),
            profile: "openai".into(),
            peer: Ipv4Addr::LOCALHOST.into(),
        };
        let usage = Arc::new(Mutex::new(None));
        let out = audited_stream(
            s, ctx, "/v1/chat/completions".into(), "m".into(), 200,
            Instant::now(), usage, None,
        );
        let got: Vec<u8> = out
            .map(|c| c.unwrap().to_vec())
            .fold(Vec::new(), |mut a, c| async move {
                a.extend(c);
                a
            })
            .await;
        assert_eq!(
            got,
            b"data: one\n\ndata: two\n\ndata: [DONE]\n\n".to_vec()
        );
    }
}
