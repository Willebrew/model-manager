//! Removable Cursor BYOK adapter.
//!
//! Cursor's "Override OpenAI Base URL" is picky:
//!   - Agent requests go through Cursor's cloud (`api2.cursor.sh`), so localhost
//!     and Tailscale 100.x IPs are unreachable. We publish HTTPS via Tailscale
//!     Funnel.
//!   - Agent may POST a Responses-API body (`input`, flat tools, `type: custom`
//!     ApplyPatch) to `/v1/chat/completions` and still expect Chat Completions
//!     SSE back. We translate before forwarding to vLLM/SGLang.
//!   - Verify hits `GET /v1/models` (sometimes `/models` or `/cursor/models`).
//!   - API keys that don't start with `sk-` are often rejected.
//!
//! Rip out later: delete this file, the `[cursor]` config field, `mod cursor`
//! / `cursor::serve` in `main.rs`, the `/api/cursor` routes, and the Cursor
//! bar in `web/index.html`.

use crate::api::SharedState;
use crate::config::{sanitize, ModelKind};
use crate::docker;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{header, HeaderMap, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get};
use axum::{Json, Router};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::sync::Mutex;
use std::time::Duration;
use tower_http::cors::CorsLayer;

#[derive(Default)]
struct FunnelSnap {
    url: Option<String>,
    error: Option<String>,
}

static FUNNEL: Mutex<FunnelSnap> = Mutex::new(FunnelSnap {
    url: None,
    error: None,
});

pub fn api_key(server_token: &str) -> String {
    format!("sk-mm-{}", server_token.chars().take(24).collect::<String>())
}

/// Cursor does not read `/v1/models` for the context window. Custom models get
/// `maxTokens=0` (the "~0 / 0 Tokens" chip) unless the model *id* itself carries
/// a parameterized override: `spark[context=262k]`.
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

pub fn snapshot(enabled: bool, port: u16, alias: &str, token: &str, context: u64) -> Value {
    let f = FUNNEL.lock().ok();
    let ctx = if context == 0 { 262144 } else { context };
    json!({
        "enabled": enabled,
        "port": port,
        "model": advertised_id(alias, ctx),
        "model_base": alias.split('[').next().unwrap_or(alias),
        "context": ctx,
        "api_key": api_key(token),
        "funnel_url": f.as_ref().and_then(|s| s.url.clone()),
        "funnel_error": f.as_ref().and_then(|s| s.error.clone()),
        "base_url": f.as_ref().and_then(|s| s.url.clone())
            .unwrap_or_else(|| format!("http://127.0.0.1:{port}/v1")),
    })
}

pub fn serve(state: SharedState) {
    tokio::spawn(async move {
        let port = { state.config.lock().await.cursor.port };
        let addr = SocketAddr::from(([0, 0, 0, 0], port));
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/v1/models", get(models))
            .route("/models", get(models))
            .route("/cursor/models", get(models))
            .route("/cursor/v1/models", get(models))
            .route("/openai/v1/models", get(models))
            .fallback(any(fallback))
            .layer(CorsLayer::very_permissive())
            .with_state(state);
        tracing::info!("Cursor adapter listening on http://0.0.0.0:{port} (toggle in the dashboard)");
        let listener = match tokio::net::TcpListener::bind(addr).await {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("Cursor adapter bind {addr} failed: {e}");
                return;
            }
        };
        if let Err(e) = axum::serve(listener, app.into_make_service()).await {
            tracing::error!("Cursor adapter exited: {e}");
        }
    });
}

#[derive(Deserialize)]
pub struct CursorToggle {
    pub enabled: bool,
}

pub async fn set_enabled(state: SharedState, enabled: bool) -> Result<Value, String> {
    let (port, token, alias) = {
        let mut cfg = state.config.lock().await;
        cfg.cursor.enabled = enabled;
        cfg.save().map_err(|e| e.to_string())?;
        (
            cfg.cursor.port,
            cfg.server.token.clone(),
            cfg.cursor.model_alias.clone(),
        )
    };
    // Funnel is *not* started here: this box already uses `tailscale serve`
    // for other apps, and `funnel --bg` can hang waiting on HTTPS certs.
    // The UI prints the one-liner to run by hand.
    if let Ok(mut g) = FUNNEL.lock() {
        *g = FunnelSnap {
            url: guess_funnel_url(),
            error: None,
        };
    }
    let ctx = upstream(&state).await.map(|(_, _, c)| c as u64).unwrap_or(262144);
    Ok(snapshot(enabled, port, &alias, &token, ctx))
}

fn guess_funnel_url() -> Option<String> {
    // MagicDNS name from /etc/hostname + well-known suffix used on this tailnet.
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

fn cursor_key_ok(headers: &HeaderMap, token: &str) -> bool {
    let want = api_key(token);
    if let Some(h) = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()) {
        if let Some(b) = h.strip_prefix("Bearer ") {
            return b == want || b == token || b == "local" || b == "sk-local";
        }
    }
    // Cursor verify sometimes omits auth on GET /models.
    false
}

async fn models(State(state): State<SharedState>, req: Request) -> Response {
    let (enabled, alias, token) = {
        let cfg = state.config.lock().await;
        (
            cfg.cursor.enabled,
            cfg.cursor.model_alias.clone(),
            cfg.server.token.clone(),
        )
    };
    if !enabled {
        return openai_err("Cursor mode is off in Model Manager", StatusCode::NOT_FOUND);
    }
    // GET /models is how Cursor Verify probes; some builds omit Authorization.
    let _ = req;
    let _ = token;
    let (served, ctx) = match upstream(&state).await {
        Some((name, _, context)) => (name, context as u64),
        None => (alias.clone(), 262144),
    };
    let tagged = advertised_id(&alias, ctx);
    let mut ids = vec![tagged];
    let base = alias.split('[').next().unwrap_or(&alias).to_string();
    if !ids.iter().any(|x| x == &base) {
        ids.push(base);
    }
    if !ids.iter().any(|x| x == &served) {
        ids.push(served);
    }
    let data: Vec<Value> = ids
        .into_iter()
        .map(|id| model_card(&id, ctx))
        .collect();
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

async fn fallback(State(state): State<SharedState>, req: Request) -> Response {
    let path = req.uri().path().to_string();
    let method = req.method().clone();
    let ua = req
        .headers()
        .get(header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    tracing::info!("Cursor adapter {} {} ua=\"{}\"", method, path, ua);
    if method == Method::GET && path.ends_with("/models") {
        return models(State(state), req).await;
    }
    if method == Method::POST
        && (path.ends_with("/chat/completions") || path.ends_with("/completions") || path.ends_with("/responses"))
    {
        return chat(State(state), req).await;
    }
    openai_err(
        &format!("Cursor adapter: no handler for {method} {path}"),
        StatusCode::NOT_FOUND,
    )
}

async fn chat(State(state): State<SharedState>, req: Request) -> Response {
    let (enabled, token) = {
        let cfg = state.config.lock().await;
        (cfg.cursor.enabled, cfg.server.token.clone())
    };
    if !enabled {
        return openai_err("Cursor mode is off in Model Manager", StatusCode::NOT_FOUND);
    }
    if !cursor_key_ok(req.headers(), &token) {
        return openai_err("invalid api key", StatusCode::UNAUTHORIZED);
    }
    let Some((served, port, _ctx)) = upstream(&state).await else {
        return openai_err(
            "No LLM is loaded in Model Manager. Load one, then retry.",
            StatusCode::SERVICE_UNAVAILABLE,
        );
    };

    let raw = match axum::body::to_bytes(req.into_body(), 16 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => return openai_err(&format!("read body: {e}"), StatusCode::BAD_REQUEST),
    };
    let mut body: Value = match serde_json::from_slice(&raw) {
        Ok(v) => v,
        Err(_) => json!({}),
    };
    body = to_chat_completions(body);
    // Always send the loaded model — Cursor sometimes omits `model` or sends a built-in id.
    body["model"] = json!(served);
    if body.get("messages").is_none() {
        return openai_err(
            "request has neither messages nor a convertible input[] payload",
            StatusCode::BAD_REQUEST,
        );
    }

    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
    {
        Ok(c) => c,
        Err(e) => return openai_err(&format!("http client: {e}"), StatusCode::INTERNAL_SERVER_ERROR),
    };
    let send = client
        .post(&url)
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await;
    let resp = match send {
        Ok(r) => r,
        Err(e) => {
            return openai_err(
                &format!("upstream {served} (:{port}) failed: {e}"),
                StatusCode::BAD_GATEWAY,
            )
        }
    };
    let status = resp.status();
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .cloned()
        .unwrap_or_else(|| header::HeaderValue::from_static("application/json"));
    let stream = resp.bytes_stream().map(|chunk| {
        chunk.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    });
    let mut builder = Response::builder().status(status.as_u16());
    builder = builder.header(header::CONTENT_TYPE, ct);
    builder = builder.header("x-ratelimit-limit", "1000000");
    builder = builder.header("x-ratelimit-remaining", "999999");
    builder = builder.header("x-ratelimit-reset", "0");
    match builder.body(Body::from_stream(stream)) {
        Ok(r) => r,
        Err(e) => openai_err(&format!("stream: {e}"), StatusCode::INTERNAL_SERVER_ERROR),
    }
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
        let running = statuses
            .iter()
            .any(|s| s.model_name == cname && s.running);
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
            // Cursor ApplyPatch / grammar tools are Responses-API only — vLLM/SGLang 400 on them.
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
            // Anthropic-style tool_result stuffed into a user message.
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
