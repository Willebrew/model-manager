//! HTTP API + dashboard for Model Manager.

use crate::config::{Config, ModelDef};
use crate::docker;
use crate::memory::{self, MemEstimate, MemSnapshot, OomVerdict};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    middleware::{from_fn_with_state, Next},
    response::{Html, IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use bollard::Docker;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use sysinfo::System;
use tokio::sync::Mutex;

pub struct AppState {
    pub config: Mutex<Config>,
    pub docker: Docker,
    pub sys: Mutex<System>,
    pub http: reqwest::Client,
    /// In-flight loads, keyed by model name, for live progress reporting.
    pub loading: Mutex<HashMap<String, LoadInfo>>,
    /// Sessions, login lockouts, TOTP anti-replay.
    pub auth: crate::auth::AuthState,
    /// Per-key gateway concurrency limiter.
    pub gateway: Arc<crate::gateway::GatewayShared>,
}

/// Snapshot taken when a load starts, so we can report progress as a fraction
/// of the model's estimated footprint that has been allocated so far.
#[derive(Clone, Copy)]
pub struct LoadInfo {
    pub baseline_used_mib: u64,
    pub estimate_mib: u64,
}

pub type SharedState = Arc<AppState>;

pub fn router(state: SharedState) -> Router {
    let protected = Router::new()
        .route("/api/state", get(get_state))
        .route("/api/models", post(upsert_model))
        .route("/api/models/:name", delete(delete_model))
        .route("/api/models/:name/estimate", get(estimate_model))
        .route("/api/models/:name/load", post(load_model))
        .route("/api/models/:name/unload", post(unload_model))
        .route("/api/models/:name/autostart", post(set_autostart))
        .route("/api/models/:name/context", post(set_context))
        .route("/api/models/:name/logs", get(model_logs))
        .route("/api/gateway", get(get_gateway).post(post_gateway))
        .layer(from_fn_with_state(state.clone(), auth));

    Router::new()
        .route("/", get(index))
        .route("/api/health", get(|| async { Json(json!({"ok": true})) }))
        .route("/api/login", post(login))
        .route("/api/logout", post(logout))
        .merge(protected)
        .with_state(state)
}

async fn index() -> impl IntoResponse {
    // Never cache the dashboard so UI updates show up on a normal refresh.
    (
        [(
            axum::http::header::CACHE_CONTROL,
            "no-store, no-cache, must-revalidate",
        )],
        Html(include_str!("web/index.html")),
    )
}

/// Session-cookie gate for every API call. Two paths in:
///   - `mm_session` cookie with a live server-side session (dashboard), or
///   - `X-MM-Admin: <key>` from a loopback peer whose SHA-256 matches
///     `[server] admin_key_hash` (local automation).
/// Mutating requests additionally require the Origin (if present) to match
/// the Host header — CSRF protection for the cookie path.
async fn auth(
    State(state): State<SharedState>,
    req: axum::extract::Request,
    next: Next,
) -> Response {
    let peer_ip = req
        .extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .map(|c| c.0.ip())
        .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));

    // Origin check first: a present-but-mismatched Origin is always a 403.
    if req.method() != axum::http::Method::GET
        && req.method() != axum::http::Method::HEAD
        && !origin_ok(&req)
    {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"error": "origin does not match host"})),
        )
            .into_response();
    }

    let mut authed = false;

    // Loopback admin key (hashed in config; never the key itself).
    if peer_ip.is_loopback() {
        let admin_hash = { state.config.lock().await.server.admin_key_hash.clone() };
        if let (Some(want), Some(got)) = (
            admin_hash,
            req.headers()
                .get("x-mm-admin")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string()),
        ) {
            if crate::auth::ct_eq(&crate::auth::sha256_hex(got.as_bytes()), &want) {
                authed = true;
            }
        }
    }

    if !authed {
        if let Some(token) = cookie(&req, "mm_session") {
            authed = state.auth.check_session(&token);
        }
    }

    if !authed {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "login required"})),
        )
            .into_response();
    }

    // Log every mutating call (who + what) so unexpected loads/unloads are
    // traceable to a client.
    if req.method() != axum::http::Method::GET {
        let ua = req
            .headers()
            .get("user-agent")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        tracing::info!(
            "API {} {} from {} ua=\"{}\"",
            req.method(),
            req.uri().path(),
            peer_ip,
            ua
        );
    }
    next.run(req).await
}

/// If an Origin header is present, its host[:port] must equal the Host header.
/// Absent Origin (curl, scripts) is fine — no cookies are sent cross-origin.
fn origin_ok(req: &axum::extract::Request) -> bool {
    let Some(origin) = req
        .headers()
        .get("origin")
        .and_then(|v| v.to_str().ok())
    else {
        return true;
    };
    let host = req
        .headers()
        .get("host")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    // Strip scheme, take host[:port].
    let o = origin
        .split("://")
        .nth(1)
        .unwrap_or(origin)
        .split('/')
        .next()
        .unwrap_or("");
    o == host
}

fn cookie(req: &axum::extract::Request, name: &str) -> Option<String> {
    let hdr = req.headers().get("cookie")?.to_str().ok()?;
    for part in hdr.split(';') {
        let part = part.trim();
        if let Some(v) = part.strip_prefix(&format!("{name}=")) {
            return Some(v.to_string());
        }
    }
    None
}

// ---------- login / logout ----------

#[derive(Deserialize)]
struct LoginBody {
    password: String,
    #[serde(default)]
    totp: String,
}

async fn login(
    State(state): State<SharedState>,
    axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<std::net::SocketAddr>,
    Json(body): Json<LoginBody>,
) -> Response {
    let ip = peer.ip();
    if state.auth.is_locked(ip) {
        tracing::warn!("login locked out: {ip}");
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"error": "too many failures; locked for 15 minutes"})),
        )
            .into_response();
    }
    let (pw_hash, totp_enc) = {
        let cfg = state.config.lock().await;
        (
            cfg.server.password_hash.clone(),
            cfg.server.totp_secret_enc.clone(),
        )
    };
    let Some(hash) = pw_hash else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "no password set — run `model-manager set-password` on the server"})),
        )
            .into_response();
    };
    let pw_ok = crate::auth::verify_password(&hash, &body.password);
    // TOTP only enforced once configured; before that password alone works so
    // first-run setup can complete from the dashboard.
    let totp_ok = match &totp_enc {
        None => true,
        Some(enc) => {
            let secret = crate::auth::secret_key()
                .and_then(|k| crate::auth::open(&k, enc))
                .unwrap_or_default();
            if secret.is_empty() {
                false
            } else {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                match crate::auth::verify_totp(&secret, &body.totp, now) {
                    Some(step) => state.auth.claim_totp_step(step),
                    None => false,
                }
            }
        }
    };
    if !pw_ok || !totp_ok {
        let backoff = state.auth.record(ip, false);
        tracing::warn!("login failed from {ip}");
        if backoff > 0 {
            tokio::time::sleep(Duration::from_millis(backoff)).await;
        }
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "invalid password or code"})),
        )
            .into_response();
    }
    state.auth.record(ip, true);
    let token = state.auth.new_session();
    tracing::info!("login ok from {ip}");
    (
        [(
            axum::http::header::SET_COOKIE,
            format!(
                "mm_session={token}; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=28800"
            ),
        )],
        Json(json!({"ok": true})),
    )
        .into_response()
}

async fn logout(State(state): State<SharedState>, req: axum::extract::Request) -> Response {
    if let Some(t) = cookie(&req, "mm_session") {
        state.auth.drop_session(&t);
    }
    (
        [(
            axum::http::header::SET_COOKIE,
            "mm_session=; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=0".to_string(),
        )],
        Json(json!({"ok": true})),
    )
        .into_response()
}

// ----- response models -----

#[derive(Serialize)]
struct ModelView {
    def: ModelDef,
    running: bool,
    state: String,
    status: String,
    healthy: bool,
    estimate: MemEstimate,
    oom: OomVerdict,
    /// Human-readable load stage while a model is starting (None once healthy).
    phase: Option<String>,
    /// Approximate load progress 0–99 (from memory allocated so far).
    load_pct: Option<u32>,
    /// Native max context from GGUF / config.json, if known. UI uses this to
    /// hide context-window chips the checkpoint cannot serve.
    max_context: Option<u32>,
}

/// Determine the current load stage of a starting container by scanning its
/// recent logs for engine-specific markers.
async fn load_phase(docker: &Docker, def: &ModelDef) -> String {
    let logs = docker::logs(docker, def, 80).await.unwrap_or_default();
    let l = logs.to_lowercase();
    let has = |s: &str| l.contains(s);
    match def.engine {
        crate::config::Engine::Llamacpp => {
            if has("server is listening") {
                "Starting API server".into()
            } else if has("cuda0 model buffer") || has("offloaded") {
                // pull "offloaded N/M layers" if present
                if let Some(p) = logs.split("offloaded ").nth(1) {
                    if let Some(frac) = p.split_whitespace().next() {
                        return format!("Offloading to GPU ({frac})");
                    }
                }
                "Loading weights to GPU".into()
            } else if has("loading model tensors") {
                "Loading tensors".into()
            } else if has("llama_model_loader") {
                "Reading model metadata".into()
            } else {
                "Starting…".into()
            }
        }
        crate::config::Engine::Vllm => {
            if has("application startup complete") || has("uvicorn running") {
                "Starting API server".into()
            } else if has("model loaded") || has("loading model weights took") {
                "Model loaded".into()
            } else if has("loading weights") || has("loading model") {
                "Loading weights".into()
            } else if has("initializing") || has("started engine") {
                "Initializing engine".into()
            } else {
                "Starting…".into()
            }
        }
        crate::config::Engine::Nemo => {
            if has("application startup complete") || has("uvicorn running") {
                "Starting API server".into()
            } else if has("diarization model ready") {
                "Diarization ready".into()
            } else if has("loading diarization model") {
                "Loading diarization model".into()
            } else if has("asr model ready") {
                "ASR ready".into()
            } else if has("loading asr model") {
                "Loading ASR model".into()
            } else if has("importing nemo") {
                "Importing NeMo".into()
            } else {
                "Starting…".into()
            }
        }
        crate::config::Engine::Audiogen => {
            if has("application startup complete") || has("uvicorn running") || has("serving on") {
                "Starting API server".into()
            } else if has("pipeline ready") || has("model ready") {
                "Model ready".into()
            } else if has("loading components") || has("loading pipeline") {
                "Loading pipeline".into()
            } else if has("loading model") {
                "Loading weights".into()
            } else {
                "Starting…".into()
            }
        }
        crate::config::Engine::Trellis => {
            if has("application startup complete") || has("uvicorn running") || has("serving on") {
                "Starting API server".into()
            } else if has("pipeline ready") || has("model ready") {
                "Model ready".into()
            } else if has("loading pipeline") || has("from_pretrained") {
                "Loading TRELLIS.2 pipeline".into()
            } else if has("loading model") {
                "Loading weights".into()
            } else {
                "Starting…".into()
            }
        }
    }
}

#[derive(Serialize)]
struct StateView {
    memory: MemSnapshot,
    docker_ok: bool,
    models: Vec<ModelView>,
    cursor: serde_json::Value,
}

async fn snapshot(state: &SharedState) -> MemSnapshot {
    let mut sys = state.sys.lock().await;
    memory::snapshot(&mut sys)
}

async fn model_healthy(state: &SharedState, port: u16) -> bool {
    let url = format!("http://127.0.0.1:{port}/health");
    matches!(
        // SGLang /health runs a dummy generate (~1s with sleep-on-idle).
        state.http.get(&url).timeout(Duration::from_millis(2500)).send().await,
        Ok(r) if r.status().is_success()
    )
}

async fn get_state(State(state): State<SharedState>) -> impl IntoResponse {
    let mem = snapshot(&state).await;
    let docker_ok = docker::ping(&state.docker).await;

    let (models, overhead, safety) = {
        let cfg = state.config.lock().await;
        (
            cfg.models.clone(),
            cfg.server.overhead_mib,
            cfg.server.safety_margin_mib,
        )
    };

    let statuses = docker::list_managed(&state.docker).await.unwrap_or_default();
    let by_name: HashMap<String, docker::ContainerStatus> = statuses
        .into_iter()
        .map(|s| (s.model_name.clone(), s))
        .collect();

    let mut views = Vec::new();
    for def in models {
        let st = by_name.get(&crate::config::sanitize(&def.name));
        let running = st.map(|s| s.running).unwrap_or(false);
        let estimate = memory::estimate(&def, overhead, mem.total_mib);
        let oom = memory::oom_check(estimate.total_mib, &mem, safety);
        let healthy = if running {
            model_healthy(&state, def.host_port).await
        } else {
            false
        };

        // While a model is up but not yet answering /health, report its stage
        // and an approximate progress % from how much memory has filled since
        // the load began.
        let (phase, load_pct) = if running && !healthy {
            let ph = load_phase(&state.docker, &def).await;
            let pct = {
                let lg = state.loading.lock().await;
                lg.get(&def.name).map(|li| {
                    let grown = mem.used_mib.saturating_sub(li.baseline_used_mib);
                    ((grown as f64 / li.estimate_mib.max(1) as f64) * 100.0).clamp(0.0, 99.0)
                        as u32
                })
            };
            (Some(ph), pct)
        } else {
            (None, None)
        };

        let max_context = if def.uses_context() {
            crate::gguf::native_context_len(std::path::Path::new(&def.model_path), def.engine)
        } else {
            None
        };

        views.push(ModelView {
            running,
            state: st.map(|s| s.state.clone()).unwrap_or_else(|| "absent".into()),
            status: st.map(|s| s.status.clone()).unwrap_or_default(),
            healthy,
            estimate,
            oom,
            phase,
            load_pct,
            max_context,
            def,
        });
    }

    let cursor = {
        let cfg = state.config.lock().await;
        let ctx = views
            .iter()
            .find(|v| v.def.kind == crate::config::ModelKind::Llm && v.running && v.healthy)
            .map(|v| v.def.context as u64)
            .unwrap_or(262144);
        crate::gateway::snapshot(
            cfg.gateway.enabled,
            cfg.gateway.port,
            &cfg.gateway.bind,
            &cfg.gateway.model_alias,
            &cfg.gateway.keys,
            ctx,
        )
    };

    Json(StateView {
        memory: mem,
        docker_ok,
        models: views,
        cursor,
    })
}

async fn get_gateway(State(state): State<SharedState>) -> impl IntoResponse {
    let (enabled, port, bind, alias, keys, ctx) = {
        let cfg = state.config.lock().await;
        let ctx = cfg
            .models
            .iter()
            .find(|m| m.kind == crate::config::ModelKind::Llm)
            .map(|m| m.context as u64)
            .unwrap_or(262144);
        (
            cfg.gateway.enabled,
            cfg.gateway.port,
            cfg.gateway.bind.clone(),
            cfg.gateway.model_alias.clone(),
            cfg.gateway.keys.clone(),
            ctx,
        )
    };
    Json(crate::gateway::snapshot(enabled, port, &bind, &alias, &keys, ctx))
}

async fn post_gateway(
    State(state): State<SharedState>,
    Json(body): Json<crate::gateway::GatewayToggle>,
) -> impl IntoResponse {
    match crate::gateway::set_enabled(state, body.enabled).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e})),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
struct UpsertBody {
    #[serde(flatten)]
    model: ModelDef,
}

async fn upsert_model(
    State(state): State<SharedState>,
    Json(body): Json<UpsertBody>,
) -> impl IntoResponse {
    if body.model.name.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "name required"}))).into_response();
    }
    let mut model = body.model;
    model.sync_context_into_launch();
    let mut cfg = state.config.lock().await;
    cfg.upsert(model);
    if let Err(e) = cfg.save() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("save failed: {e}")})),
        )
            .into_response();
    }
    (StatusCode::OK, Json(json!({"ok": true}))).into_response()
}

async fn delete_model(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    // Unload first if it exists, then drop from config.
    let def = { state.config.lock().await.find(&name).cloned() };
    if let Some(def) = def {
        let _ = docker::unload(&state.docker, &def).await;
    }
    let mut cfg = state.config.lock().await;
    let removed = cfg.remove(&name);
    let _ = cfg.save();
    Json(json!({"ok": true, "removed": removed}))
}

async fn estimate_model(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let (def, overhead, safety) = {
        let cfg = state.config.lock().await;
        match cfg.find(&name) {
            Some(d) => (d.clone(), cfg.server.overhead_mib, cfg.server.safety_margin_mib),
            None => {
                return (StatusCode::NOT_FOUND, Json(json!({"error": "unknown model"})))
                    .into_response()
            }
        }
    };
    let mem = snapshot(&state).await;
    let estimate = memory::estimate(&def, overhead, mem.total_mib);
    let oom = memory::oom_check(estimate.total_mib, &mem, safety);
    Json(json!({"estimate": estimate, "oom": oom, "memory": mem})).into_response()
}

#[derive(Deserialize)]
struct LoadQuery {
    #[serde(default)]
    force: bool,
}

async fn load_model(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    Query(q): Query<LoadQuery>,
) -> impl IntoResponse {
    let (def, overhead, safety) = {
        let cfg = state.config.lock().await;
        match cfg.find(&name) {
            Some(d) => (d.clone(), cfg.server.overhead_mib, cfg.server.safety_margin_mib),
            None => {
                return (StatusCode::NOT_FOUND, Json(json!({"error": "unknown model"})))
                    .into_response()
            }
        }
    };

    let baseline = snapshot(&state).await;
    let estimate = memory::estimate(&def, overhead, baseline.total_mib);
    let oom = memory::oom_check(estimate.total_mib, &baseline, safety);

    if oom.would_oom && !q.force {
        // Refuse and report why. Client can retry with ?force=true.
        return (
            StatusCode::CONFLICT,
            Json(json!({
                "error": "would_oom",
                "message": format!(
                    "Loading '{}' needs ~{} MiB but only {} MiB is available (safety margin {} MiB). This would likely OOM.",
                    def.name, oom.needed_mib, oom.available_mib, oom.safety_margin_mib
                ),
                "estimate": estimate,
                "oom": oom,
            })),
        )
            .into_response();
    }

    if let Err(e) = docker::load(&state.docker, &def).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("load failed: {e}")})),
        )
            .into_response();
    }

    // Record the starting point so /api/state can report live load progress.
    {
        let mut lg = state.loading.lock().await;
        lg.insert(
            def.name.clone(),
            LoadInfo {
                baseline_used_mib: baseline.used_mib,
                estimate_mib: estimate.total_mib,
            },
        );
    }

    // Measure the real footprint in the background once the model is healthy,
    // then persist it so future OOM checks use the measured value.
    spawn_measure(state.clone(), def.clone(), baseline.available_mib);

    (
        StatusCode::OK,
        Json(json!({"ok": true, "estimate": estimate, "oom": oom, "forced": q.force})),
    )
        .into_response()
}

/// Background task: wait for the model to become healthy, let memory settle,
/// then record the whole-system memory drop as this model's measured peak.
fn spawn_measure(state: SharedState, def: ModelDef, baseline_available_mib: u64) {
    tokio::spawn(async move {
        // Wait up to ~15 minutes for /health (big models mmap slowly).
        let mut healthy = false;
        for _ in 0..180 {
            if model_healthy(&state, def.host_port).await {
                healthy = true;
                break;
            }
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
        // Load finished (healthy) or gave up — stop reporting progress either way.
        {
            state.loading.lock().await.remove(&def.name);
        }
        if !healthy {
            return;
        }
        // Let allocations settle.
        tokio::time::sleep(Duration::from_secs(10)).await;
        let now = snapshot(&state).await;
        let footprint = baseline_available_mib.saturating_sub(now.available_mib);
        if footprint == 0 {
            return;
        }
        let mut cfg = state.config.lock().await;
        if let Some(m) = cfg.find_mut(&def.name) {
            // Keep the larger of any prior measurement and this one.
            let new_val = match m.measured_peak_mib {
                Some(prev) => prev.max(footprint),
                None => footprint,
            };
            m.measured_peak_mib = Some(new_val);
            let _ = cfg.save();
            tracing::info!("measured {} footprint = {} MiB", def.name, new_val);
        }
    });
}

async fn unload_model(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let def = { state.config.lock().await.find(&name).cloned() };
    let def = match def {
        Some(d) => d,
        None => {
            return (StatusCode::NOT_FOUND, Json(json!({"error": "unknown model"})))
                .into_response()
        }
    };
    state.loading.lock().await.remove(&name);
    match docker::unload(&state.docker, &def).await {
        Ok(_) => Json(json!({"ok": true})).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("unload failed: {e}")})),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
struct AutostartBody {
    enabled: bool,
}

async fn set_autostart(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    Json(body): Json<AutostartBody>,
) -> impl IntoResponse {
    // Persist the flag. If the container is currently running, recreate it so
    // the Docker restart policy takes effect immediately.
    let def = {
        let mut cfg = state.config.lock().await;
        match cfg.find_mut(&name) {
            Some(m) => {
                m.autostart = body.enabled;
                let d = m.clone();
                let _ = cfg.save();
                d
            }
            None => {
                return (StatusCode::NOT_FOUND, Json(json!({"error": "unknown model"})))
                    .into_response()
            }
        }
    };

    // Apply the new policy to the existing container live (no reload). This
    // works whether the container is running or stopped.
    let exists = docker::status_of(&state.docker, &def)
        .await
        .ok()
        .flatten()
        .is_some();
    if exists {
        if let Err(e) = docker::set_restart_policy(&state.docker, &def, body.enabled).await {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("could not update restart policy: {e}")})),
            )
                .into_response();
        }
    }
    Json(json!({"ok": true, "autostart": body.enabled, "applied": exists})).into_response()
}

#[derive(Deserialize)]
struct ContextBody {
    context: u32,
    /// If true and the model is running, unload and reload so the new window
    /// takes effect. If the model is stopped, the value is just persisted.
    #[serde(default)]
    reload: bool,
    #[serde(default)]
    force: bool,
}

async fn set_context(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    Json(body): Json<ContextBody>,
) -> impl IntoResponse {
    if body.context == 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "context must be > 0"})),
        )
            .into_response();
    }

    let (def, overhead, safety, old_ctx) = {
        let mut cfg = state.config.lock().await;
        match cfg.find_mut(&name) {
            Some(m) => {
                if !m.uses_context() {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({"error": "this model has no selectable context window"})),
                    )
                        .into_response();
                }
                let old = m.context;
                m.context = body.context;
                // llama.cpp KV grows with context; drop a stale measurement so
                // the next load re-estimates. vLLM sizes KV from gpu_mem_util.
                if m.engine == crate::config::Engine::Llamacpp && body.context > old {
                    m.measured_peak_mib = None;
                }
                m.sync_context_into_launch();
                let d = m.clone();
                if let Err(e) = cfg.save() {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({"error": format!("save failed: {e}")})),
                    )
                        .into_response();
                }
                (
                    d,
                    cfg.server.overhead_mib,
                    cfg.server.safety_margin_mib,
                    old,
                )
            }
            None => {
                return (StatusCode::NOT_FOUND, Json(json!({"error": "unknown model"})))
                    .into_response()
            }
        }
    };

    let running = docker::status_of(&state.docker, &def)
        .await
        .ok()
        .flatten()
        .map(|s| s.running)
        .unwrap_or(false);

    if !body.reload || !running {
        return Json(json!({
            "ok": true,
            "context": def.context,
            "previous": old_ctx,
            "reloaded": false,
            "running": running,
        }))
        .into_response();
    }

    // Free this model's reservation before the OOM check / relaunch.
    state.loading.lock().await.remove(&name);
    if let Err(e) = docker::unload(&state.docker, &def).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("unload failed: {e}")})),
        )
            .into_response();
    }

    let baseline = snapshot(&state).await;
    let estimate = memory::estimate(&def, overhead, baseline.total_mib);
    let oom = memory::oom_check(estimate.total_mib, &baseline, safety);
    if oom.would_oom && !body.force {
        return (
            StatusCode::CONFLICT,
            Json(json!({
                "error": "would_oom",
                "message": format!(
                    "Reloading '{}' at {} ctx needs ~{} MiB but only {} MiB is available. Model was unloaded.",
                    def.name, def.context, oom.needed_mib, oom.available_mib
                ),
                "estimate": estimate,
                "oom": oom,
                "context": def.context,
                "unloaded": true,
            })),
        )
            .into_response();
    }

    if let Err(e) = docker::load(&state.docker, &def).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("reload failed: {e}")})),
        )
            .into_response();
    }

    {
        let mut lg = state.loading.lock().await;
        lg.insert(
            def.name.clone(),
            LoadInfo {
                baseline_used_mib: baseline.used_mib,
                estimate_mib: estimate.total_mib,
            },
        );
    }
    spawn_measure(state.clone(), def.clone(), baseline.available_mib);

    (
        StatusCode::OK,
        Json(json!({
            "ok": true,
            "context": def.context,
            "previous": old_ctx,
            "reloaded": true,
            "estimate": estimate,
            "forced": body.force,
        })),
    )
        .into_response()
}

async fn model_logs(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let def = { state.config.lock().await.find(&name).cloned() };
    let def = match def {
        Some(d) => d,
        None => return (StatusCode::NOT_FOUND, Json(json!({"error": "unknown model"}))).into_response(),
    };
    match docker::logs(&state.docker, &def, 200).await {
        Ok(text) => Json(json!({"logs": text})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use tower::ServiceExt;

    fn state() -> SharedState {
        Arc::new(AppState {
            config: Mutex::new(Config::default()),
            docker: crate::docker::connect().expect("docker client"),
            sys: Mutex::new(System::new()),
            http: reqwest::Client::new(),
            loading: Mutex::new(HashMap::new()),
            auth: crate::auth::AuthState::default(),
            gateway: Arc::new(crate::gateway::GatewayShared::default()),
        })
    }

    fn req(uri: &str, method: &str) -> Request<Body> {
        let mut r = Request::builder()
            .uri(uri)
            .method(method)
            .body(Body::empty())
            .unwrap();
        r.extensions_mut().insert(axum::extract::ConnectInfo(
            SocketAddr::from((Ipv4Addr::LOCALHOST, 9)),
        ));
        r
    }

    #[tokio::test]
    async fn no_cookie_401_and_query_token_ignored() {
        let st = state();
        let app = router(st);
        let r = app
            .clone()
            .oneshot(req("/api/state", "GET"))
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::UNAUTHORIZED);
        // The old ?token= path is gone — even a valid-looking query gets 401.
        let r = app
            .oneshot(req("/api/state?token=whatever", "GET"))
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn session_cookie_works() {
        let st = state();
        let tok = st.auth.new_session();
        let app = router(st);
        let mut r = req("/api/state", "GET");
        r.headers_mut().insert(
            "cookie",
            format!("mm_session={tok}").parse().unwrap(),
        );
        let resp = app.oneshot(r).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn login_503_without_password() {
        let st = state();
        let app = router(st);
        let mut r = req("/api/login", "POST");
        r.headers_mut()
            .insert("content-type", "application/json".parse().unwrap());
        let mut r = r;
        *r.body_mut() = Body::from(r#"{"password":"x","totp":"000000"}"#);
        let resp = app.oneshot(r).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn origin_mismatch_403_on_post() {
        let st = state();
        let tok = st.auth.new_session();
        let app = router(st);
        let mut r = req("/api/models/x/unload", "POST");
        r.headers_mut()
            .insert("cookie", format!("mm_session={tok}").parse().unwrap());
        r.headers_mut()
            .insert("origin", "https://evil.example".parse().unwrap());
        r.headers_mut().insert("host", "127.0.0.1:8600".parse().unwrap());
        r.headers_mut()
            .insert("content-type", "application/json".parse().unwrap());
        *r.body_mut() = Body::from("{}");
        let resp = app.clone().oneshot(r).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        // Matching origin passes the check (then 404s on unknown model).
        let mut r2 = req("/api/models/x/unload", "POST");
        r2.headers_mut()
            .insert("cookie", format!("mm_session={tok}").parse().unwrap());
        r2.headers_mut()
            .insert("origin", "https://127.0.0.1:8600".parse().unwrap());
        r2.headers_mut().insert("host", "127.0.0.1:8600".parse().unwrap());
        r2.headers_mut()
            .insert("content-type", "application/json".parse().unwrap());
        *r2.body_mut() = Body::from("{}");
        let resp2 = app.oneshot(r2).await.unwrap();
        assert_eq!(resp2.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn loopback_admin_key() {
        let st = state();
        let key = "mma_test-admin-key";
        {
            let mut cfg = st.config.lock().await;
            cfg.server.admin_key_hash = Some(crate::auth::sha256_hex(key.as_bytes()));
        }
        let app = router(st);
        let mut r = req("/api/state", "GET");
        r.headers_mut()
            .insert("x-mm-admin", key.parse().unwrap());
        let resp = app.clone().oneshot(r).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // Wrong key → 401.
        let mut r2 = req("/api/state", "GET");
        r2.headers_mut()
            .insert("x-mm-admin", "mma_wrong".parse().unwrap());
        let resp2 = app.oneshot(r2).await.unwrap();
        assert_eq!(resp2.status(), StatusCode::UNAUTHORIZED);
    }
}
