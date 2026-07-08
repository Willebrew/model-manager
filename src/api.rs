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
        .route("/api/models/:name/logs", get(model_logs))
        .layer(from_fn_with_state(state.clone(), auth));

    Router::new()
        .route("/", get(index))
        .route("/api/health", get(|| async { Json(json!({"ok": true})) }))
        .merge(protected)
        .with_state(state)
}

async fn index() -> impl IntoResponse {
    Html(include_str!("web/index.html"))
}

/// Bearer-token gate for every mutating/reading API call.
async fn auth(
    State(state): State<SharedState>,
    req: axum::extract::Request,
    next: Next,
) -> Response {
    let token = { state.config.lock().await.server.token.clone() };
    let provided = extract_token(&req);
    if provided.as_deref() == Some(token.as_str()) {
        next.run(req).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "invalid or missing access token"})),
        )
            .into_response()
    }
}

fn extract_token(req: &axum::extract::Request) -> Option<String> {
    // Authorization: Bearer <token>
    if let Some(h) = req.headers().get("authorization") {
        if let Ok(s) = h.to_str() {
            if let Some(t) = s.strip_prefix("Bearer ") {
                return Some(t.to_string());
            }
        }
    }
    // ?token=<token> fallback (handy for quick curl / links)
    if let Some(q) = req.uri().query() {
        for pair in q.split('&') {
            if let Some(v) = pair.strip_prefix("token=") {
                return Some(percent_decode(v));
            }
        }
    }
    None
}

/// Minimal application/x-www-form-urlencoded decode for query tokens, so a
/// token containing characters like `!` (`%21`) still matches.
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    let hex = |c: u8| -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    };
    while i < b.len() {
        match b[i] {
            b'%' if i + 2 < b.len() => match (hex(b[i + 1]), hex(b[i + 2])) {
                (Some(h), Some(l)) => {
                    out.push(h * 16 + l);
                    i += 3;
                }
                _ => {
                    out.push(b[i]);
                    i += 1;
                }
            },
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
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
    }
}

#[derive(Serialize)]
struct StateView {
    memory: MemSnapshot,
    docker_ok: bool,
    models: Vec<ModelView>,
}

async fn snapshot(state: &SharedState) -> MemSnapshot {
    let mut sys = state.sys.lock().await;
    memory::snapshot(&mut sys)
}

async fn model_healthy(state: &SharedState, port: u16) -> bool {
    let url = format!("http://127.0.0.1:{port}/health");
    matches!(
        state.http.get(&url).timeout(Duration::from_millis(800)).send().await,
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
        let estimate = memory::estimate(&def, overhead);
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

        views.push(ModelView {
            running,
            state: st.map(|s| s.state.clone()).unwrap_or_else(|| "absent".into()),
            status: st.map(|s| s.status.clone()).unwrap_or_default(),
            healthy,
            estimate,
            oom,
            phase,
            load_pct,
            def,
        });
    }

    Json(StateView {
        memory: mem,
        docker_ok,
        models: views,
    })
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
    let mut cfg = state.config.lock().await;
    cfg.upsert(body.model);
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
    let estimate = memory::estimate(&def, overhead);
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
    let estimate = memory::estimate(&def, overhead);
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
