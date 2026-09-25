use anyhow::{Context, Result};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Prefix applied to every Docker container this tool manages, so we can
/// discover "our" containers and never touch unrelated ones.
pub const CONTAINER_PREFIX: &str = "modelmgr-";

/// Inference engine that serves a model. All expose an OpenAI-compatible API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    /// llama.cpp `llama-server` (GGUF models).
    Llamacpp,
    /// vLLM OpenAI server (HuggingFace-format models, and some GGUF).
    Vllm,
    /// NVIDIA NeMo speech server (`.nemo` checkpoints) exposing
    /// `/v1/audio/transcriptions`. Used for ASR + speaker diarization.
    Nemo,
    /// Audio-generation server exposing OpenAI `/v1/audio/speech`.
    /// Used for text-to-music (MiniMax Music 3) and TTS-style models.
    Audiogen,
    /// Image-to-3D server (TRELLIS.2 and similar). Exposes POST `/v1/3d/generations`
    /// (image in, GLB out) plus GET `/health`.
    Trellis,
}

impl Default for Engine {
    fn default() -> Self {
        Engine::Llamacpp
    }
}

/// What a model is for. Purely a categorization for the UI + grouping; both
/// kinds are served the same way (OpenAI-compatible container).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    /// Text-generation / chat model.
    Llm,
    /// Embedding model (serves /v1/embeddings).
    Embedding,
    /// Speech model: transcription + optional speaker diarization
    /// (serves /v1/audio/transcriptions).
    Speech,
    /// Audio generation: text-to-music or TTS (serves /v1/audio/speech).
    Audio,
    /// Image-to-3D generation (serves POST /v1/3d/generations, returns GLB).
    Image3d,
}

impl Default for ModelKind {
    fn default() -> Self {
        ModelKind::Llm
    }
}

impl Engine {
    /// Default Docker image when the model doesn't specify one.
    pub fn default_image(&self) -> &'static str {
        match self {
            // No universal default llama.cpp image — user supplies their build.
            Engine::Llamacpp => "",
            Engine::Vllm => "vllm/vllm-openai:latest",
            // Built locally from docker/nemo-speech (no upstream arm64 image).
            Engine::Nemo => "nemo-speech-spark:latest",
            // Built locally from docker/music3 (Spark aarch64; no official arm64
            // sglang-omni image that we can rely on).
            Engine::Audiogen => "music3-spark:latest",
            Engine::Trellis => "trellis2-spark:latest",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Address to bind the dashboard to. Default 0.0.0.0 so it is reachable
    /// from other devices on the LAN (e.g. a Mac managing the Spark).
    #[serde(default = "default_bind")]
    pub bind: String,
    #[serde(default = "default_port")]
    pub port: u16,
    /// Serve over HTTPS with a self-signed certificate (auto-generated).
    #[serde(default = "default_true")]
    pub tls: bool,
    /// Optional custom cert/key PEM paths. If unset, a self-signed pair is
    /// generated once and stored next to the config.
    #[serde(default)]
    pub tls_cert_path: Option<String>,
    #[serde(default)]
    pub tls_key_path: Option<String>,
    /// Shared access token required by every mutating API call. Generated on
    /// first run. Anyone on the LAN can reach the port, so this is what stops
    /// a random device from starting/stopping models.
    #[serde(default = "gen_token")]
    pub token: String,
    /// Fixed memory overhead (MiB) to add on top of weights+KV when estimating
    /// a model's footprint: CUDA context, compute buffers, fragmentation.
    #[serde(default = "default_overhead_mib")]
    pub overhead_mib: u64,
    /// Safety margin (MiB) kept free. A load is flagged as OOM-risky if the
    /// estimate would leave less than this much headroom.
    #[serde(default = "default_safety_mib")]
    pub safety_margin_mib: u64,
}

/// Removable Cursor BYOK adapter (`src/cursor.rs`). Off by default; omitted
/// from the saved config while disabled. Delete this struct + `src/cursor.rs`
/// + the UI "Cursor" bar to rip the feature out.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Plain HTTP port the Cursor adapter listens on (Tailscale Funnel targets this).
    #[serde(default = "default_cursor_port")]
    pub port: u16,
    /// Stable model id Cursor should add. Rewritten to whichever LLM is loaded.
    #[serde(default = "default_cursor_alias")]
    pub model_alias: String,
}

impl Default for CursorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            port: default_cursor_port(),
            model_alias: default_cursor_alias(),
        }
    }
}

impl CursorConfig {
    pub fn is_off(&self) -> bool {
        !self.enabled
    }
}

fn default_cursor_port() -> u16 {
    8610
}
fn default_cursor_alias() -> String {
    "spark".to_string()
}

fn default_bind() -> String {
    "0.0.0.0".to_string()
}
fn default_port() -> u16 {
    8600
}
fn default_true() -> bool {
    true
}
fn default_overhead_mib() -> u64 {
    2560
}
fn default_safety_mib() -> u64 {
    2048
}

pub fn gen_token() -> String {
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut rng = rand::thread_rng();
    (0..40)
        .map(|_| CHARS[rng.gen_range(0..CHARS.len())] as char)
        .collect()
}

/// A model the user has registered. Everything needed to launch a server
/// container for it, on either engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDef {
    /// Unique, human-friendly id (also becomes the served model id clients use).
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Optional display family. Models that share a family render as one card
    /// with variant chips (e.g. Qwen 3.8 NVFP4 DFlash2 vs FP8 vs Uncensored).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub family: String,
    /// Chip label inside a family. Empty → use `name`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub variant: String,
    /// LLM or embedding model (for UI grouping).
    #[serde(default)]
    pub kind: ModelKind,
    #[serde(default)]
    pub engine: Engine,
    /// Path to the model on this host. For llama.cpp: the first GGUF shard (or a
    /// single GGUF). For vLLM: a HuggingFace model directory (or GGUF / HF id).
    #[serde(alias = "gguf_path")]
    pub model_path: String,
    /// Docker image. Empty → engine default (`vllm/vllm-openai:latest` for vLLM).
    #[serde(default)]
    pub image: String,
    /// Host port the OpenAI-compatible API listens on.
    pub host_port: u16,
    #[serde(default = "default_context")]
    pub context: u32,
    // ---- llama.cpp-specific ----
    #[serde(default = "default_ngl")]
    pub ngl: u32,
    #[serde(default = "default_kv")]
    pub kv_type: String,
    #[serde(default = "default_threads")]
    pub threads: u32,
    // ---- vLLM-specific ----
    /// Fraction of GPU/UMA memory vLLM may use (default 0.90 if unset).
    #[serde(default)]
    pub gpu_mem_util: Option<f32>,
    /// Extra raw engine flags.
    #[serde(default)]
    pub extra_args: Vec<String>,
    /// Extra environment variables ("KEY=VALUE") for the container. Needed for
    /// images that configure themselves from the environment.
    #[serde(default)]
    pub env: Vec<String>,
    /// Extra bind mounts ("host:container[:ro]"), beyond the default /model mount.
    #[serde(default)]
    pub mounts: Vec<String>,
    /// If true, don't override the image's command — the image's own entrypoint
    /// configures itself (typically from env vars) and no default /model mount
    /// is added. Used for custom vLLM images like the DFlash entrypoint.
    #[serde(default)]
    pub use_image_entrypoint: bool,
    /// Optional Docker ENTRYPOINT override (argv). Empty = image default.
    /// Used by recipe images that wrap the vendor entrypoint (e.g. Mia DS4).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entrypoint: Vec<String>,
    /// Start automatically on boot (implemented via Docker restart policy).
    #[serde(default)]
    pub autostart: bool,
    /// Measured whole-system memory delta (MiB) from the model's first
    /// successful load. Once set, this is the authoritative OOM number.
    #[serde(default)]
    pub measured_peak_mib: Option<u64>,
}

fn default_context() -> u32 {
    8192
}
fn default_ngl() -> u32 {
    99
}
fn default_kv() -> String {
    "f16".to_string()
}
fn default_threads() -> u32 {
    num_cpus_fallback()
}
fn num_cpus_fallback() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(8)
}

impl ModelDef {
    /// Deterministic Docker container name for this model.
    pub fn container_name(&self) -> String {
        format!("{}{}", CONTAINER_PREFIX, sanitize(&self.name))
    }

    /// Effective Docker image (falls back to the engine default).
    pub fn effective_image(&self) -> String {
        if self.image.trim().is_empty() {
            self.engine.default_image().to_string()
        } else {
            self.image.clone()
        }
    }

    pub fn gpu_mem_util_or_default(&self) -> f32 {
        self.gpu_mem_util.unwrap_or(0.90)
    }

    /// LLMs and embedding models have a selectable context window. Speech and
    /// audio-gen servers do not.
    pub fn uses_context(&self) -> bool {
        matches!(self.kind, ModelKind::Llm | ModelKind::Embedding)
            && matches!(self.engine, Engine::Llamacpp | Engine::Vllm)
    }

    /// Rewrite extra_args / env so the engine actually serves `self.context`.
    ///
    /// Many Spark images take a full `vllm serve … --max-model-len N` (or
    /// SGLang `--context-length`, or `MAX_MODEL_LEN=N`) as extra_args/env.
    /// The dashboard `context` field was previously ignored for those.
    pub fn sync_context_into_launch(&mut self) {
        if !self.uses_context() {
            return;
        }
        rewrite_context_args(&mut self.extra_args, self.context, self.engine);
        rewrite_context_env(&mut self.env, self.context, self.engine);
    }

    /// Build the container command (argv appended to the image entrypoint).
    ///
    /// - llama.cpp: we invoke the binary explicitly.
    /// - vLLM: the image entrypoint is the OpenAI server, so we pass only flags.
    ///
    /// `model_ref` is the path *inside* the container.
    pub fn container_cmd(&self, model_ref: &str) -> Vec<String> {
        match self.engine {
            Engine::Llamacpp => {
                let mut a = vec![
                    crate::docker::LLAMA_SERVER_BIN.to_string(),
                    "-a".into(),
                    self.name.clone(),
                    "-m".into(),
                    model_ref.into(),
                    "--host".into(),
                    "0.0.0.0".into(),
                    "--port".into(),
                    self.host_port.to_string(),
                    "-c".into(),
                    self.context.to_string(),
                    "-ngl".into(),
                    self.ngl.to_string(),
                    "-t".into(),
                    self.threads.to_string(),
                    "-ctk".into(),
                    self.kv_type.clone(),
                    "-ctv".into(),
                    self.kv_type.clone(),
                ];
                a.extend(self.extra_args.iter().cloned());
                a
            }
            Engine::Vllm => {
                let mut a = vec![
                    "--model".into(),
                    model_ref.into(),
                    "--served-model-name".into(),
                    self.name.clone(),
                    "--host".into(),
                    "0.0.0.0".into(),
                    "--port".into(),
                    self.host_port.to_string(),
                    "--max-model-len".into(),
                    self.context.to_string(),
                    "--gpu-memory-utilization".into(),
                    format!("{:.2}", self.gpu_mem_util_or_default()),
                ];
                a.extend(self.extra_args.iter().cloned());
                a
            }
            // Our speech server takes a directory of `.nemo` checkpoints and
            // auto-detects the ASR and diarization models inside it; extra_args
            // can pin them explicitly (--asr / --diar).
            Engine::Nemo => {
                let mut a = vec![
                    "--host".into(),
                    "0.0.0.0".into(),
                    "--port".into(),
                    self.host_port.to_string(),
                    "--model-dir".into(),
                    model_ref.into(),
                    "--served-model-name".into(),
                    self.name.clone(),
                ];
                a.extend(self.extra_args.iter().cloned());
                a
            }
            // Audio-gen server (MiniMax Music 3 / similar) loads a HF directory
            // and exposes POST /v1/audio/speech.
            Engine::Audiogen => {
                let mut a = vec![
                    "--host".into(),
                    "0.0.0.0".into(),
                    "--port".into(),
                    self.host_port.to_string(),
                    "--model-dir".into(),
                    model_ref.into(),
                    "--served-model-name".into(),
                    self.name.clone(),
                ];
                a.extend(self.extra_args.iter().cloned());
                a
            }
            Engine::Trellis => {
                let mut a = vec![
                    "--host".into(),
                    "0.0.0.0".into(),
                    "--port".into(),
                    self.host_port.to_string(),
                    "--model-dir".into(),
                    model_ref.into(),
                    "--served-model-name".into(),
                    self.name.clone(),
                ];
                a.extend(self.extra_args.iter().cloned());
                a
            }
        }
    }
}

fn is_context_flag(flag: &str, engine: Engine) -> bool {
    match engine {
        Engine::Llamacpp => matches!(flag, "-c" | "--ctx-size" | "--ctx_size"),
        Engine::Vllm => matches!(
            flag,
            "--max-model-len"
                | "--max_model_len"
                | "--context-length"
                | "--context_length"
        ),
        Engine::Nemo | Engine::Audiogen | Engine::Trellis => false,
    }
}

/// Replace the value of any context-length flag in a tokenized argv.
fn rewrite_context_args(args: &mut Vec<String>, context: u32, engine: Engine) {
    let val = context.to_string();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].clone();
        if is_context_flag(&a, engine) {
            if i + 1 < args.len() {
                args[i + 1] = val.clone();
                i += 2;
                continue;
            }
        } else if let Some((flag, _)) = a.split_once('=') {
            if is_context_flag(flag, engine) {
                args[i] = format!("{flag}={val}");
            }
        }
        i += 1;
    }
}

/// Replace MAX_MODEL_LEN-style env vars, and rewrite context flags inside
/// EXTRA_ARGS / EXTRA_VLLM_ARGS blobs.
fn rewrite_context_env(env: &mut [String], context: u32, engine: Engine) {
    let val = context.to_string();
    for e in env.iter_mut() {
        let Some((k, rest)) = e.split_once('=') else {
            continue;
        };
        if matches!(
            k,
            "MAX_MODEL_LEN" | "MAX_CONTEXT_LEN" | "CONTEXT_LENGTH" | "MAX_MODEL_LENGTH"
        ) {
            *e = format!("{k}={val}");
        } else if matches!(k, "EXTRA_ARGS" | "EXTRA_VLLM_ARGS") {
            let mut tokens: Vec<String> = rest.split_whitespace().map(|s| s.to_string()).collect();
            rewrite_context_args(&mut tokens, context, engine);
            *e = format!("{k}={}", tokens.join(" "));
        }
    }
}

/// Turn an arbitrary model name into a safe container/dns fragment.
pub fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_server")]
    pub server: ServerConfig,
    #[serde(default)]
    pub models: Vec<ModelDef>,
    /// Removable. See `CursorConfig`.
    #[serde(default, skip_serializing_if = "CursorConfig::is_off")]
    pub cursor: CursorConfig,
}

fn default_server() -> ServerConfig {
    ServerConfig {
        bind: default_bind(),
        port: default_port(),
        tls: true,
        tls_cert_path: None,
        tls_key_path: None,
        token: gen_token(),
        overhead_mib: default_overhead_mib(),
        safety_margin_mib: default_safety_mib(),
    }
}

impl Default for Config {
    fn default() -> Self {
        Config {
            server: default_server(),
            models: Vec::new(),
            cursor: CursorConfig::default(),
        }
    }
}

impl Config {
    pub fn path() -> PathBuf {
        if let Ok(p) = std::env::var("MODEL_MANAGER_CONFIG") {
            return PathBuf::from(p);
        }
        let base = dirs::config_dir().unwrap_or_else(|| PathBuf::from("."));
        base.join("model-manager").join("config.toml")
    }

    pub fn dir() -> PathBuf {
        Self::path()
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."))
    }

    /// Load config, creating a default (with a fresh token) if none exists.
    pub fn load_or_init() -> Result<Config> {
        let path = Self::path();
        if path.exists() {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading config {}", path.display()))?;
            let cfg: Config = toml::from_str(&text)
                .with_context(|| format!("parsing config {}", path.display()))?;
            Ok(cfg)
        } else {
            let cfg = Config::default();
            cfg.save()?;
            Ok(cfg)
        }
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::path();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("creating config dir {}", dir.display()))?;
        }
        let text = toml::to_string_pretty(self).context("serializing config")?;
        std::fs::write(&path, text).with_context(|| format!("writing config {}", path.display()))?;
        Ok(())
    }

    pub fn find(&self, name: &str) -> Option<&ModelDef> {
        self.models.iter().find(|m| m.name == name)
    }

    pub fn find_mut(&mut self, name: &str) -> Option<&mut ModelDef> {
        self.models.iter_mut().find(|m| m.name == name)
    }

    pub fn upsert(&mut self, model: ModelDef) {
        if let Some(existing) = self.find_mut(&model.name) {
            *existing = model;
        } else {
            self.models.push(model);
        }
    }

    pub fn remove(&mut self, name: &str) -> bool {
        let before = self.models.len();
        self.models.retain(|m| m.name != name);
        self.models.len() != before
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrite_vllm_pair() {
        let mut a = vec![
            "vllm".into(),
            "serve".into(),
            "--max-model-len".into(),
            "262144".into(),
            "--kv-cache-dtype".into(),
            "fp8".into(),
        ];
        rewrite_context_args(&mut a, 1_048_576, Engine::Vllm);
        assert_eq!(a[3], "1048576");
        assert_eq!(a[5], "fp8");
    }

    #[test]
    fn rewrite_sglang_eq() {
        let mut a = vec!["--context-length=262144".into(), "--tp-size".into(), "1".into()];
        rewrite_context_args(&mut a, 8192, Engine::Vllm);
        assert_eq!(a[0], "--context-length=8192");
    }

    #[test]
    fn rewrite_env_and_blob() {
        let mut e = vec![
            "MAX_MODEL_LEN=262144".into(),
            "FOO=bar".into(),
            "EXTRA_VLLM_ARGS=--tool-call-parser qwen3_xml --max-model-len 8192".into(),
        ];
        rewrite_context_env(&mut e, 1_048_576, Engine::Vllm);
        assert_eq!(e[0], "MAX_MODEL_LEN=1048576");
        assert_eq!(e[1], "FOO=bar");
        assert!(e[2].contains("--max-model-len 1048576"));
    }
}
