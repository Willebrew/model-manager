use anyhow::{Context, Result};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Prefix applied to every Docker container this tool manages, so we can
/// discover "our" containers and never touch unrelated ones.
pub const CONTAINER_PREFIX: &str = "modelmgr-";

/// Inference engine that serves a model. Both expose an OpenAI-compatible API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    /// llama.cpp `llama-server` (GGUF models).
    Llamacpp,
    /// vLLM OpenAI server (HuggingFace-format models, and some GGUF).
    Vllm,
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
