use anyhow::{Context, Result};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Prefix applied to every Docker container this tool manages, so we can
/// discover "our" containers and never touch unrelated ones.
pub const CONTAINER_PREFIX: &str = "modelmgr-";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Address to bind the dashboard to. Default 0.0.0.0 so it is reachable
    /// from other devices on the LAN (e.g. a Mac managing the Spark).
    #[serde(default = "default_bind")]
    pub bind: String,
    #[serde(default = "default_port")]
    pub port: u16,
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

/// A model the user has registered. Everything needed to launch a llama.cpp
/// server container for it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDef {
    /// Unique, human-friendly id (also becomes the llama-server `-a` alias and
    /// the OpenAI model id clients use).
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Path to the first GGUF shard (…-00001-of-000NN.gguf) or a single GGUF.
    pub gguf_path: String,
    /// Docker image that provides /opt/llama.cpp/build-cuda/bin/llama-server.
    pub image: String,
    /// Host port llama-server listens on (OpenAI-compatible API).
    pub host_port: u16,
    #[serde(default = "default_context")]
    pub context: u32,
    #[serde(default = "default_ngl")]
    pub ngl: u32,
    #[serde(default = "default_kv")]
    pub kv_type: String,
    #[serde(default = "default_threads")]
    pub threads: u32,
    /// Extra raw llama-server flags (spec-decode, stability flags, etc.).
    #[serde(default)]
    pub extra_args: Vec<String>,
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

    /// The llama-server argv (without the binary path) built from this def.
    pub fn llama_args(&self, gguf_in_container: &str) -> Vec<String> {
        let mut a = vec![
            "-a".into(),
            self.name.clone(),
            "-m".into(),
            gguf_in_container.into(),
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
