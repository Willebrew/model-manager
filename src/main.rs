mod api;
mod auth;
mod config;
mod docker;
mod gateway;
mod gguf;
mod memory;

use anyhow::{Context, Result};
use axum_server::tls_rustls::RustlsConfig;
use clap::{Parser, Subcommand};
use std::net::{SocketAddr, UdpSocket};
use std::path::PathBuf;
use std::sync::Arc;
use sysinfo::System;
use tokio::sync::Mutex;

/// Model Manager — a memory-aware web dashboard for local llama.cpp / vLLM models.
#[derive(Parser, Debug)]
#[command(name = "model-manager", version, about)]
struct Cli {
    /// Path to config file (overrides default ~/.config/model-manager/config.toml).
    #[arg(long)]
    config: Option<String>,

    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Manage gateway API keys (mmk_...).
    Key {
        #[command(subcommand)]
        op: KeyOp,
    },
    /// Set the dashboard password (Argon2id hash stored in config).
    /// Also removes the deprecated plaintext `token` from the config.
    SetPassword,
    /// Generate a TOTP secret for dashboard login. Prints the otpauth URI
    /// and a QR code ONCE — enroll it immediately.
    SetupTotp,
    /// Rotate the loopback-only admin key (X-MM-Admin header). Printed once.
    AdminKey {
        #[command(subcommand)]
        op: AdminOp,
    },
}

#[derive(Subcommand, Debug)]
enum KeyOp {
    /// Create a key; the secret is printed once and only its hash is stored.
    Create {
        /// Client name (e.g. will-mac).
        #[arg(long)]
        name: String,
        /// "openai" (passthrough, default) or "cursor" (Cursor rewrites).
        #[arg(long, default_value = "openai")]
        profile: String,
    },
    /// List keys (id/name/profile/created/revoked — never secrets).
    List,
    /// Revoke a key by id.
    Revoke { id: String },
}

#[derive(Subcommand, Debug)]
enum AdminOp {
    Rotate,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "model_manager=info,tower_http=warn".into()),
        )
        .init();

    // rustls needs a process-wide crypto provider before any TLS work.
    let _ = rustls::crypto::ring::default_provider().install_default();

    let cli = Cli::parse();
    if let Some(p) = &cli.config {
        std::env::set_var("MODEL_MANAGER_CONFIG", p);
    }

    let mut config = config::Config::load_or_init()?;

    match &cli.cmd {
        Some(Cmd::Key { op }) => return key_cmd(&mut config, op),
        Some(Cmd::SetPassword) => return set_password(&mut config),
        Some(Cmd::SetupTotp) => return setup_totp(&mut config),
        Some(Cmd::AdminKey { op: AdminOp::Rotate }) => return admin_rotate(&mut config),
        None => {}
    }

    let bind = config.server.bind.clone();
    let port = config.server.port;
    let tls = config.server.tls;
    let cert_override = config.server.tls_cert_path.clone();
    let key_override = config.server.tls_key_path.clone();

    let docker = docker::connect()?;
    let state: api::SharedState = Arc::new(api::AppState {
        config: Mutex::new(config),
        docker,
        sys: Mutex::new(System::new()),
        http: reqwest::Client::new(),
        loading: Mutex::new(std::collections::HashMap::new()),
        auth: auth::AuthState::default(),
        gateway: Arc::new(gateway::GatewayShared::default()),
    });

    let app = api::router(state.clone());
    gateway::serve(state);

    let ip = config::resolve_bind(&bind)?;
    let addr = SocketAddr::from((ip, port));
    let lan = lan_ip().unwrap_or_else(|| "127.0.0.1".to_string());
    let scheme = if tls { "https" } else { "http" };

    println!("\n  Model Manager is running.");
    println!("  ─────────────────────────────────────────────");
    println!("  Local:    {scheme}://127.0.0.1:{port}/");
    println!("  Network:  {scheme}://{lan}:{port}/   (open this from your Mac)");
    if tls {
        println!("  Note:     self-signed cert — your browser will warn once; click through.");
    }
    println!("  ─────────────────────────────────────────────\n");

    if tls {
        let (cert_path, key_path) = ensure_cert(cert_override, key_override, &lan)?;
        let tls_config = RustlsConfig::from_pem_file(&cert_path, &key_path)
            .await
            .with_context(|| "loading TLS cert/key")?;
        axum_server::bind_rustls(addr, tls_config)
            .serve(app.into_make_service_with_connect_info::<SocketAddr>())
            .await?;
    } else {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await?;
    }
    Ok(())
}

// ---------- CLI ----------

fn key_cmd(config: &mut config::Config, op: &KeyOp) -> Result<()> {
    match op {
        KeyOp::Create { name, profile } => {
            let profile = profile.to_lowercase();
            if profile != "openai" && profile != "cursor" {
                anyhow::bail!("profile must be \"openai\" or \"cursor\"");
            }
            let (id, secret, hash) = auth::gen_gateway_key();
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            config.gateway.keys.push(config::GatewayKey {
                id: id.clone(),
                name: name.clone(),
                hash,
                created_at: now,
                revoked: false,
                profile,
            });
            config.save()?;
            // Shown exactly once; only the hash is stored.
            println!("key id:  {id}");
            println!("api key: {secret}");
            println!("Store it now — it will never be shown again.");
        }
        KeyOp::List => {
            if config.gateway.keys.is_empty() {
                println!("no gateway keys");
            }
            for k in &config.gateway.keys {
                let when = chrono_free(k.created_at);
                println!(
                    "{}\t{}\t{}\t{}{}",
                    k.id,
                    k.name,
                    k.profile,
                    when,
                    if k.revoked { "\tREVOKED" } else { "" }
                );
            }
        }
        KeyOp::Revoke { id } => {
            let Some(idx) = config.gateway.keys.iter().position(|k| k.id == *id) else {
                anyhow::bail!("no key with id {id}");
            };
            config.gateway.keys[idx].revoked = true;
            let (kid, kname) = (
                config.gateway.keys[idx].id.clone(),
                config.gateway.keys[idx].name.clone(),
            );
            config.save()?;
            println!("revoked {kid} ({kname})");
        }
    }
    Ok(())
}

/// Minimal unix-seconds → "YYYY-MM-DD HH:MM" without pulling in chrono.
fn chrono_free(unix: u64) -> String {
    let days = unix / 86400;
    let secs = unix % 86400;
    // civil-from-days algorithm
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y}-{m:02}-{d:02} {:02}:{:02}", secs / 3600, (secs % 3600) / 60)
}

fn read_password(prompt: &str) -> Result<String> {
    match rpassword::prompt_password(prompt) {
        Ok(p) => Ok(p),
        // No TTY (scripts/pipes): fall back to stdin lines.
        Err(_) => {
            eprint!("{prompt}");
            let mut s = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut s)
                .context("reading password from stdin")?;
            Ok(s.lines().next().unwrap_or("").to_string())
        }
    }
}

fn set_password(config: &mut config::Config) -> Result<()> {
    let p1 = read_password("New dashboard password: ")?;
    let p2 = read_password("Repeat password: ")?;
    if p1.is_empty() {
        anyhow::bail!("empty password");
    }
    if p1 != p2 {
        anyhow::bail!("passwords do not match");
    }
    config.server.password_hash = Some(auth::hash_password(&p1)?);
    // Retire the old plaintext token for good.
    if config.server.token.take().is_some() {
        println!("removed plaintext [server] token from config");
    }
    config.save()?;
    println!("password set (argon2id). TOTP: run `model-manager setup-totp`.");
    Ok(())
}

fn setup_totp(config: &mut config::Config) -> Result<()> {
    let secret = auth::gen_totp_secret();
    let key = auth::secret_key()?;
    config.server.totp_secret_enc = Some(auth::seal(&key, &secret)?);
    config.save()?;
    let uri = auth::totp_uri(&secret);
    println!("\nEnroll this TOTP secret now — it is never shown again:\n");
    if let Ok(code) = qrcode::QrCode::new(uri.as_bytes()) {
        let img = code
            .render::<qrcode::render::unicode::Dense1x2>()
            .dark_color(qrcode::render::unicode::Dense1x2::Light)
            .light_color(qrcode::render::unicode::Dense1x2::Dark)
            .build();
        println!("{img}");
    }
    println!("{uri}\n");
    Ok(())
}

fn admin_rotate(config: &mut config::Config) -> Result<()> {
    use rand::RngCore;
    let mut b = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut b);
    let key = format!(
        "mma_{}",
        base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            b
        )
    );
    config.server.admin_key_hash = Some(auth::sha256_hex(key.as_bytes()));
    config.save()?;
    println!("admin key (loopback X-MM-Admin header), shown once:\n{key}");
    Ok(())
}

/// Ensure a TLS cert/key pair exists, generating a self-signed one (valid for
/// localhost + this host's LAN IP) on first run. Returns their paths.
fn ensure_cert(
    cert_override: Option<String>,
    key_override: Option<String>,
    lan_ip: &str,
) -> Result<(PathBuf, PathBuf)> {
    if let (Some(c), Some(k)) = (cert_override, key_override) {
        return Ok((PathBuf::from(c), PathBuf::from(k)));
    }
    let dir = config::Config::dir();
    std::fs::create_dir_all(&dir).ok();
    let cert_path = dir.join("cert.pem");
    let key_path = dir.join("key.pem");
    if cert_path.exists() && key_path.exists() {
        return Ok((cert_path, key_path));
    }

    let mut sans = vec!["localhost".to_string(), "127.0.0.1".to_string()];
    if lan_ip != "127.0.0.1" {
        sans.push(lan_ip.to_string());
    }
    if let Ok(host) = hostname() {
        sans.push(host);
    }
    // Include the tailnet IP too so TLS works over tailscale binds.
    if let Ok(ip) = config::resolve_bind("tailscale") {
        sans.push(ip.to_string());
    }

    let rcgen::CertifiedKey { cert, key_pair } =
        rcgen::generate_simple_self_signed(sans).context("generating self-signed cert")?;
    std::fs::write(&cert_path, cert.pem())?;
    std::fs::write(&key_path, key_pair.serialize_pem())?;
    tracing::info!("generated self-signed TLS cert at {}", cert_path.display());
    Ok((cert_path, key_path))
}

fn hostname() -> Result<String> {
    let out = std::fs::read_to_string("/proc/sys/kernel/hostname")?;
    Ok(out.trim().to_string())
}

/// Best-effort LAN IP for the "open from another device" hint and cert SAN.
fn lan_ip() -> Option<String> {
    let sock = UdpSocket::bind("0.0.0.0:0").ok()?;
    sock.connect("8.8.8.8:80").ok()?;
    let addr = sock.local_addr().ok()?;
    Some(addr.ip().to_string())
}
