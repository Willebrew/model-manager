mod api;
mod config;
mod docker;
mod gguf;
mod memory;

use anyhow::{Context, Result};
use axum_server::tls_rustls::RustlsConfig;
use clap::Parser;
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

    /// Print the config path and access token, then exit.
    #[arg(long)]
    print_token: bool,
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

    let config = config::Config::load_or_init()?;

    if cli.print_token {
        println!("config: {}", config::Config::path().display());
        println!("token:  {}", config.server.token);
        return Ok(());
    }

    let bind = config.server.bind.clone();
    let port = config.server.port;
    let token = config.server.token.clone();
    let tls = config.server.tls;
    let cert_override = config.server.tls_cert_path.clone();
    let key_override = config.server.tls_key_path.clone();

    let docker = docker::connect()?;
    let state: api::SharedState = Arc::new(api::AppState {
        config: Mutex::new(config),
        docker,
        sys: Mutex::new(System::new()),
        http: reqwest::Client::new(),
    });

    let app = api::router(state);

    let addr: SocketAddr = format!("{bind}:{port}").parse()?;
    let lan = lan_ip().unwrap_or_else(|| "127.0.0.1".to_string());
    let scheme = if tls { "https" } else { "http" };

    println!("\n  Model Manager is running.");
    println!("  ─────────────────────────────────────────────");
    println!("  Local:    {scheme}://127.0.0.1:{port}/");
    println!("  Network:  {scheme}://{lan}:{port}/   (open this from your Mac)");
    println!("  Token:    {token}");
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
            .serve(app.into_make_service())
            .await?;
    } else {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;
    }
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

    let mut sans = vec![
        "localhost".to_string(),
        "127.0.0.1".to_string(),
    ];
    if lan_ip != "127.0.0.1" {
        sans.push(lan_ip.to_string());
    }
    if let Ok(host) = hostname() {
        sans.push(host);
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
