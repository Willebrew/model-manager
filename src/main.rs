mod api;
mod config;
mod docker;
mod gguf;
mod memory;

use anyhow::Result;
use clap::Parser;
use std::net::{SocketAddr, UdpSocket};
use std::sync::Arc;
use sysinfo::System;
use tokio::sync::Mutex;

/// Model Manager — a memory-aware web dashboard for local llama.cpp models.
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

    let docker = docker::connect()?;
    let state: api::SharedState = Arc::new(api::AppState {
        config: Mutex::new(config),
        docker,
        sys: Mutex::new(System::new()),
        http: reqwest::Client::new(),
    });

    let app = api::router(state);

    let addr: SocketAddr = format!("{bind}:{port}").parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;

    let lan = lan_ip().unwrap_or_else(|| "127.0.0.1".to_string());
    println!("\n  Model Manager is running.");
    println!("  ─────────────────────────────────────────────");
    println!("  Local:    http://127.0.0.1:{port}/");
    println!("  Network:  http://{lan}:{port}/   (open this from your Mac)");
    println!("  Token:    {token}");
    println!("  Tip:      http://{lan}:{port}/?token={token}  logs you straight in");
    println!("  ─────────────────────────────────────────────\n");

    axum::serve(listener, app).await?;
    Ok(())
}

/// Best-effort LAN IP for the "open from another device" hint. Opens a UDP
/// socket toward a public address (no packets sent) and reads the local addr.
fn lan_ip() -> Option<String> {
    let sock = UdpSocket::bind("0.0.0.0:0").ok()?;
    sock.connect("8.8.8.8:80").ok()?;
    let addr = sock.local_addr().ok()?;
    Some(addr.ip().to_string())
}
