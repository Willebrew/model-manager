//! Docker control plane: launch/stop llama.cpp server containers for a model,
//! and report their status. Uses the local Docker socket via bollard.

use crate::config::{ModelDef, CONTAINER_PREFIX};
use anyhow::{Context, Result};
use bollard::container::{
    Config, CreateContainerOptions, ListContainersOptions, RemoveContainerOptions,
    StartContainerOptions, StopContainerOptions,
};
use bollard::models::{
    DeviceRequest, HostConfig, RestartPolicy, RestartPolicyNameEnum,
};
use bollard::Docker;
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;

pub const LLAMA_SERVER_BIN: &str = "/opt/llama.cpp/build-cuda/bin/llama-server";
const SHM_SIZE: i64 = 32 * 1024 * 1024 * 1024; // 32 GiB

pub fn connect() -> Result<Docker> {
    Docker::connect_with_local_defaults().context("connecting to Docker daemon (is it running, and are you in the `docker` group?)")
}

#[derive(Serialize, Clone, Debug)]
pub struct ContainerStatus {
    /// Model name (container name minus our prefix).
    pub model_name: String,
    pub container: String,
    pub running: bool,
    pub state: String,
    pub status: String,
    /// Restart policy name ("always" means it comes up on boot).
    pub restart_policy: String,
}

/// List the containers this tool manages (by name prefix).
pub async fn list_managed(docker: &Docker) -> Result<Vec<ContainerStatus>> {
    let mut filters = HashMap::new();
    filters.insert("name".to_string(), vec![CONTAINER_PREFIX.to_string()]);
    let opts = ListContainersOptions {
        all: true,
        filters,
        ..Default::default()
    };
    let containers = docker.list_containers(Some(opts)).await?;
    let mut out = Vec::new();
    for c in containers {
        let name = c
            .names
            .as_ref()
            .and_then(|n| n.first())
            .map(|s| s.trim_start_matches('/').to_string())
            .unwrap_or_default();
        if !name.starts_with(CONTAINER_PREFIX) {
            continue;
        }
        let model_name = name[CONTAINER_PREFIX.len()..].to_string();
        let state = c.state.clone().unwrap_or_default();
        let restart_policy = c
            .labels
            .as_ref()
            .and_then(|l| l.get("modelmgr.restart"))
            .cloned()
            .unwrap_or_default();
        out.push(ContainerStatus {
            model_name,
            container: name,
            running: state == "running",
            state,
            status: c.status.clone().unwrap_or_default(),
            restart_policy,
        });
    }
    Ok(out)
}

/// Status of one model's container, or None if it has never been created.
pub async fn status_of(docker: &Docker, model: &ModelDef) -> Result<Option<ContainerStatus>> {
    let cname = model.container_name();
    Ok(list_managed(docker)
        .await?
        .into_iter()
        .find(|s| s.container == cname))
}

/// Launch a model: (re)create and start its container. Removes any prior
/// container with the same name first so this is idempotent.
pub async fn load(docker: &Docker, model: &ModelDef) -> Result<()> {
    let cname = model.container_name();

    // Remove any existing (stopped or running) container with this name.
    let _ = docker
        .remove_container(
            &cname,
            Some(RemoveContainerOptions {
                force: true,
                ..Default::default()
            }),
        )
        .await;

    let image = model.effective_image();
    if image.trim().is_empty() {
        anyhow::bail!(
            "no Docker image set for '{}' (llama.cpp models need an image with llama-server)",
            model.name
        );
    }

    // Mount the model read-only at /model. If model_path is a directory (e.g. a
    // HuggingFace model for vLLM) mount it directly; if it's a file (e.g. a
    // GGUF shard) mount its parent so all sibling shards are visible.
    let path = Path::new(&model.model_path);
    let (host_mount, model_ref) = if path.is_dir() {
        (path.to_path_buf(), "/model".to_string())
    } else {
        let dir = path.parent().context("model_path has no parent directory")?;
        let file = path
            .file_name()
            .and_then(|s| s.to_str())
            .context("model_path has no file name")?;
        (dir.to_path_buf(), format!("/model/{file}"))
    };
    let bind = format!("{}:/model:ro", host_mount.display());
    let cmd = model.container_cmd(&model_ref);

    // `unless-stopped` starts the model on boot but respects a manual stop,
    // unlike `always` which would resurrect it even after you stop it.
    let restart = if model.autostart {
        RestartPolicyNameEnum::UNLESS_STOPPED
    } else {
        RestartPolicyNameEnum::NO
    };

    let host_config = HostConfig {
        network_mode: Some("host".to_string()),
        binds: Some(vec![bind]),
        runtime: Some("nvidia".to_string()),
        shm_size: Some(SHM_SIZE),
        device_requests: Some(vec![DeviceRequest {
            driver: Some(String::new()),
            count: Some(-1), // all GPUs (== --gpus all)
            capabilities: Some(vec![vec!["gpu".to_string()]]),
            ..Default::default()
        }]),
        restart_policy: Some(RestartPolicy {
            name: Some(restart),
            maximum_retry_count: None,
        }),
        ..Default::default()
    };

    let mut labels = HashMap::new();
    labels.insert("modelmgr".to_string(), "1".to_string());
    labels.insert(
        "modelmgr.restart".to_string(),
        if model.autostart { "unless-stopped" } else { "no" }.to_string(),
    );

    let config = Config {
        image: Some(image),
        cmd: Some(cmd),
        labels: Some(labels),
        host_config: Some(host_config),
        ..Default::default()
    };

    docker
        .create_container(
            Some(CreateContainerOptions {
                name: cname.clone(),
                platform: None,
            }),
            config,
        )
        .await
        .with_context(|| format!("creating container {cname}"))?;

    docker
        .start_container(&cname, None::<StartContainerOptions<String>>)
        .await
        .with_context(|| format!("starting container {cname}"))?;

    Ok(())
}

/// Change a container's restart policy in place (no reload). Used by the boot
/// toggle so flipping it doesn't re-load the whole model.
pub async fn set_restart_policy(docker: &Docker, model: &ModelDef, autostart: bool) -> Result<()> {
    use bollard::container::UpdateContainerOptions;
    let name = model.container_name();
    let policy = if autostart {
        RestartPolicyNameEnum::UNLESS_STOPPED
    } else {
        RestartPolicyNameEnum::NO
    };
    docker
        .update_container(
            &name,
            UpdateContainerOptions::<String> {
                restart_policy: Some(RestartPolicy {
                    name: Some(policy),
                    maximum_retry_count: None,
                }),
                ..Default::default()
            },
        )
        .await
        .with_context(|| format!("updating restart policy for {name}"))?;
    Ok(())
}

/// Stop and remove a model's container.
pub async fn unload(docker: &Docker, model: &ModelDef) -> Result<()> {
    let cname = model.container_name();
    let _ = docker
        .stop_container(&cname, Some(StopContainerOptions { t: 30 }))
        .await;
    docker
        .remove_container(
            &cname,
            Some(RemoveContainerOptions {
                force: true,
                ..Default::default()
            }),
        )
        .await
        .with_context(|| format!("removing container {cname}"))?;
    Ok(())
}

/// Fetch recent logs (stdout+stderr) for a model's container, last `tail` lines.
pub async fn logs(docker: &Docker, model: &ModelDef, tail: usize) -> Result<String> {
    use bollard::container::LogsOptions;
    use futures_util::StreamExt;
    let cname = model.container_name();
    let mut stream = docker.logs(
        &cname,
        Some(LogsOptions::<String> {
            stdout: true,
            stderr: true,
            tail: tail.to_string(),
            ..Default::default()
        }),
    );
    let mut buf = String::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(log) => buf.push_str(&log.to_string()),
            Err(_) => break,
        }
    }
    Ok(buf)
}

/// Whether the Docker daemon is reachable.
pub async fn ping(docker: &Docker) -> bool {
    docker.ping().await.is_ok()
}
