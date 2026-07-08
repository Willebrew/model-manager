# Model Manager

A small, memory-aware **web dashboard for running local LLMs** (llama.cpp **and** vLLM) on a single machine — built for the NVIDIA DGX Spark (GB10, 128 GB unified memory), but works on any Linux box with Docker + an NVIDIA GPU. Served over **HTTPS** and reachable from any device on your network.

It answers the question that matters most on a unified-memory box: **"will this model fit, or will it OOM?"** — *before* you start it.

## Why

On a 128 GB Spark a single 4-bit 200B-class model can occupy ~117 GB. Load a second one and the whole box locks up and OOMs. Model Manager:

- **Estimates each model's memory footprint before you load it** — weights (exact, from GGUF shard sizes) + KV cache (from the GGUF hyperparameters and your context length) + overhead.
- **Refuses (or warns) when a load would OOM**, given what's already running and free.
- **Learns the real number.** After a model's first successful load it records the actual whole-system memory it took, and uses that measured value for future decisions.
- Lets you **load / unload** models from a browser, **from any device on your network** (manage the Spark from your Mac).
- Lets you pick which models **start automatically on boot** (via the Docker restart policy).

## Features

- 🌐 **Web dashboard over HTTPS** bound to your LAN — open it from your laptop, phone, anything (self-signed cert auto-generated on first run).
- 🧩 **Two engines**: run **llama.cpp** (GGUF) or **vLLM** (HuggingFace-format) models — pick per model.
- 📊 **Live memory gauge** (RAM + swap) and per-model status (running / loading / stopped / won't-fit).
- 🧠 **Memory estimator** with a measured-peak cache, so estimates get exact after first run.
- 🛑 **OOM guard** — a load that won't fit is blocked with a clear explanation; override with one click if you really mean it.
- 🚀 **One-click load / unload** of model server containers.
- ⏻ **Boot autostart** toggle per model.
- 🔒 **Access token** so only people with the token can control your models.
- 📜 **Live container logs** in the UI.

## How it works

Each registered model is launched as a Docker container running its engine's
OpenAI-compatible server — llama.cpp's `llama-server` or vLLM's API server.
Model Manager talks to the Docker daemon directly (via the socket), mounts the
model read-only at `/model`, and passes through `--gpus all` + the NVIDIA
runtime. The dashboard polls the daemon and the system for live state.

Memory estimate for a model:

```
total ≈ weights(sum of GGUF shard bytes)
      + kv_cache(n_layers · context · kv_heads · (key_dim+value_dim) · bytes_per_elem[kv_type])
      + overhead(CUDA context, compute buffers)
```

If a measured peak from a real load exists, that value is used for OOM decisions instead of the estimate.

## Requirements

- Linux, Docker, and (for GPU models) the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) configured as a Docker runtime.
- Your user in the `docker` group (`sudo usermod -aG docker $USER`).
- A Docker image that contains `llama-server` at `/opt/llama.cpp/build-cuda/bin/llama-server` (adjust in `src/docker.rs` if yours differs).
- Rust (stable) to build.

## Build & run

```bash
git clone https://github.com/Willebrew/model-manager
cd model-manager
cargo build --release
./target/release/model-manager
```

On startup it prints your dashboard URL and access token:

```
  Local:    https://127.0.0.1:8600/
  Network:  https://192.168.1.78:8600/   (open this from your Mac)
  Token:    iO4EUzsK4eexYdOTFHdYiSo9IODdCZJ5ESTzc4ls
  Note:     self-signed cert — your browser will warn once; click through.
```

Open the Network URL from any device on your LAN and paste the token (or use the
`?token=…` link to log straight in).

Print the token any time:

```bash
model-manager --print-token
```

## Configuration

Config lives at `~/.config/model-manager/config.toml` (override with
`--config` or `MODEL_MANAGER_CONFIG`). It's created on first run with a random
token. Models are added from the UI, but you can also edit the file directly:

```toml
[server]
bind = "0.0.0.0"
port = 8600
tls = true                # serve HTTPS with a self-signed cert (auto-generated)
token = "…"
overhead_mib = 2560       # fixed overhead added to each estimate
safety_margin_mib = 2048  # keep this much free; less than this = OOM warning

# A llama.cpp (GGUF) model
[[models]]
name = "MiniMax-M2.7-UD-IQ4_XS"
engine = "llamacpp"
description = "229B MoE, UD-IQ4_XS, 108K ctx + ngram spec-decode"
model_path = "/home/you/models/MiniMax-M2.7-GGUF/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf"
image = "minimax-m27-longctx-108k"
host_port = 18080
context = 108000
ngl = 99
kv_type = "q8_0"
threads = 20
extra_args = ["--jinja", "-fa", "on", "--no-warmup", "-np", "1",
              "--cache-reuse", "256", "--ctx-checkpoints", "0", "-cram", "100",
              "--spec-type", "ngram-simple", "--draft-max", "16",
              "--draft-min", "1", "--draft-p-min", "0.5", "--spec-ngram-size-n", "4"]
autostart = false

# A vLLM (HuggingFace) model — image defaults to vllm/vllm-openai:latest
[[models]]
name = "Qwen2.5-72B-Instruct"
engine = "vllm"
model_path = "/home/you/models/Qwen2.5-72B-Instruct"   # HF model directory
host_port = 18081
context = 32768
gpu_mem_util = 0.90
extra_args = ["--tensor-parallel-size", "1"]
autostart = false
```

For llama.cpp, `model_path` points at the **first GGUF shard**; all shards in
that directory are mounted and their sizes summed for the weight estimate. For
vLLM, `model_path` is the **HuggingFace model directory** (weights are summed
from its `*.safetensors`/`*.bin`, KV from its `config.json`).

## HTTP API

All `/api/*` routes (except `/api/health`) require `Authorization: Bearer <token>`
or `?token=<token>`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/state` | memory + every model's status & estimate |
| GET | `/api/models/:name/estimate` | footprint + OOM verdict |
| POST | `/api/models` | add/update a model |
| DELETE | `/api/models/:name` | remove (and unload) a model |
| POST | `/api/models/:name/load?force=<bool>` | load (409 if it would OOM unless `force`) |
| POST | `/api/models/:name/unload` | stop + remove container |
| POST | `/api/models/:name/autostart` | `{ "enabled": bool }` |
| GET | `/api/models/:name/logs` | recent container logs |

## Security note

The dashboard controls processes on your machine and binds to your LAN by
default. It serves **HTTPS** with an auto-generated self-signed cert (your
browser warns once — click through, or import the cert from
`~/.config/model-manager/cert.pem`). The access token gates every action — treat
it like a password, and don't reuse a system/sudo password for it. For untrusted
networks, set `bind = "127.0.0.1"` and reach it over SSH, or front it with a
reverse proxy that has a real certificate.

## License

MIT — see [LICENSE](LICENSE).
