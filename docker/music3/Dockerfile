# MiniMax Music 3 audio-generation server for DGX Spark / GB10.
#
# Official serve path is SGLang-Omni, but there is no reliable aarch64 + CUDA 13
# image for it. This container wraps the Hugging Face ModularPipeline on
# NVIDIA's Spark-ready PyTorch image (same base as docker/nemo-speech).
#
#   docker build -t music3-spark:latest docker/music3
#
# Do not reinstall torch on top of this base — it already ships an aarch64 +
# CUDA 13 wheel built for Blackwell.
FROM nvcr.io/nvidia/pytorch:25.11-py3

# libsndfile only — the server writes WAV via soundfile. Do not pull ffmpeg
# (it drags in 150+ X11/mesa packages on this Ubuntu ports mirror).
RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Keep the image's torch. Install the rest of the Music 3 stack around it.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
      "transformers>=4.57.0" \
      accelerate \
      huggingface_hub \
      safetensors \
      soundfile \
      fastapi \
      "uvicorn[standard]" \
      python-multipart \
      einops \
      regex \
      Pillow

# MiniMax Music 3 lives on this commit until huggingface/diffusers#14456 merges.
# --no-deps so pip does not replace the image torch.
RUN pip install --no-cache-dir --no-deps \
      "git+https://github.com/huggingface/diffusers@dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d"

COPY server.py /app/server.py
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

ENTRYPOINT ["python3", "/app/server.py"]
