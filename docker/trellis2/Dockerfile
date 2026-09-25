# TRELLIS.2 image-to-3D server for DGX Spark / GB10 (aarch64, CUDA 13, SM 121).
#
# Uses the Spark-compatible fork (raziel2001au/dgx-trellis2) compiled against
# NVIDIA's Spark PyTorch image. Flash-attn / o-voxel / nvdiffrast compile from
# source and need tens of GiB free — do not build while a 90+ GB LLM is loaded.
#
#   docker build -t trellis2-spark:latest docker/trellis2
#
FROM nvcr.io/nvidia/pytorch:25.11-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs libjpeg-dev libgl1 libglib2.0-0 ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Keep the image torch. Install the TRELLIS.2 Python stack around it.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
      imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
      trimesh transformers kornia timm ninja \
      fastapi "uvicorn[standard]" python-multipart Pillow \
      "gradio==6.0.1" tensorboard pandas lpips zstandard \
 && pip install --no-cache-dir --no-deps \
      "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# Spark-specific fork: flash-attn / nvdiffrast / nvdiffrec / cumesh / flexgemm / o-voxel.
WORKDIR /opt
RUN git clone --recurse-submodules --depth 1 \
      https://github.com/raziel2001au/dgx-trellis2.git trellis2

WORKDIR /opt/trellis2
# GB10 only. Do not let extension setup.py emit sm_80/90/100/110 fatbins.
ENV TORCH_CUDA_ARCH_LIST="12.1" \
    FLASH_ATTN_CUDA_ARCHS="121" \
    MAX_JOBS="16" \
    CUDA_HOME=/usr/local/cuda \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# NGC 25.11 already ships flash_attn 2.7.4.post1 built for this box.
# scripts/install_flash_attention.sh compiles 2.8.4 from source for five
# unused GPU archs and takes an hour+. Skip it.
RUN python3 -c "import flash_attn; print('using image flash_attn', flash_attn.__version__)"
RUN bash -lc 'set -e; source scripts/install_nvdiffrast.sh'
RUN bash -lc 'set -e; source scripts/install_nvdiffrec.sh'
RUN bash -lc 'set -e; source scripts/install_cumesh.sh'
RUN bash -lc 'set -e; source scripts/install_flex_gemm.sh'
RUN bash -lc 'set -e; source scripts/install_o_voxel.sh'
# trellis2 is imported via PYTHONPATH; it has no setup.py / pyproject.toml.

COPY rembg_BiRefNet.py /opt/trellis2/trellis2/pipelines/rembg/BiRefNet.py
COPY image_feature_extractor.py /opt/trellis2/trellis2/modules/image_feature_extractor.py
COPY server.py /app/server.py
WORKDIR /opt/trellis2

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/trellis2 \
    HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

ENTRYPOINT ["python3", "/app/server.py"]
