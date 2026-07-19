# NeMo speech server (ASR + speaker diarization) for DGX Spark / GB10.
#
# Built on NVIDIA's PyTorch container rather than nvcr.io/nvidia/nemo: that
# image is now the Megatron/LLM backend and its speech collection is not
# guaranteed. This base already ships an aarch64 + CUDA 13 torch built for
# Blackwell, which is the hard part — never reinstall torch on top of it.
#
#   docker build -t nemo-speech-spark:latest docker/nemo-speech
#
# sm_121 (GB10) is not in the wheel's kernel list; CUDA JITs the sm_120 PTX at
# first launch. The "not officially supported" warning at startup is expected.
FROM nvcr.io/nvidia/pytorch:25.11-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libsndfile1 libsndfile1-dev sox \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir Cython packaging

# lhotse >=1.32.2 is required: earlier versions use the `data_source` argument
# that newer torch removed from Sampler, which breaks NeMo's dataloaders.
RUN pip install --no-cache-dir \
      "nemo_toolkit[asr]>=2.7" \
      "lhotse>=1.32.2" \
      fastapi "uvicorn[standard]" python-multipart soundfile

COPY server.py /app/server.py
WORKDIR /app

# Unbuffered so Model Manager can read load-progress markers from the log.
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf \
    NEMO_CACHE_DIR=/cache/nemo

ENTRYPOINT ["python3", "/app/server.py"]
