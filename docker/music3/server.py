#!/usr/bin/env python3
"""OpenAI-compatible audio-generation server for MiniMax Music 3.

Serves the official speech-API contract used by SGLang-Omni:

    POST /v1/audio/speech
        input         = lyrics (required; section tags on their own lines)
        instructions  = music caption / style (required)
        seed          = int (default 0; deterministic)
        max_new_tokens= audio frames at 25 fps (default 750, max 9000)
        response_format = "wav"
        stream        = false

Response body is a 32 kHz 16-bit stereo WAV.

The pipeline is the Hugging Face ModularPipeline for MiniMax Music 3.
Weights are loaded once at startup from --model-dir and kept resident.
Inference is serialized behind a lock — the pipeline is not re-entrant.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import sys
import threading
import time
import uuid
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("music3")

FRAMES_PER_SEC = 25
MAX_FRAMES = 9000
DEFAULT_FRAMES = 750


class SpeechReq(BaseModel):
    model: Optional[str] = None
    input: str = Field(..., min_length=1)
    instructions: str = Field(..., min_length=1)
    seed: int = 0
    max_new_tokens: int = DEFAULT_FRAMES
    response_format: str = "wav"
    stream: bool = False
    voice: Optional[str] = None


class Progress:
    """Live job state. The browser polls GET /v1/audio/progress."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._t0 = 0.0
        self._wav: bytes | None = None
        self._data = {
            "state": "idle",
            "phase": "idle",
            "pct": 0.0,
            "frames_done": 0,
            "frames_total": 0,
            "job_id": "",
            "seed": 0,
            "lyrics": "",
            "caption": "",
        }

    def reset(self, frames_total: int, seed: int, lyrics: str, caption: str) -> None:
        with self._lock:
            self._t0 = time.time()
            self._wav = None
            self._data = {
                "state": "running",
                "phase": "reading",
                "pct": 2.0,
                "frames_done": 0,
                "frames_total": int(frames_total),
                "job_id": uuid.uuid4().hex,
                "seed": int(seed),
                "lyrics": lyrics,
                "caption": caption,
            }

    def update(self, **kwargs) -> None:
        with self._lock:
            self._data.update(kwargs)

    def store_wav(self, wav: bytes) -> None:
        with self._lock:
            self._wav = wav

    def wav(self) -> bytes | None:
        with self._lock:
            return self._wav

    def finish(self, ok: bool) -> None:
        with self._lock:
            self._data["state"] = "done" if ok else "error"
            self._data["phase"] = "done" if ok else "error"
            if ok:
                self._data["pct"] = 100.0

    def snapshot(self) -> dict:
        with self._lock:
            out = dict(self._data)
            out["elapsed_s"] = (
                round(time.time() - self._t0, 1) if self._t0 else 0.0
            )
            out["has_result"] = self._wav is not None
            return out


class Engine:
    def __init__(self, model_dir: str, device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device
        self.pipe = None
        self.sampling_rate = 32000
        self.lock = asyncio.Lock()
        self.ready = False
        self.progress = Progress()

    def load(self) -> None:
        # Weights are already on disk. Never phone Hugging Face at start.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        log.info("loading pipeline from %s", self.model_dir)
        # Import here so the container log shows a clear "importing" phase
        # before the heavyweight load (Model Manager watches these markers).
        log.info("importing diffusers")
        from diffusers import ModularPipeline

        log.info("loading pipeline")
        pipe = ModularPipeline.from_pretrained(
            self.model_dir, local_files_only=True
        )
        log.info("loading components")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        # Index files name the HF repo; force every submodule onto the local tree.
        pipe.load_components(
            dtype=dtype,
            local_files_only=True,
            pretrained_model_name_or_path=self.model_dir,
        )
        missing = [
            name
            for name, spec in getattr(pipe, "_component_specs", {}).items()
            if getattr(spec, "pretrained_model_name_or_path", None)
            and getattr(pipe, name, None) is None
        ]
        if missing:
            raise RuntimeError(
                "offline load missing components from "
                f"{self.model_dir}: {', '.join(sorted(missing))}"
            )
        pipe.to(self.device)
        sr = getattr(pipe, "sampling_rate", None)
        if sr:
            self.sampling_rate = int(sr)
        self.pipe = pipe
        self.ready = True
        log.info("pipeline ready (sr=%s)", self.sampling_rate)

    def _install_progress_hooks(self, max_frames: int):
        handles = []
        lm = getattr(self.pipe, "language_model", None)
        head = getattr(lm, "lm_head", None) if lm is not None else None
        if head is not None:
            counts = {"lm": 0}

            def on_lm(_module, _inp, _out):
                counts["lm"] += 1
                done = max(0, counts["lm"] - 1)
                frac = min(1.0, done / max(max_frames, 1))
                self.progress.update(
                    phase="writing",
                    frames_done=done,
                    pct=round(4.0 + 72.0 * frac, 1),
                )

            handles.append(head.register_forward_hook(on_lm))

        transformer = getattr(self.pipe, "transformer", None)
        if transformer is not None:
            counts_tf = {"n": 0}

            def on_tf(_module, _inp, _out):
                counts_tf["n"] += 1
                self.progress.update(
                    phase="painting",
                    pct=round(min(96.0, 76.0 + counts_tf["n"] * 0.35), 1),
                )

            handles.append(transformer.register_forward_hook(on_tf))

        vocoder = getattr(self.pipe, "vocoder", None)
        if vocoder is not None:
            def on_voc(_module, _inp, _out):
                self.progress.update(phase="pressing", pct=98.0)

            handles.append(vocoder.register_forward_hook(on_voc))
        return handles

    def generate(self, lyrics: str, caption: str, seed: int, frames: int) -> bytes:
        duration = max(1.0, float(frames) / FRAMES_PER_SEC)
        gen = None
        if seed is not None and seed >= 0:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
        log.info(
            "generate seed=%s frames=%s duration=%.1fs lyrics=%d chars caption=%d chars",
            seed, frames, duration, len(lyrics), len(caption),
        )
        self.progress.reset(frames, seed, lyrics, caption)
        handles = self._install_progress_hooks(frames)
        t0 = time.time()
        try:
            out = self.pipe(
                prompt=caption,
                lyrics=lyrics,
                audio_duration=duration,
                generator=gen,
                output="audios",
            )
        except Exception:
            self.progress.finish(False)
            raise
        finally:
            for handle in handles:
                handle.remove()
        audio = out[0]
        if hasattr(audio, "detach"):
            audio = audio.detach().float().cpu().numpy()
        else:
            audio = np.asarray(audio, dtype=np.float32)
        # (channels, samples) or (samples, channels) or (samples,)
        if audio.ndim == 1:
            audio = audio[:, None]
        elif audio.shape[0] <= 8 and audio.shape[0] < audio.shape[-1]:
            audio = audio.T
        audio = np.clip(audio, -1.0, 1.0)
        buf = io.BytesIO()
        sf.write(buf, audio, self.sampling_rate, format="WAV", subtype="PCM_16")
        wav = buf.getvalue()
        log.info("generate done in %.1fs (%d bytes)", time.time() - t0, len(wav))
        self.progress.store_wav(wav)
        self.progress.finish(True)
        return wav


def build_app(engine: Engine, served: str) -> FastAPI:
    app = FastAPI(title="MiniMax Music 3", version="0.1.0")

    @app.get("/health")
    def health():
        if not engine.ready:
            raise HTTPException(status_code=503, detail="model still loading")
        return {"ok": True, "model": served}

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{"id": served, "object": "model", "owned_by": "local"}],
        }

    @app.get("/v1/audio/progress")
    def progress():
        return engine.progress.snapshot()

    @app.get("/v1/audio/result")
    def result():
        snap = engine.progress.snapshot()
        if snap["state"] == "running":
            raise HTTPException(status_code=409, detail="still cutting")
        wav = engine.progress.wav()
        if not wav:
            raise HTTPException(status_code=404, detail="no take waiting")
        return Response(content=wav, media_type="audio/wav")

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechReq):
        if not engine.ready:
            raise HTTPException(status_code=503, detail="model still loading")
        if req.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported")
        fmt = (req.response_format or "wav").lower()
        if fmt not in ("wav", "pcm"):
            raise HTTPException(status_code=400, detail="response_format must be wav")
        frames = int(req.max_new_tokens or DEFAULT_FRAMES)
        if frames < 1 or frames > MAX_FRAMES:
            raise HTTPException(
                status_code=400,
                detail=f"max_new_tokens must be 1..{MAX_FRAMES} (25 frames/sec)",
            )
        async with engine.lock:
            try:
                wav = await asyncio.to_thread(
                    engine.generate, req.input, req.instructions, req.seed, frames
                )
            except Exception as e:
                log.exception("generation failed")
                raise HTTPException(status_code=500, detail=str(e)[:800])
        return Response(content=wav, media_type="audio/wav")

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="MiniMax Music 3 audio-gen server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--model-dir", default="/model")
    ap.add_argument("--served-model-name", default="minimax-music3")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    if not os.path.isdir(args.model_dir):
        log.error("model dir does not exist: %s", args.model_dir)
        sys.exit(1)

    engine = Engine(args.model_dir, device=args.device)
    app = build_app(engine, args.served_model_name)
    engine.load()

    import uvicorn

    log.info("serving on %s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
