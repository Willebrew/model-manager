#!/usr/bin/env python3
"""OpenAI-compatible speech server backed by NVIDIA NeMo.

Serves transcription (Parakeet-TDT) with optional speaker diarization
(Sortformer) on a single endpoint:

    POST /v1/audio/transcriptions

The request shape follows OpenAI's Whisper API (multipart `file`, `model`,
`response_format`, `timestamp_granularities[]`) plus one extension, `diarize`,
which attributes each word to a speaker and returns speaker-labelled turns.

Both models are loaded once at startup and kept resident; inference is
serialized behind a lock because the NeMo models are not re-entrant.
"""

import argparse
import asyncio
import contextlib
import glob
import json
import logging
import os
import subprocess
import sys
import tempfile
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("nemo-speech")

# Sortformer v1 is memory-bound and degrades past roughly this much audio.
# Beyond it we still transcribe, but skip diarization rather than OOM.
DEFAULT_MAX_DIAR_SECONDS = 720

ASR_HINTS = ("parakeet", "conformer", "citrinet", "canary", "asr")
DIAR_HINTS = ("diar", "sortformer")


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------

def to_wav16k(src: str, dst: str) -> None:
    """Decode anything ffmpeg understands into 16 kHz mono PCM, which is what
    both NeMo models expect."""
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-i", src, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError(f"could not decode audio: {p.stderr.strip()[:400]}")


def wav_duration(path: str) -> float:
    try:
        import soundfile as sf
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return 0.0


# --------------------------------------------------------------------------
# speaker attribution
# --------------------------------------------------------------------------

def parse_segments(raw) -> list:
    """Normalize Sortformer output into {start,end,speaker} dicts.

    `diarize()` returns one entry per audio file; each entry is a list of
    "start end speaker" strings (comma- or space-separated depending on
    version).
    """
    segs = []
    for line in raw or []:
        if not isinstance(line, str):
            continue
        parts = [x.strip() for x in (line.split(",") if "," in line else line.split())]
        if len(parts) < 3:
            continue
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        spk = parts[2]
        # Speaker may arrive as "speaker_1" or a bare index.
        digits = "".join(ch for ch in spk if ch.isdigit())
        segs.append({
            "start": start,
            "end": end,
            "speaker": int(digits) if digits else 0,
        })
    segs.sort(key=lambda s: s["start"])
    return segs


def attribute(words: list, segs: list) -> list:
    """Tag each word with a speaker by maximum temporal overlap.

    Overlap beats the midpoint-anchor approach for words straddling a turn
    boundary: the word goes to whichever speaker actually covers more of it.
    """
    out = []
    for w in words:
        ws, we = float(w["start"]), float(w["end"])
        best, best_ov = None, 0.0
        for sg in segs:
            ov = min(we, sg["end"]) - max(ws, sg["start"])
            if ov > best_ov:
                best, best_ov = sg["speaker"], ov
        if best is None and segs:
            # No overlap at all (e.g. word inside a silence gap): nearest turn.
            mid = (ws + we) / 2.0
            best = min(
                segs,
                key=lambda s: min(abs(mid - s["start"]), abs(mid - s["end"])),
            )["speaker"]
        out.append({**w, "speaker": best if best is not None else 0})
    return out


def to_turns(tagged: list, max_gap: float = 1.5) -> list:
    """Collapse speaker-tagged words into contiguous speaker turns."""
    turns = []
    for w in tagged:
        start, end = float(w["start"]), float(w["end"])
        if turns and turns[-1]["speaker"] == w["speaker"] and start - turns[-1]["end"] <= max_gap:
            turns[-1]["text"] += " " + w["word"]
            turns[-1]["end"] = end
        else:
            turns.append({
                "speaker": w["speaker"],
                "text": w["word"],
                "start": start,
                "end": end,
            })
    return turns


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

def ts(seconds: float, comma: bool = True) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def as_srt(items: list, label: bool) -> str:
    out = []
    for i, it in enumerate(items, 1):
        prefix = f"Speaker {it['speaker']}: " if label and it.get("speaker") is not None else ""
        out.append(f"{i}\n{ts(it['start'])} --> {ts(it['end'])}\n{prefix}{it['text'].strip()}\n")
    return "\n".join(out)


def as_vtt(items: list, label: bool) -> str:
    out = ["WEBVTT\n"]
    for it in items:
        prefix = f"Speaker {it['speaker']}: " if label and it.get("speaker") is not None else ""
        out.append(
            f"{ts(it['start'], False)} --> {ts(it['end'], False)}\n{prefix}{it['text'].strip()}\n"
        )
    return "\n".join(out)


# --------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------

def find_ckpt(model_dir: str, explicit: str, hints: tuple, anti: tuple):
    """Resolve a checkpoint: an explicit path/filename/HF id, else auto-detect
    a matching *.nemo in the model directory."""
    if explicit:
        if os.path.isabs(explicit) or os.path.exists(explicit):
            return explicit
        cand = os.path.join(model_dir, explicit)
        if os.path.exists(cand):
            return cand
        return explicit  # treat as a HF/NGC model id
    files = sorted(glob.glob(os.path.join(model_dir, "**", "*.nemo"), recursive=True))
    for f in files:
        base = os.path.basename(f).lower()
        if any(h in base for h in hints) and not any(a in base for a in anti):
            return f
    return None


class Engine:
    def __init__(self, args):
        self.args = args
        self.asr = None
        self.diar = None
        self.lock = asyncio.Lock()
        self.ready = False

    def load(self):
        log.info("importing nemo")
        import torch
        import nemo.collections.asr as nemo_asr
        from nemo.collections.asr.models import SortformerEncLabelModel

        self.torch = torch
        dev = self.args.device
        log.info("torch %s cuda=%s device=%s", torch.__version__, torch.version.cuda, dev)
        if dev == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA not available — falling back to CPU (this will be slow)")
            dev = "cpu"
        self.device = dev

        asr_ref = find_ckpt(self.args.model_dir, self.args.asr, ASR_HINTS, DIAR_HINTS)
        if not asr_ref:
            raise SystemExit(
                f"no ASR checkpoint found in {self.args.model_dir} "
                "(expected a *.nemo file, or pass --asr)"
            )
        log.info("loading asr model: %s", asr_ref)
        if os.path.exists(asr_ref):
            self.asr = nemo_asr.models.ASRModel.restore_from(asr_ref, map_location=dev)
        else:
            self.asr = nemo_asr.models.ASRModel.from_pretrained(asr_ref, map_location=dev)
        self.asr.eval()
        if dev == "cuda" and self.args.bf16:
            # Halves activation memory on long-form audio; Blackwell has native bf16.
            self.asr = self.asr.to(torch.bfloat16)
        if self.args.local_attention:
            # Bounds attention growth so multi-hour audio stays in memory.
            with contextlib.suppress(Exception):
                self.asr.change_attention_model(
                    self_attention_model="rel_pos_local_attn",
                    att_context_size=[256, 256],
                )
                log.info("enabled local attention for long-form audio")
        log.info("asr model ready")

        if not self.args.no_diar:
            diar_ref = find_ckpt(self.args.model_dir, self.args.diar, DIAR_HINTS, ())
            if diar_ref:
                log.info("loading diarization model: %s", diar_ref)
                if os.path.exists(diar_ref):
                    self.diar = SortformerEncLabelModel.restore_from(
                        restore_path=diar_ref, map_location=dev, strict=False
                    )
                else:
                    self.diar = SortformerEncLabelModel.from_pretrained(
                        diar_ref, map_location=dev
                    )
                self.diar.eval()
                log.info("diarization model ready")
            else:
                log.warning("no diarization checkpoint found — diarization disabled")

        self.ready = True
        log.info("all models ready")

    # -- inference ---------------------------------------------------------

    def _transcribe(self, wav: str):
        hyp = self.asr.transcribe([wav], timestamps=True)[0]
        text = getattr(hyp, "text", None)
        if text is None:  # older NeMo returns plain strings
            return str(hyp), [], []
        stamps = getattr(hyp, "timestamp", None) or {}
        return text, stamps.get("word", []) or [], stamps.get("segment", []) or []

    def _diarize(self, wav: str):
        raw = self.diar.diarize(audio=[wav], batch_size=1)
        return parse_segments(raw[0] if raw else [])

    def run(self, wav: str, want_diar: bool):
        t0 = time.time()
        text, words, segments = self._transcribe(wav)
        result = {
            "text": text,
            "words": words,
            "segments": segments,
            "turns": None,
            "warning": None,
            "duration": wav_duration(wav),
        }

        if want_diar and self.diar is not None:
            dur = result["duration"]
            if self.args.max_diar_seconds and dur > self.args.max_diar_seconds:
                result["warning"] = (
                    f"audio is {dur:.0f}s, longer than the {self.args.max_diar_seconds}s "
                    "diarization limit — returned transcript without speaker labels"
                )
            elif not words:
                result["warning"] = "no word timestamps available — cannot attribute speakers"
            else:
                try:
                    segs = self._diarize(wav)
                    if segs:
                        result["turns"] = to_turns(
                            attribute(words, segs), self.args.turn_gap
                        )
                        result["speakers"] = sorted({s["speaker"] for s in segs})
                    else:
                        result["warning"] = "diarization returned no speaker segments"
                except Exception as e:  # transcript is still useful without speakers
                    log.exception("diarization failed")
                    result["warning"] = f"diarization failed: {e}"
        elif want_diar and self.diar is None:
            result["warning"] = "diarization requested but no diarization model is loaded"

        result["elapsed"] = round(time.time() - t0, 2)
        return result


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def build_app(engine: "Engine", model_name: str):
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse, PlainTextResponse

    app = FastAPI(title="nemo-speech", docs_url="/docs")

    @app.get("/health")
    def health():
        if not engine.ready:
            raise HTTPException(status_code=503, detail="loading")
        return {"status": "ok", "model": model_name, "diarization": engine.diar is not None}

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{
                "id": model_name,
                "object": "model",
                "owned_by": "nemo",
                "capabilities": {
                    "transcription": True,
                    "diarization": engine.diar is not None,
                },
            }],
        }

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        model: str = Form(default=model_name),
        response_format: str = Form(default="json"),
        diarize: str = Form(default="false"),
        language: str = Form(default=None),          # accepted, auto-detected by v3
        temperature: float = Form(default=0.0),      # accepted for compatibility
        timestamp_granularities: str = Form(default=None),
    ):
        if not engine.ready:
            raise HTTPException(status_code=503, detail="models still loading")

        want_diar = str(diarize).lower() in ("1", "true", "yes", "on")
        suffix = os.path.splitext(file.filename or "")[1] or ".bin"

        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "in" + suffix)
            wav = os.path.join(td, "audio.wav")
            with open(src, "wb") as fh:
                fh.write(await file.read())
            try:
                to_wav16k(src, wav)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

            async with engine.lock:
                try:
                    r = await asyncio.to_thread(engine.run, wav, want_diar)
                except Exception as e:
                    log.exception("transcription failed")
                    raise HTTPException(status_code=500, detail=str(e))

        turns = r.get("turns")
        fmt = (response_format or "json").lower()

        if fmt == "text":
            if turns:
                return PlainTextResponse(
                    "\n".join(f"Speaker {t['speaker']}: {t['text'].strip()}" for t in turns)
                )
            return PlainTextResponse(r["text"])

        if fmt in ("srt", "vtt"):
            items = turns or r["segments"] or [
                {"start": 0.0, "end": r["duration"], "text": r["text"], "speaker": None}
            ]
            body = as_srt(items, bool(turns)) if fmt == "srt" else as_vtt(items, bool(turns))
            return PlainTextResponse(body)

        payload = {"text": r["text"]}
        if fmt == "verbose_json":
            payload.update({
                "task": "transcribe",
                "duration": r["duration"],
                "segments": r["segments"],
                "words": r["words"],
            })
        if turns:
            payload["turns"] = turns
            payload["speakers"] = r.get("speakers", [])
            # Convenience rendering so callers don't have to assemble turns.
            payload["diarized_text"] = "\n".join(
                f"Speaker {t['speaker']}: {t['text'].strip()}" for t in turns
            )
        if r.get("warning"):
            payload["warning"] = r["warning"]
        return JSONResponse(payload)

    return app


def main():
    ap = argparse.ArgumentParser(description="NeMo speech server (ASR + diarization)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--model-dir", default="/model", help="directory of .nemo checkpoints")
    ap.add_argument("--served-model-name", default="nemo-speech")
    ap.add_argument("--asr", default="", help="ASR .nemo filename/path or HF id")
    ap.add_argument("--diar", default="", help="diarization .nemo filename/path or HF id")
    ap.add_argument("--no-diar", action="store_true", help="disable diarization entirely")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no-bf16", dest="bf16", action="store_false")
    ap.add_argument("--local-attention", action="store_true",
                    help="bound attention memory for multi-hour audio")
    ap.add_argument("--max-diar-seconds", type=int, default=DEFAULT_MAX_DIAR_SECONDS,
                    help="skip diarization above this duration (0 = no limit)")
    ap.add_argument("--turn-gap", type=float, default=1.5,
                    help="merge same-speaker turns separated by less than this many seconds")
    args = ap.parse_args()

    import uvicorn
    engine = Engine(args)
    app = build_app(engine, args.served_model_name)
    engine.load()
    log.info("serving on %s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
