#!/usr/bin/env python3
"""Image-to-3D server for TRELLIS.2 on DGX Spark.

    GET  /health
    GET  /v1/models
    GET  /                — small upload form
    POST /v1/3d/generations
        multipart: image (file, required)
        form fields:
            seed, resolution (512|1024|1536), preprocess_image (true/false)
            decimation_target, texture_size
        Response: model/gltf-binary (GLB)

Weights load once at startup from --model-dir. Inference is serialized.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import tempfile
import threading
import time
from typing import Optional

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("trellis2")

PIPE_TYPE = {
    "512": "512",
    "1024": "1024_cascade",
    "1536": "1536_cascade",
}

INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>TRELLIS.2</title>
<style>
body{font-family:system-ui,sans-serif;max-width:40rem;margin:2rem auto;padding:0 1rem;color:#e8e4df;background:#1c1917}
label,button{display:block;margin:.6rem 0} input,select{margin-left:.4rem}
button{padding:.4rem .8rem}
a{color:#a8c5ff}
</style></head>
<body>
<h1>TRELLIS.2 image-to-3D</h1>
<p>Upload an image. You get a GLB back. Default resolution is 512³ (fits Spark).</p>
<form action="/v1/3d/generations" method="post" enctype="multipart/form-data">
  <label>Image <input type="file" name="image" accept="image/*" required></label>
  <label>Resolution
    <select name="resolution">
      <option value="512" selected>512</option>
      <option value="1024">1024</option>
      <option value="1536">1536</option>
    </select>
  </label>
  <label>Seed <input type="number" name="seed" value="0"></label>
  <button type="submit">Generate GLB</button>
</form>
<p><a href="/health">/health</a> · <a href="/v1/models">/v1/models</a></p>
</body></html>
"""


class Engine:
    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        self.ready = False
        self._lock = threading.Lock()
        self._pipeline = None
        self._layout = None

    def load(self) -> None:
        log.info("loading pipeline from %s", self.model_dir)
        import torch  # noqa: F401 — CUDA init
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        pipe = Trellis2ImageTo3DPipeline.from_pretrained(self.model_dir)
        pipe.cuda()
        self._pipeline = pipe
        self._layout = getattr(pipe, "pbr_attr_layout", None)
        self.ready = True
        log.info("pipeline ready")

    def generate(
        self,
        image: Image.Image,
        seed: int,
        resolution: str,
        preprocess: bool,
        decimation_target: int,
        texture_size: int,
    ) -> bytes:
        if not self.ready or self._pipeline is None:
            raise RuntimeError("pipeline not loaded")
        ptype = PIPE_TYPE.get(resolution)
        if ptype is None:
            raise ValueError(f"resolution must be one of {list(PIPE_TYPE)}")
        import o_voxel
        import torch

        with self._lock:
            t0 = time.time()
            log.info(
                "generate seed=%s res=%s preprocess=%s",
                seed,
                resolution,
                preprocess,
            )
            outputs = self._pipeline.run(
                image,
                seed=int(seed),
                preprocess_image=preprocess,
                pipeline_type=ptype,
            )
            mesh = outputs[0]
            mesh.simplify(16777216)
            layout = self._layout
            if layout is None:
                layout = getattr(mesh, "layout", None)
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=layout,
                voxel_size=getattr(mesh, "voxel_size", None),
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=int(decimation_target),
                texture_size=int(texture_size),
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=False,
            )
            tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
            tmp.close()
            try:
                glb.export(tmp.name, extension_webp=True)
                data = open(tmp.name, "rb").read()
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
            torch.cuda.empty_cache()
            log.info("generate done in %.1fs (%d bytes)", time.time() - t0, len(data))
            return data


def build_app(engine: Engine, served: str) -> FastAPI:
    app = FastAPI(title="TRELLIS.2", version="0.1.0")

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

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.post("/v1/3d/generations")
    async def generate(
        image: UploadFile = File(...),
        seed: int = Form(0),
        resolution: str = Form("512"),
        preprocess_image: str = Form("true"),
        decimation_target: int = Form(500000),
        texture_size: int = Form(2048),
    ):
        if not engine.ready:
            raise HTTPException(status_code=503, detail="model still loading")
        raw = await image.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty image")
        try:
            im = Image.open(io.BytesIO(raw))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"not an image: {e}") from e
        preprocess = str(preprocess_image).lower() not in ("0", "false", "no")
        try:
            glb = await asyncio_to_thread(
                engine.generate,
                im,
                seed,
                resolution,
                preprocess,
                decimation_target,
                texture_size,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            log.exception("generate failed")
            raise HTTPException(status_code=500, detail=str(e)) from e
        return Response(
            content=glb,
            media_type="model/gltf-binary",
            headers={"Content-Disposition": 'attachment; filename="trellis2.glb"'},
        )

    return app


def asyncio_to_thread(fn, *args):
    import asyncio

    return asyncio.to_thread(fn, *args)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8014)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--served-model-name", default="trellis2-4b")
    args = p.parse_args()

    engine = Engine(args.model_dir)
    app = build_app(engine, args.served_model_name)

    def _load():
        try:
            engine.load()
        except Exception:
            log.exception("pipeline load failed")
            os._exit(1)

    threading.Thread(target=_load, daemon=True).start()

    import uvicorn

    log.info("serving on http://%s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
