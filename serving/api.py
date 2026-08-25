"""
serving/api.py — FastAPI counterpart to serving/app.py.

This is the other half of the diagram's "FastAPI / Streamlit" node.
Streamlit is the human demo UI; this is the machine-callable HTTP
interface a real deployment (or a monitoring canary, or another service)
would hit instead of a browser. Both share the exact same model-loading
and search logic via serving/search_core.SearchEngine — there is no
second copy of crop/embed/search here.

Run:
  cd MLOPS && uvicorn serving.api:app --host 0.0.0.0 --port 8000

Then:
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/search \\
       -F "file=@some_image.jpg" -F "k=5"
"""

import io
import os
import sys
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.search_core import SearchEngine  # noqa: E402

app = FastAPI(
    title="Visual Product Search API",
    description="FastAPI serving layer over the DVC/MLflow-produced search index. "
                 "Champion model is resolved from the MLflow Model Registry "
                 "(Production stage), falling back to params.yaml.",
    version="1.0.0",
)

_engine: Optional[SearchEngine] = None


def get_engine() -> SearchEngine:
    """Lazy singleton — loads all models/index once, on first request, not at import time.

    Importing this module (e.g. for OpenAPI schema generation, or by a test)
    should never trigger a multi-second model load; only an actual request should.
    """
    global _engine
    if _engine is None:
        _engine = SearchEngine()
    return _engine


@app.get("/health")
def health():
    """
    Liveness/readiness probe — what src/monitoring/canary_check.py and
    docker-compose healthchecks hit. Deliberately does NOT force a full
    model load (that happens lazily on first /search), so this stays fast
    even right after container start.
    """
    return {"status": "ok", "engine_loaded": _engine is not None}


@app.get("/champion")
def champion():
    """Which config is currently being served, and where it came from."""
    engine = get_engine()
    return {"champion": engine.champion, "fusion_alpha": engine.fusion_alpha}


@app.post("/search")
async def search(
    file: UploadFile = File(...),
    k: int = Form(5),
    clothes_type: Optional[int] = Form(None),
):
    """
    clothes_type: null/omitted = all, 1 = upper-body, 2 = lower-body, 3 = full-body
    (same encoding as DeepFashion / src/common.py CLOTHES_CLASS).
    """
    if k < 1 or k > 50:
        raise HTTPException(status_code=400, detail="k must be between 1 and 50")

    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    engine = get_engine()
    results, meta = engine.search(pil_image, k=k, requested_type=clothes_type)

    return {"results": results, "meta": meta}
