"""
src/common.py — shared functions used by every pipeline stage.

This is a straight port of the logic that already lives in the team's
training.ipynb / app.py / eval.py, refactored into an importable module so
every MLOps stage (YOLO training, embedding, fine-tuning, evaluation,
serving, monitoring) calls the *same* code instead of copy-pasted notebook
cells. That single change is most of what "reproducibility" (principle P3
in the MLOps paper) means in practice.

Nothing here changes the modeling approach from the original project —
same CLIP arch, same fusion formula, same InfoNCE loss, same YOLO class
mapping. Only the dataset size and the surrounding plumbing are new.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_params(params_path: str = "params.yaml") -> dict:
    """Load params.yaml. Every script takes this as its single config source."""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# DeepFashion metadata parsing (list_bbox_inshop.txt / list_description_inshop.json)
# ---------------------------------------------------------------------------

def parse_bbox_file(bbox_path: str) -> dict:
    """
    Parse list_bbox_inshop.txt into {(item_id, flattened_filename): {...}}.
    Mirrors the parsing already used in training.ipynb Cell 3b, unchanged,
    so bbox lookups behave identically to the original notebook pipeline.
    """
    bbox_map = {}
    with open(bbox_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for line in lines[2:]:  # line 0 = count, line 1 = header
        parts = line.split()
        if len(parts) < 7:
            continue
        p = Path(parts[0])  # img/WOMEN/Category/id_00000001/02_1_front.jpg
        full_filename = f"{p.parts[1]}_{p.parts[2]}_{p.parts[3]}_{p.name}"
        bbox_map[(p.parent.name, full_filename)] = {
            "clothes_type": int(parts[1]),
            "pose_type": int(parts[2]),
            "bbox": (int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])),
        }
    return bbox_map


def parse_description_file(desc_path: str) -> dict:
    """Parse list_description_inshop.json into {item_id: short text}."""
    if not desc_path or not os.path.exists(desc_path):
        return {}
    with open(desc_path) as f:
        desc_data = json.load(f)
    item_descriptions = {}
    for entry in desc_data:
        iid = entry["item"]
        color = entry.get("color", "")
        lines = entry.get("description", [])
        text = f"{color}. " + " ".join(lines[:2]) if lines else color
        item_descriptions[iid] = text.strip()
    return item_descriptions


CLOTHES_CLASS = {1: 0, 2: 1, 3: 2}  # DeepFashion clothes_type -> YOLO class id
CLOTHES_LABEL = {1: "Upper-body", 2: "Lower-body", 3: "Full-body", None: "All"}


# ---------------------------------------------------------------------------
# Clothing-aware cropping (YOLO primary, GT bbox fallback, full image last resort)
# ---------------------------------------------------------------------------

def crop_with_yolo(yolo_model, img, requested_type=None, bbox_map=None, item_id=None,
                    filename=None, use_gt_fallback=True):
    """
    Same three-tier crop logic as training.ipynb / app.py:
      1. fine-tuned YOLO detection (confidence > 0.4, filtered by requested_type)
      2. ground-truth DeepFashion bbox, if available
      3. full image
    `img` may be a path (str) or a PIL.Image.
    """
    from PIL import Image

    if isinstance(img, str):
        pil_image = Image.open(img).convert("RGB")
        item_id = item_id or Path(img).parent.name
        filename = filename or Path(img).name
    else:
        pil_image = img.convert("RGB")

    gt_info = None
    if use_gt_fallback and bbox_map is not None and item_id is not None and filename is not None:
        gt_info = bbox_map.get((item_id, filename))

    requested_yolo_class = CLOTHES_CLASS.get(requested_type) if requested_type else None

    results = yolo_model(pil_image, verbose=False)
    boxes = results[0].boxes

    matching = [
        b for b in (boxes or [])
        if float(b.conf) > 0.4
        and (requested_yolo_class is None or int(b.cls[0]) == requested_yolo_class)
    ]

    if matching:
        best = max(matching, key=lambda b: float(b.conf))
        x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
        W, H = pil_image.size
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if (x2 - x1) >= 20 and (y2 - y1) >= 20:
            return pil_image.crop((x1, y1, x2, y2)), True, (x1, y1, x2, y2)

    if gt_info is not None:
        x1, y1, x2, y2 = gt_info["bbox"]
        W, H = pil_image.size
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if (x2 - x1) >= 20 and (y2 - y1) >= 20:
            return pil_image.crop((x1, y1, x2, y2)), True, (x1, y1, x2, y2)

    return pil_image, False, None


# ---------------------------------------------------------------------------
# CLIP embedding + fusion
# ---------------------------------------------------------------------------

def get_image_embedding(clip_model, clip_preprocess, pil_image, device):
    tensor = clip_preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")


def get_text_embedding(clip_model, clip_tokenizer, text, device):
    tokens = clip_tokenizer([text]).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")


def fuse_embeddings(image_emb, text_emb, alpha):
    fused = alpha * image_emb + (1 - alpha) * text_emb
    fused = fused / (np.linalg.norm(fused, axis=-1, keepdims=True) + 1e-9)
    return fused.astype("float32")


# ---------------------------------------------------------------------------
# HNSW index
# ---------------------------------------------------------------------------

def build_index(embeddings: np.ndarray):
    import hnswlib

    idx = hnswlib.Index(space="cosine", dim=embeddings.shape[1])
    idx.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
    idx.add_items(embeddings)
    idx.set_ef(50)
    return idx


def load_index(path: str, dim: int = 512):
    import hnswlib

    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.load_index(path)
    return idx
