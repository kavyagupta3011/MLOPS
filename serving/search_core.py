"""
serving/search_core.py — the shared logic behind both serving/app.py
(Streamlit UI) and serving/api.py (FastAPI). Neither file duplicates
crop/embed/search logic; they both call into this module.

This is also where "MLflow Registry" actually becomes the source of
truth for which model is serving, matching the diagram's Registry ->
Docker Image -> serving arrow: resolve_champion() checks the MLflow
Model Registry's "Production"-stage version first, and only falls back
to params.yaml's regression_gate.baseline_config if the registry is
unreachable (e.g. mlflow.db wasn't brought along, or nothing's been
promoted yet — see src/ci/promote_to_registry.py). Either way, the rest
of resolution (which index file, which alpha) is the same
config-name-to-paths logic the original app.py had.
"""

import os

import numpy as np
import open_clip
import pandas as pd
import torch
import hnswlib
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

from src.common import load_params

MLOPS_ROOT = os.path.join(os.path.dirname(__file__), "..")
BLIP_CAPTION_ID = "Salesforce/blip-image-captioning-base"


def resolve_champion(params: dict) -> str:
    """
    Return the config name to serve, e.g. "C_alpha0.7_seed16".
    Tries the MLflow Model Registry's Production stage first; falls back
    to params.yaml if the registry can't be reached or nothing is
    promoted yet — this fallback is what keeps grading/demos working on
    a machine that never ran promote_to_registry.py.
    """
    mf = params["mlflow"]
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(mf["tracking_uri"])
        client = MlflowClient()
        model_name = mf.get("registry_model_name", "visual-search-clip")
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            config_name = versions[0].tags.get("config_name")
            if config_name:
                print(f"[search_core] Champion resolved from MLflow Registry: {config_name}")
                return config_name
    except Exception as e:
        print(f"[search_core] MLflow Registry lookup failed ({e}); falling back to params.yaml.")

    fallback = params["regression_gate"]["baseline_config"]
    print(f"[search_core] Champion resolved from params.yaml (fallback): {fallback}")
    return fallback


def resolve_champion_paths(champion: str, artifacts_dir: str):
    """Turn a champion config name like 'C_alpha0.7_seed16' into concrete file paths."""
    if champion.startswith("A"):
        return None, os.path.join(artifacts_dir, "index_A.bin"), None
    if champion.startswith("B"):
        alpha = champion.split("alpha")[-1]
        alpha_tag = alpha.replace("0.", "")
        return None, os.path.join(artifacts_dir, f"index_B_{alpha_tag}.bin"), float(alpha)
    if champion.startswith("C"):
        alpha = champion.split("_")[1].replace("alpha", "")
        seed = champion.split("seed")[-1]
        alpha_tag = alpha.replace("0.", "")
        ckpt = os.path.join(artifacts_dir, "clip_checkpoints", f"clip_finetuned_{seed}.pt")
        idx = os.path.join(artifacts_dir, f"index_C_{alpha_tag}_{seed}.bin")
        return ckpt, idx, float(alpha)
    raise ValueError(f"Unrecognized champion config format: {champion}")


class SearchEngine:
    """Loads all models/indices once (expensive) and answers queries (cheap)."""

    def __init__(self, params: dict | None = None):
        self.params = params or load_params(os.path.join(MLOPS_ROOT, "params.yaml"))
        self.artifacts_dir = os.path.join(MLOPS_ROOT, self.params["paths"]["artifacts_dir"])
        self.gallery_dir = os.path.join(MLOPS_ROOT, self.params["paths"]["small_split_dir"], "gallery")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.champion = resolve_champion(self.params)
        ckpt_path, index_path, self.fusion_alpha = resolve_champion_paths(self.champion, self.artifacts_dir)

        yolo_path = os.path.join(self.artifacts_dir, "yolo", "best.pt")
        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"Missing {yolo_path}. Run the pipeline first: `dvc repro` from MLOPS/.")
        self.yolo = YOLO(yolo_path)

        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            self.params["clip"]["arch"], pretrained=self.params["clip"]["pretrained"]
        )
        self.clip_tokenizer = open_clip.get_tokenizer(self.params["clip"]["arch"])
        if ckpt_path:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Missing {ckpt_path}. Run `dvc repro -s finetune_clip` first.")
            self.clip_model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.clip_model = self.clip_model.to(self.device).eval()

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Missing {index_path}. Run the pipeline first: `dvc repro` from MLOPS/.")
        self.index = hnswlib.Index(space="cosine", dim=512)
        self.index.load_index(index_path)

        self.metadata = pd.read_csv(os.path.join(self.artifacts_dir, "gallery_metadata.csv"))

        self.blip_processor = BlipProcessor.from_pretrained(BLIP_CAPTION_ID)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_CAPTION_ID).to(self.device).eval()

    # -- crop / embed / fuse, ported unchanged from the original app.py --

    def crop_with_yolo(self, pil_image, requested_type=None):
        class_map = {1: 0, 2: 1, 3: 2}
        requested_yolo_class = class_map.get(requested_type)
        results = self.yolo(pil_image, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return pil_image, False, None
        matching = [
            b for b in boxes
            if float(b.conf) > 0.5 and (requested_yolo_class is None or int(b.cls[0]) == requested_yolo_class)
        ]
        if not matching:
            return pil_image, False, None
        best = max(matching, key=lambda b: float(b.conf))
        x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return pil_image, False, None
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(pil_image.width, x2), min(pil_image.height, y2)
        return pil_image.crop((x1, y1, x2, y2)), True, (x1, y1, x2, y2)

    def get_image_embedding(self, pil_image):
        tensor = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype("float32")

    def get_text_embedding(self, text):
        tokens = self.clip_tokenizer([text]).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype("float32")

    @staticmethod
    def fuse_embeddings(image_emb, text_emb, alpha):
        fused = alpha * image_emb + (1 - alpha) * text_emb
        fused = fused / (np.linalg.norm(fused, axis=-1, keepdims=True) + 1e-9)
        return fused.astype("float32")

    def generate_blip_caption(self, pil_image):
        inputs = self.blip_processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_new_tokens=40)
        return self.blip_processor.decode(out[0], skip_special_tokens=True).strip()

    def search(self, pil_image: Image.Image, k: int = 5, requested_type: int | None = None):
        """Full query pipeline: crop -> embed (+caption fuse) -> HNSW search -> filter. Returns (results, meta)."""
        cropped_img, was_cropped, yolo_bbox = self.crop_with_yolo(pil_image, requested_type)
        use_image = cropped_img if was_cropped else pil_image

        query_img_emb = self.get_image_embedding(use_image)
        query_emb = query_img_emb
        query_caption = ""
        if self.fusion_alpha is not None:
            try:
                query_caption = self.generate_blip_caption(use_image)
                txt_emb = self.get_text_embedding(query_caption)
                query_emb = self.fuse_embeddings(query_img_emb, txt_emb, self.fusion_alpha)
            except Exception:
                pass

        search_k = max(k * 3, 15)
        labels, distances = self.index.knn_query(query_emb, k=min(search_k, self.index.get_current_count()))

        results = []
        for lbl, dist in zip(labels[0], distances[0]):
            row = self.metadata.iloc[int(lbl)]
            if requested_type is not None and row.get("clothes_type") != requested_type:
                continue
            results.append({
                "item_id": str(row["item_id"]),
                "relative_path": row["relative_path"],
                "image_path": os.path.join(self.gallery_dir, row["relative_path"]),
                "caption": row.get("caption", ""),
                "similarity": float(1 - dist),
            })
            if len(results) >= k:
                break

        meta = {
            "champion": self.champion,
            "was_cropped": was_cropped,
            "yolo_bbox": yolo_bbox,
            "query_caption": query_caption,
        }
        return results, meta
