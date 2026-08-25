"""
src/pipeline/03_embed_config_a.py — Stage 4: Configuration A (vision-only).

Crops every gallery image with the fine-tuned YOLO detector, embeds it with
base OpenCLIP ViT-B/32, and builds the HNSW index for the vision-only
baseline. Same math as the report's section 5.2, just wrapped as a script
that logs its output as a tracked artifact instead of a notebook cell.

Reads:
  artifacts/yolo/best.pt
  paths.small_split_dir/gallery/<item_id>/*.jpg
Writes:
  artifacts/index_A.bin
  artifacts/gallery_embeddings_A.npy   (kept for stage 4 caption fusion + Config C base)
  artifacts/gallery_manifest.csv       (item_id, relative_path, clothes_type) — feeds gallery_metadata.csv later

Run:
  python -m src.pipeline.03_embed_config_a
"""

import os
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from src.common import (build_index, crop_with_yolo, get_device,
                         get_image_embedding, load_params, parse_bbox_file)


def main():
    params = load_params()
    p = params["paths"]
    device = get_device()
    os.makedirs(p["artifacts_dir"], exist_ok=True)

    bbox_map = parse_bbox_file(p["bbox_file"])

    gallery_dir = os.path.join(p["small_split_dir"], "gallery")
    gallery_data = []
    for item_folder in sorted(Path(gallery_dir).iterdir()):
        if item_folder.is_dir():
            for img_file in sorted(item_folder.glob("*.jpg")):
                gallery_data.append((str(img_file), item_folder.name))
    print(f"[embed_config_a] {len(gallery_data)} gallery images.")

    yolo_model = YOLO(os.path.join(p["artifacts_dir"], "yolo", "best.pt"))
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model = clip_model.to(device).eval()

    embeddings, item_ids, rel_paths, clothes_types = [], [], [], []
    for img_path, item_id in tqdm(gallery_data, desc="Embedding gallery (Config A)"):
        try:
            filename = Path(img_path).name
            cropped, _, _ = crop_with_yolo(
                yolo_model, img_path, bbox_map=bbox_map, item_id=item_id, filename=filename
            )
            emb = get_image_embedding(clip_model, clip_preprocess, cropped, device).squeeze(0)
            embeddings.append(emb)
            item_ids.append(item_id)
            rel_paths.append(f"{item_id}/{filename}")
            clothes_types.append(bbox_map.get((item_id, filename), {}).get("clothes_type"))
        except Exception as e:
            print(f"  skipped {img_path}: {e}")

    embeddings = np.array(embeddings).astype("float32")
    index_a = build_index(embeddings)
    index_a.save_index(os.path.join(p["artifacts_dir"], "index_A.bin"))
    np.save(os.path.join(p["artifacts_dir"], "gallery_embeddings_A.npy"), embeddings)

    manifest = pd.DataFrame({
        "item_id": item_ids, "relative_path": rel_paths, "clothes_type": clothes_types
    })
    manifest.to_csv(os.path.join(p["artifacts_dir"], "gallery_manifest.csv"), index=False)

    print(f"[embed_config_a] index_A.bin built with {len(embeddings)} vectors.")


if __name__ == "__main__":
    main()
