"""
src/pipeline/06_embed_config_c.py — Stage 7: Configuration C indices + final metadata.

Re-embeds the gallery with each fine-tuned CLIP checkpoint, fuses with the
BLIP captions from stage 5 at both alpha values, and writes the final
gallery_metadata.csv the Streamlit app reads (same schema as the
existing gallery_metadata.csv: item_id, relative_path, caption, clothes_type).

Reads:
  artifacts/clip_checkpoints/clip_finetuned_{seed}.pt   (one per seed in params.yaml)
  artifacts/gallery_captions.csv
Writes:
  artifacts/index_C_{alpha}_{seed}.bin   for every (alpha, seed) pair
  artifacts/gallery_metadata.csv

Run:
  python -m src.pipeline.06_embed_config_c
"""

import os
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from src.common import (build_index, crop_with_yolo, fuse_embeddings, get_device,
                         get_image_embedding, get_text_embedding, load_params,
                         parse_bbox_file)


def main():
    params = load_params()
    p = params["paths"]
    device = get_device()

    bbox_map = parse_bbox_file(p["bbox_file"])
    yolo_model = YOLO(os.path.join(p["artifacts_dir"], "yolo", "best.pt"))
    manifest = pd.read_csv(os.path.join(p["artifacts_dir"], "gallery_captions.csv"))

    clip_tokenizer = open_clip.get_tokenizer(params["clip"]["arch"])

    # metadata.csv only needs to be written once — it doesn't depend on the seed
    manifest.to_csv(os.path.join(p["artifacts_dir"], "gallery_metadata.csv"), index=False)
    print(f"[embed_config_c] gallery_metadata.csv written ({len(manifest)} rows).")

    for seed in params["clip"]["seeds"]:
        ckpt_path = os.path.join(p["artifacts_dir"], "clip_checkpoints", f"clip_finetuned_{seed}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  skipping seed {seed}: {ckpt_path} not found — run stage 6 (finetune_clip) first.")
            continue

        import torch
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            params["clip"]["arch"], pretrained=params["clip"]["pretrained"]
        )
        clip_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        clip_model = clip_model.to(device).eval()

        embeddings, text_embeddings = [], []
        for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc=f"Re-embedding (seed {seed})"):
            item_id, rel_path = row["item_id"], row["relative_path"]
            img_path = os.path.join(p["small_split_dir"], "gallery", rel_path)
            try:
                filename = Path(rel_path).name
                cropped, _, _ = crop_with_yolo(
                    yolo_model, img_path, bbox_map=bbox_map, item_id=item_id, filename=filename
                )
                emb = get_image_embedding(clip_model, clip_preprocess, cropped, device).squeeze(0)
            except Exception:
                emb = np.zeros(512, dtype="float32")
            embeddings.append(emb)
            text_embeddings.append(
                get_text_embedding(clip_model, clip_tokenizer, row.get("caption", ""), device).squeeze(0)
            )

        embeddings = np.array(embeddings).astype("float32")
        text_embeddings = np.array(text_embeddings).astype("float32")

        for alpha in params["fusion"]["alphas"]:
            fused = fuse_embeddings(embeddings, text_embeddings, alpha)
            idx = build_index(fused)
            alpha_tag = str(alpha).replace("0.", "")
            out_path = os.path.join(p["artifacts_dir"], f"index_C_{alpha_tag}_{seed}.bin")
            idx.save_index(out_path)
            print(f"  index_C_{alpha_tag}_{seed}.bin built (alpha={alpha}, seed={seed}).")


if __name__ == "__main__":
    main()
