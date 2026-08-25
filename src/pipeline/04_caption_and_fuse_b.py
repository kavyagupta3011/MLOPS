"""
src/pipeline/04_caption_and_fuse_b.py — Stage 5: Configuration B (image+text fusion).

Generates a BLIP caption for every gallery image, encodes it with CLIP's
text tower, and fuses it with the Config A image embedding at both alpha
values from params.yaml. BLIP stays frozen — same design choice as the
report (section 4.3): it's a caption generator here, nothing is trained.

Reads:
  artifacts/gallery_embeddings_A.npy, artifacts/gallery_manifest.csv
Writes:
  artifacts/gallery_captions.csv
  artifacts/index_B_{alpha}.bin   for each alpha in params.yaml fusion.alphas

Run:
  python -m src.pipeline.04_caption_and_fuse_b
"""

import os

import numpy as np
import open_clip
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.common import (build_index, fuse_embeddings, get_device,
                         get_text_embedding, load_params)

BLIP_CAPTION_ID = "Salesforce/blip-image-captioning-base"
GALLERY_ROOT_FMT = "{split_dir}/gallery/{rel_path}"


def generate_caption(processor, model, pil_image, device):
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40)
    return processor.decode(out[0], skip_special_tokens=True).strip()


def main():
    params = load_params()
    p = params["paths"]
    device = get_device()

    manifest = pd.read_csv(os.path.join(p["artifacts_dir"], "gallery_manifest.csv"))
    embeddings_a = np.load(os.path.join(p["artifacts_dir"], "gallery_embeddings_A.npy"))

    print(f"[caption_and_fuse_b] Loading BLIP ({BLIP_CAPTION_ID}) ...")
    processor = BlipProcessor.from_pretrained(BLIP_CAPTION_ID)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_CAPTION_ID).to(device).eval()

    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    from PIL import Image

    captions = []
    for rel_path in tqdm(manifest["relative_path"], desc="Generating captions"):
        img_path = GALLERY_ROOT_FMT.format(split_dir=p["small_split_dir"], rel_path=rel_path)
        try:
            img = Image.open(img_path).convert("RGB")
            captions.append(generate_caption(processor, model, img, device))
        except Exception:
            captions.append("a clothing item")

    manifest["caption"] = captions
    manifest.to_csv(os.path.join(p["artifacts_dir"], "gallery_captions.csv"), index=False)

    text_embeddings = np.array([
        get_text_embedding(clip_model, tokenizer, cap, device).squeeze(0) for cap in captions
    ]).astype("float32")

    for alpha in params["fusion"]["alphas"]:
        fused = fuse_embeddings(embeddings_a, text_embeddings, alpha)
        idx = build_index(fused)
        alpha_tag = str(alpha).replace("0.", "")
        idx.save_index(os.path.join(p["artifacts_dir"], f"index_B_{alpha_tag}.bin"))
        print(f"[caption_and_fuse_b] index_B_{alpha_tag}.bin built (alpha={alpha}).")


if __name__ == "__main__":
    main()
