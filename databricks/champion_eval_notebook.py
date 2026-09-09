# Databricks notebook source
# MAGIC %md
# MAGIC # Champion model evaluation — running on Databricks compute
# MAGIC
# MAGIC This notebook is the "real compute on Databricks" half of the project's
# MAGIC Databricks connection (the other half — `src/pipeline/07_evaluate.py`
# MAGIC automatically mirroring every config's real metrics here — already runs
# MAGIC on every local evaluation; see `docs/principles_mapping.md`).
# MAGIC
# MAGIC This notebook does something that logging-only step can't: it actually
# MAGIC downloads the real trained model files (published as GitHub Release
# MAGIC assets — see `.github/workflows/monitor.yml` for the same pattern used
# MAGIC in CI) and runs the full serving pipeline — YOLO crop -> CLIP embed ->
# MAGIC BLIP caption -> fuse -> HNSW search — on Databricks' own cluster, against
# MAGIC a batch of real query images, computing a genuine Recall@5 on this
# MAGIC compute, not just mirroring a number computed elsewhere.
# MAGIC
# MAGIC Run all cells top to bottom (Run > Run all).

# COMMAND ----------

# MAGIC %pip install open-clip-torch ultralytics transformers hnswlib pillow requests
dbutils.library.restartPython()

# COMMAND ----------

import os
import zipfile
from pathlib import Path

import requests

WORKDIR = "/tmp/visual_search_mlops"
os.makedirs(WORKDIR, exist_ok=True)

RELEASE_BASE = "https://github.com/kavyagupta3011/MLOPS/releases/download/model-v1"
FILES = [
    "best.pt",                 # YOLO weights
    "clip_finetuned_16.pt",    # fine-tuned CLIP checkpoint (the champion, seed 16)
    "gallery_metadata.csv",    # gallery item_id / relative_path / caption lookup
    "demo_small_split.zip",    # the actual query + gallery images
]

for fname in FILES:
    dest = os.path.join(WORKDIR, fname)
    if os.path.exists(dest):
        print(f"[setup] {fname} already downloaded, skipping.")
        continue
    print(f"[setup] downloading {fname} ...")
    r = requests.get(f"{RELEASE_BASE}/{fname}", stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print(f"[setup] {fname} -> {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")

# unpack the images (train/query/gallery folders)
zip_path = os.path.join(WORKDIR, "demo_small_split.zip")
extract_dir = os.path.join(WORKDIR, "data")
if not os.path.exists(os.path.join(extract_dir, "data", "processed", "small_split")):
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
SMALL_SPLIT_DIR = os.path.join(extract_dir, "data", "processed", "small_split")
print(f"[setup] images extracted to {SMALL_SPLIT_DIR}")

# COMMAND ----------

import open_clip
import pandas as pd
import torch
import hnswlib
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[load] using device: {device}")

yolo_model = YOLO(os.path.join(WORKDIR, "best.pt"))

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model.load_state_dict(torch.load(os.path.join(WORKDIR, "clip_finetuned_16.pt"), map_location=device))
clip_model = clip_model.to(device).eval()
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device).eval()

metadata = pd.read_csv(os.path.join(WORKDIR, "gallery_metadata.csv"))
print(f"[load] gallery has {len(metadata)} items")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build the HNSW index fresh, on this cluster
# MAGIC
# MAGIC The prebuilt `.bin` index file is tied to the exact hnswlib build that
# MAGIC created it and doesn't reliably load across different machines/versions,
# MAGIC so instead of downloading `index_C_7_16.bin`, this notebook re-embeds the
# MAGIC gallery images with the fine-tuned CLIP model and builds the index here,
# MAGIC live, on Databricks' own compute — which is itself part of "real work
# MAGIC happening on Databricks," not just a logging call.

# COMMAND ----------

GALLERY_DIR = os.path.join(SMALL_SPLIT_DIR, "gallery")
FUSION_ALPHA = 0.7  # matches the champion config: C_alpha0.7_seed16


def crop_with_yolo(pil_image):
    results = yolo_model(pil_image, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return pil_image, False
    best = max(boxes, key=lambda b: float(b.conf))
    if float(best.conf) <= 0.5:
        return pil_image, False
    x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(pil_image.width, x2), min(pil_image.height, y2)
    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return pil_image, False
    return pil_image.crop((x1, y1, x2, y2)), True


def get_image_embedding(pil_image):
    tensor = clip_preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")


def get_text_embedding(text):
    tokens = clip_tokenizer([text]).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")


def caption(pil_image):
    inputs = blip_processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=30)
    return blip_processor.decode(out[0], skip_special_tokens=True).strip()


def fuse(img_emb, txt_emb, alpha):
    fused = alpha * img_emb + (1 - alpha) * txt_emb
    return (fused / (((fused ** 2).sum(axis=-1, keepdims=True)) ** 0.5 + 1e-9)).astype("float32")


def embed_one(pil_image):
    cropped, _ = crop_with_yolo(pil_image)
    img_emb = get_image_embedding(cropped)
    cap = caption(cropped)
    txt_emb = get_text_embedding(cap)
    return fuse(img_emb, txt_emb, FUSION_ALPHA)


print("[build] embedding gallery images (this is real compute, running now)...")
gallery_vectors = []
for _, row in metadata.iterrows():
    img_path = os.path.join(GALLERY_DIR, row["relative_path"])
    if not os.path.exists(img_path):
        continue
    img = Image.open(img_path).convert("RGB")
    gallery_vectors.append((row["item_id"], embed_one(img)))

print(f"[build] embedded {len(gallery_vectors)} gallery images, building HNSW index...")
index = hnswlib.Index(space="cosine", dim=512)
index.init_index(max_elements=len(gallery_vectors), ef_construction=200, M=16)
item_ids = []
for i, (item_id, vec) in enumerate(gallery_vectors):
    index.add_items(vec, [i])
    item_ids.append(item_id)
index.set_ef(50)
print("[build] index ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run real queries through the pipeline and score Recall@5
# MAGIC
# MAGIC Capped at 20 query images to keep this fast on Free Edition's shared
# MAGIC compute — the point is proving the pipeline runs here for real, not
# MAGIC reproducing the full 108-query local evaluation number exactly.

# COMMAND ----------

QUERY_DIR = os.path.join(SMALL_SPLIT_DIR, "query")
MAX_QUERIES = 20

query_paths = []
for item_folder in sorted(Path(QUERY_DIR).iterdir()):
    if not item_folder.is_dir():
        continue
    for img_file in sorted(item_folder.glob("*.jpg"))[:1]:
        query_paths.append((str(img_file), item_folder.name))
    if len(query_paths) >= MAX_QUERIES:
        break

hits = 0
for img_path, true_item_id in query_paths:
    img = Image.open(img_path).convert("RGB")
    query_vec = embed_one(img)
    labels, _ = index.knn_query(query_vec, k=5)
    predicted_ids = [item_ids[int(lbl)] for lbl in labels[0]]
    if true_item_id in predicted_ids:
        hits += 1

recall_at_5 = hits / len(query_paths)
print(f"[eval] Recall@5 over {len(query_paths)} real queries, computed on Databricks: {recall_at_5:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log this run to MLflow — no extra setup needed
# MAGIC
# MAGIC Notebooks running on Databricks are automatically connected to this
# MAGIC workspace's MLflow — no `DATABRICKS_HOST` / `DATABRICKS_TOKEN` needed
# MAGIC here, unlike the local script, since the notebook already *is* running
# MAGIC inside Databricks.

# COMMAND ----------

import mlflow

mlflow.set_experiment("/Users/kavya.gupta@iiitb.ac.in/visual-search-mlops")
with mlflow.start_run(run_name="C_alpha0.7_seed16_databricks_compute"):
    mlflow.log_param("config", "C_alpha0.7_seed16")
    mlflow.log_param("fusion_alpha", FUSION_ALPHA)
    mlflow.log_param("n_queries", len(query_paths))
    mlflow.log_param("ran_on", "Databricks Free Edition compute")
    mlflow.log_metric("recall_at_5", recall_at_5)

print("Done — check this workspace's Experiments page for the new run.")
