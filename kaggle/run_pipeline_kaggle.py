# ============================================================
# run_pipeline_kaggle.py — paste this into ONE cell of a new Kaggle notebook.
#
# Setup on Kaggle (same as your original training.ipynb):
#   1. New Notebook -> Add Input -> attach dataset "nainika0305/deepfashionretrieval"
#      (it already contains list_eval_partition.txt, list_bbox_inshop.txt,
#      list_description_inshop.json and img/img/... — nothing else to upload)
#   2. Settings -> Accelerator -> GPU T4 (or whatever's available)
#   3. Settings -> Internet -> On (needed for pip installs + yolov8n.pt download)
#   4. Paste this whole file into one cell, run it.
#
# This is the small-dataset version of training.ipynb: it samples
# N_ITEMS identities instead of using the full ~38k, and every stage
# below is a straight port of the same functions in MLOPS/src/ — kept
# as one flat script (not imports) because that's the pattern your
# team's existing notebooks already use, and it's simplest to paste
# into Kaggle with zero extra setup.
#
# At the end it zips artifacts/ (checkpoints, indices, metadata) and
# mlruns/ (every run's tracked params/metrics) into one file you
# download and unpack into your local MLOPS/artifacts/ and MLOPS/mlruns/
# — see MLOPS/kaggle/README.md for that half of the process.
# ============================================================

CURRENT_SEEDS = [16, 34]     # keep in sync with MLOPS/params.yaml clip.seeds
N_ITEMS = 60                  # keep in sync with MLOPS/params.yaml sampling.n_items
MIN_IMAGES_PER_ITEM = 2
SPLIT_SEED = 42
YOLO_EPOCHS = 8
YOLO_IMGSZ = 320
CLIP_EPOCHS = 4
CLIP_LR = 1e-5
CLIP_WD = 1e-4
CLIP_TEMP = 0.07
TRAINABLE_BLOCKS = 4
ALPHAS = [0.5, 0.7]
K_VALUES = [5, 10, 15]

print("[STATUS] ---> Installing dependencies...")
get_ipython().system('pip install open-clip-torch hnswlib ultralytics transformers accelerate mlflow -q')

import os, random, json, yaml, shutil, gc
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
import open_clip, hnswlib, pandas as pd
import mlflow
import warnings
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[STATUS] ---> Device: {device}")
OUTPUT = "/kaggle/working"
os.makedirs(f"{OUTPUT}/artifacts", exist_ok=True)

mlflow.set_tracking_uri(f"file:{OUTPUT}/mlruns")
mlflow.set_experiment("visual-search-mlops")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ── Locate dataset (same walk your original notebooks use) ──────────────
print("[STATUS] ---> Locating dataset in /kaggle/input...")
partition_file = bbox_file = None
base_img_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'list_eval_partition.txt' in files:
        partition_file = os.path.join(root, 'list_eval_partition.txt')
    if 'list_bbox_inshop.txt' in files:
        bbox_file = os.path.join(root, 'list_bbox_inshop.txt')
    if root.endswith(os.path.join('img', 'img')):
        base_img_dir = root

if not (partition_file and bbox_file and base_img_dir):
    raise FileNotFoundError(
        "Could not find list_eval_partition.txt / list_bbox_inshop.txt / img/img — "
        "did you attach the nainika0305/deepfashionretrieval dataset as input?"
    )
print(f" partition_file: {partition_file}\n bbox_file: {bbox_file}\n base_img_dir: {base_img_dir}")


# ── Stage 1: sample N_ITEMS and copy into train/query/gallery ───────────
print(f"\n[STATUS] ---> Sampling {N_ITEMS} item_ids...")
with open(partition_file, 'r') as f:
    lines = f.readlines()[2:]
entries = []
for line in lines:
    parts = line.strip().split()
    if len(parts) == 3:
        entries.append(tuple(parts))  # (img_path, item_id, split)

from collections import defaultdict
by_item = defaultdict(list)
for img_path, item_id, split in entries:
    by_item[item_id].append((img_path, split))

eligible = [iid for iid, imgs in by_item.items() if len(imgs) >= MIN_IMAGES_PER_ITEM]
rng = random.Random(SPLIT_SEED)
rng.shuffle(eligible)
chosen = set(eligible[:N_ITEMS])
print(f" {len(chosen)} item_ids chosen (of {len(eligible)} eligible, {len(by_item)} total).")

split_dir = f"{OUTPUT}/small_split"
for split in ["train", "query", "gallery"]:
    os.makedirs(f"{split_dir}/{split}", exist_ok=True)

copied = 0
for img_path, item_id, split in tqdm(entries, desc="Copying"):
    if item_id not in chosen:
        continue
    src = os.path.join(base_img_dir, img_path.replace("img/", "", 1))
    if not os.path.exists(src):
        continue
    dest_folder = f"{split_dir}/{split}/{item_id}"
    os.makedirs(dest_folder, exist_ok=True)
    file_name = img_path.replace("img/", "", 1).replace("/", "_")
    shutil.copy2(src, f"{dest_folder}/{file_name}")
    copied += 1
print(f" Copied {copied} images.")


def load_split(split_dir_):
    data = []
    for item_folder in sorted(Path(split_dir_).iterdir()):
        if item_folder.is_dir():
            for img_file in sorted(item_folder.glob("*.jpg")):
                data.append((str(img_file), item_folder.name))
    return data


train_data = load_split(f"{split_dir}/train")
gallery_data = load_split(f"{split_dir}/gallery")
query_data = load_split(f"{split_dir}/query")
print(f" train:{len(train_data)}  gallery:{len(gallery_data)}  query:{len(query_data)}")


# ── Parse GT bboxes (same logic as training.ipynb Cell 3b) ──────────────
def parse_bbox_file(bbox_path):
    bbox_map = {}
    with open(bbox_path) as f:
        lines_ = [l.strip() for l in f if l.strip()]
    for line in lines_[2:]:
        parts = line.split()
        if len(parts) < 7: continue
        p = Path(parts[0])
        full_filename = f"{p.parts[1]}_{p.parts[2]}_{p.parts[3]}_{p.name}"
        bbox_map[(p.parent.name, full_filename)] = {
            "clothes_type": int(parts[1]), "pose_type": int(parts[2]),
            "bbox": (int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])),
        }
    return bbox_map

print("\n[STATUS] ---> Parsing GT bboxes...")
bbox_map = parse_bbox_file(bbox_file)
print(f" {len(bbox_map):,} GT bbox entries loaded.")

CLOTHES_CLASS = {1: 0, 2: 1, 3: 2}


# ── Stage 2+3: YOLO labels + fine-tune ───────────────────────────────────
print("\n[STATUS] ---> Writing YOLO labels...")
YOLO_DIR = f"{OUTPUT}/yolo_dataset"
os.makedirs(f"{YOLO_DIR}/images/train", exist_ok=True)
os.makedirs(f"{YOLO_DIR}/labels/train", exist_ok=True)

for img_path, item_id in tqdm(train_data, desc="YOLO labels"):
    filename = Path(img_path).name
    info = bbox_map.get((item_id, filename))
    if info is None: continue
    try:
        img = Image.open(img_path); W, H = img.size
    except Exception: continue
    x1, y1, x2, y2 = info["bbox"]
    cx, cy = ((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H
    bw, bh = (x2 - x1) / W, (y2 - y1) / H
    cx, cy, bw, bh = [min(max(v, 0), 1) for v in (cx, cy, bw, bh)]
    if bw < 0.01 or bh < 0.01: continue
    class_id = CLOTHES_CLASS[info["clothes_type"]]
    stem = f"{item_id}_{Path(filename).stem}"
    dst = f"{YOLO_DIR}/images/train/{stem}.jpg"
    if not os.path.exists(dst): shutil.copy2(img_path, dst)
    with open(f"{YOLO_DIR}/labels/train/{stem}.txt", "w") as f:
        f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

yaml_path = f"{YOLO_DIR}/clothing.yaml"
with open(yaml_path, "w") as f:
    yaml.dump({"path": YOLO_DIR, "train": "images/train", "val": "images/train",
               "nc": 3, "names": ["upper-body", "lower-body", "full-body"]}, f)

from ultralytics import YOLO
print(f"\n[STATUS] ---> Fine-tuning YOLOv8n ({YOLO_EPOCHS} epochs)...")
with mlflow.start_run(run_name="yolo_finetune"):
    mlflow.log_params({"epochs": YOLO_EPOCHS, "imgsz": YOLO_IMGSZ, "n_items": N_ITEMS})
    yolo_base = YOLO("yolov8n.pt")
    yolo_base.train(data=yaml_path, epochs=YOLO_EPOCHS, imgsz=YOLO_IMGSZ, batch=16,
                     project=OUTPUT, name="yolo_clothing", exist_ok=True, verbose=False)
    YOLO_BEST = f"{OUTPUT}/yolo_clothing/weights/best.pt"
    shutil.copy2(YOLO_BEST, f"{OUTPUT}/artifacts/best.pt")
    mlflow.log_artifact(f"{OUTPUT}/artifacts/best.pt")
yolo_model = YOLO(f"{OUTPUT}/artifacts/best.pt")
print(f" Fine-tuned YOLO -> {OUTPUT}/artifacts/best.pt")


def crop_with_yolo(img_path_or_pil, requested_type=None):
    class_map = {1: 0, 2: 1, 3: 2}
    requested_yolo_class = class_map.get(requested_type)
    if isinstance(img_path_or_pil, str):
        img = Image.open(img_path_or_pil).convert("RGB")
        item_id, filename = Path(img_path_or_pil).parent.name, Path(img_path_or_pil).name
        gt_info = bbox_map.get((item_id, filename))
    else:
        img, gt_info = img_path_or_pil.convert("RGB"), None
    results = yolo_model(img, verbose=False)
    boxes = results[0].boxes
    matching = [b for b in (boxes or []) if float(b.conf) > 0.4 and
                (requested_yolo_class is None or int(b.cls[0]) == requested_yolo_class)]
    if matching:
        best = max(matching, key=lambda b: float(b.conf))
        x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
        W, H = img.size
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if (x2 - x1) >= 20 and (y2 - y1) >= 20:
            return img.crop((x1, y1, x2, y2))
    if gt_info is not None:
        x1, y1, x2, y2 = gt_info["bbox"]
        W, H = img.size
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if (x2 - x1) >= 20 and (y2 - y1) >= 20:
            return img.crop((x1, y1, x2, y2))
    return img


# ── Stage 4: CLIP base + Config A embedding ──────────────────────────────
print("\n[STATUS] ---> Loading base CLIP + embedding gallery (Config A)...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.to(device).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")


def get_image_embedding(pil_image, model=None):
    m = model if model is not None else clip_model
    tensor = clip_preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = m.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().squeeze().astype("float32")


def get_text_embedding(text, model=None):
    m = model if model is not None else clip_model
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        emb = m.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().squeeze().astype("float32")


def fuse(img_emb, txt_emb, alpha):
    f = alpha * img_emb + (1 - alpha) * txt_emb
    return (f / (np.linalg.norm(f) + 1e-9)).astype("float32")


def build_index(embeddings):
    idx = hnswlib.Index(space="cosine", dim=embeddings.shape[1])
    idx.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
    idx.add_items(embeddings)
    idx.set_ef(50)
    return idx


gallery_emb_A, gallery_item_ids, gallery_img_paths = [], [], []
for img_path, item_id in tqdm(gallery_data, desc="Embedding (Config A)"):
    try:
        emb = get_image_embedding(crop_with_yolo(img_path))
        gallery_emb_A.append(emb); gallery_item_ids.append(item_id); gallery_img_paths.append(img_path)
    except Exception: pass
gallery_emb_A = np.array(gallery_emb_A).astype("float32")
index_A = build_index(gallery_emb_A)
index_A.save_index(f"{OUTPUT}/artifacts/index_A.bin")
print(f" index_A.bin built with {len(gallery_emb_A)} vectors.")


# ── Stage 5: BLIP captions + Config B fusion ─────────────────────────────
print("\n[STATUS] ---> Loading BLIP, generating captions (Config B)...")
from transformers import BlipProcessor, BlipForConditionalGeneration
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device).eval()

gallery_captions = []
for img_path, _ in tqdm(gallery_data, desc="Captioning"):
    try:
        inputs = blip_processor(images=crop_with_yolo(img_path), return_tensors="pt").to(device)
        with torch.no_grad(): out = blip_model.generate(**inputs, max_new_tokens=40)
        gallery_captions.append(blip_processor.decode(out[0], skip_special_tokens=True).strip())
    except Exception:
        gallery_captions.append("a clothing item")

text_emb_A = np.array([get_text_embedding(c) for c in gallery_captions]).astype("float32")
for alpha in ALPHAS:
    fused = np.array([fuse(gallery_emb_A[i], text_emb_A[i], alpha) for i in range(len(gallery_emb_A))])
    idx = build_index(fused)
    idx.save_index(f"{OUTPUT}/artifacts/index_B_{str(alpha).replace('0.', '')}.bin")
print(" Config B indices built.")

del blip_model; gc.collect(); torch.cuda.empty_cache()


# ── Stage 6: CLIP fine-tune per seed (Config C) ──────────────────────────
class FashionDataset(Dataset):
    def __init__(self, data, transform):
        self.data, self.transform = data, transform
        all_ids = sorted(set(iid for _, iid in data))
        self.id_to_int = {iid: i for i, iid in enumerate(all_ids)}
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        path, item_id = self.data[idx]
        try: img = crop_with_yolo(path)
        except Exception: img = Image.new("RGB", (224, 224))
        return self.transform(img), self.id_to_int[item_id]


class InfoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__(); self.temp = temperature
    def forward(self, embeddings, labels):
        emb = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-9)
        sim = torch.matmul(emb, emb.T) / self.temp
        lab = labels.view(-1, 1)
        pos_mask = (lab == lab.T).float(); pos_mask.fill_diagonal_(0.0)
        eye = torch.eye(len(labels), device=emb.device).bool()
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp = torch.exp(sim).masked_fill(eye, 0.0)
        log_prob = sim - torch.log(exp.sum(dim=1, keepdim=True) + 1e-9)
        num_pos = pos_mask.sum(dim=1).clamp(min=1.0)
        return (-(log_prob * pos_mask).sum(dim=1) / num_pos).mean()


for seed in CURRENT_SEEDS:
    print(f"\n[STATUS] ---> Fine-tuning CLIP, seed {seed}...")
    set_seed(seed)
    with mlflow.start_run(run_name=f"clip_finetune_seed{seed}"):
        mlflow.log_params({"seed": seed, "epochs": CLIP_EPOCHS, "lr": CLIP_LR,
                            "trainable_blocks": TRAINABLE_BLOCKS, "n_train_images": len(train_data)})
        clip_ft, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        clip_ft = clip_ft.to(device)
        for p in clip_ft.visual.parameters(): p.requires_grad = False
        for block in list(clip_ft.visual.transformer.resblocks)[-TRAINABLE_BLOCKS:]:
            for p in block.parameters(): p.requires_grad = True

        loader = DataLoader(FashionDataset(train_data, clip_preprocess), batch_size=16,
                             shuffle=True, num_workers=0, pin_memory=True)
        criterion = InfoNCE(temperature=CLIP_TEMP)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, clip_ft.parameters()),
                                 lr=CLIP_LR, weight_decay=CLIP_WD)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=1),
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, CLIP_EPOCHS - 1))
        ], milestones=[1])

        clip_ft.train()
        for epoch in range(CLIP_EPOCHS):
            total_loss, n_batches = 0.0, 0
            for imgs, labels in tqdm(loader, desc=f"seed {seed} epoch {epoch+1}/{CLIP_EPOCHS}", leave=False):
                loss = criterion(clip_ft.encode_image(imgs.to(device)), labels.to(device))
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(clip_ft.parameters(), max_norm=1.0)
                optimizer.step(); total_loss += loss.item(); n_batches += 1
            scheduler.step()
            avg_loss = total_loss / max(1, n_batches)
            print(f" epoch {epoch+1}/{CLIP_EPOCHS} | loss={avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        ckpt_path = f"{OUTPUT}/artifacts/clip_finetuned_{seed}.pt"
        torch.save(clip_ft.state_dict(), ckpt_path)
        mlflow.log_artifact(ckpt_path)

    # Config C embedding + fused indices for this seed
    clip_ft.eval()
    gallery_emb_C = []
    for img_path, _ in tqdm(gallery_data, desc=f"Re-embedding seed {seed}"):
        try: gallery_emb_C.append(get_image_embedding(crop_with_yolo(img_path), model=clip_ft))
        except Exception: gallery_emb_C.append(np.zeros(512, dtype="float32"))
    gallery_emb_C = np.array(gallery_emb_C).astype("float32")
    for alpha in ALPHAS:
        fused = np.array([fuse(gallery_emb_C[i], text_emb_A[i], alpha) for i in range(len(gallery_emb_C))])
        idx = build_index(fused)
        alpha_tag = str(alpha).replace("0.", "")
        idx.save_index(f"{OUTPUT}/artifacts/index_C_{alpha_tag}_{seed}.bin")
    print(f" Config C indices built for seed {seed}.")
    del clip_ft; gc.collect(); torch.cuda.empty_cache()


# ── Stage 7: metadata.csv ────────────────────────────────────────────────
metadata_df = pd.DataFrame([
    {"item_id": iid, "relative_path": f"{iid}/{Path(p).name}", "caption": cap,
     "clothes_type": bbox_map.get((iid, Path(p).name), {}).get("clothes_type")}
    for p, iid, cap in zip(gallery_img_paths, gallery_item_ids, gallery_captions)
])
metadata_df.to_csv(f"{OUTPUT}/artifacts/gallery_metadata.csv", index=False)
print(f"\n[STATUS] ---> gallery_metadata.csv written ({len(metadata_df)} rows).")


# ── Package everything for download ──────────────────────────────────────
print("\n[STATUS] ---> Zipping artifacts + mlruns + gallery images for download...")
shutil.make_archive(f"{OUTPUT}/mlops_kaggle_output", "zip", root_dir=OUTPUT,
                     base_dir="artifacts")
get_ipython().system(f'cd {OUTPUT} && zip -rq mlops_kaggle_output.zip mlruns small_split')

from IPython.display import FileLink, display
print("\n ALL DONE. Download this and follow MLOPS/kaggle/README.md to unpack it locally:")
display(FileLink("mlops_kaggle_output.zip"))
