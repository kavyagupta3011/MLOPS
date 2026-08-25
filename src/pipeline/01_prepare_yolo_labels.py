"""
src/pipeline/01_prepare_yolo_labels.py — Stage 2.

Converts DeepFashion ground-truth bounding boxes (corner format) into YOLO
label files (normalized center/width/height format), exactly as described
in the report's section 3.2. Only touches images in the small `train`
split produced by make_small_dataset.py.

Reads:
  paths.small_split_dir/train/<item_id>/*.jpg
  paths.bbox_file
Writes:
  paths.yolo_dataset_dir/images/train/*.jpg
  paths.yolo_dataset_dir/labels/train/*.txt
  paths.yolo_dataset_dir/clothing.yaml

Run:
  python -m src.pipeline.01_prepare_yolo_labels
"""

import os
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

from src.common import CLOTHES_CLASS, load_params, parse_bbox_file


def main():
    params = load_params()
    p = params["paths"]

    train_dir = os.path.join(p["small_split_dir"], "train")
    yolo_dir = p["yolo_dataset_dir"]
    img_dir = os.path.join(yolo_dir, "images", "train")
    lbl_dir = os.path.join(yolo_dir, "labels", "train")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    print(f"[prepare_yolo_labels] Parsing {p['bbox_file']} ...")
    bbox_map = parse_bbox_file(p["bbox_file"])
    print(f"[prepare_yolo_labels] {len(bbox_map):,} GT bbox entries loaded.")

    train_data = []
    for item_folder in sorted(Path(train_dir).iterdir()):
        if item_folder.is_dir():
            for img_file in sorted(item_folder.glob("*.jpg")):
                train_data.append((str(img_file), item_folder.name))

    count = 0
    for img_path, item_id in tqdm(train_data, desc="Writing YOLO labels"):
        filename = Path(img_path).name
        info = bbox_map.get((item_id, filename))
        if info is None:
            continue
        try:
            img = Image.open(img_path)
            W, H = img.size
        except Exception:
            continue

        x1, y1, x2, y2 = info["bbox"]
        cx = ((x1 + x2) / 2) / W
        cy = ((y1 + y2) / 2) / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
        bw, bh = min(max(bw, 0), 1), min(max(bh, 0), 1)
        if bw < 0.01 or bh < 0.01:
            continue

        class_id = CLOTHES_CLASS[info["clothes_type"]]
        stem = f"{item_id}_{Path(filename).stem}"
        dst_img = os.path.join(img_dir, f"{stem}.jpg")
        if not os.path.exists(dst_img):
            import shutil
            shutil.copy2(img_path, dst_img)
        with open(os.path.join(lbl_dir, f"{stem}.txt"), "w") as f:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        count += 1

    yaml_path = os.path.join(yolo_dir, "clothing.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump({
            "path": os.path.abspath(yolo_dir),
            "train": "images/train",
            "val": "images/train",
            "nc": 3,
            "names": ["upper-body", "lower-body", "full-body"],
        }, f)

    print(f"[prepare_yolo_labels] {count} images labelled. Dataset yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
