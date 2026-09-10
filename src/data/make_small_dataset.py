"""
src/data/make_small_dataset.py — Stage 1 of the pipeline.

Same job as the team's splitintoquerygallerytrain.ipynb, but instead of
copying all 52,712 images it samples `sampling.n_items` item_ids (see
params.yaml) and only copies those — this is the literal "smaller dataset"
version of the retrain.

Reads:
  paths.partition_file   (list_eval_partition.txt: image_name item_id split)
Writes:
  paths.small_split_dir/{train,query,gallery}/<item_id>/<flattened_name>.jpg
  paths.full_gallery_dir/<item_id>/<flattened_name>.jpg

Run:
  python -m src.data.make_small_dataset
"""

import os
import random
import shutil
from collections import defaultdict

from src.common import load_params


def read_partition_file(partition_file: str):
    with open(partition_file, "r") as f:
        lines = f.readlines()[2:]  # line 0 = count, line 1 = header
    entries = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            img_path, item_id, split = parts
            entries.append((img_path, item_id, split))
    return entries


def main():
    params = load_params()
    p = params["paths"]
    s = params["sampling"]

    partition_file = p["partition_file"]
    base_img_dir = os.path.join(p["base_data_dir"], "img", "img")
    output_dir = p["small_split_dir"]
    full_gallery_dir = p["full_gallery_dir"]

    if not os.path.exists(partition_file):
        raise FileNotFoundError(
            f"{partition_file} not found. See README.md 'Getting the dataset' — "
            "this repo does not ship the raw DeepFashion files."
        )

    print(f"[make_small_dataset] Reading {partition_file} ...")
    entries = read_partition_file(partition_file)
    print(f"[make_small_dataset] {len(entries):,} total entries in the full dataset.")

    # Group by item_id so we sample whole products, not random loose images —
    # a query/gallery pair only makes sense if the item's images span >=2 files.
    by_item = defaultdict(list)
    for img_path, item_id, split in entries:
        by_item[item_id].append((img_path, split))

    eligible_items = [
        iid for iid, imgs in by_item.items() if len(imgs) >= s["min_images_per_item"]
    ]
    rng = random.Random(s["split_seed"])
    rng.shuffle(eligible_items)
    chosen_items = set(eligible_items[: s["n_items"]])
    print(f"[make_small_dataset] Sampled {len(chosen_items)} item_ids "
          f"(of {len(eligible_items)} eligible, {len(by_item)} total).")

    for split in ["train", "query", "gallery"]:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    os.makedirs(full_gallery_dir, exist_ok=True)

    copied, missing, full_gallery_images = 0, 0, 0
    full_gallery_items = set()
    for img_path, item_id, split in entries:
        src = os.path.join(base_img_dir, img_path.replace("img/", "", 1))
        if not os.path.exists(src):
            if item_id in chosen_items or split == "gallery":
                missing += 1
            continue

        file_name = img_path.replace("img/", "", 1).replace("/", "_")
        if split == "gallery":
            full_dest_folder = os.path.join(full_gallery_dir, item_id)
            os.makedirs(full_dest_folder, exist_ok=True)
            shutil.copy2(src, os.path.join(full_dest_folder, file_name))
            full_gallery_images += 1
            full_gallery_items.add(item_id)

        if item_id in chosen_items:
            dest_folder = os.path.join(output_dir, split, item_id)
            os.makedirs(dest_folder, exist_ok=True)
            shutil.copy2(src, os.path.join(dest_folder, file_name))
            copied += 1

    print(f"[make_small_dataset] Copied {copied} images, {missing} missing on disk.")
    for split in ["train", "query", "gallery"]:
        split_path = os.path.join(output_dir, split)
        n = sum(len(files) for _, _, files in os.walk(split_path))
        print(f"  {split}: {n} images")
    print(f"[make_small_dataset] Full gallery: {full_gallery_images} images "
          f"across {len(full_gallery_items)} items -> {full_gallery_dir}")


if __name__ == "__main__":
    main()
