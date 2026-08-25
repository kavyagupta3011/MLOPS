"""
scripts/generate_ci_fixture.py — builds ci_fixture/, a tiny, synthetic
stand-in for the DeepFashion In-Shop dataset, used ONLY by CI.

Why this exists: `.github/workflows/ci.yml`'s `pipeline` job runs `dvc
repro` on every push, to demonstrate P1 (CI/CD) actually exercising the
ML pipeline, not just linting. The real dataset is gigabytes and
deliberately gitignored (see README "Getting the dataset") — it can
never live in the repo. This script generates a handful of tiny
(128x128, solid-color) placeholder images plus correctly-formatted
`list_eval_partition.txt` / `list_bbox_inshop.txt` files that match the
real dataset's exact format (same columns, same path conventions), so
every pipeline stage (YOLO label conversion, YOLO fine-tune, CLIP
fine-tune, evaluation) runs against real code paths with real file I/O —
it just trains/evaluates on nonsense images. The point is proving the
*pipeline mechanics* work end to end automatically; it says nothing about
model quality. Real training happens on Kaggle against the real dataset
(see kaggle/README.md) — this fixture is CI-only.

Output layout mirrors the real dataset exactly, so the exact same
src/pipeline/*.py code paths run unmodified:
  ci_fixture/DeepFashionRetrieval/
    list_eval_partition.txt
    list_bbox_inshop.txt
    img/img/WOMEN/Blouses_Shirts/id_000000XX/NN_K_view.jpg

12 synthetic item_ids, 3 images each (train/query/gallery), each image
tagged with a real, non-degenerate bounding box (so the "skip if bbox
too small" check in 01_prepare_yolo_labels.py doesn't just skip
everything) — this is what lets validate_dataset's eligible-item-count
and bbox-coverage checks pass with room to spare even under a shrunk
`sampling.n_items` (see ci.yml's "Shrink params for CI fixture" step).

Committed output, regenerate with:
  python scripts/generate_ci_fixture.py
"""

import os
import shutil

from PIL import Image, ImageDraw

OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "ci_fixture", "DeepFashionRetrieval")
N_ITEMS = 12
CATEGORY = "Blouses_Shirts"
GENDER = "WOMEN"
IMG_SIZE = 128
BBOX = (20, 20, 108, 108)  # x1, y1, x2, y2 — well above the 0.01-fraction minimum
CLOTHES_TYPES = [1, 2, 3]  # cycles upper/lower/full-body across items
VIEWS = [("1", "front", "train"), ("2", "side", "gallery"), ("3", "back", "query")]

# A distinct fill color per item so the tiny images aren't literally identical
# (helps a human skimming ci_fixture/ visually tell items apart; irrelevant to the pipeline).
COLORS = [
    (200, 60, 60), (60, 200, 60), (60, 60, 200), (200, 200, 60),
    (200, 60, 200), (60, 200, 200), (150, 100, 50), (100, 150, 50),
    (50, 100, 150), (150, 50, 100), (100, 50, 150), (50, 150, 100),
]


def main():
    if os.path.exists(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)
    img_root = os.path.join(OUT_ROOT, "img", "img", GENDER, CATEGORY)
    os.makedirs(img_root, exist_ok=True)

    partition_lines = []
    bbox_lines = []
    n_partition_rows = 0
    n_bbox_rows = 0

    for i in range(1, N_ITEMS + 1):
        item_id = f"id_{i:08d}"
        clothes_type = CLOTHES_TYPES[(i - 1) % len(CLOTHES_TYPES)]
        color = COLORS[(i - 1) % len(COLORS)]
        item_dir = os.path.join(img_root, item_id)
        os.makedirs(item_dir, exist_ok=True)

        for seq, view, split in VIEWS:
            filename = f"{seq}_1_{view}.jpg"
            img_path = f"img/{GENDER}/{CATEGORY}/{item_id}/{filename}"

            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=color)
            draw = ImageDraw.Draw(img)
            draw.rectangle(BBOX, outline=(255, 255, 255), width=3)
            img.save(os.path.join(item_dir, filename), "JPEG", quality=70)

            partition_lines.append(f"{img_path} {item_id} {split}")
            n_partition_rows += 1

            x1, y1, x2, y2 = BBOX
            bbox_lines.append(f"{img_path} {clothes_type} 1 {x1} {y1} {x2} {y2}")
            n_bbox_rows += 1

    partition_path = os.path.join(OUT_ROOT, "list_eval_partition.txt")
    with open(partition_path, "w") as f:
        f.write(f"{n_partition_rows}\n")
        f.write("image_name item_id evaluation_status\n")
        f.write("\n".join(partition_lines) + "\n")

    bbox_path = os.path.join(OUT_ROOT, "list_bbox_inshop.txt")
    with open(bbox_path, "w") as f:
        f.write(f"{n_bbox_rows}\n")
        f.write("image_name clothes_type pose_type x_1 y_1 x_2 y_2\n")
        f.write("\n".join(bbox_lines) + "\n")

    n_images = sum(len(files) for _, _, files in os.walk(img_root))
    print(f"[generate_ci_fixture] {N_ITEMS} items, {n_partition_rows} partition rows, "
          f"{n_bbox_rows} bbox rows, {n_images} images -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
