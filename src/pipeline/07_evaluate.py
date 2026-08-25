"""
src/pipeline/07_evaluate.py — Stage 8: evaluation + regression baseline.

Runs Recall@K / NDCG@K / mAP@K (same formulas as eval.py / the report's
section 9) for every built index — Config A, both Config B alphas, and
Config C for every (alpha, seed) pair — against the small query split.

To keep a small-dataset run fast, the query pipeline here fuses the query
image embedding with its own BLIP caption (mirroring app.py's live query
flow) rather than eval.py's heavier BLIP-2 ITM re-ranking — a deliberate
simplification for iteration speed, noted in docs/principles_mapping.md.

Writes:
  artifacts/metrics.json         — every config's mean Recall/NDCG/mAP per K
Logs to MLflow:
  one run per config, with all K-value metrics

This is also what src/ci/regression_gate.py reads to decide whether a
new set of results is worse than the currently-approved baseline.

Run:
  python -m src.pipeline.07_evaluate
"""

import json
import os
from pathlib import Path

import mlflow
import numpy as np
import open_clip
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor
from ultralytics import YOLO

from src.common import (crop_with_yolo, fuse_embeddings, get_device,
                         get_image_embedding, get_text_embedding, load_index,
                         load_params, parse_bbox_file)

BLIP_CAPTION_ID = "Salesforce/blip-image-captioning-base"


def compute_metrics(retrieved_ids, gt_id, total_relevant, k_values):
    out = {}
    for k in k_values:
        top_k = retrieved_ids[:k]
        recall = 1 if gt_id in top_k else 0
        dcg = sum(1.0 / np.log2(i + 2) for i, r in enumerate(top_k) if r == gt_id)
        ideal_hits = min(total_relevant, k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        hits, ap = 0, 0.0
        for i, r in enumerate(top_k):
            if r == gt_id:
                hits += 1
                ap += hits / (i + 1)
        normalizer = min(total_relevant, k)
        out[k] = {"recall": recall, "ndcg": ndcg, "map": ap / normalizer if normalizer > 0 else 0.0}
    return out


def load_query_data(query_dir, metadata):
    query_data = []
    for item_folder in Path(query_dir).iterdir():
        if not item_folder.is_dir():
            continue
        gt_id = item_folder.name
        rows = metadata[metadata["item_id"] == gt_id]
        requested_type = rows.iloc[0]["clothes_type"] if len(rows) else None
        for img_file in sorted(item_folder.glob("*.jpg")):
            query_data.append((str(img_file), gt_id, requested_type))
    return query_data


def evaluate_index(config_name, index, query_data, metadata, clip_model, clip_preprocess,
                    clip_tokenizer, yolo_model, blip_processor, blip_model, bbox_map,
                    device, k_values, alpha=None):
    gallery_id_counts = metadata["item_id"].value_counts().to_dict()
    results = {k: {"recall": [], "ndcg": [], "map": []} for k in k_values}

    for img_path, gt_id, requested_type in tqdm(query_data, desc=config_name, leave=False):
        try:
            filename = Path(img_path).name
            cropped, _, _ = crop_with_yolo(
                yolo_model, img_path, requested_type=requested_type,
                bbox_map=bbox_map, item_id=gt_id, filename=filename,
            )
            img_emb = get_image_embedding(clip_model, clip_preprocess, cropped, device)

            query_emb = img_emb
            if alpha is not None:
                inputs = blip_processor(images=cropped, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = blip_model.generate(**inputs, max_new_tokens=30)
                caption = blip_processor.decode(out[0], skip_special_tokens=True).strip()
                txt_emb = get_text_embedding(clip_model, clip_tokenizer, caption, device)
                query_emb = fuse_embeddings(img_emb, txt_emb, alpha)

            search_k = max(k_values) * 5
            labels, _ = index.knn_query(query_emb, k=min(search_k, index.get_current_count()))
            retrieved_ids = [metadata.iloc[int(lbl)]["item_id"] for lbl in labels[0]][: max(k_values)]

            m = compute_metrics(retrieved_ids, gt_id, gallery_id_counts.get(gt_id, 1), k_values)
            for k in k_values:
                results[k]["recall"].append(m[k]["recall"])
                results[k]["ndcg"].append(m[k]["ndcg"])
                results[k]["map"].append(m[k]["map"])
        except Exception as e:
            print(f"  error on {img_path}: {e}")

    return {
        k: {
            "recall": float(np.mean(results[k]["recall"])) if results[k]["recall"] else 0.0,
            "ndcg": float(np.mean(results[k]["ndcg"])) if results[k]["ndcg"] else 0.0,
            "map": float(np.mean(results[k]["map"])) if results[k]["map"] else 0.0,
            "n_queries": len(results[k]["recall"]),
        }
        for k in k_values
    }


def main():
    params = load_params()
    p, e, mf = params["paths"], params["eval"], params["mlflow"]
    device = get_device()
    k_values = e["k_values"]

    mlflow.set_tracking_uri(mf["tracking_uri"])
    mlflow.set_experiment(mf["experiment_name"])

    metadata = pd.read_csv(os.path.join(p["artifacts_dir"], "gallery_metadata.csv"))
    query_dir = os.path.join(p["small_split_dir"], "query")
    query_data = load_query_data(query_dir, metadata)
    if e["max_queries"] and len(query_data) > e["max_queries"]:
        query_data = query_data[: e["max_queries"]]
    print(f"[evaluate] {len(query_data)} query images.")

    bbox_map = parse_bbox_file(p["bbox_file"])
    yolo_model = YOLO(os.path.join(p["artifacts_dir"], "yolo", "best.pt"))
    blip_processor = BlipProcessor.from_pretrained(BLIP_CAPTION_ID)
    blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_CAPTION_ID).to(device).eval()
    tokenizer = open_clip.get_tokenizer(params["clip"]["arch"])

    all_metrics = {}

    # --- Config A ---
    clip_base, _, clip_preprocess = open_clip.create_model_and_transforms(
        params["clip"]["arch"], pretrained=params["clip"]["pretrained"]
    )
    clip_base = clip_base.to(device).eval()
    index_a = load_index(os.path.join(p["artifacts_dir"], "index_A.bin"))
    all_metrics["A"] = evaluate_index(
        "A (vision-only)", index_a, query_data, metadata, clip_base, clip_preprocess,
        tokenizer, yolo_model, blip_processor, blip_model, bbox_map, device, k_values, alpha=None,
    )

    # --- Config B ---
    for alpha in params["fusion"]["alphas"]:
        alpha_tag = str(alpha).replace("0.", "")
        idx_path = os.path.join(p["artifacts_dir"], f"index_B_{alpha_tag}.bin")
        if not os.path.exists(idx_path):
            continue
        index_b = load_index(idx_path)
        all_metrics[f"B_alpha{alpha}"] = evaluate_index(
            f"B alpha={alpha}", index_b, query_data, metadata, clip_base, clip_preprocess,
            tokenizer, yolo_model, blip_processor, blip_model, bbox_map, device, k_values, alpha=alpha,
        )

    # --- Config C ---
    for seed in params["clip"]["seeds"]:
        ckpt_path = os.path.join(p["artifacts_dir"], "clip_checkpoints", f"clip_finetuned_{seed}.pt")
        if not os.path.exists(ckpt_path):
            continue
        clip_ft, _, _ = open_clip.create_model_and_transforms(
            params["clip"]["arch"], pretrained=params["clip"]["pretrained"]
        )
        clip_ft.load_state_dict(torch.load(ckpt_path, map_location=device))
        clip_ft = clip_ft.to(device).eval()

        for alpha in params["fusion"]["alphas"]:
            alpha_tag = str(alpha).replace("0.", "")
            idx_path = os.path.join(p["artifacts_dir"], f"index_C_{alpha_tag}_{seed}.bin")
            if not os.path.exists(idx_path):
                continue
            index_c = load_index(idx_path)
            all_metrics[f"C_alpha{alpha}_seed{seed}"] = evaluate_index(
                f"C alpha={alpha} seed={seed}", index_c, query_data, metadata, clip_ft,
                clip_preprocess, tokenizer, yolo_model, blip_processor, blip_model, bbox_map,
                device, k_values, alpha=alpha,
            )

    os.makedirs(p["artifacts_dir"], exist_ok=True)
    metrics_path = os.path.join(p["artifacts_dir"], "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[evaluate] metrics written -> {metrics_path}")

    for config_name, per_k in all_metrics.items():
        with mlflow.start_run(run_name=f"eval_{config_name}"):
            mlflow.log_param("config", config_name)
            for k, vals in per_k.items():
                # MLflow metric names may only contain alphanumerics, "_", "-",
                # ".", " ", ":", "/" — "@" (as in "recall@5") is rejected outright.
                mlflow.log_metric(f"recall_at_{k}", vals["recall"])
                mlflow.log_metric(f"ndcg_at_{k}", vals["ndcg"])
                mlflow.log_metric(f"map_at_{k}", vals["map"])

    print("\n" + "=" * 70)
    print(f"{'Config':<22} | {'K':<4} | {'Recall':<8} | {'NDCG':<8} | {'mAP':<8}")
    print("=" * 70)
    for config_name, per_k in all_metrics.items():
        for i, (k, vals) in enumerate(per_k.items()):
            label = config_name if i == 0 else ""
            print(f"{label:<22} | {k:<4} | {vals['recall']:.4f}  | {vals['ndcg']:.4f}  | {vals['map']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
