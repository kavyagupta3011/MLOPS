"""
src/pipeline/05_finetune_clip.py — Stage 6: Configuration C (fine-tuned CLIP).

Same supervised-contrastive fine-tune as training.ipynb Cell 8 — only the
last 4 visual transformer blocks are unfrozen, InfoNCE loss, AdamW,
warmup+cosine schedule — but parameterized by seed and logged to MLflow
per run (params, per-epoch loss, final checkpoint as an artifact). This
replaces hand-copying Table 4's losses into the report: query MLflow
instead.

Run one seed:
  python -m src.pipeline.05_finetune_clip --seed 16

Run every seed listed in params.yaml clip.seeds:
  python -m src.pipeline.05_finetune_clip --all-seeds
"""

import argparse
import os
from pathlib import Path

import mlflow
import open_clip
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.common import crop_with_yolo, get_device, load_params, parse_bbox_file, set_seed
from ultralytics import YOLO


class FashionDataset(Dataset):
    """Crops each training image with YOLO, returns (tensor, int_label)."""

    def __init__(self, data, transform, yolo_model, bbox_map):
        self.data = data
        self.transform = transform
        self.yolo_model = yolo_model
        self.bbox_map = bbox_map
        all_ids = sorted({iid for _, iid in data})
        self.id_to_int = {iid: i for i, iid in enumerate(all_ids)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, item_id = self.data[idx]
        try:
            filename = Path(path).name
            img, _, _ = crop_with_yolo(
                self.yolo_model, path, bbox_map=self.bbox_map, item_id=item_id, filename=filename
            )
        except Exception:
            from PIL import Image
            img = Image.new("RGB", (224, 224))
        return self.transform(img), self.id_to_int[item_id]


class InfoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, embeddings, labels):
        emb = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-9)
        sim = torch.matmul(emb, emb.T) / self.temp
        lab = labels.view(-1, 1)
        pos_mask = (lab == lab.T).float()
        pos_mask.fill_diagonal_(0.0)
        eye = torch.eye(len(labels), device=emb.device).bool()
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp = torch.exp(sim).masked_fill(eye, 0.0)
        log_prob = sim - torch.log(exp.sum(dim=1, keepdim=True) + 1e-9)
        num_pos = pos_mask.sum(dim=1).clamp(min=1.0)
        loss_per_item = -(log_prob * pos_mask).sum(dim=1) / num_pos
        return loss_per_item.mean()


def run_one_seed(params, seed):
    p, c, mf = params["paths"], params["clip"], params["mlflow"]
    device = get_device()
    set_seed(seed)

    mlflow.set_tracking_uri(mf["tracking_uri"])
    mlflow.set_experiment(mf["experiment_name"])

    bbox_map = parse_bbox_file(p["bbox_file"])
    yolo_model = YOLO(os.path.join(p["artifacts_dir"], "yolo", "best.pt"))

    train_dir = os.path.join(p["small_split_dir"], "train")
    train_data = []
    for item_folder in sorted(Path(train_dir).iterdir()):
        if item_folder.is_dir():
            for img_file in sorted(item_folder.glob("*.jpg")):
                train_data.append((str(img_file), item_folder.name))

    with mlflow.start_run(run_name=f"clip_finetune_seed{seed}"):
        mlflow.log_params({
            "stage": "clip_finetune",
            "seed": seed,
            "arch": c["arch"],
            "trainable_blocks": c["trainable_blocks"],
            "epochs": c["epochs"],
            "batch_size": c["batch_size"],
            "lr": c["lr"],
            "weight_decay": c["weight_decay"],
            "temperature": c["temperature"],
            "n_train_images": len(train_data),
        })

        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            c["arch"], pretrained=c["pretrained"]
        )
        clip_model = clip_model.to(device)
        for param in clip_model.visual.parameters():
            param.requires_grad = False
        for block in list(clip_model.visual.transformer.resblocks)[-c["trainable_blocks"]:]:
            for param in block.parameters():
                param.requires_grad = True

        dataset = FashionDataset(train_data, clip_preprocess, yolo_model, bbox_map)
        loader = DataLoader(dataset, batch_size=c["batch_size"], shuffle=True,
                             num_workers=0, pin_memory=(device == "cuda"))

        criterion = InfoNCE(temperature=c["temperature"])
        optimizer = optim.AdamW(
            filter(lambda p_: p_.requires_grad, clip_model.parameters()),
            lr=c["lr"], weight_decay=c["weight_decay"],
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                             total_iters=c["warmup_epochs"]),
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, c["epochs"] - c["warmup_epochs"])
                ),
            ],
            milestones=[c["warmup_epochs"]],
        )

        clip_model.train()
        print(f"[finetune_clip] seed={seed} — {c['epochs']} epochs over {len(train_data)} images")
        for epoch in range(c["epochs"]):
            total_loss, n_batches = 0.0, 0
            for imgs, labels in tqdm(loader, desc=f"seed {seed} | epoch {epoch + 1}/{c['epochs']}", leave=False):
                loss = criterion(clip_model.encode_image(imgs.to(device)), labels.to(device))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clip_model.parameters(), max_norm=c["grad_clip_norm"])
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            scheduler.step()
            avg_loss = total_loss / max(1, n_batches)
            print(f"  epoch {epoch + 1}/{c['epochs']} | loss={avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        out_dir = os.path.join(p["artifacts_dir"], "clip_checkpoints")
        os.makedirs(out_dir, exist_ok=True)
        ckpt_path = os.path.join(out_dir, f"clip_finetuned_{seed}.pt")
        torch.save(clip_model.state_dict(), ckpt_path)
        mlflow.log_artifact(ckpt_path)
        print(f"[finetune_clip] seed={seed} checkpoint -> {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--all-seeds", action="store_true")
    args = parser.parse_args()

    params = load_params()
    seeds = params["clip"]["seeds"] if args.all_seeds else [args.seed or params["clip"]["seeds"][0]]
    for seed in seeds:
        run_one_seed(params, seed)


if __name__ == "__main__":
    main()
