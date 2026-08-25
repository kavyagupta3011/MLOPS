"""
src/pipeline/02_train_yolo.py — Stage 3.

Fine-tunes YOLOv8n on the small clothing-bbox dataset built by stage 1,
same as training.ipynb Cell 4 — but every run is now logged to MLflow
(component C7 / principle P7: metadata tracking) instead of only
being visible in a notebook's scrollback.

Writes:
  artifacts/yolo/best.pt
Logs to MLflow:
  params: epochs, imgsz, batch, n_train_images
  metrics: final box/cls/dfl loss, mAP50 (from Ultralytics results)
  artifact: best.pt

Run:
  python -m src.pipeline.02_train_yolo
"""

import os
import shutil

import mlflow
from ultralytics import YOLO

from src.common import load_params


def main():
    params = load_params()
    p, y, mf = params["paths"], params["yolo"], params["mlflow"]

    mlflow.set_tracking_uri(mf["tracking_uri"])
    mlflow.set_experiment(mf["experiment_name"])

    yaml_path = os.path.join(p["yolo_dataset_dir"], "clothing.yaml")
    n_train_images = sum(
        len(files) for _, _, files in os.walk(os.path.join(p["yolo_dataset_dir"], "images", "train"))
    )

    with mlflow.start_run(run_name="yolo_finetune"):
        mlflow.log_params({
            "stage": "yolo_finetune",
            "base_weights": y["base_weights"],
            "epochs": y["epochs"],
            "imgsz": y["imgsz"],
            "batch": y["batch"],
            "n_train_images": n_train_images,
        })

        # project must be an ABSOLUTE path: some Ultralytics versions resolve a
        # relative `project` against their own configured settings["runs_dir"]
        # (e.g. a path inside site-packages on a fresh CI runner) instead of
        # the current working directory — which silently writes best.pt
        # somewhere other than where this script looks for it afterward.
        # Passing an absolute path sidesteps that resolution entirely.
        project_dir = os.path.abspath(os.path.join(p["artifacts_dir"], "yolo_runs"))

        model = YOLO(y["base_weights"])
        results = model.train(
            data=yaml_path,
            epochs=y["epochs"],
            imgsz=y["imgsz"],
            batch=y["batch"],
            project=project_dir,
            name="yolo_clothing",
            exist_ok=True,
            verbose=False,
        )

        # Ultralytics writes metrics into results.results_dict on the trainer;
        # log whatever numeric metrics are available without hard failing
        # if the Ultralytics version changes its internal result shape.
        try:
            for k, v in results.results_dict.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k.replace("(", "").replace(")", ""), v)
        except Exception as e:
            print(f"[train_yolo] Could not log Ultralytics metrics: {e}")

        # Prefer the actual save_dir Ultralytics used (belt-and-suspenders
        # against any further path-resolution surprises across versions),
        # falling back to the path we asked for if that attribute is absent.
        save_dir = getattr(results, "save_dir", None) or os.path.join(project_dir, "yolo_clothing")
        best_src = os.path.join(str(save_dir), "weights", "best.pt")
        os.makedirs(os.path.join(p["artifacts_dir"], "yolo"), exist_ok=True)
        best_dst = os.path.join(p["artifacts_dir"], "yolo", "best.pt")
        shutil.copy2(best_src, best_dst)
        mlflow.log_artifact(best_dst)

        print(f"[train_yolo] Fine-tuned weights -> {best_dst}")


if __name__ == "__main__":
    main()
