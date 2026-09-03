"""
src/ci/auto_promote_best.py — "Best Trial -> MLflow Registry", automated.

src/ci/promote_to_registry.py promotes one specific, named run/config —
useful when a human has looked at champion_challenger.py's table and
picked a winner. This script is the unattended version the diagram
actually draws: after evaluate.py has scored every config (including
every AutoML/Optuna trial that was fine-tuned and evaluated), pick the
single best-performing *trainable* config automatically and promote it,
no human in the loop.

"Trainable" here means Configuration C (fine-tuned CLIP) — Config A/B
reuse the frozen pretrained CLIP checkpoint with no model artifact of
their own to register, so they're not candidates for the registry (the
registry tracks model *versions*; A/B don't have MLflow-run checkpoints
to attach one to).

Selection metric: regression_gate.metric from params.yaml (default
recall@5), same metric the CI gate itself is judged against, so
"best" and "passes CI" mean the same number.

Run, after `dvc repro` (or an Optuna sweep) has produced artifacts/metrics.json:
  python -m src.ci.auto_promote_best
Exit code 0 = promoted (or metrics didn't contain any Config C entries — a
no-op, not a failure), 1 = an actual error occurred while promoting.
"""

import json
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient

from src.ci.promote_to_registry import find_run_id_by_name
from src.common import load_params


def main():
    params = load_params()
    p, rg, mf = params["paths"], params["regression_gate"], params["mlflow"]

    metrics_path = os.path.join(p["artifacts_dir"], "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"[auto_promote_best] {metrics_path} not found — run evaluate.py first.")
        sys.exit(1)

    with open(metrics_path) as f:
        metrics = json.load(f)

    metric_name, k = rg["metric"].split("@")

    candidates = [
        (config_name, per_k[k][metric_name])
        for config_name, per_k in metrics.items()
        if config_name.startswith("C_") and k in per_k
    ]
    if not candidates:
        print("[auto_promote_best] No Configuration C entries in metrics.json yet "
              "(nothing fine-tuned/evaluated) — nothing to promote. Not an error.")
        sys.exit(0)

    best_config, best_score = max(candidates, key=lambda c: c[1])
    print(f"[auto_promote_best] Best Config C by {rg['metric']}: "
          f"{best_config} ({rg['metric']}={best_score:.4f}) among {len(candidates)} candidate(s).")

    # "C_alpha0.7_seed16" -> seed "16"
    seed = best_config.split("seed")[-1]
    run_name = f"clip_finetune_seed{seed}"
    artifact = f"clip_finetuned_{seed}.pt"

    try:
        mlflow.set_tracking_uri(mf["tracking_uri"])
        client = MlflowClient()

        run_id = find_run_id_by_name(client, mf["experiment_name"], run_name)
        model_uri = f"runs:/{run_id}/{artifact}"
        model_name = mf.get("registry_model_name", "visual-search-clip")

        print(f"[auto_promote_best] Registering {model_uri} as '{model_name}'...")
        result = mlflow.register_model(model_uri=model_uri, name=model_name)
        client.set_model_version_tag(model_name, result.version, "config_name", best_config)
        client.set_model_version_tag(model_name, result.version, "promoted_by", "auto_promote_best")
        client.set_model_version_tag(model_name, result.version, f"metric_{rg['metric'].replace('@', '_at_')}", f"{best_score:.4f}")  # mlflow tag/param keys can't contain '@' 

        client.transition_model_version_stage(
            name=model_name, version=result.version,
            stage="Production", archive_existing_versions=True,
        )
        print(f"[auto_promote_best] '{model_name}' version {result.version} "
              f"(config_name={best_config}) is now in Production.")
        sys.exit(0)

    except Exception as e:
        print(f"[auto_promote_best] FAIL — {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
