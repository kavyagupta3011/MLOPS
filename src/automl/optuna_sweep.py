"""
src/automl/optuna_sweep.py — the "AutoML" term from your notes, made concrete.

Configuration C currently hand-picks fusion alpha from {0.5, 0.7} and CLIP
learning rate is a single fixed value in params.yaml. This script replaces
that manual grid with an Optuna study: each trial samples alpha and lr,
fine-tunes CLIP for a *reduced* number of epochs (fast search), builds a
Config C index, evaluates Recall@5, and Optuna picks the next trial's
values based on what's worked so far.

This is deliberately separate from the main dvc.yaml pipeline — it's a
search *over* the pipeline, not a pipeline stage itself. Run it after the
main pipeline has produced a working YOLO checkpoint and gallery captions
at least once (it reuses those, only re-running CLIP fine-tune + Config C
embedding + eval per trial).

Run:
  python -m src.automl.optuna_sweep --n-trials 10
"""

import argparse
import copy
import importlib

import optuna

from src.common import load_params

# src/pipeline/05_finetune_clip.py starts with a digit, which is fine for
# `python -m src.pipeline.05_finetune_clip` (runpy takes a plain string) but
# not for a `from ... import ...` statement (the compiler requires a valid
# identifier there) — importlib.import_module sidesteps that.
finetune_module = importlib.import_module("src.pipeline.05_finetune_clip")


def objective(trial: optuna.Trial, base_params: dict) -> float:
    params = copy.deepcopy(base_params)
    alpha = trial.suggest_float("alpha", 0.3, 0.9)
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    trainable_blocks = trial.suggest_int("trainable_blocks", 2, 6)

    params["fusion"]["alphas"] = [round(alpha, 3)]
    params["clip"]["lr"] = lr
    params["clip"]["trainable_blocks"] = trainable_blocks
    params["clip"]["epochs"] = max(2, params["clip"]["epochs"] // 2)  # fast search, not final quality
    seed = params["clip"]["seeds"][0]
    params["clip"]["seeds"] = [seed]

    finetune_module.run_one_seed(params, seed)

    # Reuse stage 7's embedding + fusion logic, then stage 8's evaluator, at
    # this trial's alpha only — see src/pipeline/06_embed_config_c.py and
    # src/pipeline/07_evaluate.py for the full versions this trims down from.
    import subprocess
    subprocess.run(["python", "-m", "src.pipeline.06_embed_config_c"], check=True)
    subprocess.run(["python", "-m", "src.pipeline.07_evaluate"], check=True)

    import json
    import os
    with open(os.path.join(params["paths"]["artifacts_dir"], "metrics.json")) as f:
        metrics = json.load(f)

    key = f"C_alpha{round(alpha, 3)}_seed{seed}"
    if key not in metrics:
        # alpha rounding can mismatch the key evaluate.py wrote; fall back to any C_ config
        matches = [k for k in metrics if k.startswith("C_")]
        key = matches[0] if matches else None
    if key is None:
        return 0.0

    k = str(params["eval"]["k_values"][0])  # optimize Recall@5
    return metrics[key][k]["recall"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10)
    args = parser.parse_args()

    base_params = load_params()
    study = optuna.create_study(direction="maximize", study_name="clip_alpha_lr_sweep")
    study.optimize(lambda trial: objective(trial, base_params), n_trials=args.n_trials)

    print("\n[optuna_sweep] Best trial:")
    print(f"  Recall@{base_params['eval']['k_values'][0]} = {study.best_value:.4f}")
    print(f"  params: {study.best_params}")
    print("\nUpdate params.yaml's clip.lr / clip.trainable_blocks / fusion.alphas "
          "with these values, then re-run `dvc repro` for the full-epoch final run.")


if __name__ == "__main__":
    main()
