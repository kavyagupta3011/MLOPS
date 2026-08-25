"""
src/ci/promote_to_registry.py — the "MLflow Registry" node in the diagram,
made real (not just tracking runs, actually registering + promoting one).

Everything up to this point (src/pipeline/05_finetune_clip.py,
src/automl/optuna_sweep.py) logs runs to MLflow tracking — params,
metrics, and the checkpoint as a run artifact. That's necessary but not
the same as a *registry*: a registry is the thing that answers "which
version is currently in production" for a client (serving/app.py,
serving/api.py) to query, independent of which run produced it.

This script finds a finished MLflow run by name, registers its logged
checkpoint as a new version of the model named in
params.yaml -> mlflow.registry_model_name, and promotes that version to
the "Production" stage (archiving whatever was Production before).

Requires a database-backed MLflow tracking URI (params.yaml's
mlflow.tracking_uri is sqlite:///mlflow.db by default) — the Model
Registry API isn't available against a plain file store.

The registry entry stores more than the checkpoint: `--config-name` (e.g.
"C_alpha0.7_seed16") is written as a tag on the registered version. That
config name is what serving/search_core.py actually reads back — the
checkpoint file resolves the CLIP weights, but the index/alpha/config
identity is still resolved from artifacts/ by name (see
serving/search_core.py resolve_champion_paths, ported from app.py's
original version). Registering a bare checkpoint with no config tag
would tell a client *a* model is in Production without saying which
fused index or alpha goes with it — the tag is what closes that gap.

Run, after a training run has finished (e.g. clip_finetune_seed16):
  python -m src.ci.promote_to_registry \\
      --run-name clip_finetune_seed16 \\
      --artifact clip_finetuned_16.pt \\
      --config-name C_alpha0.7_seed16
"""

import argparse

import mlflow
from mlflow.tracking import MlflowClient

from src.common import load_params


def find_run_id_by_name(client: MlflowClient, experiment_name: str, run_name: str) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment '{experiment_name}' not found — has anything been logged yet?")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No run named '{run_name}' found in experiment '{experiment_name}'.")
    return runs[0].info.run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True, help="e.g. clip_finetune_seed16")
    parser.add_argument("--artifact", required=True, help="artifact filename logged by that run, e.g. clip_finetuned_16.pt")
    parser.add_argument("--config-name", required=True,
                         help="full config identity, e.g. C_alpha0.7_seed16 — must match a key evaluate.py wrote to metrics.json")
    args = parser.parse_args()

    params = load_params()
    mf = params["mlflow"]

    mlflow.set_tracking_uri(mf["tracking_uri"])
    client = MlflowClient()

    run_id = find_run_id_by_name(client, mf["experiment_name"], args.run_name)
    model_uri = f"runs:/{run_id}/{args.artifact}"
    model_name = mf.get("registry_model_name", "visual-search-clip")

    print(f"[promote_to_registry] Registering {model_uri} as '{model_name}'...")
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    client.set_model_version_tag(model_name, result.version, "config_name", args.config_name)

    print(f"[promote_to_registry] Registered version {result.version} (config_name={args.config_name}). "
          f"Promoting to stage 'Production' (archiving any existing Production version)...")
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production",
        archive_existing_versions=True,
    )

    print(f"[promote_to_registry] Done. '{model_name}' version {result.version} "
          f"(config_name={args.config_name}) is now in Production.")
    print("serving/app.py and serving/api.py resolve the champion config from this registry entry "
          "first, falling back to params.yaml's regression_gate.baseline_config if the registry "
          "is unreachable (e.g. grading on a machine without mlflow.db present).")


if __name__ == "__main__":
    main()
