"""
src/ci/import_kaggle_runs.py — one-time bridge between Kaggle's file-store
mlruns/ (where training actually logged to) and the local database-backed
mlflow.db (where the Model Registry API — used by promote_to_registry.py /
auto_promote_best.py — actually lives). See params.yaml's mlflow.tracking_uri
comment: these are two separate MLflow backends by design.

For each seed in params.yaml's clip.seeds, finds the matching
"clip_finetune_seed{N}" run in the Kaggle file-store, copies its real
params/metrics, and re-logs it as a new run of the same name in the local
sqlite backend with the actual checkpoint attached as an artifact — so
find_run_id_by_name() in promote_to_registry.py can locate it.

Run once, after bringing artifacts/ and mlruns/ back from Kaggle:
  python -m src.ci.import_kaggle_runs
"""
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"  # newer mlflow blocks file-store tracking by default; we read Kaggle's file-store mlruns/ here
import mlflow
from mlflow.tracking import MlflowClient

from src.common import load_params


def main():
    params = load_params()
    mf = params["mlflow"]
    seeds = params["clip"]["seeds"]

    # Read side: Kaggle's file-store
    file_client = MlflowClient(tracking_uri="file:./mlruns")
    # Search across every experiment in the file store, not just mf["experiment_name"] —
    # Ultralytics' built-in MLflow auto-logging can silently redirect the active experiment
    # during YOLO training (observed: CLIP runs landed under an experiment literally named
    # "/kaggle/working" instead of "visual-search-mlops"). Matching by run name only, across
    # all experiments, is robust to whichever experiment a run actually ended up in.
    all_experiments = file_client.search_experiments()
    if not all_experiments:
        print("[import_kaggle_runs] No experiments found in file:./mlruns — "
              "did you unpack Kaggle's mlruns/ into the repo root?")
        return
    all_exp_ids = [e.experiment_id for e in all_experiments]

    # Write side: local database-backed store (params.yaml's mlflow.tracking_uri)
    mlflow.set_tracking_uri(mf["tracking_uri"])
    mlflow.set_experiment(mf["experiment_name"])
    write_client = MlflowClient()

    for seed in seeds:
        run_name = f"clip_finetune_seed{seed}"
        matches = file_client.search_runs(
            experiment_ids=all_exp_ids,
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            max_results=1,
        )
        if not matches:
            print(f"[import_kaggle_runs] No run named '{run_name}' found in Kaggle's mlruns/ — skipping.")
            continue
        src_run = matches[0]

        artifact_path = os.path.join(params["paths"]["artifacts_dir"], "clip_checkpoints", f"clip_finetuned_{seed}.pt")
        if not os.path.exists(artifact_path):
            print(f"[import_kaggle_runs] {artifact_path} not found — skipping {run_name}.")
            continue

        with mlflow.start_run(run_name=run_name) as new_run:
            for k, v in src_run.data.params.items():
                mlflow.log_param(k, v)
            for k, v in src_run.data.metrics.items():
                mlflow.log_metric(k, v)
            mlflow.set_tag("imported_from", "kaggle_file_store")
            mlflow.log_artifact(artifact_path)
            print(f"[import_kaggle_runs] Re-created '{run_name}' locally as run {new_run.info.run_id} "
                  f"({len(src_run.data.params)} params, {len(src_run.data.metrics)} metrics, checkpoint attached).")

    print("[import_kaggle_runs] Done. promote_to_registry.py / auto_promote_best.py can now find these runs.")


if __name__ == "__main__":
    main()
