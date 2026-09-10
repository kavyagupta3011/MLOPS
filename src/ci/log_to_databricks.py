"""
src/ci/log_to_databricks.py -- mirrors the champion run's real params/metrics
to a Databricks-hosted MLflow experiment, as a SECOND tracking destination
alongside the local sqlite-backed mlflow.db.

Why a mirror, not a full migration: promote_to_registry.py / auto_promote_best.py
rely on mlflow<3 registry semantics (register_model() on a plain log_artifact()).
Databricks' current registry is Unity-Catalog-based (mlflow 3.x model), a
different API shape -- moving the registry there would mean rewriting the
already-working promotion/serving code. So the local mlflow.db stays the
single source of truth for the Model Registry (what auto_promote_best.py
promotes to and serving/search_core.py::resolve_champion() reads from);
this script just logs the same real numbers to Databricks too, for
experiment-tracking visibility on a hosted platform (component C7 / the
"MLOps platforms exist" point the paper makes about Databricks specifically).

One-time setup before running this:
    pip install --upgrade "mlflow<3"
    python -c "import mlflow; mlflow.login()"
    (prompts for your Databricks workspace URL, email, password --
     saves credentials locally so you won't be asked again)

Run:
    python -m src.ci.log_to_databricks
"""
import json

import mlflow

from src.common import load_params

# Replace with your own Databricks account email -- experiment paths live
# under your user folder in the workspace.
DATABRICKS_EXPERIMENT_PATH = "/Users/kavya.gupta@iiitb.ac.in/visual-search-mlops"


def main():
    params = load_params()
    with open("artifacts/metrics.json") as f:
        metrics = json.load(f)

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(DATABRICKS_EXPERIMENT_PATH)

    champion = params["regression_gate"]["baseline_config"]
    # metrics.json is nested: {config_name: {"5": {"recall":..,"ndcg":..,"map":..,"n_queries":..}, "10": {...}, "15": {...}}}
    # -- not a flat dict -- so pull the champion's own per-K breakdown specifically.
    champion_metrics = metrics.get(champion, {})
    if not champion_metrics:
        print(f"[log_to_databricks] WARNING: no entry for '{champion}' in artifacts/metrics.json -- "
              f"available configs: {list(metrics.keys())}")

    with mlflow.start_run(run_name=champion):
        mlflow.log_param("config", champion)
        mlflow.log_param("clip_seeds", params["clip"]["seeds"])
        mlflow.log_param("fusion_alphas", params["fusion"]["alphas"])
        for k_value, k_metrics in champion_metrics.items():
            for metric_name, value in k_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{metric_name}_at_{k_value}", value)
        print(f"[log_to_databricks] Logged '{champion}' ({len(champion_metrics)} K-values) "
              f"to {DATABRICKS_EXPERIMENT_PATH} on Databricks.")


if __name__ == "__main__":
    main()
