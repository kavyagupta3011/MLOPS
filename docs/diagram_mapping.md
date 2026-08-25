# Mapping the architecture diagram to this repo

This is the node-by-node answer to "it should be done like this right?" —
your pasted diagram, and exactly which file implements each arrow. Two
nodes are built as a **pragmatic equivalent** rather than the literal
tool named, and one loop is **laptop-demoable** rather than live/24-7.
Both trade-offs are explained inline, not hidden.

```
GitHub
  -> GitHub Actions (CI/CD)
    -> Airflow DAG
      Validate Dataset -> Prepare Metadata -> Prepare YOLO Labels -> Train YOLO -> Generate Crops
    -> AutoML / Kubeflow Katib
      Hyperparameter Search -> Multiple CLIP Trials -> Retrieval Evaluation -> Best Trial
    -> MLflow Registry
    -> Docker Image
    -> Kubernetes
    -> FastAPI / Streamlit
      -> HNSW Search -> Top-K Results
    -> Monitoring
    -> Retraining Trigger -> (back to Airflow DAG)
```

## Node by node

| Diagram node | Built as | File(s) |
|---|---|---|
| GitHub | this git repo | — |
| GitHub Actions (CI/CD) | lint + `dvc repro` + regression gate on every push; scheduled canary every 6h | `.github/workflows/ci.yml`, `.github/workflows/monitor.yml` |
| Airflow DAG — Validate Dataset | dataset sanity checks (files exist/parse, enough eligible items, bbox coverage) before anything downstream runs | `src/pipeline/00_validate_dataset.py`, `dvc.yaml` stage `validate_dataset`, DAG task `validate_dataset` |
| Airflow DAG — Prepare Metadata / Prepare YOLO Labels | sampling + label conversion | `src/data/make_small_dataset.py`, `src/pipeline/01_prepare_yolo_labels.py` |
| Airflow DAG — Train YOLO | fine-tune YOLOv8n on 3 clothing classes, logged to MLflow | `src/pipeline/02_train_yolo.py` |
| Airflow DAG — Generate Crops | applied inline wherever an image is embedded (train, eval, serving) — not a separate offline stage, since crops depend on which model/config is asking | `src/common.py::crop_with_yolo`, called from `03_embed_config_a.py`, `06_embed_config_c.py`, `07_evaluate.py`, `serving/search_core.py` |
| AutoML / Kubeflow Katib — Hyperparameter Search | **pragmatic equivalent: Optuna**, not a Kubeflow/Katib cluster (see "Substitutions" below) | `src/automl/optuna_sweep.py` |
| AutoML — Multiple CLIP Trials | each Optuna trial calls the same fine-tune function real training uses, one seed/param-set at a time | `src/automl/optuna_sweep.py` → `src/pipeline/05_finetune_clip.py::run_one_seed` |
| AutoML — Retrieval Evaluation | each trial is scored the same way `07_evaluate.py` scores a real run (recall/ndcg/map) | `src/automl/optuna_sweep.py` objective function |
| AutoML — Best Trial | Optuna's `study.best_trial`; separately, across *all* evaluated Config C variants (grid + sweep), the actual best-by-metric one is what gets promoted | `src/automl/optuna_sweep.py`, `src/ci/auto_promote_best.py` |
| MLflow Registry | a real registry — DB-backed tracking store, register + promote to "Production", tagged with which index/alpha config goes with the checkpoint | `params.yaml` (`mlflow.tracking_uri: sqlite:///mlflow.db`), `src/ci/promote_to_registry.py` (manual, named run), `src/ci/auto_promote_best.py` (automatic, picks the winner) |
| Docker Image | one image, two entrypoints (Streamlit or FastAPI) via an entrypoint script | `serving/Dockerfile`, `docker-entrypoint.sh` |
| Kubernetes | **pragmatic equivalent: Docker Compose**, not a k8s cluster (see "Substitutions" below) | `docker-compose.yml` (repo root) — runs `app`, `api`, and `mlflow` server together |
| FastAPI / Streamlit | both, sharing one implementation | `serving/app.py` (Streamlit UI), `serving/api.py` (FastAPI `/search`, `/health`, `/champion`), `serving/search_core.py` (the shared `SearchEngine` + registry-first `resolve_champion()`) |
| HNSW Search -> Top-K Results | unchanged from the original project's approach | `serving/search_core.py::SearchEngine.search` |
| Monitoring | fixed-probe canary query through the real serving path, not just "did the pipeline exit 0" | `src/monitoring/canary_check.py`, `.github/workflows/monitor.yml` |
| Retraining Trigger -> back to Airflow DAG | canary failure writes a signal file; the DAG reports it at the start of its next run and clears it once a fresh model is evaluated + promoted | `src/monitoring/canary_check.py` (writes `artifacts/retrain_signal.json`), `orchestration/airflow/dags/fashion_retrieval_dag.py` tasks `check_retrain_signal` / `clear_retrain_signal` — **see "What's not live-wired" below, this loop is observe-and-log, not autonomous** |

## Substitutions, and why

**Kubeflow Katib -> Optuna.** Katib is Kubernetes-native hyperparameter
tuning — it assumes you already have a cluster running trials as pods.
Optuna does the same search algorithm (and the same "try many configs,
keep the best" idea the diagram draws) as a Python library with zero
extra infrastructure, which is the right trade for a project training on
a laptop/Kaggle notebook, not a cluster. If this were being handed off to
run continuously on real production traffic at scale, Katib would be the
right call — the search algorithm doesn't change, only where it executes.

**Kubernetes -> Docker Compose.** A k8s Deployment gives you replica
count, rolling updates, and a Service/Ingress for traffic — real value
when you have real traffic and need to survive a pod crashing at 3am.
For a class project demoed on one laptop, Compose gives the same "the
served app is a container, not a `streamlit run` on bare metal" property
(the actual MLOps point: it's C8 model serving, decoupled from the host
environment) without a cluster to provision and keep alive. `docker-compose.yml`
runs the exact same image k8s would run — moving to k8s later is a
Deployment YAML pointing at this image, not a rewrite.

## What's not live-wired (the "keep it laptop-demoable" choice)

The diagram draws one continuous loop: monitoring detects a problem and
retraining kicks off automatically, no human involved. Building that for
real needs somewhere for Airflow to run *continuously* (so it's listening
for the trigger at 3am, not just when your laptop happens to be open) —
that's a hosting decision, not a code one, and standing up always-on
infrastructure for a class project is effort with no audience.

What's built instead, so the loop is real code you can demo, not just a
description:

1. `src/monitoring/canary_check.py` fails -> writes `artifacts/retrain_signal.json`
   (reason, timestamp, detail).
2. Next time the Airflow DAG runs (manually triggered, or on a schedule
   if you set `schedule="@weekly"` in the DAG), `check_retrain_signal`
   reads and prints that file first, so it's visible *why* this run
   matters.
3. The DAG runs the full pipeline through `evaluate` -> `regression_gate`
   -> `champion_challenger` -> `auto_promote_best` (promotes whichever
   Config C variant now scores best) -> `clear_retrain_signal` (consumes
   the signal — the next run starts clean unless a new failure writes a
   fresh one).

The gap, stated plainly for the report: step 2 requires a human (or a
cron schedule) to actually start the DAG run. A production system would
have Airflow always running and either poll for the signal file or have
canary_check call Airflow's REST API directly to trigger the DAG the
moment it fails. That's a small code change (`requests.post` to
Airflow's `/dags/{dag_id}/dagRuns` endpoint) — deliberately not built
here because it only means something once Airflow has somewhere to run
continuously, which a laptop demo doesn't have.

## Try it end to end

```bash
# 1. Run the pipeline (produces models, indices, mlflow.db runs)
dvc repro

# 2. Promote the best fine-tuned config to the registry
python -m src.ci.auto_promote_best
# or, to promote a specific named run yourself:
python -m src.ci.promote_to_registry --run-name clip_finetune_seed16 \
    --artifact clip_finetuned_16.pt --config-name C_alpha0.7_seed16

# 3. Serve it — either directly...
streamlit run serving/app.py
uvicorn serving.api:app --port 8000
# ...or containerized, all three pieces together:
docker compose up --build

# 4. Simulate the monitoring/retrain loop
python -m src.monitoring.canary_check     # pass: nothing happens
# (to see a failure, temporarily rename data/canary/canary.jpg)
cat artifacts/retrain_signal.json          # present only after a failure

# 5. Run the DAG (reports + consumes the signal above)
cd orchestration/airflow && docker compose up
# open http://localhost:8080, trigger fashion_retrieval_pipeline manually
```
