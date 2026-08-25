# Mapping this repo to Kreuzberger, Kühl & Hirschl (2023)

Direct reference table for the report: every principle (P1–P9) and
component (C1–C9) from the paper, next to the file in this repo that
implements it. Use this table itself as a report section if useful —
it's the honest version of "here's how we applied the paper," not a
restatement of the paper's abstract.

## Principles (Section IV-A)

| # | Principle | Where it lives here |
|---|---|---|
| P1 | CI/CD automation | `.github/workflows/ci.yml` — lint, `dvc repro`, regression gate, on every push |
| P2 | Workflow orchestration | `dvc.yaml` (primary) and `orchestration/airflow/` (optional, named explicitly in the paper) |
| P3 | Reproducibility | `params.yaml` as the single config source + `dvc.yaml` dependency hashing — same params + same code = same outputs |
| P4 | Versioning | DVC for data/model artifacts (see README "Set up DVC"), git for code; retires the old `previous_version_files/` folder |
| P5 | Collaboration | Shared `params.yaml`, MLflow experiment shared across the team (point `mlflow.tracking_uri` at a shared server, not just `file:./mlruns`) |
| P6 | Continuous ML training & evaluation | `orchestration/airflow/dags/fashion_retrieval_dag.py` can be scheduled (e.g. `@weekly`) instead of triggered manually |
| P7 | ML metadata tracking/logging | MLflow calls in `src/pipeline/02_train_yolo.py`, `05_finetune_clip.py`, `07_evaluate.py` — replaces manually copying Table 4's losses into the report |
| P8 | Continuous monitoring | `src/monitoring/canary_check.py` + `.github/workflows/monitor.yml` |
| P9 | Feedback loops | `src/monitoring/canary_check.py` writes `artifacts/retrain_signal.json` on failure; the Airflow DAG's `check_retrain_signal` task reports it at the start of the next run and `clear_retrain_signal` consumes it at the end. Kicking off that DAG run is still a human action, not a live webhook (see "Open gaps" below and `docs/diagram_mapping.md`) |

## Components (Section IV-B)

| # | Component | Where it lives here |
|---|---|---|
| C1 | CI/CD | `.github/workflows/ci.yml` |
| C2 | Source code repository | this git repo |
| C3 | Workflow orchestration | `dvc.yaml`, `orchestration/airflow/` |
| C4 | Feature store | `artifacts/gallery_manifest.csv` / `gallery_metadata.csv` — a flat-file stand-in, honestly labeled as such (a real feature store buys you online/offline serving parity, which this project doesn't need) |
| C5 | Model training infrastructure | whatever runs `dvc repro` — a laptop, Kaggle, or Colab; not distributed, and doesn't need to be at this scale |
| C6 | Model registry | The real MLflow Model Registry now (`mlflow.tracking_uri: sqlite:///mlflow.db` in `params.yaml` — a DB-backed store, required for the Registry API) — `src/ci/promote_to_registry.py` and `src/ci/auto_promote_best.py` register + promote to "Production"; `serving/search_core.py resolve_champion()` reads it back, falling back to `params.yaml`'s `regression_gate.baseline_config` if the registry is unreachable |
| C7 | ML metadata store | MLflow (`mlruns/` locally for tracking, or a shared tracking server; the registry itself lives in `mlflow.db`) |
| C8 | Model serving | `serving/app.py` (Streamlit) + `serving/api.py` (FastAPI) + `serving/search_core.py` (shared logic) + `serving/Dockerfile` / root `docker-compose.yml` |
| C9 | Monitoring | `src/monitoring/canary_check.py` |

## Terms from the assignment notes, and where they land

| Term | Where it lands |
|---|---|
| AutoML | `src/automl/optuna_sweep.py` |
| Regression testing | `src/ci/regression_gate.py`, gated in `ci.yml` |
| A/B testing / Split testing | `src/ci/champion_challenger.py` |
| Apache Airflow | `orchestration/airflow/` |
| Databricks | discussed, not built (see root `README.md` "What we didn't build, on purpose") |
| Swift (object storage) | discussed, not built — DVC + a remote is the zero-infra equivalent (see root README) |
| "Uber button" story (silent failure) | `src/monitoring/canary_check.py` — checks the user-visible outcome, not just "did the pipeline exit 0" |

## Open gaps (say this in the report, don't hide it)

- **P9 feedback loop is signal-and-log, not fully autonomous.** A failed
  canary check now writes `artifacts/retrain_signal.json` and the Airflow
  DAG reports/consumes it — but nothing automatically *starts* that DAG
  run when the signal appears. A real system would poll or subscribe to
  the signal and kick off retraining unattended; here, starting the run
  is still a human action (or you, running `airflow dags trigger` by
  hand). Worth naming as a known limitation rather than implying it's
  fully closed-loop. See `docs/diagram_mapping.md` for the full reasoning.
- **C4 feature store is a flat file**, not a real online/offline store —
  fine at this scale, called out so it doesn't read as a misunderstanding
  of what a feature store is for.
- **Evaluation in `07_evaluate.py` is simplified** vs. the original
  `eval.py` (BLIP caption fusion instead of BLIP-2 ITM re-ranking) to keep
  the small-dataset loop fast — noted here so the two scripts' numbers
  aren't mistaken for directly comparable.
