# Visual Search Engine — MLOps rebuild

A small-scale rebuild of the DeepFashion retrieval pipeline (YOLOv8n crop
→ CLIP embed → BLIP caption → fused HNSW index), wrapped in the MLOps
scaffolding described in Kreuzberger, Kühl & Hirschl (2023): versioned
data/models, a DAG-orchestrated pipeline, tracked experiments, a CI/CD
regression gate, containerized serving, and a monitoring check.

This is a standalone repo, separate from the original Visual-Search-Engine
project — a parallel, smaller-dataset build that demonstrates the MLOps
layer around the same modeling approach (same YOLO class mapping, same
CLIP arch, same InfoNCE fine-tuning, same fusion formula). See
`docs/principles_mapping.md` for exactly which file answers which part
of the paper, and `docs/diagram_mapping.md` if you're working from the
GitHub -> Airflow -> AutoML -> MLflow Registry -> Docker -> Kubernetes ->
FastAPI/Streamlit -> Monitoring -> Retraining diagram — it maps every
node to a file, and explains the two places this repo uses a pragmatic
equivalent (Optuna instead of Kubeflow Katib, Docker Compose instead of
Kubernetes) plus the one loop that's laptop-demoable rather than live.

## What "smaller dataset" means here

Everything is driven by `params.yaml`. `sampling.n_items: 60` samples 60
DeepFashion item identities (instead of the full ~26k training + ~12.6k
gallery identities) and only copies their images. Bump it up once the
pipeline runs clean end to end; the code doesn't change, only the number.

## Prerequisites

- Python 3.11 (a GPU helps for CLIP/YOLO fine-tuning but isn't required —
  60 items / a few hundred images finishes on CPU in a reasonable time)
- Git
- Docker Desktop — only needed for the optional Airflow demo and for
  containerized serving
- A Kaggle account — only needed to fetch the raw dataset (see below)

## 1. Set up the environment

```bash
cd MLOPS
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Getting the dataset

The raw DeepFashion In-Shop files are **not** in this repo (they're
gigabytes — that's the same reason your original `list_eval_partition.txt`
was never committed either). Get them from the same source the team
already used:

```bash
# using the Kaggle CLI (pip install kaggle; needs a kaggle.json API token)
kaggle datasets download -d nainika0305/deepfashionretrieval -p data/raw --unzip
```

Or download manually from Kaggle and unzip so you end up with:

```
data/raw/DeepFashionRetrieval/
  list_eval_partition.txt   <- from the Kaggle download, not included here
  list_bbox_inshop.txt      <- already placed here
  list_description_inshop.json  <- already placed here
  img/img/...                <- from the Kaggle download, not included here
```

`list_bbox_inshop.txt` and `list_description_inshop.json` are already sitting
in `data/raw/DeepFashionRetrieval/` (copied over from the original project).
You only need to add `list_eval_partition.txt` and `img/img/...` from Kaggle
— and only if you're running the pipeline locally; if training stays on
Kaggle (see `kaggle/README.md`), you don't need any of this locally at all.

`params.yaml`'s `paths.*` block already points at these locations — edit
it if you put the dataset somewhere else.

## 3. Set up DVC (principle P4 — versioning)

```bash
dvc init --subdir          # run once
dvc remote add -d local_remote ../dvc_remote_storage   # or any remote you have (S3, Drive, etc.)
```

This replaces what the old `previous_version_files/` folder was doing by
hand: `dvc repro` only re-runs a stage whose inputs actually changed, and
`dvc push` sends versioned artifacts to the remote instead of committing
27&nbsp;MB `.bin` files straight into git.

## 4. Run the pipeline

```bash
dvc repro
```

This runs all nine stages in `dvc.yaml` in order — dataset validation,
dataset sampling, YOLO label prep, YOLO fine-tune, Config A embedding,
BLIP captioning + Config B fusion, CLIP fine-tuning (Config C, one run
per seed in `params.yaml`), Config C embedding, and evaluation.
Re-running after editing e.g. `clip.lr` in `params.yaml` only re-runs the
fine-tune stage onward — DVC skips stages whose inputs didn't change.

To run one stage at a time (useful while debugging):

```bash
dvc repro -s make_small_dataset
dvc repro -s train_yolo
# ... etc, see dvc.yaml for the full stage list
```

## 5. Look at what you tracked (principle P7) and the model registry (C6)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://localhost:5000 — every YOLO training run, every CLIP
fine-tune (per seed), and every evaluated config is logged with its
params and metrics. This is what replaces manually copying numbers into
a report table.

`mlflow.tracking_uri` in `params.yaml` is `sqlite:///mlflow.db` (a real
database-backed store), not the default `file:./mlruns` — that's what
makes the **Model Registry** tab in this same UI work. After a training
run finishes, promote its checkpoint:

```bash
# promote whichever Config C variant currently scores best (automatic):
python -m src.ci.auto_promote_best

# or promote one specific run yourself:
python -m src.ci.promote_to_registry \
  --run-name clip_finetune_seed16 \
  --artifact clip_finetuned_16.pt \
  --config-name C_alpha0.7_seed16
```

This registers the checkpoint as a new version of the `visual-search-clip`
model and promotes it to the "Production" stage — `serving/search_core.py`
reads that stage back at startup to decide which config to serve (falling
back to `params.yaml`'s `regression_gate.baseline_config` if nothing's
been promoted yet, e.g. right after a fresh clone).

## 6. Evaluate, gate, and compare (principles P1, "regression testing", "A/B testing")

```bash
python -m src.pipeline.07_evaluate           # already runs as part of `dvc repro`
python -m src.ci.regression_gate             # exits non-zero on a real regression
python -m src.ci.champion_challenger         # prints the offline A/B comparison table
```

The first time you run `regression_gate.py` there's no baseline yet, so
it adopts the current run as the baseline (`artifacts/baseline_metrics.json`)
and tells you to commit it. After that, a real drop in Recall@5 fails the
check — try it by hand once (e.g. temporarily set `clip.epochs: 1`) to see
what a caught regression looks like before you rely on it in CI.

## 7. Run the app

Streamlit UI and FastAPI share the same underlying `SearchEngine`
(`serving/search_core.py`) — run either, or both:

```bash
streamlit run serving/app.py                       # UI:  http://localhost:8501
uvicorn serving.api:app --port 8000                 # API: http://localhost:8000/docs
```

Or containerized, all three pieces (Streamlit, FastAPI, and an MLflow UI
server) together — the pragmatic, laptop-sized stand-in for the
diagram's "Docker Image -> Kubernetes" step, see `docs/diagram_mapping.md`:

```bash
docker compose up --build
```

Which model/index gets served is resolved by `serving/search_core.py`'s
`resolve_champion()`: it checks the MLflow Model Registry's Production
stage first (see step 5 — `promote_to_registry.py` / `auto_promote_best.py`),
and only falls back to `params.yaml`'s `regression_gate.baseline_config`
if the registry is unreachable or nothing's been promoted yet. Either
way, promoting a new champion is a registry action or a one-line
`params.yaml` change — never an `app.py`/`api.py` edit.

## 8. Monitoring and the retraining trigger (principle P8/P9 — the "Uber button" lesson)

```bash
# put any clothing photo at data/canary/canary.jpg first
python -m src.monitoring.canary_check
```

This is the check that would have caught a silently-broken feature: it
runs one fixed image through the exact same path the app/api use (via
`SearchEngine`) and fails if the result set comes back empty or
nonsensical — not just "did the code throw an exception."
`.github/workflows/monitor.yml` runs it every 6 hours once this is
deployed somewhere reachable by CI.

On failure, it also writes `artifacts/retrain_signal.json` — this is the
diagram's "Monitoring -> Retraining Trigger -> back to Airflow" arrow.
The Airflow DAG's `check_retrain_signal` task reports that file's reason
at the start of its next run, and `clear_retrain_signal` consumes it once
a fresh model has been evaluated and promoted. Starting that DAG run is
still a human action in this laptop-demoable setup, not a live webhook —
see `docs/diagram_mapping.md` "What's not live-wired" for exactly why and
what a fully-automated version would add.

## 9. CI/CD

`.github/workflows/ci.yml` runs lint + `dvc repro` + the regression gate
on every push. It needs a small dataset fixture available in CI (the
workflow has a placeholder step — decide as a team whether that's a
tiny committed fixture or a download step, see the comment in the file).

## 10. Optional: Airflow

See `orchestration/airflow/README.md`. Not required to run the pipeline
day to day — `dvc repro` already does that — but demonstrates the
DAG-based orchestration the paper names explicitly. The DAG now runs the
full loop: `check_retrain_signal` (reports why this run started, if a
canary failure triggered it) -> `validate_dataset` -> the pipeline
stages -> `evaluate` -> `regression_gate` -> `champion_challenger` ->
`auto_promote_best` (promotes the best Config C to the MLflow Registry)
-> `clear_retrain_signal`.

## 11. Optional: AutoML sweep

```bash
python -m src.automl.optuna_sweep --n-trials 10
```

Searches fusion alpha, CLIP learning rate, and trainable-block count
instead of the hand-picked grid ({0.5, 0.7} × {16, 34, 59}) — replaces a
manual choice with a logged search.

## What we didn't build, on purpose

**Databricks** and **OpenStack Swift / S3** are named in the paper (and
your notes) as production-scale tools for orchestration+compute and
artifact storage respectively. Standing either up for a class project
would be infrastructure with no audience. Instead: DVC + any remote is
the zero-infra version of an artifact store, and a laptop/Kaggle running
`dvc repro` (or the optional Airflow container) is the zero-infra version
of a managed workflow platform. Say this explicitly in the report —
naming the production equivalent of what you actually built is more
convincing than an unused trial account.

## Repository layout

```
MLOPS/
  params.yaml                  single config source (P3, reproducibility)
  dvc.yaml                     pipeline DAG (P2, C3) — 9 stages, validate_dataset first
  docker-compose.yml           app + api + mlflow, one command (pragmatic Kubernetes equivalent)
  docker-entrypoint.sh         picks Streamlit vs FastAPI inside the same image
  src/
    common.py                  shared functions, ported from training.ipynb/app.py/eval.py
    data/make_small_dataset.py stage 2 — sample N item_ids
    pipeline/                  stages 1, 3-9 — validate, labels, YOLO train, embed A,
                                caption+fuse B, CLIP finetune, embed C, evaluate
    ci/regression_gate.py      "regression testing"
    ci/champion_challenger.py  "A/B testing" / "split testing"
    ci/promote_to_registry.py  MLflow Registry promotion, named run (manual)
    ci/auto_promote_best.py    MLflow Registry promotion, best Config C (automatic)
    monitoring/canary_check.py "Uber button" lesson, P8/P9 — writes retrain_signal.json
    automl/optuna_sweep.py     "AutoML" (pragmatic Kubeflow Katib equivalent)
  orchestration/airflow/       optional Airflow DAG + docker-compose
  serving/
    search_core.py             shared SearchEngine + registry-first champion resolution
    app.py                     Streamlit UI (C8)
    api.py                     FastAPI (C8)
    Dockerfile                 one image, both entrypoints
  .github/workflows/           CI (ci.yml) + scheduled monitoring (monitor.yml)
  docs/principles_mapping.md   paper P1-P9/C1-C9 -> file, for the report
  docs/diagram_mapping.md      the GitHub->...->Retraining diagram, node by node -> file
```
