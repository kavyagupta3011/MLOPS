# Airflow orchestration (optional)

The default way to run this pipeline is `dvc repro` from the `MLOPS/` root
— it's one command, needs nothing but the Python environment, and is what
`.github/workflows/ci.yml` uses. Use it for local iteration.

This folder is the Airflow version of the same dvc.yaml stages, plus the
CI/registry/retrain-signal steps around them (`check_retrain_signal` ->
`validate_dataset` -> ... -> `evaluate` -> `regression_gate` ->
`champion_challenger` -> `auto_promote_best` -> `clear_retrain_signal`),
provided because the paper (and probably your lecture) names Apache
Airflow explicitly as the workflow-orchestration component. Stand it up
when you want to *demonstrate* DAG-based orchestration — scheduling,
retries, the Airflow UI's graph view — for the report or a demo, not as
your everyday way of running the pipeline. See `docs/diagram_mapping.md`
at the repo root for how this DAG maps onto the full architecture
diagram, including the retraining-trigger loop and why it's
report-and-log rather than a live webhook.

## Requirements

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- The rest of `MLOPS/` set up already (`dvc.yaml`, `params.yaml`, a Python
  env with `requirements.txt` installed at least once so `dvc repro` has
  something to run inside the container)

## Run it

```bash
cd MLOPS/orchestration/airflow
docker compose up
```

First boot takes a few minutes (installs requirements.txt inside the
container). Once it's up:

1. Open http://localhost:8080 — log in with `admin` / `admin`.
2. Find the `fashion_retrieval_pipeline` DAG, un-pause it, and trigger a
   run manually (the schedule is `None` by default — see the DAG file to
   set e.g. `@weekly` if you want to demonstrate principle P6, continuous
   training).
3. Watch the graph view — each node is one `dvc repro -s <stage>` call.

## Why SequentialExecutor + SQLite

Airflow's "real" setup needs Postgres and usually Redis/Celery for
parallel task execution. None of that buys anything here — this DAG is a
straight line (each stage depends on the last), so there's nothing to
parallelize, and SQLite is one less moving part for a laptop demo. If
you extend the DAG to fan out (e.g. training multiple CLIP seeds in
parallel instead of via `--all-seeds` in one task), switch to
`LocalExecutor` + Postgres at that point.
