"""
orchestration/airflow/dags/fashion_retrieval_dag.py

The paper names Apache Airflow explicitly as the workflow-orchestration
component (C3). This DAG wires the dvc.yaml stages into an Airflow graph
with real task dependencies, so the pipeline can run on a schedule / be
retried per-task / show up in the Airflow UI, instead of notebook cells
run in a fixed order by hand.

Each pipeline task shells out to `dvc repro -s <stage>`, so Airflow gets
the scheduling and retry/backfill machinery while DVC keeps doing the
actual dependency-hashing and skip-if-unchanged work — the two tools
aren't competing here, DVC defines *what* a stage needs, Airflow decides
*when* it runs.

This DAG is the optional/heavier orchestration path — see
orchestration/airflow/README.md for why `dvc repro` alone (no Airflow) is
the path used by CI and recommended for local iteration.

Diagram nodes this DAG covers end to end, laptop-demoable version:
  Validate Dataset -> Prepare Metadata/Labels -> Train YOLO -> Generate
  Crops -> [AutoML/Optuna happens separately, see src/automl/optuna_sweep.py
  and orchestration/airflow/README.md "About the AutoML step"] -> finetune_clip
  -> embed_config_c -> evaluate -> regression_gate (champion/challenger CI
  gate) -> auto_promote_best (MLflow Registry) -> Monitoring/Retrain-Trigger
  loop, closed by check_retrain_signal / clear_retrain_signal below.

What's NOT live-wired (the "keep it laptop-demoable" choice): this DAG
does not poll a webhook or a live queue for GitHub pushes, and it does
not automatically re-trigger itself when src/monitoring/canary_check.py
fails on a schedule (.github/workflows/monitor.yml). Instead:
  - check_retrain_signal runs first and *reports* whether a prior canary
    failure left artifacts/retrain_signal.json behind, so a human (or you,
    manually) can see *why* a run is happening.
  - clear_retrain_signal runs last, once this run has produced a fresh
    evaluated + promoted model, consuming that signal.
The actual "kick off this DAG because the canary failed" step is a human
action (or, in a real deployment, an Airflow REST API call / sensor you'd
add once you have somewhere for Airflow to run continuously — out of
scope for a laptop demo). See docs/diagram_mapping.md for the full
node-by-node mapping and why each simplification was made.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/project"  # see docker-compose.yaml volume mount

default_args = {
    "owner": "visual-search-mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="fashion_retrieval_pipeline",
    description="Small-scale DeepFashion retrieval pipeline (validate -> crop -> embed -> caption -> fuse -> finetune -> index -> eval -> promote)",
    default_args=default_args,
    schedule=None,  # trigger manually, or set e.g. "@weekly" for continuous training (principle P6)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["visual-search", "mlops"],
) as dag:

    def dvc_task(task_id: str, stage: str) -> BashOperator:
        return BashOperator(
            task_id=task_id,
            bash_command=f"cd {PROJECT_DIR} && dvc repro -s {stage}",
        )

    check_retrain_signal = BashOperator(
        task_id="check_retrain_signal",
        bash_command=(
            f"cd {PROJECT_DIR} && python -c \""
            "import json, os; p = 'artifacts/retrain_signal.json'; "
            "print('[check_retrain_signal] Triggered following a canary failure: ' "
            "+ json.dumps(json.load(open(p))) if os.path.exists(p) "
            "else '[check_retrain_signal] No pending retrain signal (manual/scheduled run).')\""
        ),
    )

    validate_dataset = dvc_task("validate_dataset", "validate_dataset")
    make_small_dataset = dvc_task("make_small_dataset", "make_small_dataset")
    prepare_yolo_labels = dvc_task("prepare_yolo_labels", "prepare_yolo_labels")
    train_yolo = dvc_task("train_yolo", "train_yolo")
    embed_config_a = dvc_task("embed_config_a", "embed_config_a")
    caption_and_fuse_b = dvc_task("caption_and_fuse_b", "caption_and_fuse_b")
    finetune_clip = dvc_task("finetune_clip", "finetune_clip")
    embed_config_c = dvc_task("embed_config_c", "embed_config_c")
    evaluate = dvc_task("evaluate", "evaluate")

    regression_gate = BashOperator(
        task_id="regression_gate",
        bash_command=f"cd {PROJECT_DIR} && python -m src.ci.regression_gate",
    )

    champion_challenger = BashOperator(
        task_id="champion_challenger",
        bash_command=f"cd {PROJECT_DIR} && python -m src.ci.champion_challenger",
    )

    # Only reached if regression_gate passed (default trigger_rule=all_success) —
    # the "Best Trial -> MLflow Registry" arrow, made automatic: whichever
    # Config C variant scores highest on regression_gate.metric gets promoted
    # to the registry's Production stage. See src/ci/auto_promote_best.py.
    auto_promote_best = BashOperator(
        task_id="auto_promote_best",
        bash_command=f"cd {PROJECT_DIR} && python -m src.ci.auto_promote_best",
    )

    clear_retrain_signal = BashOperator(
        task_id="clear_retrain_signal",
        bash_command=(
            f"cd {PROJECT_DIR} && python -c \""
            "import os; p = 'artifacts/retrain_signal.json'; "
            "os.remove(p) if os.path.exists(p) else None; "
            "print('[clear_retrain_signal] Signal cleared — this run produced a fresh promoted model.')\""
        ),
    )

    (
        check_retrain_signal
        >> validate_dataset
        >> make_small_dataset
        >> prepare_yolo_labels
        >> train_yolo
        >> embed_config_a
        >> caption_and_fuse_b
        >> finetune_clip
        >> embed_config_c
        >> evaluate
        >> regression_gate
        >> champion_challenger
        >> auto_promote_best
        >> clear_retrain_signal
    )
