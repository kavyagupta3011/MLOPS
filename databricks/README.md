# databricks/

`champion_eval_notebook.py` is a Databricks source-format notebook: it
downloads the real promoted model files from this repo's GitHub Release
(`model-v1`), runs the actual serving pipeline (YOLO crop -> CLIP embed ->
BLIP caption -> fuse -> HNSW search) against real query images, and logs
the result to this workspace's MLflow — all executed on Databricks' own
compute, not just receiving a logged number from elsewhere.

This is the second half of the project's Databricks connection. The first
half — `src/pipeline/07_evaluate.py` automatically mirroring every real
evaluation run's metrics to Databricks — runs on every local `dvc repro`;
see `docs/principles_mapping.md` and the root `README.md`'s "What we
didn't build, on purpose" section for the full reasoning on both halves,
and on why the Model Registry itself stays local.

## How to run it

1. In your Databricks workspace, click **Workspace** in the left sidebar,
   then **Import** (or the "+" / "Create" menu -> Import).
2. Upload `champion_eval_notebook.py` from this folder — Databricks
   auto-detects it as a notebook from its `# Databricks notebook source`
   header.
3. Attach it to a cluster (Free Edition's default serverless compute is
   fine) and click **Run all**.
4. Takes a few minutes — it's downloading ~600MB of model files plus the
   CLIP/BLIP pretrained weights the first time, then genuinely running
   YOLO/CLIP/BLIP inference over real images.
5. Check this workspace's **Experiments** page afterward for a new run
   named `C_alpha0.7_seed16_databricks_compute`.
