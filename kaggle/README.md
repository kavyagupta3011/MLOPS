# Training on Kaggle, then finishing locally

`run_pipeline_kaggle.py` is a paste-into-one-cell version of stages 1–7
(`src/data/make_small_dataset.py` through `src/pipeline/06_embed_config_c.py`),
adapted for Kaggle the same way your original `training.ipynb` was written
— because that's where the GPU and the dataset already are.

## 1. Run it on Kaggle

1. New Notebook → **Add Input** → attach `nainika0305/deepfashionretrieval`
   (already has everything: partition file, bboxes, descriptions, images).
2. **Settings → Accelerator → GPU T4** (or whatever's offered).
3. **Settings → Internet → On** (needed for `pip install` and the
   `yolov8n.pt` auto-download).
4. Paste the whole contents of `run_pipeline_kaggle.py` into one cell, run it.
5. It prints progress the same way your existing notebooks do, and ends
   with a `FileLink` to `mlops_kaggle_output.zip` — click it to download.

Runtime for `N_ITEMS = 60` on a T4 should be well under an hour total
(YOLO fine-tune + 2 CLIP fine-tunes + embedding + captioning). Bump
`N_ITEMS` at the top of the script once this runs clean.

## 2. Bring the results back

The zip contains:

```
artifacts/                  -> unpack into MLOPS/artifacts/
mlruns/                     -> unpack into MLOPS/mlruns/
small_split/                -> unpack into MLOPS/data/processed/small_split/
```

On your machine, from `MLOPS/`:

```bash
unzip ~/Downloads/mlops_kaggle_output.zip -d /tmp/kaggle_output
cp -r /tmp/kaggle_output/artifacts/*      artifacts/
cp -r /tmp/kaggle_output/mlruns/*         mlruns/
mkdir -p data/processed
cp -r /tmp/kaggle_output/small_split      data/processed/small_split
```

(Or drag-and-drop the same folders in Explorer if that's easier — the
point is just: `artifacts/`, `mlruns/`, and `data/processed/small_split/`
need to exist locally with this content before the next step.)

## 3. Finish locally

Now everything in the main `MLOPS/README.md` from step 5 onward works:

```bash
mlflow ui --backend-store-uri file:./mlruns    # see every tracked run
python -m src.pipeline.07_evaluate             # Recall/NDCG/mAP -> artifacts/metrics.json
python -m src.ci.regression_gate               # pass/fail gate
python -m src.ci.champion_challenger           # A/B comparison table
streamlit run serving/app.py                   # the actual demo
```

## Why this script duplicates code instead of importing `src/`

Kaggle notebooks run standalone — importing a local package would mean
uploading `src/` as a separate Kaggle Dataset and keeping it in sync,
which is more moving parts than a ~400-line paste-and-run cell. This is
the same tradeoff your original `training.ipynb` already made (it never
imported from a package either). The two versions are meant to be kept
in sync by hand when the modeling logic changes — `docs/principles_mapping.md`
notes this as a known duplication, not an oversight.
