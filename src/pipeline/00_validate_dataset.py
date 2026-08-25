"""
src/pipeline/00_validate_dataset.py — "Validate Dataset", the first node
in the Airflow DAG.

Checks the raw DeepFashion files are actually usable before anything
downstream wastes time on them: the partition file parses, the bbox file
parses, there are enough eligible item_ids for the configured sample
size, and bbox coverage for a same-size random sample isn't suspiciously
low (a sign the bbox/partition files are mismatched versions).

This is deliberately cheap (a few seconds) — it's a gate, not an audit.
Writes a small report so a human (or a later Airflow task) can see *why*
a run was rejected instead of just seeing a stack trace three stages
later.

Reads:
  paths.partition_file, paths.bbox_file
Writes:
  artifacts/dataset_validation_report.json
Exit code 0 = pass, 1 = fail (missing files or below thresholds).

Run:
  python -m src.pipeline.00_validate_dataset
"""

import json
import os
import random
import sys
from pathlib import Path

from src.common import load_params, parse_bbox_file

MIN_ELIGIBLE_ITEMS = 10          # need at least this many item_ids with >= min_images_per_item
MIN_BBOX_COVERAGE = 0.5          # at least this fraction of a sample must have a matching GT bbox


def read_partition_entries(partition_file):
    with open(partition_file, "r") as f:
        lines = f.readlines()[2:]
    entries = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            entries.append(tuple(parts))
    return entries


def main():
    params = load_params()
    p, s = params["paths"], params["sampling"]
    report = {"checks": [], "passed": True}

    def check(name, ok, detail):
        report["checks"].append({"name": name, "passed": bool(ok), "detail": detail})
        if not ok:
            report["passed"] = False
        status = "PASS" if ok else "FAIL"
        print(f"[validate_dataset] [{status}] {name}: {detail}")

    partition_exists = os.path.exists(p["partition_file"])
    check("partition_file_exists", partition_exists, p["partition_file"])
    bbox_exists = os.path.exists(p["bbox_file"])
    check("bbox_file_exists", bbox_exists, p["bbox_file"])

    if not (partition_exists and bbox_exists):
        os.makedirs(p["artifacts_dir"], exist_ok=True)
        with open(os.path.join(p["artifacts_dir"], "dataset_validation_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print("[validate_dataset] FAIL — required files missing. See README 'Getting the dataset'.")
        sys.exit(1)

    entries = read_partition_entries(p["partition_file"])
    check("partition_file_parses", len(entries) > 0, f"{len(entries)} rows")

    from collections import defaultdict
    by_item = defaultdict(list)
    for img_path, item_id, split in entries:
        by_item[item_id].append(img_path)
    eligible = [iid for iid, imgs in by_item.items() if len(imgs) >= s["min_images_per_item"]]
    check(
        "enough_eligible_items",
        len(eligible) >= max(MIN_ELIGIBLE_ITEMS, s["n_items"]),
        f"{len(eligible)} eligible (need >= {max(MIN_ELIGIBLE_ITEMS, s['n_items'])} "
        f"for sampling.n_items={s['n_items']})",
    )

    bbox_map = parse_bbox_file(p["bbox_file"])
    check("bbox_file_parses", len(bbox_map) > 0, f"{len(bbox_map):,} entries")

    rng = random.Random(s["split_seed"])
    sample_entries = rng.sample(entries, min(500, len(entries)))
    hits = 0
    for img_path, item_id, _ in sample_entries:
        p_ = Path(img_path)
        full_filename = f"{p_.parts[1]}_{p_.parts[2]}_{p_.parts[3]}_{p_.name}" if len(p_.parts) >= 4 else p_.name
        if (item_id, full_filename) in bbox_map:
            hits += 1
    coverage = hits / len(sample_entries) if sample_entries else 0.0
    check(
        "bbox_coverage_sane",
        coverage >= MIN_BBOX_COVERAGE,
        f"{coverage:.1%} of a {len(sample_entries)}-row sample matched a GT bbox "
        f"(threshold {MIN_BBOX_COVERAGE:.0%}) — low coverage usually means the "
        f"partition and bbox files are from different dataset versions",
    )

    os.makedirs(p["artifacts_dir"], exist_ok=True)
    with open(os.path.join(p["artifacts_dir"], "dataset_validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    if not report["passed"]:
        print("\n[validate_dataset] FAIL — see artifacts/dataset_validation_report.json")
        sys.exit(1)

    print("\n[validate_dataset] PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
