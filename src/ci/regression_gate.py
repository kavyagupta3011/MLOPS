"""
src/ci/regression_gate.py — the "Regression (testing)" term from your notes,
made concrete.

Compares artifacts/metrics.json (just produced by stage 8, evaluate.py)
against artifacts/baseline_metrics.json (the last metrics.json that a human
reviewed and approved, committed to the repo). If the champion config's
Recall@5 drops by more than params.yaml's regression_gate.max_allowed_drop,
this exits non-zero — which is what makes .github/workflows/ci.yml fail
the build. This is principle P1 (CI/CD automation) applied to model
quality instead of just "does the code compile."

To approve a new baseline on purpose (e.g. after a genuine improvement or
an intentional dataset change), copy the new metrics.json over
baseline_metrics.json and commit that — a deliberate, reviewable action,
not something CI does silently.

Run:
  python -m src.ci.regression_gate
Exit code 0 = pass, 1 = regression detected, 2 = no baseline yet (first run).
"""

import json
import os
import sys

from src.common import load_params


def main():
    params = load_params()
    p, rg = params["paths"], params["regression_gate"]

    metrics_path = os.path.join(p["artifacts_dir"], "metrics.json")
    baseline_path = os.path.join(p["artifacts_dir"], "baseline_metrics.json")

    if not os.path.exists(metrics_path):
        print(f"[regression_gate] {metrics_path} not found — run evaluate.py first.")
        sys.exit(2)

    with open(metrics_path) as f:
        new_metrics = json.load(f)

    if not os.path.exists(baseline_path):
        print(f"[regression_gate] No baseline at {baseline_path} yet — "
              f"treating this run as the baseline. Commit it to lock it in.")
        with open(baseline_path, "w") as f:
            json.dump(new_metrics, f, indent=2)
        sys.exit(0)

    with open(baseline_path) as f:
        baseline_metrics = json.load(f)

    config = rg["baseline_config"]
    metric_name, k = rg["metric"].split("@")
    k = int(k)

    if config not in new_metrics or config not in baseline_metrics:
        print(f"[regression_gate] Config '{config}' missing from metrics — "
              f"check regression_gate.baseline_config in params.yaml matches "
              f"a key evaluate.py actually produced.")
        sys.exit(2)

    new_val = new_metrics[config][str(k)][metric_name]
    baseline_val = baseline_metrics[config][str(k)][metric_name]
    drop = baseline_val - new_val

    print(f"[regression_gate] {config} {rg['metric']}: "
          f"baseline={baseline_val:.4f} new={new_val:.4f} drop={drop:.4f} "
          f"(max allowed={rg['max_allowed_drop']})")

    if drop > rg["max_allowed_drop"]:
        print(f"[regression_gate] FAIL — regression exceeds threshold.")
        sys.exit(1)

    print(f"[regression_gate] PASS.")
    sys.exit(0)


if __name__ == "__main__":
    main()
