"""
src/monitoring/canary_check.py — component C9 (continuous monitoring),
the direct answer to the "Uber button" story from your notes.

The lesson that story illustrates: a deploy finishing without errors is
not proof the feature works for users. `dvc repro` finishing green only
proves the pipeline ran end to end — it doesn't prove a real query still
comes back with sane results. This script is the check that actually
looks at the user-visible outcome: send one fixed "canary" image through
the exact same crop -> embed -> fuse -> search path serving/app.py and
serving/api.py use (via serving/search_core.SearchEngine — same code,
not a re-implementation), and assert the result is non-empty and
plausible.

Put a stable canary image at data/canary/canary.jpg (any clothing photo,
committed to the repo — it never changes, that's the point: same input,
every run, so a change in output means the *system* changed).

This is also the "Monitoring -> Retraining Trigger" arrow in the diagram,
made concrete: on FAIL, this script writes artifacts/retrain_signal.json.
That file is what closes the loop back to Airflow — see
orchestration/airflow/dags/fashion_retrieval_dag.py's check_retrain_signal
task, which reads it and (in this laptop-demoable setup) logs a clear
"retraining would be triggered here" message with the reason, rather than
actually kicking off an unattended multi-minute training run on a
schedule. That's the documented gap: P9 (continuous training via feedback)
is real code, wired end to end, but the trigger is observe-and-log, not
autonomous-and-blind — a deliberate choice for a laptop demo, spelled out
in docs/diagram_mapping.md.

Run manually:
  python -m src.monitoring.canary_check

Run on a schedule: see .github/workflows/monitor.yml
Exit code 0 = healthy, 1 = canary failed (results empty / similarity too low).
"""

import json
import os
import sys
import time

from PIL import Image

from serving.search_core import SearchEngine
from src.common import load_params

MIN_SIMILARITY = 0.15  # a real match should clear this; tune once you see real numbers
CANARY_IMAGE = "data/canary/canary.jpg"
RETRAIN_SIGNAL_PATH = "artifacts/retrain_signal.json"


def write_retrain_signal(reason: str, detail: str):
    os.makedirs(os.path.dirname(RETRAIN_SIGNAL_PATH), exist_ok=True)
    signal = {
        "triggered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": reason,
        "detail": detail,
        "source": "src.monitoring.canary_check",
    }
    with open(RETRAIN_SIGNAL_PATH, "w") as f:
        json.dump(signal, f, indent=2)
    print(f"[canary] Wrote {RETRAIN_SIGNAL_PATH} — "
          f"orchestration/airflow's check_retrain_signal task will pick this up next run.")


def clear_retrain_signal():
    """A passing canary means whatever last triggered a retrain concern is resolved."""
    if os.path.exists(RETRAIN_SIGNAL_PATH):
        os.remove(RETRAIN_SIGNAL_PATH)
        print(f"[canary] Cleared stale {RETRAIN_SIGNAL_PATH} (canary now passes).")


def main():
    params = load_params()

    if not os.path.exists(CANARY_IMAGE):
        print(f"[canary] {CANARY_IMAGE} not found. Place a stable clothing photo there "
              f"(any image works — it's the fixed 'known good' probe).")
        write_retrain_signal("canary_missing", f"{CANARY_IMAGE} not found")
        sys.exit(1)

    try:
        engine = SearchEngine(params)
        print(f"[canary] Checking champion config: {engine.champion}")

        img = Image.open(CANARY_IMAGE).convert("RGB")
        results, meta = engine.search(img, k=5)

        n_results = len(results)
        top_similarity = results[0]["similarity"] if n_results else 0.0

        print(f"[canary] champion={engine.champion}  was_cropped={meta['was_cropped']}  "
              f"n_results={n_results}  top_similarity={top_similarity:.4f}")

        if n_results == 0:
            detail = "zero results returned — the silent-failure case: the app would render an empty grid with no exception thrown."
            print(f"[canary] FAIL — {detail}")
            write_retrain_signal("zero_results", detail)
            sys.exit(1)

        if top_similarity < MIN_SIMILARITY:
            detail = f"top similarity {top_similarity:.4f} below threshold {MIN_SIMILARITY} (champion={engine.champion})"
            print(f"[canary] FAIL — {detail}. Results came back but look nonsensical.")
            write_retrain_signal("low_similarity", detail)
            sys.exit(1)

        print("[canary] PASS — deployed system returns sane results for the fixed probe image.")
        clear_retrain_signal()
        sys.exit(0)

    except Exception as e:
        detail = f"exception during the check itself: {e}"
        print(f"[canary] FAIL — {detail}")
        write_retrain_signal("exception", detail)
        sys.exit(1)


if __name__ == "__main__":
    main()
