"""
src/ci/champion_challenger.py — the "A/B testing" / "Split testing" terms,
made concrete using the metrics you already produce.

This project doesn't have live production traffic to split, so this script
does the honest offline version: it reads artifacts/metrics.json (every
config evaluated in stage 8) and reports it as a champion vs. challenger
comparison — same numbers as the report's Table 5, reframed the way the
paper's section V-D describes ("deploying a challenger model in addition
to an existing champion model to find out which one performs better").

Champion = params.yaml regression_gate.baseline_config (the currently
deployed / approved config). Everything else in metrics.json is a
challenger.

Run:
  python -m src.ci.champion_challenger
"""

import json
import os

from src.common import load_params


def main():
    params = load_params()
    p, rg, e = params["paths"], params["regression_gate"], params["eval"]

    metrics_path = os.path.join(p["artifacts_dir"], "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"[champion_challenger] {metrics_path} not found — run evaluate.py first.")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    champion = rg["baseline_config"]
    k = e["k_values"][-1]  # report at the largest K, e.g. 15

    if champion not in metrics:
        print(f"[champion_challenger] champion config '{champion}' not present in metrics.json.")
        return

    champ_recall = metrics[champion][str(k)]["recall"]

    print(f"\nChampion: {champion}  (Recall@{k} = {champ_recall:.4f})\n")
    print(f"{'Challenger':<24} | {'Recall@'+str(k):<10} | {'Δ vs champion':<14} | Verdict")
    print("-" * 70)

    for config_name, per_k in metrics.items():
        if config_name == champion:
            continue
        challenger_recall = per_k[str(k)]["recall"]
        delta = challenger_recall - champ_recall
        verdict = "beats champion" if delta > 0 else "loses to champion"
        print(f"{config_name:<24} | {challenger_recall:<10.4f} | {delta:+.4f}        | {verdict}")

    print(
        "\nOnline version (not built here, described for the report): route a "
        "fixed percentage of live Streamlit queries to the challenger's index, "
        "log which result the user clicks as the outcome, and compare "
        "click-through-on-correct-result between champion and challenger "
        "traffic rather than this offline query set."
    )


if __name__ == "__main__":
    main()
