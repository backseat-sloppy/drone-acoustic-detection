"""Score per-hop CSVs (file, t_end_s, score, truth) at one frozen threshold, per test source.

The CSVs come from `dad wav --out` or from scripts/edge_impulse/run_wasm.mjs, over the
clips written by export_edge_impulse.py. `truth` is 1 drone, 0 no drone, -1 left out.
The source is read from the clip name (`drone.halmstad_...`).

    uv run python -I scripts/score_hops.py TEST.csv --val VAL.csv --fpr 0.05   # freeze on validation
    uv run python -I scripts/score_hops.py TEST.csv --threshold 0.297          # a threshold frozen elsewhere
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(path: str, truth_from: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if truth_from:  # take window labels from another CSV over the same files and hops
        with open(truth_from, newline="", encoding="utf-8") as f:
            ref = {(Path(r["file"]).name, round(float(r["t_end_s"]), 3)): r["truth"] for r in csv.DictReader(f)}
        for r in rows:
            r["truth"] = ref[(Path(r["file"]).name, round(float(r["t_end_s"]), 3))]
    names = [Path(r["file"]).name for r in rows]
    source = np.array([n.split(".", 1)[1].split("_", 1)[0] for n in names])
    kind = np.array([n.split(".", 1)[0] for n in names])
    return (np.array([float(r["score"]) for r in rows]), np.array([int(r["truth"]) for r in rows]), source, kind)


def freeze(score: np.ndarray, truth: np.ndarray, fpr: float) -> float:
    """Lowest threshold whose false-positive rate on no-drone windows is at most fpr."""
    neg = np.sort(score[truth == 0])
    k = int(np.floor(fpr * len(neg)))
    return float(np.nextafter(neg[len(neg) - 1 - k], np.inf)) if k < len(neg) else float(neg[0])


def rates(score, truth, thr) -> dict:
    alarm = score >= thr
    tp, fn = int(np.sum(alarm & (truth == 1))), int(np.sum(~alarm & (truth == 1)))
    fp, tn = int(np.sum(alarm & (truth == 0))), int(np.sum(~alarm & (truth == 0)))
    div = lambda a, b: round(a / b, 3) if b else None  # noqa: E731
    return dict(recall=div(tp, tp + fn), false_positive_rate=div(fp, fp + tn), precision=div(tp, tp + fp),
                drone_windows=tp + fn, no_drone_windows=fp + tn)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("test")
    p.add_argument("--val")
    p.add_argument("--fpr", type=float, default=0.05)
    p.add_argument("--threshold", type=float)
    p.add_argument("--truth-from", help="CSV from `dad wav --out` over the same clips, for its window labels")
    p.add_argument("--val-truth-from")
    a = p.parse_args()
    if a.threshold is None:
        s, t, _, _ = load(a.val, a.val_truth_from)
        a.threshold = freeze(s, t, a.fpr)
        print(f"threshold {a.threshold:.6g} frozen at {a.fpr:.0%} false positives on {a.val}: {rates(s, t, a.threshold)}")
    s, t, src, _ = load(a.test, a.truth_from)
    out = {name: rates(s[src == name], t[src == name], a.threshold) for name in sorted(set(src))}
    out["all"] = rates(s, t, a.threshold)
    print(json.dumps(dict(threshold=a.threshold, **out), indent=1))


if __name__ == "__main__":
    # self-check of the threshold rule: 5% of 100 negatives may alarm
    neg = np.arange(100.0)
    assert abs(rates(neg, np.zeros(100, int), freeze(neg, np.zeros(100, int), 0.05))["false_positive_rate"] - 0.05) < 1e-9
    main()
