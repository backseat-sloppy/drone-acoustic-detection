"""Mean and range over seeds of the figures that decide an iteration.

    uv run python -I scripts/aggregate_range.py it4_int_it3_screen it4_int_it3_seed1 it4_int_it3_seed2

Groups run_range.py results by arm, recipe, model, window length and number
of windows averaged (k), and prints one row per group.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dad.rangerun import _get  # noqa: E402

FIGURES = {
    "recall 100-200 m": ("test", "own", "0.10", "by_range", "100-200 m", "recall"),
    "recall 0-50 m": ("test", "own", "0.10", "by_range", "0-50 m", "recall"),
    "within 5 s": ("test", "own", "0.10", "latency_100_200", "within_5s"),
    "no drone 5 s": ("test", "own", "0.10", "latency_background", "within_5s"),
    "within 10 s": ("test", "own", "0.10", "latency_100_200", "within_10s"),
    "no drone 10 s": ("test", "own", "0.10", "latency_background", "within_10s"),
    "control >300 m": ("test", "auc_session", "300-inf m", "auc"),
    "AUC 100-200 m": ("test", "auc_session", "100-200 m", "auc"),
}


def main(names: list[str]) -> None:
    groups = collections.defaultdict(list)
    for n in names:
        for r in json.loads((REPO / "artifacts" / "runs" / f"{n}.json").read_text())["results"]:
            key = (r["arm"], r["recipe"], r["scorer"], r.get("window_s", 1.0), r.get("integrate", 1),
                   r.get("decision_span_s", r.get("window_s", 1.0)))
            groups[key].append(r)
    print("| Held out | Recipe | Window x k (span) | Seeds | " + " | ".join(FIGURES) + " |")
    print("|---|---|---|---|" + "---|" * len(FIGURES))
    for (arm, recipe, scorer, win, k, span), rs in sorted(groups.items()):
        cells = []
        for path in FIGURES.values():
            v = np.array([x for x in (_get(r, *path) for r in rs) if x is not None], dtype=float)
            cells.append("-" if not len(v) else f"{v.mean():.3f}" + (f" ({v.min():.2f}-{v.max():.2f})" if len(v) > 1 else ""))
        print(f"| {arm.replace('hold out ', '')} | {recipe} {scorer} | {win:g} s x {k} ({span:g} s) | "
              f"{len(rs)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
