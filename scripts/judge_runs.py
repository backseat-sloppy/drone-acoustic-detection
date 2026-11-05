"""Seed means per recipe from scripts/run_range.py run files: the figures iteration 11's rule judges.

    python scripts/judge_runs.py artifacts/runs/it11c_s0.json artifacts/runs/it11c_s1.json ...

Per recipe (and scorer), mean over the run files (seeds), with the per-seed values:
  pooled arm ("hold out none"): validation recall at the deployment threshold and at 10% FA, mower
      false alarms at the deployment threshold, Lund recall at the deployment threshold (all ranges
      and 0-50 m)
  each held-out unit: recall on all its drone windows at 10% false alarms on its own background
      (as iteration 9), and the FPV mean over the three FPV campaigns
Reads only the single-window results (integrate = 1).
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dad.rangerun import headline  # noqa: E402

FPV = ("hold out fpv:ariel_25.02", "hold out fpv:palmahim_10.02", "hold out fpv:palmahim_27.05")


def deploy_recall(chain: dict, rng: str = "all"):
    return ((chain.get("deploy") or {}).get("by_range") or {}).get(rng, {}).get("recall")


def main(paths):
    vals = collections.defaultdict(list)            # (scorer, recipe, figure) -> per-seed values
    for p in paths:
        for r in json.loads(Path(p).read_text())["results"]:
            if r.get("integrate", 1) != 1:
                continue
            key = (r["scorer"], r["recipe"])
            if r["arm"] == "hold out none":
                vals[key + ("val recall @deploy",)].append(deploy_recall(r["val"]))
                vals[key + ("val recall @10%",)].append(headline(r["val"])["recall_all_at_10pct"])
                vals[key + ("mower FA @deploy",)].append(r["confuser"]["fa_deploy"])
                if "lund" in r["unseen"]:
                    vals[key + ("Lund recall @deploy",)].append(deploy_recall(r["unseen"]["lund"]))
                    vals[key + ("Lund 0-50 m @deploy",)].append(deploy_recall(r["unseen"]["lund"], "0-50 m"))
            elif "test" in r:
                vals[key + (r["arm"].removeprefix("hold out ") + " @10% own",)].append(
                    headline(r["test"])["recall_all_at_10pct"])
    for key in list(vals):
        if key[2].startswith("fpv:ariel"):
            per = [vals.get(key[:2] + (f.removeprefix("hold out ") + " @10% own",), []) for f in FPV]
            if all(len(x) == len(per[0]) for x in per):
                vals[key[:2] + ("FPV mean @10% own",)] = [sum(s) / 3 for s in zip(*per)]
    fmt = lambda v: "-" if v is None else f"{v:.3f}"
    for key in sorted(vals, key=lambda k: (k[0], k[1], k[2])):
        v = [x for x in vals[key] if x is not None]
        mean = sum(v) / len(v) if v else None
        print(f"{key[0]:<6} {key[1]:<20} {key[2]:<34} {fmt(mean)}  ({' / '.join(fmt(x) for x in vals[key])})")


if __name__ == "__main__":
    main(sys.argv[1:])
