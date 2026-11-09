"""Iteration 16 (report entry 155): the numbers the box judges by, per arm and network.

    .venv/bin/python scripts/it16_judge.py      # reads artifacts/runs/it16_*_s*.json

Per run: validation recall and false alarms at the deployment threshold (5% FA on
validation), validation recall at 10% FA, Lund recall at the deployment threshold
and false alarms on the mower-only audio at it. Then the mean over seeds, and the
verdict of the box against `ref` for the same network.
"""
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "artifacts" / "runs"
ARMS = ("ref", "600k", "win2s", "wide", "mel128", "win2s600k")


def get(d, *keys):
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


rows = defaultdict(list)   # (arm, net) -> list of metric dicts
for f in sorted(RUNS.glob("it16_*_s*.json")):
    m = re.fullmatch(r"it16_(\w+?)_s(\d)", f.stem)
    if not m:
        continue
    arm, seed = m.group(1), int(m.group(2))
    for r in json.loads(f.read_text())["results"]:
        if r.get("integrate", 1) != 1:
            continue
        net = r["scorer"].replace("wide", "").split(" ")[0]   # dscnnwide -> dscnn, "comb (no ML)" -> comb
        rows[(arm, net)].append(dict(
            seed=seed,
            val=get(r, "val", "deploy", "by_range", "all", "recall"),
            val_fa=get(r, "val", "deploy_fa"),
            val10=get(r, "val", "own", "0.10", "by_range", "all", "recall"),
            lund=get(r, "unseen", "lund", "deploy", "by_range", "all", "recall"),
            mower=get(r, "confuser", "fa_deploy"),
            train_s=r.get("train_seconds")))

KEYS = ("val", "val_fa", "val10", "lund", "mower")
fmt = lambda v: "  -  " if v is None else f"{v:.3f}"
mean = lambda xs: statistics.mean(xs) if xs else None
print(f"{'arm':7} {'net':6} seeds  " + "  ".join(f"{k:>6}" for k in KEYS) + "   lund per seed")
means = {}
for arm in ARMS:
    for net in ("comb", "lgbm", "dscnn", "crnn"):
        rs = rows.get((arm, net))
        if not rs:
            continue
        mm = {k: mean([r[k] for r in rs if r[k] is not None]) for k in KEYS}
        means[(arm, net)] = mm
        seeds = " ".join(f"s{r['seed']}={fmt(r['lund'])}" for r in sorted(rs, key=lambda r: r["seed"]))
        print(f"{arm:7} {net:6} {len(rs):5}  " + "  ".join(f"{fmt(mm[k]):>6}" for k in KEYS) + f"   {seeds}")

print("\nVerdict (box of entry 155): Lund >= ref + 0.05, validation >= ref - 0.02, mower <= ref + 0.05")
for net in ("dscnn", "crnn"):
    ref = means.get(("ref", net))
    for arm in ARMS[1:]:
        v = means.get((arm, net))
        if not ref or not v or None in (v["lund"], v["val"], v["mower"]):
            continue
        d = {k: v[k] - ref[k] for k in ("lund", "val", "mower")}
        win = d["lund"] >= 0.05 and d["val"] >= -0.02 and d["mower"] <= 0.05
        print(f"  {net:6} {arm:7} Lund {d['lund']:+.3f}  val {d['val']:+.3f}  mower {d['mower']:+.3f}  "
              f"-> {'WINS' if win else 'no'}")
