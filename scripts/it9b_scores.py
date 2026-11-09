"""It9B (report entry 160): mean network score on drone and no-drone windows.

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/it9b_scores.py CKPT.pt [CKPT.pt ...] --device cuda

Scores the same windows run_range.py evaluates for an arm with nothing held out (1 s window,
0.5 s hop): validation (training rigs), Lund (never trained on, drone only) and the mower
session (drone + mower, then mower alone). The score is what the GUIs plot as confidence.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dad import rangerun as rg  # noqa: E402
from dad import realrun as rr  # noqa: E402
from dad.config import DEFAULT  # noqa: E402
from dad.realdata import load_manifest  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("ckpts", nargs="+")
p.add_argument("--device", default="cpu")
a = p.parse_args()

arm = rg.make_pool_arm(load_manifest(), "none")
sets = {"val": arm.val, "lund": arm.unseen["lund"], "mower": arm.confuser}
print(f"{'checkpoint':40} {'set':6} {'drone mean':>10} {'n':>6} {'no-drone mean':>13} {'n':>6}")
for ck in a.ckpts:
    net = rr.NetScorer.load(ck, DEFAULT, a.device)
    for name, recs in sets.items():
        sc = rg.score_chain(net, recs, net.cfg, 0.5)
        y = sc.cw.y
        cell = lambda m: (f"{sc.s[m].mean():10.3f}", f"{int(m.sum()):6d}") if m.any() else (f"{'-':>10}", f"{0:6d}")
        d, nd = cell(y == 1), cell(y == 0)
        print(f"{Path(ck).stem:40} {name:6} {d[0]} {d[1]} {nd[0]:>13} {nd[1]}", flush=True)
