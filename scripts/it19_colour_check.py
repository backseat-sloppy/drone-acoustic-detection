"""Iteration 19 (report entry 169): pink and brown noise alarms at each model's frozen threshold.

    .venv/bin/python scripts/it19_colour_check.py [run ...]   # default: it16_600k_s0-2 and it19_colour_s0-2

Fresh noise, 20 s per seed, seeds 1000-1004 (training draws its noise from the run's own generator),
at -50, -30 and -10 dBFS RMS. The box of iteration 19 judges -30 dBFS. Scored as the GUIs score
(dad.wavio.stream: 1 s window, 0.125 s hop). A tripwire on synthetic noise, not detection performance.
"""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dad import steady  # noqa: E402
from dad.config import DEFAULT  # noqa: E402
from dad.wavio import Clip, load_scorer, stream, threshold_from_run  # noqa: E402

SLOPES = {"pink": -3.0, "brown": -6.0}
LEVELS_DBFS = (-50.0, -30.0, -10.0)
SEEDS = range(1000, 1005)
SECONDS = 20
RUNS = sys.argv[1:] or [f"{a}_s{s}" for a in ("it16_600k", "it19_colour") for s in (0, 1, 2)]
device = "cuda" if torch.cuda.is_available() else "cpu"

noise = {(c, lv, s): (steady.slope_noise(SECONDS * DEFAULT.sample_rate, db, np.random.default_rng(s))
                      * 10 ** (lv / 20)).astype(np.float32)
         for c, db in SLOPES.items() for lv in LEVELS_DBFS for s in SEEDS}
rows = []
for run in RUNS:
    rj = ROOT / "artifacts/runs" / f"{run}.json"
    args = json.loads(rj.read_text())["args"]
    for arch in args["models"]:
        ckpt = next((ROOT / "artifacts/models").glob(f"{run}_none_*_{arch}.pt"))
        # Iteration 20: a filtered model is scored with its own frontend (run args `bandpass`, `wiener`).
        cfg = dataclasses.replace(DEFAULT, window_seconds=float(args.get("window_seconds", 1.0)),   # iteration 23: 2 s models
                                  bandpass=tuple(args["bandpass"]) if args.get("bandpass") else None,
                                  wiener=bool(args.get("wiener", False)))
        thr = threshold_from_run(rj, str(ckpt), cfg.window_seconds)
        sc = load_scorer(str(ckpt), cfg, device)
        for c in SLOPES:
            for lv in LEVELS_DBFS:
                d = [stream(Clip(c, noise[(c, lv, s)], cfg.sample_rate, 1, "FLOAT"), sc, cfg, thr)
                     for s in SEEDS]
                rows.append(dict(run=run, arch=arch, colour=c, dbfs=lv, threshold=thr,
                                 alarm=float(np.mean(np.concatenate([x.drone for x in d]))),
                                 mean_score=float(np.mean(np.concatenate([x.score for x in d])))))
                print(f"{run:15} {arch:6} {c:5} {lv:+5.0f} dBFS  alarms {rows[-1]['alarm']:6.1%}  "
                      f"mean score {rows[-1]['mean_score']:.3f}  (threshold {thr:.3f})", flush=True)
# Default runs keep entry 171's file; other runs write <prefix of the last run>_colour_check.json.
out = ROOT / "artifacts/runs" / (f"{RUNS[-1].rsplit('_s', 1)[0]}_colour_check.json" if sys.argv[1:] else "it19_colour_check.json")
out.write_text(json.dumps(rows, indent=1))
print("wrote", out)
