"""Clean drone vs no-drone: synthetic Pd drones against real no-drone audio, MFE frontend.

    .venv/bin/python scripts/train_clean.py --drones ~/drone/drones_5h5 --nodrone ~/nodrone \\
        --name clean_s0 --device cuda

Drone windows (label 1): 1 s windows lying wholly inside a scene's drone label, from the
scenes of `scripts/make_scenes.py --drone-only` (seed < --val-seed trains, the rest validates).
No-drone windows (label 0): 1 s windows of nodrone/recordings and nodrone/stems (listed in
recordings.csv and stems.csv), skipping digital silence; the rig --val-rig validates, the rest
trains. Nothing is mixed: each window is one drone or one no-drone clip, as Marius chose
(2026-11-04). Batches are half drone, half no-drone; variant "level" (each window's own
in-band level removed, entry 014), so loudness says nothing about the label.

Models: dscnn (MFE, 64 log-mel, the iteration 6 and 9 recipe through NetScorer, --windows),
and the baselines `comb` (no training) and `lgbm` (--lgbm-windows from the same sampler).
Validation: every 1 s window at --eval-hop inside the validation drone labels, and every 1 s
window of the validation rig's clips. Threshold at 5% false alarms on validation.

Writes <out>/<name>.json, <name>_dscnn.pt, <name>_lgbm.pkl. There is no real drone here, so
nothing in this file is a test result (CLAUDE.md section 5: synthetic scaffolding).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dad import realrun as rr  # noqa: E402
from dad.config import DEFAULT  # noqa: E402
from dad.features import LogMel  # noqa: E402

FS = DEFAULT.sample_rate
N = DEFAULT.samples_per_window


def read16k(path: Path) -> np.ndarray:
    x, fs = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if fs != FS:
        g = np.gcd(fs, FS)
        x = resample_poly(x, FS // g, fs // g).astype(np.float32)
    return x


def load_drones(root: Path, val_seed: int, limit: int | None):
    """[(audio, lo, hi)] per scene, train and validation; lo/hi: drone label in samples."""
    train, val = [], []
    for k, lab in enumerate(sorted((root / "labels").glob("*.json"))):
        if limit and k >= limit:
            break
        d = json.loads(lab.read_text())
        ev = [e for e in d["events"] if e["role"] == "drone"]
        if not ev:
            continue
        lo, hi = int(ev[0]["onset"] * FS), int(ev[0]["offset"] * FS)
        if hi - lo < N:
            continue
        (val if d["seed"] >= val_seed else train).append((read16k(root / "audio" / f"{lab.stem}.wav"), lo, hi))
    return train, val


def load_nodrone(root: Path, val_rig: str, limit: int | None):
    """[(audio, group)] per clip; the clips of rig val_rig validate."""
    rows = [(r["wav"], r["source"]) for r in csv.DictReader((root / "recordings.csv").open())]
    rows += [(r["file"], "stem:" + r["group"]) for r in csv.DictReader((root / "stems.csv").open())]
    train, val = [], []
    for k, (rel, group) in enumerate(rows):
        if limit and k >= limit * 2:
            break
        x = read16k(root / rel)
        if len(x) < N:
            continue
        (val if group == val_rig else train).append((x, group))
    return train, val


class CleanSampler:
    """Batch `index` is the same every time for a given seed: half drone, half no-drone windows."""

    def __init__(self, drones, nodrone, seed: int = 0):
        self.drones, self.nodrone, self.seed = drones, nodrone, seed
        self.fe = LogMel(DEFAULT)

    def _drone(self, rng):
        x, lo, hi = self.drones[int(rng.integers(len(self.drones)))]
        s = int(rng.integers(lo, hi - N + 1))
        return x[s:s + N]

    def _nodrone(self, rng):
        for _ in range(50):
            x, _ = self.nodrone[int(rng.integers(len(self.nodrone)))]
            s = int(rng.integers(0, len(x) - N + 1))
            w = x[s:s + N]
            if rr.inband_power(w) > rr.SILENCE_POWER:
                return w
        raise RuntimeError("could not draw a non-silent no-drone window")

    def batch(self, index: int, batch_size: int):
        rng = np.random.default_rng([self.seed, index])
        y = (np.arange(batch_size) % 2).astype(np.int64)
        L = np.stack([self.fe.frames(self._drone(rng) if t else self._nodrone(rng)) for t in y])
        return rr.level_normalise(L), y


def eval_windows(clips, hop: int):
    """Every 1 s window at hop: inside (lo, hi) for drones, the whole clip otherwise."""
    out = []
    for c in clips:
        x, lo, hi = c if len(c) == 3 else (c[0], 0, len(c[0]))
        out += [x[s:s + N] for s in range(lo, hi - N + 1, hop)
                if len(c) == 3 or rr.inband_power(x[s:s + N]) > rr.SILENCE_POWER]
    return np.stack(out)


def scores(scorer, pos, neg) -> dict:
    sp, sn = scorer.score(pos), scorer.score(neg)
    thr = float(np.quantile(sn, 0.95))
    return dict(auc=rr._auc(sp, sn), recall_at_5pct_fa=rr._recall_at_fa(sp, sn),
                fa_at_95_recall=rr._fa_at_recall(sp, sn), threshold_5pct_fa=thr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drones", type=Path, required=True)
    p.add_argument("--nodrone", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", type=Path, default=REPO / "artifacts" / "runs")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--windows", type=int, default=200_000)
    p.add_argument("--lgbm-windows", type=int, default=40_000)
    p.add_argument("--val-seed", type=int, default=1800, help="drone scenes with seed >= this validate")
    p.add_argument("--val-rig", default="nightjar", help="no-drone source held out for validation")
    p.add_argument("--eval-hop", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None, help="load only this many scenes (smoke tests)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)  # noqa: E731

    d_train, d_val = load_drones(a.drones, a.val_seed, a.limit)
    n_train, n_val = load_nodrone(a.nodrone, a.val_rig, a.limit)
    if not (d_train and d_val and n_train and n_val):
        raise SystemExit(f"empty split: drones {len(d_train)}/{len(d_val)}, no-drone {len(n_train)}/{len(n_val)}")
    hop = int(a.eval_hop * FS)
    pos, neg = eval_windows(d_val, hop), eval_windows(n_val, hop)
    res = dict(name=a.name, seed=a.seed, frontend=DEFAULT.describe(), variant="level", window_s=1.0,
               eval_hop_s=a.eval_hop, windows=a.windows, lgbm_windows=a.lgbm_windows,
               train=dict(drone_scenes=len(d_train), nodrone_clips=len(n_train),
                          nodrone_groups=sorted({g for _, g in n_train})),
               val=dict(drone_scenes=len(d_val), drone_windows=len(pos), nodrone_clips=len(n_val),
                        nodrone_windows=len(neg), rig=a.val_rig), models={})
    log(f"train: {len(d_train)} drone scenes, {len(n_train)} no-drone clips; "
        f"validation: {len(pos)} drone and {len(neg)} no-drone windows")

    sampler = CleanSampler(d_train, n_train, a.seed)
    variant = rr.VARIANTS["level"]
    res["models"]["comb"] = scores(rr.CombScorer(DEFAULT), pos, neg)
    log(f"comb: {res['models']['comb']}")
    lg = rr.LgbmScorer(variant, DEFAULT).fit(sampler, a.lgbm_windows)
    lg.save(a.out / f"{a.name}_lgbm.pkl")
    res["models"]["lgbm"] = scores(lg, pos, neg)
    log(f"lgbm: {res['models']['lgbm']}")
    t0 = time.time()
    net = rr.NetScorer("dscnn", variant, DEFAULT, a.device).fit(sampler, a.windows, log=log)
    net.save(a.out / f"{a.name}_dscnn.pt")
    res["models"]["dscnn"] = dict(scores(net, pos, neg), train_seconds=round(time.time() - t0, 1),
                                  history=net.history)
    log(f"dscnn: { {k: v for k, v in res['models']['dscnn'].items() if k != 'history'} }")
    (a.out / f"{a.name}.json").write_text(json.dumps(res, indent=1))
    log(f"wrote {a.out / (a.name + '.json')}")


if __name__ == "__main__":
    main()
