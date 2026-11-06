"""Clean drone vs no-drone: synthetic Pd drones against real no-drone audio, MFE frontend.

    .venv/bin/python scripts/train_clean.py --drones ~/drone/drones_5h5 --nodrone ~/nodrone \\
        --name clean_s0 --device cuda

Drone windows (label 1): 1 s windows lying wholly inside a scene's drone label, from the
scenes of `scripts/make_scenes.py --drone-only` (seed < --val-seed trains, the rest validates).
No-drone windows (label 0): 1 s windows of nodrone/recordings and nodrone/stems (listed in
recordings.csv and stems.csv), skipping digital silence; the rig --val-rig validates, the rest
trains. By default nothing is mixed: each window is one drone or one no-drone clip, as Marius chose
(2026-11-04); --mix-real, --speed and --eq add it9's mixing and augmentation (entry 123). Batches are half drone, half no-drone; variant "level" (each window's own
in-band level removed, entry 014), so loudness says nothing about the label.
With soundscape scenes (`make_scenes.py` without --drone-only), the drone windows carry their
scene's background, and the scenes without a drone join the no-drone clips (split by the same
--val-seed), so a soundscape alone is a negative as well. Scenes marked heldout (an FSD50K eval
stem in them, a test source) are skipped.

Models: dscnn (MFE, 64 log-mel, the iteration 6 and 9 recipe through NetScorer, --windows),
and the baselines `comb` (no training) and `lgbm` (--lgbm-windows from the same sampler).
Validation: every 1 s window at --eval-hop inside the validation drone labels, and every 1 s
window of the validation rig's clips. Threshold at 5% false alarms on validation.

--arch picks the network from dad.models.zoo (dscnn by default; crnn: conv blocks, a bidirectional
GRU over time and attention pooling, as master's entry 096 trained on real rigs). It is stored under
its own name: <name>_<arch>.pt and models.<arch> in the JSON.

Writes <out>/<name>.json, <name>_<arch>.pt, <name>_lgbm.pkl. There is no real drone here, so
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
    """[(audio, lo, hi)] per scene, train and validation; lo/hi: drone label in samples.
    Scenes without a drone come back as no-drone clips [(audio, "scene")]; heldout scenes are skipped."""
    train, val, neg_train, neg_val = [], [], [], []
    for k, lab in enumerate(sorted((root / "labels").glob("*.json"))):
        if limit and k >= limit:
            break
        d = json.loads(lab.read_text())
        if d.get("heldout"):
            continue
        ev = [e for e in d["events"] if e["role"] == "drone"]
        if not ev:
            (neg_val if d["seed"] >= val_seed else neg_train).append((read16k(root / "audio" / f"{lab.stem}.wav"), "scene"))
            continue
        lo, hi = int(ev[0]["onset"] * FS), int(ev[0]["offset"] * FS)
        if hi - lo < N:
            continue
        (val if d["seed"] >= val_seed else train).append((read16k(root / "audio" / f"{lab.stem}.wav"), lo, hi))
    return train, val, neg_train, neg_val


# Test sources (REQUIREMENTS.md): never in training. Before entry 132 the it11 line trained on the
# Halmstad and FPV no-drone recordings of nodrone/recordings.csv.
TEST_SOURCES = frozenset({"halmstad", "fpv", "dronenoise"})


def load_nodrone(root: Path, val_rig: str, limit: int | None):
    """[(audio, group)] per clip; the clips of rig val_rig validate. Test sources are skipped."""
    rows = [(r["wav"], r["source"]) for r in csv.DictReader((root / "recordings.csv").open())
            if r["source"] not in TEST_SOURCES]
    rows += [(r["file"], "stem:" + r["group"]) for r in csv.DictReader((root / "stems.csv").open())
             if r.get("source") != "fsd50k_eval"]          # FSD50K eval is a held-out test source
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
    """Batch `index` is the same every time for a given seed: half drone, half no-drone windows.

    speed (lo, hi): every window is resampled by a factor from lo-hi, as rangerun's --speed (it9).
    eq: a random smooth gain over the mel bands on every window (realrun._eq_curve, as it9).
    mix_real: the share of scene windows (drone and no-drone scenes alike) summed into a real
    no-drone recording at mix_snr dB in-band, the scene over the recording, as rangerun's simdist.
    Both classes get it, so "real recording texture" says nothing about the label (entry 123)."""

    def __init__(self, drones, nodrone, seed: int = 0, speed=None, eq: bool = False,
                 mix_real: float = 0.0, mix_snr=(-10.0, 5.0)):
        from dad.rangerun import SPEED_UP, speed_down
        self.seed, self.speed, self.eq, self.mix_real, self.mix_snr = seed, speed, eq, mix_real, mix_snr
        self._up, self._down = SPEED_UP, speed_down
        self.n = N if speed is None else int(np.ceil(N * speed[1])) + 32   # long enough to speed up
        self.drones = [d for d in drones if d[2] - d[1] >= self.n]
        self.nodrone = [c for c in nodrone if len(c[0]) >= self.n]
        self.real = [c for c in self.nodrone if c[1] != "scene" and not c[1].startswith("stem:")]
        if mix_real and not self.real:
            raise SystemExit("--mix-real needs real no-drone recordings (--nodrone)")
        self.fe = LogMel(DEFAULT)

    def _clip(self, clips, rng):
        """A non-silent window of n samples from a random clip; its group."""
        for _ in range(50):
            x, g = clips[int(rng.integers(len(clips)))]
            s = int(rng.integers(0, len(x) - self.n + 1))
            w = x[s:s + self.n]
            if rr.inband_power(w) > rr.SILENCE_POWER:
                return w, g
        raise RuntimeError("could not draw a non-silent no-drone window")

    def _mix(self, w, rng):
        if not (self.mix_real and rng.random() < self.mix_real):
            return w
        a, _ = self._clip(self.real, rng)
        pa, pw = rr.inband_power(a), max(rr.inband_power(w), 1e-20)
        return a + w * np.sqrt(pa * 10.0 ** (rng.uniform(*self.mix_snr) / 10.0) / pw)

    def _drone(self, rng):
        x, lo, hi = self.drones[int(rng.integers(len(self.drones)))]
        s = int(rng.integers(lo, hi - self.n + 1))
        return self._mix(x[s:s + self.n], rng)

    def _nodrone(self, rng):
        w, g = self._clip(self.nodrone, rng)
        return self._mix(w, rng) if g == "scene" else w

    def raw_batch(self, index: int, batch_size: int):
        """batch()'s windows before the frontend, as gpufront.GpuFrontend.train_features takes them:
        audio (batch, n), the speed as resample_down's `down`, the EQ curves and the labels. With it,
        NetScorer runs the frontend on a CUDA device and the CPU only draws windows (report entry 119)."""
        rng = np.random.default_rng([self.seed, index])
        y = (np.arange(batch_size) % 2).astype(np.int64)
        audio = np.stack([self._drone(rng) if t else self._nodrone(rng) for t in y]).astype(np.float32)
        down = np.array([self._up if self.speed is None else self._down(rng.uniform(*self.speed)) for _ in y])
        bands = self.fe._fb.shape[0]
        eq = (np.stack([rr._eq_curve(rng, bands) for _ in y]) if self.eq
              else np.zeros((batch_size, bands), np.float32))
        return audio, down, eq, y

    def batch(self, index: int, batch_size: int):
        from dad.rangerun import resample_down
        audio, down, eq, y = self.raw_batch(index, batch_size)       # the same draws as the GPU path
        L = np.stack([self.fe.frames(resample_down(x, d)[:N]) for x, d in zip(audio, down)])
        return rr.level_normalise(L + eq[:, :, None]), y


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
    p.add_argument("--drones", type=Path, nargs="+", required=True,
                   help="one or more scene folders (make_scenes.py --out), e.g. it11/drone it11/nodrone")
    p.add_argument("--nodrone", type=Path, default=None,
                   help="real no-drone clips (recordings.csv, stems.csv); without it only no-drone scenes are negatives")
    p.add_argument("--name", required=True)
    p.add_argument("--out", type=Path, default=REPO / "artifacts" / "runs")
    p.add_argument("--device", default="cpu")
    p.add_argument("--arch", default="dscnn", choices=("dscnn", "crnn"), help="the network (dad.models.zoo)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--windows", type=int, default=200_000)
    p.add_argument("--lgbm-windows", type=int, default=40_000)
    p.add_argument("--val-seed", type=int, default=1800, help="drone scenes with seed >= this validate")
    p.add_argument("--val-rig", default="nightjar", help="no-drone source held out for validation")
    p.add_argument("--eval-hop", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None, help="load only this many scenes (smoke tests)")
    p.add_argument("--speed", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                   help="resample every training window by a factor from LO-HI (it9: 0.8 1.25)")
    p.add_argument("--eq", action="store_true", help="random spectral EQ on every training window (as it9)")
    p.add_argument("--mix-real", type=float, default=0.0,
                   help="share of scene windows summed into a real no-drone recording (entry 123)")
    p.add_argument("--mix-snr", nargs=2, type=float, default=(-10.0, 5.0), help="scene over recording, dB in-band")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)  # noqa: E731

    d_train, d_val, w_train, w_val = (sum(z, []) for z in zip(*(load_drones(r, a.val_seed, a.limit) for r in a.drones)))
    n_train, n_val = load_nodrone(a.nodrone, a.val_rig, a.limit) if a.nodrone else ([], [])
    n_train, n_val = n_train + w_train, n_val + w_val       # no-drone scenes, if any, are no-drone too
    if not (d_train and d_val and n_train and n_val):
        raise SystemExit(f"empty split: drones {len(d_train)}/{len(d_val)}, no-drone {len(n_train)}/{len(n_val)}")
    hop = int(a.eval_hop * FS)
    pos, neg = eval_windows(d_val, hop), eval_windows(n_val, hop)
    res = dict(name=a.name, seed=a.seed, frontend=DEFAULT.describe(), variant="level", window_s=1.0,
               eval_hop_s=a.eval_hop, windows=a.windows, lgbm_windows=a.lgbm_windows,
               train=dict(drone_scenes=len(d_train), nodrone_clips=len(n_train),
                          nodrone_groups=sorted({g for _, g in n_train})),
               val=dict(drone_scenes=len(d_val), drone_windows=len(pos), nodrone_clips=len(n_val),
                        nodrone_windows=len(neg), rig=a.val_rig if a.nodrone else None), models={})
    log(f"train: {len(d_train)} drone scenes, {len(n_train)} no-drone clips; "
        f"validation: {len(pos)} drone and {len(neg)} no-drone windows")

    sampler = CleanSampler(d_train, n_train, a.seed, a.speed, a.eq, a.mix_real, tuple(a.mix_snr))
    res["augment"] = dict(speed=a.speed, eq=a.eq, mix_real=a.mix_real, mix_snr=list(a.mix_snr))
    variant = rr.VARIANTS["level"]
    res["models"]["comb"] = scores(rr.CombScorer(DEFAULT), pos, neg)
    log(f"comb: {res['models']['comb']}")
    lg = rr.LgbmScorer(variant, DEFAULT).fit(sampler, a.lgbm_windows)
    lg.save(a.out / f"{a.name}_lgbm.pkl")
    res["models"]["lgbm"] = scores(lg, pos, neg)
    log(f"lgbm: {res['models']['lgbm']}")
    t0 = time.time()
    net = rr.NetScorer(a.arch, variant, DEFAULT, a.device).fit(sampler, a.windows, log=log)
    net.save(a.out / f"{a.name}_{a.arch}.pt")
    res["models"][a.arch] = dict(scores(net, pos, neg), train_seconds=round(time.time() - t0, 1),
                                  history=net.history)
    log(f"{a.arch}: { {k: v for k, v in res['models'][a.arch].items() if k != 'history'} }")
    (a.out / f"{a.name}.json").write_text(json.dumps(res, indent=1))
    log(f"wrote {a.out / (a.name + '.json')}")


if __name__ == "__main__":
    main()
