"""Iteration 11, step 0: what in a window makes the networks say "drone"?

    uv run python scripts/phase_probe.py [--per-source 300] [--seed 0]

Real 1 s windows from the processed store (drone recordings: label 1; no-drone: label 0; the
lawn-mower session and Lund's suspect recording left out) are scored as they are and in two copies:
  phase     the window's own FFT magnitude with random phases: the same fine spectrum, harmonic lines
            included, but no time structure (no blade modulation, no onsets)
  envelope  the magnitude smoothed over about one octave (at least 300 Hz) before the random phases:
            only the coarse spectral shape is left, no harmonic lines, no time structure
Each copy keeps the window's RMS. Plus a sweep of Gaussian noise with spectral slopes from -12 to
+6 dB/octave (20 s each).

Per network (it6 CNN, it9 CNN, it9 CRNN, each at its frozen threshold):
  drone windows the network detects as they are: the share still detected as phase / envelope copies
  no-drone windows: the alarm share as they are, as phase copies, as envelope copies
  noise sweep: the alarm share per slope

These windows come from rigs the networks trained on (Lund excepted): a diagnostic of what the
networks use, not detection performance. Writes artifacts/phase_probe/phase_probe.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import drone_war_grid as dwg  # noqa: E402

from dad.config import DEFAULT  # noqa: E402
from dad.rangerun import CONFUSER, LUND_SUSPECT  # noqa: E402
from dad.realdata import load_manifest  # noqa: E402

FS, N = DEFAULT.sample_rate, DEFAULT.samples_per_window
NETS = ["it6 CNN", "it9 CNN", "it9 CRNN"]
SOURCES = ["nightjar", "d3set", "fpv", "halmstad", "lund"]
SLOPES = np.arange(-12, 7, 3)
OUT = dwg.ROOT / "artifacts/phase_probe/phase_probe.csv"


def random_phase(mag: np.ndarray, rng) -> np.ndarray:
    ph = np.exp(2j * np.pi * rng.random(mag.shape))
    ph[..., 0] = 1.0
    return np.fft.irfft(mag * ph, N, axis=-1)


def smooth(power: np.ndarray) -> np.ndarray:
    """Mean power over [f - h, f + h], h = max(150 Hz, 0.35 f): about one octave, at least 300 Hz wide."""
    f = np.fft.rfftfreq(N, 1 / FS)
    h = np.maximum(150.0, 0.35 * f) * N / FS                  # half-width in bins
    k = np.arange(len(f))
    lo = np.clip(np.round(k - h).astype(int), 0, len(f) - 1)
    hi = np.clip(np.round(k + h).astype(int), 0, len(f) - 1)
    c = np.concatenate([np.zeros(power.shape[:-1] + (1,)), np.cumsum(power, axis=-1)], axis=-1)
    return (c[..., hi + 1] - c[..., lo]) / (hi - lo + 1)


def copies(w: np.ndarray, rng) -> dict[str, np.ndarray]:
    X = np.fft.rfft(w.astype(np.float64), axis=-1)
    rms = np.sqrt(np.mean(w.astype(np.float64) ** 2, axis=-1, keepdims=True))
    out = {"phase": random_phase(np.abs(X), rng), "envelope": random_phase(np.sqrt(smooth(np.abs(X) ** 2)), rng)}
    for k, y in out.items():
        out[k] = (y * rms / np.maximum(np.sqrt(np.mean(y ** 2, axis=-1, keepdims=True)), 1e-12)).astype(np.float32)
    return {"original": w.astype(np.float32), **out}


def slope_noise(db_per_oct: float, seconds: float, rng) -> np.ndarray:
    n = int(seconds * FS)
    f = np.fft.rfftfreq(n, 1 / FS)
    f[0] = f[1]
    g = (f / 1000.0) ** (db_per_oct / (20 * np.log10(2)))    # amplitude: db_per_oct dB of power per octave
    g[0] = 0.0
    x = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * g, n)
    return (x / np.sqrt(np.mean(x ** 2)) * 10 ** (-30 / 20)).astype(np.float32)


def windows(recs, per_source: int, rng) -> np.ndarray:
    """Up to per_source windows, a few from each recording, skipping digital silence."""
    take = max(1, -(-per_source // max(len(recs), 1)))
    out = []
    for r in recs:
        if r.n_samples < N:
            continue
        for s in rng.integers(0, r.n_samples - N + 1, size=min(take, r.n_samples // N)):
            a = r.audio(int(s), N)
            if np.any(a):
                out.append(a)
    rng.shuffle(out)
    return np.stack(out[:per_source]) if out else np.empty((0, N), np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-source", type=int, default=300, help="windows per source and label")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    recs = [r for r in load_manifest() if r.group not in CONFUSER and r.recording_id not in LUND_SUSPECT
            and r.source in SOURCES and r.label in (0, 1)]
    sets = {}
    for src in SOURCES:
        for lab in (1, 0):
            w = windows([r for r in recs if r.source == src and r.label == lab], a.per_source, rng)
            if len(w):
                sets[(src, lab)] = copies(w, rng)
    noise = {s: slope_noise(s, 20.0, rng) for s in SLOPES}
    rows = []
    for m in NETS:
        thr, sc = dwg._detectors(m)
        t, net = thr["CNN"], sc["CNN"]
        for (src, lab), cp in sets.items():
            s = {k: np.asarray(net.score(x)) for k, x in cp.items()}
            hit = s["original"] >= t
            row = dict(model=m, source=src, label=lab, windows=len(hit), alarms_original=hit.mean())
            for k in ("phase", "envelope"):
                row[f"alarms_{k}"] = (s[k] >= t).mean()
                row[f"kept_{k}"] = (s[k][hit] >= t).mean() if hit.any() else np.nan   # of the windows detected as they are
            rows.append(row)
        for slope, x in noise.items():
            w = np.lib.stride_tricks.sliding_window_view(x, N)[:: DEFAULT.samples_per_hop]
            rows.append(dict(model=m, source=f"noise {slope:+d} dB/oct", label=0, windows=len(w),
                             alarms_original=(np.asarray(net.score(np.ascontiguousarray(w))) >= t).mean()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    for r in rows:
        print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
