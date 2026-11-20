"""False alarms per hour of a detector on the complete test's no-drone layers, at a list of thresholds.

The audio is what the complete test plays with no drone: the war soundscape (38.5 s, looped), the Pure
Data wind renders and the Pure Data car pass-bys in artifacts/complete_test/layers (the sealed car is
skipped), each scaled the way the test scales a layer (RMS of the whole signal to a dB SPL, full scale =
110 dB SPL), alone and mixed. Windows of 1 s every 0.125 s, as the live test. An alarm is a stretch of
consecutive windows at or above the threshold (as live_test_engine._false_alarms): a car that holds the
score up for 3 s is one alarm.

These are not real recordings of a war zone: the war soundscape is one 38.5 s file, so its hours are
that file again and again; the cars and the wind are synthetic. A number here is a floor on what happens
in practice, not an estimate of it.

    uv run python scripts/fa_rate.py [--detector "it9 CRNN"] [--thresholds 0.483 0.9 0.95 0.99]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import complete_test_engine as E                              # noqa: E402

SR = E.SR
WIN, HOP = SR, SR // 8


def load16(p: Path) -> np.ndarray:
    x, sr = sf.read(p, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    return resample_poly(x, SR, sr) if sr != SR else x


def at_level(x: np.ndarray, db: float) -> np.ndarray:
    return x * (E.rms_for(db) / np.sqrt(np.mean(x ** 2)))


def score(sc, x: np.ndarray, batch: int = 512) -> np.ndarray:
    ends = np.arange(WIN, len(x) + 1, HOP)
    out = []
    for i in range(0, len(ends), batch):
        out.append(np.asarray(sc.score(np.stack([x[e - WIN:e].astype(np.float32) for e in ends[i:i + batch]])), float))
    return np.concatenate(out) if out else np.zeros(0)


def alarms(s: np.ndarray, thr: float) -> int:
    a = s >= thr
    return int(np.sum(a & ~np.concatenate([[False], a[:-1]])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="it9 CRNN")
    ap.add_argument("--thresholds", type=float, nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/complete_test/fa_rate")
    a = ap.parse_args()
    thr, sc = E.load_detectors([a.detector])
    frozen = thr[a.detector]
    ths = a.thresholds or [round(frozen, 3), 0.9, 0.95, 0.99]
    sc = sc[a.detector]

    war = load16(E.WAR_SOUNDSCAPE)
    layers = sorted((ROOT / "artifacts/complete_test/layers").glob("*.wav"))
    cars = [p for p in layers if p.name.startswith("cars_") and "sealed" not in p.name]
    winds = [p for p in layers if p.name.startswith("wind_")]

    cases = []                                                   # (group, name, signal)
    for db in (60, 75, 90):
        cases.append((f"war {db} dB", "WarSoundScape", at_level(np.tile(war, 4), db)))   # 154 s: the loop 4 times
    for p in winds:
        x = load16(p)
        for db in (25, 50):
            cases.append((f"wind {db} dB", p.stem, at_level(x, db)))
    for p in cars:
        x = load16(p)
        for db in (50, 70):
            cases.append((f"cars {db} dB", p.stem, at_level(x, db)))
        w = np.resize(war, len(x))
        cases.append(("war 60 + cars 50 dB", p.stem, at_level(w, 60) + at_level(x, 50)))
    for p in winds:
        x = load16(p)
        w = np.resize(war, len(x))
        cases.append(("war 60 + wind 25 dB", p.stem, at_level(w, 60) + at_level(x, 25)))

    rows = []
    for g, name, x in cases:
        s = score(sc, x)
        rows.append(dict(group=g, file=name, seconds=round(len(x) / SR, 1), windows=len(s), max_conf=round(float(s.max()), 4),
                         mean_conf=round(float(s.mean()), 4),
                         **{f"alarms@{t:g}": alarms(s, t) for t in ths},
                         **{f"windows@{t:g}": int((s >= t).sum()) for t in ths}))
    a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / f"{a.detector.replace(' ', '_')}.csv"
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{a.detector}, frozen threshold {frozen:.3f}; {f}")
    print(f"{'background':22s} {'minutes':>7s} {'max':>6s} {'mean':>6s} " + " ".join(f"{'FA/h@' + format(t, 'g'):>10s}" for t in ths))
    for g in dict.fromkeys(r["group"] for r in rows):
        rs = [r for r in rows if r["group"] == g]
        sec = sum(r["seconds"] for r in rs)
        print(f"{g:22s} {sec / 60:7.1f} {max(r['max_conf'] for r in rs):6.3f} {np.mean([r['mean_conf'] for r in rs]):6.3f} "
              + " ".join(f"{sum(r[f'alarms@{t:g}'] for r in rs) / sec * 3600:10.1f}" for t in ths))


if __name__ == "__main__":
    main()
