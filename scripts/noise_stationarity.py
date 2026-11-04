"""How steady is war-shaped noise, next to the war audio it was built from?

    uv run python scripts/noise_stationarity.py

Sources, all read as the frontend reads them (mono, 16 kHz):
  shaped  war-shaped noise, 60 s, seed 1000 (scripts/war_shaped_noise.py, war_gain.txt)
  track   the war track, data/processed/war/war_track.wav, whole length
  clips   the 23 FSD50K war clips in data/processed/stems/war, each on its own (median and range over clips)

Measures, per source:
  L10/L50/L90   level exceeded 10/50/90 % of the time, dBFS, 125 ms RMS frames (digital silence skipped)
  crest         peak over RMS of each 1 s window, dB, median over windows
  within        std over the 100 frames of a 1 s window of each level-normalised log-mel band, dB,
                mean over bands, median over windows: how much the picture the network sees changes
                inside one window
  between       std over 1 s windows (1 s apart) of each band's window mean, dB, mean over bands:
                how much the window's spectrum changes from window to window
  to_shaped     RMS over bands of a window's mean spectrum minus the shaped noise's, dB, median over
                windows: how far a real window's spectrum sits from the long-term average one
  alarms        share of 1 s windows (0.125 s hop) each network calls drone at its frozen threshold;
                for the clips also per war class

Writes artifacts/noise_stationarity/stationarity.csv. Synthetic noise numbers are not detection performance.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
import drone_war_grid as dwg  # noqa: E402

from dad.config import DEFAULT  # noqa: E402
from dad.features import LogMel  # noqa: E402
from dad.realrun import level_normalise  # noqa: E402
from dad.wavio import Clip, read_wav, stream  # noqa: E402

FS, N = DEFAULT.sample_rate, DEFAULT.samples_per_window
DB = 10 / np.log(10)                                         # natural-log power -> dB
OUT = dwg.ROOT / "artifacts/noise_stationarity/stationarity.csv"
NETS = ["it6 CNN", "it9 CNN", "it9 CRNN"]


def shaped(seconds=60.0, seed=dwg.WSN_SEED):
    x = dwg.wsn.shaped_noise(np.loadtxt(dwg.WSN_GAIN), dwg.es.FS, seconds + 0.1, seed)
    return resample_poly(x, FS, dwg.es.FS)[: int(seconds * FS)].astype(np.float32)


def measures(x: np.ndarray, fe: LogMel, ref: np.ndarray | None = None) -> dict:
    hop = FS // 8
    fr = x[: len(x) // hop * hop].reshape(-1, hop).astype(np.float64)
    rms = np.sqrt(np.mean(fr ** 2, axis=1))
    lv = 20 * np.log10(rms[rms > 0])
    l10, l50, l90 = np.percentile(lv, [90, 50, 10])          # L10 = exceeded 10 % of the time
    w = x[: len(x) // N * N].reshape(-1, N)                  # 1 s windows, 1 s apart
    w = w[np.sqrt(np.mean(w.astype(np.float64) ** 2, axis=1)) > 0]
    crest = 20 * np.log10(np.abs(w).max(axis=1) / np.sqrt(np.mean(w.astype(np.float64) ** 2, axis=1)))
    L = level_normalise(np.stack([fe.frames(a) for a in w])) * DB   # (windows, bands, frames), dB
    return dict(L10=l10, L50=l50, L90=l90, crest=float(np.median(crest)),
                within=float(np.median(L.std(axis=2).mean(axis=1))),
                between=float(L.mean(axis=2).std(axis=0).mean()) if len(L) > 1 else np.nan, windows=len(L),
                spectrum=L.mean(axis=(0, 2)),
                to_shaped=np.nan if ref is None else float(np.median(np.sqrt(((L.mean(axis=2) - ref) ** 2).mean(axis=1)))))


def alarms(xs: list[np.ndarray]) -> dict:
    """{network: decisions per x}"""
    out = {}
    for m in NETS:
        thr, sc = dwg._detectors(m)
        out[m] = [stream(Clip("", x, FS, 1, "FLOAT"), sc["CNN"], DEFAULT, thr["CNN"]).drone for x in xs]
    return out


def main():
    fe = LogMel(DEFAULT)
    clips = sorted((dwg.ROOT / "data/processed/stems/war").rglob("*.wav"))
    src = {"shaped": [shaped()], "track": [read_wav(dwg.ROOT / "data/processed/war/war_track.wav").audio],
           "clips": [read_wav(p).audio for p in clips]}
    assert len(src["clips"]) == 23, len(src["clips"])
    keys = ["L10", "L50", "L90", "crest", "within", "between", "to_shaped"]
    ref = measures(src["shaped"][0], fe)["spectrum"]
    rows = []
    for name, xs in src.items():
        per = [measures(x, fe, ref) for x in xs]
        row = dict(source=name, files=len(xs), seconds=round(sum(len(x) for x in xs) / FS, 1))
        for k in keys:
            v = np.array([p[k] for p in per], dtype=float)
            row[k] = round(float(np.nanmedian(v)), 2)
            if len(xs) > 1:
                row[k + "_range"] = f"{np.nanmin(v):.1f}..{np.nanmax(v):.1f}"
        row["L10_minus_L90"] = round(float(np.nanmedian([p["L10"] - p["L90"] for p in per])), 2)
        dec = alarms(xs)
        row.update({f"alarms {m}": round(float(np.concatenate(d).mean()), 3) for m, d in dec.items()})
        rows.append(row)
        print(row)
        if name == "clips":                                  # which war classes alarm
            for c in sorted({p.parent.name for p in clips}):
                i = [k for k, p in enumerate(clips) if p.parent.name == c]
                print(f"  {c:17s} {len(i)} clips  " + "  ".join(
                    f"{m} {np.concatenate([d[k] for k in i]).mean():5.1%}" for m, d in dec.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
