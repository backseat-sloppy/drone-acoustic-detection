"""Listening list: the Nightjar no-drone stretches that sound most like a drone.

Nightjar's no-drone label means "no logged DJI flight", not "no drone" (entry 226).
Every 10 s segment of every label-0 Nightjar recording is scored by two detectors:
`comb` (untrained harmonic comb, unbiased) and the it19 CRNN (trained on these very
windows as negatives, so it is biased low here). Segments are ranked by the mean of
their two percentile ranks; the top N are written as unscaled WAVs for listening.

    uv run python -I scripts/nightjar_listen_list.py --top 40
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import drone_war_grid as dwg  # noqa: E402
from dad.realdata import SAMPLE_RATE as SR, load_manifest  # noqa: E402
from dad.wavio import Clip, stream  # noqa: E402

SEG = 10  # seconds per listening segment
MODEL = "it19 CRNN colour s1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--out", type=Path, default=ROOT / "data/extracted/nightjar_listen")
    ap.add_argument("--gain-db", type=float, default=50.0,
                    help="one common gain for the audible copies in OUT/plus<G>dB (Nightjar sits near -80 dBFS)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    loud = a.out / f"plus{a.gain_db:g}dB"
    loud.mkdir(exist_ok=True)
    g = 10 ** (a.gain_db / 20)

    thr, sc = dwg._detectors(MODEL, "cpu")
    recs = [r for r in load_manifest() if r.source == "nightjar" and r.label == 0]
    rows = []
    for r in recs:
        x = r.audio()
        dec = {k: stream(Clip(r.recording_id, x, SR, 1, "FLOAT"), s, s.cfg, thr[k]) for k, s in sc.items()}
        for i in range(r.n_samples // (SEG * SR)):
            row = dict(recording_id=r.recording_id, group=r.group, start_s=i * SEG, end_s=(i + 1) * SEG,
                       parent=r.attributes.get("parent", ""),
                       parent_start_s=r.attributes.get("parent_start_s", 0.0) + i * SEG)
            for k, d in dec.items():
                m = (d.t_end > i * SEG) & (d.t_end <= (i + 1) * SEG)
                row[f"{k}_max"] = float(d.score[m].max()) if m.any() else float("nan")
                row[f"{k}_alarm_frac"] = float(d.drone[m].mean()) if m.any() else float("nan")
            seg = x[i * SEG * SR:(i + 1) * SEG * SR]
            row["rms_dbfs"] = float(20 * np.log10(np.sqrt(np.mean(seg.astype(np.float64) ** 2)) + 1e-12))
            rows.append(row)
        print(f"{r.recording_id}: {r.n_samples // (SEG * SR)} segments", flush=True)

    for k in sc:  # percentile rank per detector, then the mean of the two
        v = np.array([q[f"{k}_max"] for q in rows])
        rank = v.argsort().argsort() / max(len(v) - 1, 1)
        for q, p in zip(rows, rank):
            q[f"{k}_pct"] = float(p)
    for q in rows:
        q["rank_score"] = float(np.mean([q[f"{k}_pct"] for k in sc]))
    rows.sort(key=lambda q: -q["rank_score"])

    by_id = {r.recording_id: r for r in recs}
    for n, q in enumerate(rows, 1):
        q["rank"] = n
        q["wav"] = ""
        if n <= a.top:
            r = by_id[q["recording_id"]]
            seg = r.audio(q["start_s"] * SR, SEG * SR)  # unscaled: the stored level, never normalised
            name = f"{n:03d}_{r.recording_id.replace('/', '_').replace('#', '_')}_{q['start_s']:04d}s.wav"
            sf.write(a.out / name, seg, SR, subtype="FLOAT")
            # same gain for every clip, so relative loudness survives; tanh only bends the few transients
            # that would pass full scale, to protect headphones
            sf.write(loud / name, np.tanh(seg.astype(np.float64) * g), SR, subtype="FLOAT")
            q["wav"] = name
            q["mads_verdict"] = ""  # drone / no drone / unsure, filled in by ear
    keys = ["rank", "wav", "mads_verdict", "recording_id", "group", "start_s", "end_s", "parent", "parent_start_s",
            "rms_dbfs", "rank_score"] + [f"{k}_{s}" for k in sc for s in ("max", "alarm_frac", "pct")]
    with open(a.out / "listen_list.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} segments from {len(recs)} recordings; top {a.top} written to {a.out}")


if __name__ == "__main__":
    main()
