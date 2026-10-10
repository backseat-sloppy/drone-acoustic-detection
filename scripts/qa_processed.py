"""Quality checks on data/processed. Reads only; prints figures for the report.

    uv run python -I scripts/qa_processed.py
"""
from __future__ import annotations

import collections
import csv
import datetime as dt
import sys
from pathlib import Path

import numpy as np
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dad.config import DEFAULT  # noqa: E402
from dad.features import LogMel  # noqa: E402

SR = 16_000
OUT = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_manifest() -> list[dict]:
    with open(OUT / "manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def audio(r: dict) -> np.ndarray:
    m = np.memmap(OUT / r["file"], dtype=r["dtype"], mode="r")
    x = np.asarray(m[int(r["offset"]): int(r["offset"]) + int(r["n_samples"])], dtype=np.float32)
    return x / 32768 if r["dtype"] == "int16" else x


def pct(v, qs=(5, 25, 50, 75, 95)) -> str:
    return " ".join(f"{np.percentile(v, q):7.1f}" for q in qs)


def levels(rows: list[dict]) -> None:
    print("\n== Recording level, dBFS RMS (percentiles 5 25 50 75 95) and duration")
    groups = collections.defaultdict(list)
    for r in rows:
        key = r["source"] if r["source"] != "halmstad" else f"halmstad/{r['kind']}"
        if r["source"] == "ddl":
            key = f"ddl/{r['subset']}"
        groups[key].append(r)
    for key in sorted(groups):
        g = groups[key]
        rms = [float(r["rms_dbfs"]) for r in g]
        dur = [float(r["duration_s"]) for r in g]
        print(f"  {key:<22} n={len(g):>6}  rms {pct(rms)}   dur_s {pct(dur)}   total {sum(dur) / 3600:6.2f} h")


def ddl_cross_session(rows: list[dict]) -> None:
    """Search for the same audio appearing in two sessions.

    Exact duplicates were removed by hash in prepare_data.py. That cannot see
    two sessions cut from overlapping stretches of one recording at different
    offsets, so sessions close in time are compared by cross-correlation.
    """
    print("\n== DDL: same audio in two sessions? (normalised cross-correlation, 1.0 = identical)")
    spans = collections.defaultdict(list)
    with open(OUT / "ddl_clips.csv", newline="") as f:
        for c in csv.DictReader(f):
            if c["kept"] == "1":
                spans[c["session"]].append(dt.datetime.strptime(c["timestamp"], "%Y%m%d%H%M%S"))
    ddl = {r["recording_id"]: r for r in rows if r["source"] == "ddl"}
    names = sorted(ddl)
    slack = dt.timedelta(seconds=120)  # labels drift by up to ~80 s in some sessions
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
             if min(spans[a]) - slack <= max(spans[b]) and min(spans[b]) - slack <= max(spans[a])]
    worst = 0.0
    for a, b in pairs:
        best = 0.0
        for src, dst in ((a, b), (b, a)):
            x, y = audio(ddl[src]), audio(ddl[dst]).astype(np.float64)
            n = 2 * SR
            if len(y) < n:
                continue
            total = np.concatenate([[0.0], np.cumsum(y * y)])
            energy = np.sqrt(np.maximum(total[n:] - total[:-n], 1e-20))
            for start in range(0, len(x) - n, 20 * SR):
                s = x[start:start + n].astype(np.float64)
                c = correlate(y, s, mode="valid", method="fft")
                best = max(best, float(np.max(np.abs(c) / (np.linalg.norm(s) * energy))))
        worst = max(worst, best)
        flag = "  <-- DUPLICATE AUDIO" if best > 0.9 else ""
        print(f"  {a} ~ {b}: {best:.3f}{flag}")
    print(f"  pairs checked: {len(pairs)}, highest correlation: {worst:.3f}")


def dronenoise_profile(rows: list[dict]) -> None:
    """How long is the drone actually loud in each DroneNoise recording?"""
    print("\n== DroneNoise: 1 s level relative to the loudest second of the same file, dB")
    drops = {"flyover": [], "hover": []}
    for r in rows:
        if r["source"] != "dronenoise" or r["subset"] != "M5":
            continue
        x = audio(r)
        n = len(x) // SR
        db = 10 * np.log10(np.mean(x[: n * SR].reshape(n, SR) ** 2, axis=1) + 1e-20)
        drops["flyover" if '"flyover"' in r["attributes"] else "hover"].append(db - db.max())
    for op, d in drops.items():
        if d:
            allv = np.concatenate(d)
            print(f"  {op:<8} files={len(d):>3}  quietest second per file: {pct([v.min() for v in d])}"
                  f"   seconds within 10 dB of peak: {np.mean(allv > -10) * 100:.0f}%"
                  f"   within 20 dB: {np.mean(allv > -20) * 100:.0f}%")


def spectra(rows: list[dict]) -> None:
    """Average log-mel spectrum per source, level removed, so sources can be compared by shape."""
    print("\n== Mean log-mel spectrum per source, level removed (dB, 8 groups of 8 mel bands, low to high)")
    fe = LogMel(DEFAULT)
    rng = np.random.default_rng(0)
    groups = collections.defaultdict(list)
    for r in rows:
        key = r["source"] if r["source"] not in ("halmstad", "ddl") else (
            f"halmstad/{r['kind']}" if r["source"] == "halmstad" else f"ddl/{r['subset']}")
        groups[key].append(r)
    n = DEFAULT.samples_per_window
    for key in sorted(groups):
        g = [r for r in groups[key] if int(r["n_samples"]) >= n]
        picks = [g[i] for i in rng.choice(len(g), size=min(300, len(g)), replace=False)]
        acc = []
        for r in picks:
            x = audio(r)
            start = int(rng.integers(0, len(x) - n + 1))
            m = fe.frames(x[start:start + n]).mean(axis=1)  # natural log power per mel band
            acc.append(m - m.mean())
        shape = np.mean(acc, axis=0) * (10 / np.log(10))
        print(f"  {key:<22} " + " ".join(f"{v:6.1f}" for v in shape.reshape(8, 8).mean(axis=1)))


if __name__ == "__main__":
    rows = load_manifest()
    checks = {"levels": levels, "ddl": ddl_cross_session, "dronenoise": dronenoise_profile, "spectra": spectra}
    for name in sys.argv[1:] or list(checks):
        checks[name](rows)
