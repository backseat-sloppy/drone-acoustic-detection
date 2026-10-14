"""Quality checks on the sources added by prepare_new_data.py. Reads only.

    uv run python -I scripts/qa_new_data.py

1. Clock alignment. Each drone segment starts GUARD_S after the motors start
   by the drone's own log. If the audio clock and the log clock agree, the
   in-band level should step up at that moment. Prints the step time relative
   to the logged motor start, per flight, and saves spectrograms.
2. Lund before and after its ground truth: is a drone audible there?
   Spectrograms of the first and last minutes of each recording.
3. Level by source and state (the shortcut check of report entry 008).

Figures go to artifacts/qa/.
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, spectrogram

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from prepare_new_data import GUARD_S  # noqa: E402

SR = 16_000
OUT = REPO / "data" / "processed"
FIG = REPO / "artifacts" / "qa"
NEW = ("nightjar", "d3set", "lund", "glasgow")


def load() -> list[dict]:
    with open(OUT / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["source"] in NEW]
    for r in rows:
        r["attr"] = json.loads(r["attributes"] or "{}")
    return rows


def parent_audio(rows: list[dict]) -> np.ndarray:
    """The whole parent recording that these segment rows were cut from."""
    first = min(rows, key=lambda r: int(r["offset"]))
    last = max(rows, key=lambda r: int(r["offset"]) + int(r["n_samples"]))
    start = int(first["offset"]) - int(round(first["attr"]["parent_start_s"] * SR))
    end = int(last["offset"]) + int(last["n_samples"])
    m = np.memmap(OUT / first["file"], dtype=first["dtype"], mode="r")
    x = np.asarray(m[start:end], dtype=np.float32)
    return x / 32768 if first["dtype"] == "int16" else x


_BAND = butter(4, [150, 4000], "bandpass", fs=SR, output="sos")


def band_level(x: np.ndarray, step_s: float = 0.5) -> np.ndarray:
    y = sosfiltfilt(_BAND, x.astype(np.float64))
    n = int(step_s * SR)
    k = len(y) // n
    return 10 * np.log10(np.mean(y[: k * n].reshape(k, n) ** 2, axis=1) + 1e-20)


def save_spec(x: np.ndarray, t0: float, title: str, path: Path, marks: list[float] = ()) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, t, s = spectrogram(x, SR, nperseg=1024, noverlap=512)
    keep = f <= 4000
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.pcolormesh(t + t0, f[keep], 10 * np.log10(s[keep] + 1e-20), shading="auto", cmap="magma",
                  vmin=np.percentile(10 * np.log10(s[keep] + 1e-20), 5),
                  vmax=np.percentile(10 * np.log10(s[keep] + 1e-20), 99.5))
    for m in marks:
        ax.axvline(m, color="cyan", lw=1, ls="--")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("s")
    ax.set_ylabel("Hz")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def alignment(rows: list[dict]) -> None:
    print("\n== 1. Level step at the logged motor start (and stop)")
    print("   step = first 0.5 s bin, within +-30 s, at least 6 dB above the median of the 20 s before")
    by_parent = collections.defaultdict(list)
    for r in rows:
        if r["source"] in ("nightjar", "d3set"):
            by_parent[(r["source"], r["attr"]["session"], r["attr"]["parent"])].append(r)
    for (src, sess, parent), segs in sorted(by_parent.items()):
        segs.sort(key=lambda r: r["attr"]["parent_start_s"])
        drone = [r for r in segs if r["label"] == "1"]
        if not drone or not any(r["label"] == "0" for r in segs):
            continue
        x = parent_audio(segs)
        lev = band_level(x)
        for r in drone:
            on = r["attr"]["parent_start_s"] - GUARD_S
            off = r["attr"]["parent_start_s"] + float(r["duration_s"]) + GUARD_S
            out = []
            for name, edge, sign in (("start", on, 1), ("stop", off, -1)):
                if not (25 < edge < len(x) / SR - 25):
                    continue
                i = int(edge / 0.5)
                win = lev[i - 60:i + 60]
                if sign > 0:
                    hit = np.flatnonzero(win > np.median(lev[i - 60:i - 20]) + 6)
                    step = (hit[0] - 60) * 0.5 if len(hit) else np.nan
                else:
                    hit = np.flatnonzero(win > np.median(lev[i + 20:i + 60]) + 6)
                    step = (hit[-1] - 60) * 0.5 if len(hit) else np.nan
                out.append(f"{name} {step:+5.1f} s")
                a = max(0, int((edge - 60) * SR))
                save_spec(x[a:int((edge + 60) * SR)], a / SR, f"{src} {sess} {parent}: logged motor {name} "
                          f"at {edge:.1f} s (dashed)", FIG / f"align_{src}_{Path(parent).stem}_{sess.replace('/', '_')}"
                          f"_{name}_{int(edge)}.png".replace(" ", "_"), [edge])
            print(f"   {src:8s} {sess[:34]:34s} {Path(parent).name[:22]:22s} " + "   ".join(out), flush=True)


def lund_edges(rows: list[dict]) -> None:
    print("\n== 2. Lund: in-band level before / during / after the ground truth, dB re the flight median")
    tracks = collections.defaultdict(lambda: ([], []))
    with open(OUT / "range_track.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source"] == "lund":
                tracks[r["recording_id"]][0].append(float(r["t_s"]))
                tracks[r["recording_id"]][1].append(float(r["range_m"]))
    tracks = {k: (np.array(a), np.array(b)) for k, (a, b) in tracks.items()}
    by = collections.defaultdict(list)
    for r in rows:
        if r["source"] == "lund":
            by[r["attr"]["session"]].append(r)
    for sess, segs in sorted(by.items()):
        segs.sort(key=lambda r: r["attr"]["parent_start_s"])
        x = parent_audio(segs)
        lev = band_level(x)
        d = [r for r in segs if r["label"] == "1"][0]
        a = d["attr"]["parent_start_s"]
        b = a + float(d["duration_s"])
        med = np.median(lev[int(a / 0.5):int(b / 0.5)])
        pre = np.median(lev[: int(a / 0.5)]) - med if a > 5 else np.nan
        post = np.median(lev[int(b / 0.5):]) - med if len(lev) - b / 0.5 > 10 else np.nan
        first = np.median(lev[int(a / 0.5):int(a / 0.5) + 60]) - med
        # Clock check: the level should follow -20 log10(range). Find the time
        # shift of the ground truth that best matches the audio.
        t, rng = tracks[d["recording_id"]]
        tt = np.arange(0, float(d["duration_s"]) - 1, 0.5)
        expect = -20 * np.log10(np.interp(tt, t, rng))
        lags = np.arange(-60, 61)          # half-second steps, +-30 s
        i0 = int(a / 0.5)
        corr = []
        for k in lags:
            seg = lev[max(0, i0 + k):i0 + k + len(tt)]
            m = min(len(seg), len(expect)) if i0 + k >= 0 else 0
            corr.append(np.corrcoef(seg[:m], expect[:m])[0, 1] if m > 60 else np.nan)
        corr = np.array(corr)
        best = int(np.nanargmax(corr))
        print(f"   {sess:20s} pre {a:5.0f} s {pre:+5.1f} dB | first 30 s of GT {first:+5.1f} dB | "
              f"post {len(x) / SR - b:5.0f} s {post:+5.1f} dB | take-off range {d['attr']['takeoff_range_m']} m | "
              f"level~range r {corr[60]:.2f} at 0 s, best {corr[best]:.2f} at {lags[best] * 0.5:+.1f} s",
              flush=True)
        s0 = max(0.0, a - 90)
        save_spec(x[int(s0 * SR):int((a + 60) * SR)], s0, f"lund {sess}: ground truth starts at {a:.1f} s",
                  FIG / f"lund_{sess}_start.png", [a])
        e1 = min(len(x) / SR, b + 90)
        save_spec(x[int((b - 60) * SR):int(e1 * SR)], b - 60, f"lund {sess}: ground truth ends at {b:.1f} s",
                  FIG / f"lund_{sess}_end.png", [b])


def levels(rows: list[dict]) -> None:
    print("\n== 3. Segment level, dBFS RMS: median (5th-95th percentile), weighted by duration")
    groups = collections.defaultdict(list)
    for r in rows:
        key = (r["source"], r["subset"] if r["source"] == "nightjar" else "", r["kind"])
        groups[key].append((float(r["rms_dbfs"]), float(r["duration_s"])))
    for key, v in sorted(groups.items()):
        lv, w = np.array(v).T
        o = np.argsort(lv)
        c = np.cumsum(w[o]) / w.sum()
        q = lambda p: lv[o][np.searchsorted(c, p)]
        print(f"   {key[0]:8s} {key[1]:6s} {key[2]:10s} {q(0.5):6.1f} ({q(0.05):6.1f} to {q(0.95):6.1f})  "
              f"{w.sum() / 60:6.1f} min in {len(v)} segments")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    rows = load()
    levels(rows)
    alignment(rows)
    lund_edges(rows)


if __name__ == "__main__":
    main()
