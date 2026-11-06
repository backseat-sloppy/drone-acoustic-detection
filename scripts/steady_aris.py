"""The steadiest drone stretches in ITU ARIS: hovering drones with a near-constant sound.

ITU ARIS is published as 1 s segments with 50 % overlap, cut from 35 annotated drone intervals of
one 3 h recording. Each interval is joined back into continuous audio (every second segment, plus
the last half-segment when needed; the overlaps are identical, entry 101). Every window of 5-15 s
(hop 0.5 s) is then measured on 125 ms frames:

    spread    L90-L10 of the frame levels in 120-5000 Hz, dB         (low = steady loudness)
    flux      mean over 48 log bands (120-5000 Hz) of each band's    (low = steady spectrum)
              std over the frames, after removing each frame's level
    line      std, in semitones, of the frequency (120-1500 Hz) that  (low = constant rotor speed)
              stands highest above the background spectrum
    tonal     the window's mean spectrum over the background, dB,    (rotor lines stand out)
              mean of the 10 highest bins in 120-3000 Hz
    snr       median frame level in 120-5000 Hz over the background  (the drone dominates)
              spectrum's level there, dB

The background spectrum is the median per bin over all frames of the 1000 background segments.
Measuring against it matters: the site has a steady machine tone near 178 Hz and rumble below
~80 Hz, and a faint drone over them otherwise passes for the steadiest "drone" of all (the first
two versions of this script did exactly that; report entry 137). Windows need snr >= 10 dB and
tonal >= 20 dB (gates). Score = z(spread) + z(flux) + z(line), z over all 10 s windows; lower is
steadier. Per interval the best 10 s window is kept, then grown towards 15 s
while the score stays within 0.1 of it (intervals of 5-10 s use their best window of any length).
The 10 intervals with the best score are written, one excerpt each, unchanged samples
(22.05 kHz PCM16), to OUT/Steady_01.wav ... and OUT/picks.json with where each came from.

    uv run python scripts/steady_aris.py [--out data/extracted/steady_peek] [--n 10] [--min-s 5]

The complete test takes real drone files of 5-10 s (complete_test_engine.MIN_DRONE_S, DRONE_S).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
ARIS = ROOT / "data/extracted/ITU_ARIS_Lab/itu-aris-lab-acoustic-drone-dataset"
FS = 22050
HOP = int(0.125 * FS)                      # frame hop, 125 ms
NFFT = 4096                                # ~5.4 Hz bins: rotor lines resolve
STEP = 4                                   # window hop in frames (0.5 s)
LO = 120                                   # Hz: the background is wind/traffic rumble below ~80 Hz
EDGES = np.geomspace(LO, 5000, 49)
TONAL_MIN = 20.0                           # dB above the background spectrum: rotor lines stand out (gate)
SNR_MIN = 10.0                             # dB, drone band over the background: the drone dominates (gate)


def intervals():
    g = json.loads((ARIS / "groups.json").read_text(encoding="utf-8"))
    segs = {}
    for s in g["segments"]:
        if s["label"] == "drone":
            segs.setdefault(s["group"], []).append(s["file"])
    out = {}
    for grp, files in segs.items():
        files.sort()
        x = [sf.read(ARIS / f, dtype="int16")[0] for f in files]
        half = len(x[0]) // 2
        parts = x[0::2] + ([x[-1][half:]] if len(x) % 2 == 0 else [])
        out[grp] = (np.concatenate(parts), files)
    return out


F = np.fft.rfftfreq(NFFT, 1 / FS)
BAND = (F >= LO) & (F <= 5000)
LINE = (F >= LO) & (F <= 1500)


def stft(x):
    pad = np.pad(np.asarray(x, np.float64), (NFFT // 2, NFFT // 2))
    n = 1 + (len(pad) - NFFT) // HOP
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    return np.abs(np.fft.rfft(pad[idx] * np.hanning(NFFT), axis=1)) ** 2


def background_spectrum():
    """Median power per bin over every frame of the 1000 background segments: the site's own sound,
    including its steady machine tone near 178 Hz."""
    fr = [stft(sf.read(p, dtype="float64")[0]) for p in sorted((ARIS / "audio/background").glob("*.wav"))]
    return np.median(np.concatenate(fr), 0) + 1e-20


def frames(x, bg):
    """Per 125 ms frame: drone-band level dB, log band shape (48), strongest line above the background
    (Hz), drone-band SNR over the background dB; and the excess spectrum dB."""
    p = stft(x.astype(np.float64) / 32768) + 1e-20
    rms = 10 * np.log10(p[:, BAND].sum(1))
    bands = np.stack([p[:, (F >= lo) & (F < hi)].sum(1) for lo, hi in zip(EDGES[:-1], EDGES[1:])], 1)
    lb = 10 * np.log10(bands)
    lb -= lb.mean(1, keepdims=True)
    ex = 10 * np.log10(p / bg)
    line = F[LINE][np.argmax(ex[:, LINE], 1)]
    snr = rms - 10 * np.log10(bg[BAND].sum())
    return rms, lb, line, snr, ex


def measure(x, bg, lengths):
    rms, lb, line, snr, ex = frames(x, bg)
    t = (F >= LO) & (F <= 3000)
    rows = []
    for L in lengths:
        w = int(L / 0.125)
        for s in range(0, len(rms) - w + 1, STEP):
            e = slice(s, s + w)
            spec = ex[e][:, t].mean(0)
            rows.append(dict(
                start_s=s * 0.125, length_s=float(L),
                spread=float(np.percentile(rms[e], 90) - np.percentile(rms[e], 10)),
                flux=float(lb[e].std(0).mean()),
                line=float(np.std(12 * np.log2(line[e] / np.median(line[e])))),
                line_hz=float(np.median(line[e])),
                tonal=float(np.sort(spec)[-10:].mean()),
                snr=float(np.median(snr[e]))))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "data/extracted/steady_peek")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--min-s", type=int, default=5,
                    help="shortest excerpt, s (complete_test_engine takes real drone files of 5-10 s)")
    a = ap.parse_args()

    bg = background_spectrum()
    ints = intervals()
    cand = {}
    for grp, (x, files) in ints.items():
        dur = len(x) / FS
        if dur < a.min_s:
            continue
        lengths = [L for L in range(a.min_s, 16) if L <= dur]
        cand[grp] = [r for r in measure(x, bg, lengths) if r["snr"] >= SNR_MIN and r["tonal"] >= TONAL_MIN]

    ten = [r for rs in cand.values() for r in rs if r["length_s"] == 10]
    keys = ("spread", "flux", "line")
    mu = {k: np.mean([r[k] for r in ten]) for k in keys}
    sd = {k: np.std([r[k] for r in ten]) for k in keys}

    def score(r):
        return sum((r[k] - mu[k]) / sd[k] for k in keys)

    best = []
    for grp, rs in cand.items():
        if not rs:
            continue
        for r in rs:
            r["score"] = score(r)
        r10 = [r for r in rs if r["length_s"] == 10]
        if r10:
            b = min(r10, key=lambda r: r["score"])
            grown = [r for r in rs if r["length_s"] > 10 and r["start_s"] <= b["start_s"]
                     and r["start_s"] + r["length_s"] >= b["start_s"] + 10 and r["score"] <= b["score"] + 0.1]
            if grown:
                b = max(grown, key=lambda r: (r["length_s"], -r["score"]))
        else:
            b = min(rs, key=lambda r: r["score"])
        best.append(dict(group=grp, **b))
    best.sort(key=lambda r: r["score"])
    picks = best[: a.n]

    a.out.mkdir(parents=True, exist_ok=True)
    meta = dict(source="ITU ARIS Lab acoustic drone dataset v1.1.0 (Zenodo 22980617), CC BY 4.0",
                method=__doc__, picks=[])
    for i, r in enumerate(picks, 1):
        x, files = ints[r["group"]]
        s, n = int(r["start_s"] * FS), int(r["length_s"] * FS)
        name = f"Steady_{i:02d}.wav"
        sf.write(a.out / name, x[s:s + n], FS, subtype="PCM_16")
        meta["picks"].append(dict(file=name, group=r["group"], first_segment=files[0], last_segment=files[-1],
                                  interval_s=round(len(x) / FS, 1),
                                  **{k: round(v, 3) for k, v in r.items() if k != "group"}))
        print(f"{name}  {r['group']:<20} start {r['start_s']:6.1f} s  {r['length_s']:4.0f} s  "
              f"score {r['score']:6.2f}  spread {r['spread']:4.1f} dB  flux {r['flux']:4.2f}  "
              f"line {r['line']:5.2f} st @ {r['line_hz']:4.0f} Hz  tonal {r['tonal']:4.1f} dB  snr {r['snr']:4.1f} dB")
    meta["all_intervals"] = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in best]
    (a.out / "picks.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
