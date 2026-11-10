"""Cleaned bee loops for bee_distractor.pd: real bee recordings with the background removed.

For each recording, in a short-time Fourier transform (4096-point Hann, hop 256):
  1. the wingbeat f0 per frame (harmonic sum in the bee's f0 range, median over 9 frames);
  2. the background per frame: median power between the harmonics (more than 0.3 f0 from any
     harmonic, up to FMAX), then median over 0.5 s;
  3. a soft mask that keeps bins within the harmonic's band (+-(15 Hz + 2 % of its frequency),
     at most +-0.35 f0) with gain max(0, 1 - A*background/power);
  4. everywhere up to the recording's top frequency, also a broadband mask max(0, 1 - B*floor/power)
     with floor = the 20th percentile of that bin's power over the whole recording (the steady
     background). It keeps the bee's own sound between and above the harmonics, which the
     harmonic mask alone removed (entry 163: 40-130 dB lost above 4 kHz, "muffled").
  The mask is the larger of the two, zero elsewhere.
The masked transform is inverted with the original phase. What is left is the bee's own harmonics
with their natural wingbeat-to-wingbeat variation; wind, birds and hiss between them are gone.
Then the loop segment is cut (LOOP_S, the steadiest in level), faded 20 ms at both ends,
and scaled to -26 dBFS A-weighted RMS (= 84 dB(A) at WAV 1.0 = 110 dB SPL), so the patch's
10^((L-84)/20) gives L dB(A) at 1 m.

bee_clean<k>.wav  the cleaned loops (44.1 kHz, float)
bee_types.txt     one line per type: title median f0 (Hz)

Run from the repo root: uv run python Pure-data+Script/BeeDistractor/make_tables.py
"""
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import istft, stft

HERE = Path(__file__).parent
STEMS = Path("data/processed/stems/distracter/buzzing_insect")
SR, NF, HOP, A, B = 44100, 4096, 256, 3.0, 6.0
# id, title, f0 range (Hz), highest harmonic kept (Hz; 17350 has birds at 3-8 kHz), loop (start s, length s)
# fmax: top frequency kept; floor: use the broadband floor mask (step 4). 17350 (swarm) was dropped in
# entry 163: everything it has above 3 kHz is birds, and below that alone it sounds muffled.
TYPES = [("59451", "honeybee", (190, 260), 16000, True, None),
         ("233783", "bumblebee", (120, 180), 16000, True, (4.6, 3.7)),   # the longest stretch without its gate dropouts or clicks
         ("79931", "bumblebee_2", (140, 200), 16000, True, None)]
LOOP_S = 8.0


def a_weight_db(f):
    f2 = np.asarray(f, float) ** 2
    ra = 12194 ** 2 * f2 ** 2 / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194 ** 2))
    return 20 * np.log10(np.maximum(ra, 1e-12)) + 2.0


def a_rms(x):
    P = np.abs(np.fft.rfft(x)) ** 2; f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.sqrt(np.sum(P * 10 ** (a_weight_db(f) / 10)) * 2 / len(x) ** 2)


def clean(x, lo, hi, fmax, use_floor):
    f, t, Z = stft(x, SR, window="hann", nperseg=NF, noverlap=NF - HOP)
    P = np.abs(Z) ** 2; L = np.log(P + 1e-20)
    cand = np.arange(lo, hi, 0.5)
    idx = lambda q: np.clip(np.round(q * NF / SR).astype(int), 0, len(f) - 1)
    I = idx(np.arange(1, 9)[:, None] * cand[None, :])                  # bins of harmonics 1-8 of every candidate
    f0 = cand[np.argmax(L[I, :].mean(0), axis=0)]
    f0 = median_filter(f0, 9, mode="nearest")
    M = np.zeros_like(P); bg = np.zeros(P.shape[1])
    for k in range(P.shape[1]):
        hn = f / f0[k]; dist = np.abs(hn - np.round(hn)) * f0[k]          # Hz to the nearest harmonic
        between = (dist > 0.3 * f0[k]) & (f > f0[k] * 0.5) & (f < fmax)
        bg[k] = np.median(P[between, k])
    bg = median_filter(bg, int(0.5 * SR / HOP) | 1, mode="nearest")
    for k in range(P.shape[1]):
        hn = f / f0[k]; h = np.round(hn); dist = np.abs(hn - h) * f0[k]
        band = (h >= 1) & (f < fmax) & (dist < np.minimum(15 + 0.02 * h * f0[k], 0.35 * f0[k]))
        M[band, k] = np.maximum(0, 1 - A * bg[k] / np.maximum(P[band, k], 1e-30))
    nz = P[:, P.sum(0) > 1e-12]                                        # 233783 has gated (silent) frames
    floor = np.percentile(nz, 20, axis=1)[:, None]
    Mb = np.maximum(0, 1 - B * floor / np.maximum(P, 1e-30)) * (f < fmax)[:, None]
    if use_floor:
        M = np.maximum(M, median_filter(Mb, size=(3, 5)))               # smoothed: fewer isolated bins (musical noise)
    _, y = istft(Z * M, SR, window="hann", nperseg=NF, noverlap=NF - HOP)
    return y[:len(x)], float(np.median(f0))


def steadiest(x, n):
    w = SR // 4; lv = np.array([20 * np.log10(np.std(x[k:k + w]) + 1e-12) for k in range(0, len(x) - w, w)])
    m = n // w; sc = [lv[k:k + m].std() for k in range(len(lv) - m + 1)]
    return int(np.argmin(sc)) * w


lines = []
for k, (i, title, (lo, hi), fmax, use_floor, loop) in enumerate(TYPES):
    x, sr = sf.read(STEMS / f"fsd50k_dev_{i}.wav"); assert sr == SR and x.ndim == 1
    y, f0 = clean(x, lo, hi, fmax, use_floor)
    n = int((loop[1] if loop else LOOP_S) * SR); st = int(loop[0] * SR) if loop else steadiest(y, n)
    seg = y[st:st + n].copy(); fade = int(0.02 * SR); r = np.linspace(0, 1, fade)
    seg[:fade] *= r; seg[-fade:] *= r[::-1]
    seg *= 10 ** (-26 / 20) / a_rms(seg)
    sf.write(HERE / f"bee_clean{k}.wav", seg.astype(np.float32), SR, subtype="FLOAT")
    raw = x[st:st + n]
    print(f"{i} {title}: f0 {f0:.1f} Hz, loop {st / SR:.2f}-{(st + n) / SR:.2f} s, removed {10 * np.log10(np.sum(raw ** 2) / np.sum(y[st:st + n] ** 2)):.1f} dB "
          f"of power, peak {np.abs(seg).max():.2f}")
    lines.append(f"{title} {f0:.1f}")
(HERE / "bee_types.txt").write_text("\n".join(lines) + "\n")
