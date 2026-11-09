"""Tables for car_distractor.pd (FFT 1024 at 44.1 kHz, bins 0..512).

car_alpha.txt      ISO 9613-1 air absorption per bin, dB/m (dad.propagation, 10 C, 70 % RH)
car_tyre_gain.txt  tyre/road noise shape: amplitude gain per bin, mean(g^2) = 1. Mean power
                   spectrum of the loudest 1 s (closest approach) of real dry-road FSD50K dev
                   pass-bys, smoothed over 1/3 octave, rolled off below 400 Hz (12 dB/octave).
car_hann.txt       1024-point Hann window
car_engines.txt    the engine recordings: path, title, A-weighting correction of the engine at the
                   pitch it has at 45 km/h (dB(A) minus dB), and of the tyre noise; the patch reads it
car_gw*.txt        ISO 9613-2 ground term: per-bin weights for the octave bands 125, 250, 500, 1k and
                   2k-8k ("hi"), linear in log2(f) between band centres (63 Hz has no excess term)

Run from the repo root: uv run python Pure-data+Script/CarDistractor/make_tables.py
"""
from pathlib import Path

import numpy as np
import soundfile as sf

from dad import propagation

HERE = Path(__file__).parent
STEMS = Path("data/processed/stems/distracter/car")
# single cars, dry road, fast (tyre noise dominates); wet-road and multi-vehicle clips left out
PASSBYS = ["425492", "425865", "425868", "425381"]
N, SR = 1024, 44100
f = np.arange(N // 2 + 1) * SR / N


def loudest_second(x):
    hop = SR // 10
    e = [np.sum(x[i:i + SR] ** 2) for i in range(0, len(x) - SR, hop)]
    i = int(np.argmax(e)) * hop
    return x[i:i + SR]


def psd(x):
    w = np.hanning(N)
    frames = [x[i:i + N] * w for i in range(0, len(x) - N, N // 4)]
    return np.mean([np.abs(np.fft.rfft(fr)) ** 2 for fr in frames], axis=0)


spectra = []
for i in PASSBYS:
    x, sr = sf.read(STEMS / f"fsd50k_dev_{i}.wav")
    assert sr == SR and x.ndim == 1, i
    p = psd(loudest_second(x))
    spectra.append(p / p.sum())
p = np.mean(spectra, axis=0)
sm = np.array([p[(f >= fk * 2 ** -(1 / 6)) & (f <= fk * 2 ** (1 / 6))].mean() if fk > 0 else p[0] for fk in f])
FC = 400.0                       # below: the recorded cars' engines and wind, not tyres; our engine source covers it
sm = sm * (f / FC) ** 4 / (1 + (f / FC) ** 4)                     # 2nd-order high-pass on amplitude
g = np.sqrt(sm / sm.mean())
assert abs(np.mean(g ** 2) - 1) < 1e-9

np.savetxt(HERE / "car_tyre_gain.txt", g, fmt="%.6g")
np.savetxt(HERE / "car_alpha.txt", propagation.air_absorption_db_per_m(f), fmt="%.6g")
np.savetxt(HERE / "car_hann.txt", 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N), fmt="%.6g")
peak = f[1:][np.argmax(sm[1:])]
print(f"tyre shape from {len(PASSBYS)} pass-bys: peak {peak:.0f} Hz; "
      f"gain at 125/500/1k/2k/4k/8k Hz: " + " ".join(f"{20*np.log10(g[int(fk*N/SR)]):.1f}" for fk in (125, 500, 1000, 2000, 4000, 8000)))

BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
lf = np.log2(np.maximum(f, 1.0))
W = np.array([np.interp(lf, np.log2(BANDS), np.eye(len(BANDS))[i]) for i in range(len(BANDS))])   # flat outside
for name, rows in [("125", [1]), ("250", [2]), ("500", [3]), ("1k", [4]), ("hi", [5, 6, 7])]:
    np.savetxt(HERE / f"car_gw{name}.txt", W[rows].sum(0), fmt="%.6g")
assert np.allclose(W.sum(0), 1)


# ---- engines and A-weighting corrections (dB(A) minus dB), so engine and tyres can be balanced in dB(A)
ENGINES = [("369049", "car_idle"), ("369053", "car_idle_loop"), ("387676", "car_engine_running_4"),
           ("387677", "car_engine_running_3"), ("387679", "car_engine_idle_running"), ("401552", "car_engine_outside_idle")]
PITCH_45 = 45 * 31 / 800        # gear model in the patch: 45 km/h = 4th gear, 1395 rpm; recordings taken as 800 rpm


def a_weight_db(fr):
    f2 = np.asarray(fr, float) ** 2
    ra = 12194 ** 2 * f2 ** 2 / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194 ** 2))
    return 20 * np.log10(np.maximum(ra, 1e-12)) + 2.0


def a_corr(power, freqs):
    return 10 * np.log10(np.sum(power * 10 ** (a_weight_db(freqs) / 10)) / np.sum(power))


c_tyre = a_corr(g[1:] ** 2, f[1:])
lines = []
for i, title in ENGINES:
    x, sr = sf.read(Path("data/processed/stems/distracter/idling") / f"fsd50k_dev_{i}.wav")
    P = np.abs(np.fft.rfft(x)) ** 2; fx = np.fft.rfftfreq(len(x), 1 / sr) * PITCH_45   # played back at PITCH_45
    keep = (fx > 0) & (fx < sr / 2)
    c = a_corr(P[keep], fx[keep])
    lines.append(f"../../data/processed/stems/distracter/idling/fsd50k_dev_{i}.wav {title} {c:.2f} {c_tyre:.2f}")
    print(f"{i} {title}: engine A-correction {c:.2f} dB")
(HERE / "car_engines.txt").write_text("\n".join(lines) + "\n")
print(f"tyre noise A-correction {c_tyre:.2f} dB")
