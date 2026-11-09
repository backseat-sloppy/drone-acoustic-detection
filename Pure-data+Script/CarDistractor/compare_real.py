"""Real FSD50K dev pass-bys against car_distractor.pd at the same speed and distance.

The speed comes from the clip title. The closest distance is estimated from the clip: for a point
source at speed v and closest distance d the level is within 3 dB of its peak for 2 d / v seconds.
Pd renders with that speed and distance, defaults otherwise, engine ENGINE. Writes car_vs_real.png
and prints a table. Run from the repo root:
    uv run python Pure-data+Script/CarDistractor/compare_real.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

HERE = Path(__file__).parent.resolve()
STEMS = Path("data/processed/stems/distracter/car")
PD = next(Path("/Applications").glob("Pd*.app/Contents/Resources/bin/pd"), None) or sys.exit("Pd not found")
SR, ENGINE = 44100, 3
REAL = [("425650", 30, "Kangoo 30 km/h"), ("425649", 40, "Kangoo 40 km/h"),
        ("425381", 70, "Audi A4 70 km/h *"), ("425505", 120, "BMW 120 km/h, wet")]
NF, HOP = 2048, 256


def a_weight_db(f):
    f2 = np.asarray(f, float) ** 2
    ra = 12194 ** 2 * f2 ** 2 / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194 ** 2))
    return 20 * np.log10(np.maximum(ra, 1e-12)) + 2.0


def stft(x):
    w = np.hanning(NF)
    fr = np.array([np.abs(np.fft.rfft(x[i:i + NF] * w)) ** 2 for i in range(0, len(x) - NF, HOP)])
    return fr, np.fft.rfftfreq(NF, 1 / SR), (np.arange(len(fr)) * HOP + NF / 2) / SR


def envelope_dba(fr, f):
    e = 10 * np.log10(fr @ 10 ** (a_weight_db(f) / 10) + 1e-20)
    k = int(0.2 * SR / HOP)                                           # 0.2 s smoothing
    return np.convolve(e, np.ones(k) / k, mode="same")


def width_3db(env, t):
    i = int(np.argmax(env)); lo, hi = i, i
    while lo > 0 and env[lo] > env[i] - 3: lo -= 1
    while hi < len(env) - 1 and env[hi] > env[i] - 3: hi += 1
    return t[hi] - t[lo], i


def render(v, d, out):
    subprocess.run([str(PD), "-nogui", "-nosound", "-r", str(SR), "-open", str(HERE / "car_distractor.pd"),
                    "-send", "car-batch 1", "-send", f"car-clip {ENGINE}", "-send", f"car-speed {v}",
                    "-send", f"car-dist {d:.2f}", "-send", "car-dur 10", "-send", f"car-rec symbol {out}",
                    "-send", "car-go bang"], timeout=60, capture_output=True)


def centre(x, i, half=5.0):                                           # 10 s around frame i, zero-padded
    c = int(i * HOP + NF / 2); n = int(half * SR)
    y = np.zeros(2 * n); a, b = max(0, c - n), min(len(x), c + n)
    y[a - (c - n): b - (c - n)] = x[a:b]
    return y


rows, tmp = [], Path(tempfile.mkdtemp())
for fid, v, label in REAL:
    x, sr = sf.read(STEMS / f"fsd50k_dev_{fid}.wav"); assert sr == SR
    fr, f, t = stft(x); env = envelope_dba(fr, f); w, ip = width_3db(env, t)
    d = float(np.clip(v / 3.6 * w / 2, 2, 50))
    out = tmp / f"pd_{fid}.wav"; render(v, d, out)
    y, _ = sf.read(out)
    fry, _, ty = stft(y); envy = envelope_dba(fry, f); wy, iy = width_3db(envy, ty)
    rows.append(dict(fid=fid, v=v, label=label, d=d, real=centre(x, ip), pd=centre(y, iy), w=w, wy=wy))

fig, ax = plt.subplots(4, len(rows), figsize=(4.2 * len(rows), 14), constrained_layout=True)
print(f"{'clip':<22}{'d est m':>8}{'-3 dB s real/pd':>17}{'centroid Hz real/pd':>21}{'<200 Hz share real/pd':>23}{'level 2 s off peak real/pd':>28}")
for j, r in enumerate(rows):
    out = {}
    for k, row in (("real", 0), ("pd", 1)):
        fr, f, t = stft(r[k]); S = 10 * np.log10(fr.T + 1e-20); S -= S.max()
        a = ax[row, j]; a.pcolormesh(t - 5, f, S, vmin=-70, vmax=0, shading="auto", cmap="magma")
        a.set_yscale("log"); a.set_ylim(30, 16000); a.set_xlim(-5, 5)
        a.set_title(f"{'REAL' if k == 'real' else 'PD'}: {r['label']}" + (f", d = {r['d']:.1f} m" if k == "pd" else ""), fontsize=10)
        a.set_ylabel("Hz"); a.set_xlabel("s from loudest moment")
        mid = len(fr) // 2; h = int(0.5 * SR / HOP)
        p = fr[mid - h: mid + h].mean(0); p /= p.sum()
        env = envelope_dba(fr, f); env -= env.max()
        out[k] = dict(p=p, env=env, t=t - 5, cen=float((p * f).sum()), low=float(p[f < 200].sum()),
                      off=float(np.mean([env[np.argmin(abs(t - 5 - s))] for s in (-2, 2)])))
    a = ax[2, j]
    for k, c in (("real", "C0"), ("pd", "C3")):
        sm = np.convolve(out[k]["p"], np.ones(9) / 9, mode="same")
        a.semilogx(f[1:], 10 * np.log10(sm[1:] + 1e-20), c, label=k)
    a.set_xlim(30, 16000); a.set_ylim(-80, -10); a.set_title("spectrum, 1 s at the loudest moment (power share)", fontsize=10)
    a.set_xlabel("Hz"); a.set_ylabel("dB"); a.legend(fontsize=9); a.grid(alpha=0.3)
    a = ax[3, j]
    for k, c in (("real", "C0"), ("pd", "C3")):
        a.plot(out[k]["t"], out[k]["env"], c, label=k)
    a.set_ylim(-35, 2); a.set_xlim(-5, 5); a.set_title("A-weighted level, 0 dB = peak", fontsize=10)
    a.set_xlabel("s from loudest moment"); a.set_ylabel("dB"); a.legend(fontsize=9); a.grid(alpha=0.3)
    print(f"{r['fid'] + ' ' + str(r['v']) + ' km/h':<22}{r['d']:>8.1f}{r['w']:>9.2f} /{r['wy']:>5.2f}"
          f"{out['real']['cen']:>12.0f} /{out['pd']['cen']:>6.0f}{out['real']['low']:>15.2f} /{out['pd']['low']:>5.2f}"
          f"{out['real']['off']:>19.1f} /{out['pd']['off']:>6.1f}")
fig.suptitle(f"Real FSD50K dev pass-bys (top) and car_distractor.pd at the same speed and estimated distance (engine {ENGINE}, defaults). "
             "* 425381 is one of the four clips the tyre spectrum was averaged from.", fontsize=11)
fig.savefig(HERE / "car_vs_real.png", dpi=110)
print("wrote", HERE / "car_vs_real.png")
