"""Real bee recordings (FSD50K dev, not used as loops) against bee_distractor.pd.

Each real clip gets a Pd render of the same kind of bee: 2 bees, 1 m away, flying in +-1 m,
defaults otherwise. Compares spectrograms, the long-term spectrum, the wingbeat pitch track and the
level track: how much pitch and level wander in 0.25 s steps (the 'pitch glides' knob and the
flight paths set this in Pd). Writes bee_vs_real.png and prints a table. Run from the repo root:
    uv run python Pure-data+Script/BeeDistractor/compare_real.py
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
STEMS = Path("data/processed/stems/distracter/buzzing_insect")
PD = next(Path("/Applications").glob("Pd*.app/Contents/Resources/bin/pd"), None) or sys.exit("Pd not found")
SR, DUR = 44100, 8
# real clip, label, Pd bee type (bee_types.txt line). No held-out clean bumblebee clip exists in dev.
REAL = [("201613", "Bees.wav (honeybees)", 0), ("17349", "beeswarm01 (swarm)", 0)]
NF, HOP = 4096, 1024


def render(clip, out, extra=()):
    a = ["-send", "bee-batch 1", "-send", f"bee-type {clip}", "-send", f"bee-dur {DUR}", "-send", "bee-dist 1"]
    for s in extra:
        a += ["-send", s]
    subprocess.run([str(PD), "-nogui", "-nosound", "-r", str(SR), "-open", str(HERE / "bee_distractor.pd"), *a,
                    "-send", f"bee-rec symbol {out}", "-send", "bee-go bang"], timeout=60, capture_output=True)


def stft(x):
    w = np.hanning(NF)
    fr = np.array([np.abs(np.fft.rfft(x[i:i + NF] * w)) ** 2 for i in range(0, len(x) - NF, HOP)])
    return fr, np.fft.rfftfreq(NF, 1 / SR), (np.arange(len(fr)) * HOP + NF / 2) / SR


def tracks(x):
    """wingbeat f0 (harmonic sum, 80-400 Hz) and level, in 0.25 s steps"""
    n = SR // 4; f = np.fft.rfftfreq(n, 1 / SR); c = np.arange(80, 400, 0.5); f0, lv = [], []
    for k in range(0, len(x) - n, n):
        P = np.abs(np.fft.rfft(x[k:k + n] * np.hanning(n))) ** 2; L = np.log(P + 1e-20)
        f0.append(c[np.argmax([sum(np.interp(h * q, f, L) for h in range(1, 6)) for q in c])])
        lv.append(10 * np.log10(P.sum() + 1e-30))
    return np.array(f0), np.array(lv)


def wander(f0, lv):
    """median step-to-step pitch change (%) and level change (dB) between 0.25 s frames"""
    return float(np.median(np.abs(np.diff(np.log(f0)))) * 100), float(np.median(np.abs(np.diff(lv))))


tmp = Path(tempfile.mkdtemp()); rows = []
for fid, label, clip in REAL:
    x, sr = sf.read(STEMS / f"fsd50k_dev_{fid}.wav"); assert sr == SR
    out = tmp / f"pd_{fid}.wav"; render(clip, out)
    y, _ = sf.read(out); y = y[: SR * DUR]
    rows.append((fid, label, clip, x, y))

fig, ax = plt.subplots(4, len(rows), figsize=(4.6 * len(rows), 13), constrained_layout=True)
print(f"{'real clip':<26}{'f0 med Hz real/pd':>19}{'pitch step % real/pd':>22}{'level step dB real/pd':>23}")
for j, (fid, label, clip, x, y) in enumerate(rows):
    tr = {}
    for k, z in (("real", x), ("pd", y)):
        fr, f, t = stft(z); S = 10 * np.log10(fr.T + 1e-20); S -= S.max()
        a = ax[0 if k == "real" else 1, j]
        a.pcolormesh(t, f, S, vmin=-70, vmax=0, shading="auto", cmap="magma"); a.set_yscale("log"); a.set_ylim(60, 16000)
        a.set_title(f"{'REAL' if k == 'real' else 'PD'}: {label if k == 'real' else 'type ' + str(clip) + ', 2 bees, 1 m'}", fontsize=10)
        a.set_ylabel("Hz"); a.set_xlabel("s")
        p = fr.mean(0); p /= p.sum(); f0, lv = tracks(z); tr[k] = (p, f, f0, lv)
    a = ax[2, j]
    for k, c in (("real", "C0"), ("pd", "C3")):
        p, f = tr[k][0], tr[k][1]; a.semilogx(f[1:], 10 * np.log10(np.convolve(p, np.ones(5) / 5, "same")[1:] + 1e-20), c, label=k)
    a.set_xlim(60, 16000); a.set_ylim(-90, -10); a.set_title("long-term spectrum (power share)", fontsize=10); a.legend(fontsize=9); a.grid(alpha=0.3)
    a = ax[3, j]
    for k, c in (("real", "C0"), ("pd", "C3")):
        f0 = tr[k][2]; a.plot(np.arange(len(f0)) / 4, f0, c, label=k)
    a.set_ylim(80, 400); a.set_title("wingbeat pitch track, 0.25 s frames", fontsize=10); a.set_xlabel("s"); a.set_ylabel("Hz"); a.legend(fontsize=9); a.grid(alpha=0.3)
    wr, wp = wander(*tr["real"][2:]), wander(*tr["pd"][2:])
    print(f"{fid + ' ' + label[:18]:<26}{np.median(tr['real'][2]):>10.0f} /{np.median(tr['pd'][2]):>5.0f}"
          f"{wr[0]:>15.1f} /{wp[0]:>5.1f}{wr[1]:>17.1f} /{wp[1]:>5.1f}")
fig.suptitle("Real FSD50K dev bee recordings (top, not used by the patch) and bee_distractor.pd renders (cleaned loops) (2 bees, 1 m, +-1 m, defaults otherwise)", fontsize=11)
fig.savefig(HERE / "bee_vs_real.png", dpi=100)
print("wrote", HERE / "bee_vs_real.png")
