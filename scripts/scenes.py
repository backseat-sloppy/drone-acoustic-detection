"""Test scenes from the stems, with a controlled drone-to-distracter ratio, and
how much of the drone a listener could still hear (supervisor notes S1, S10).

Two commands.

    describe   S10: an acoustic description of every stem class, measured from
               the stems, so the classes can be told apart and their masking
               potential read off: docs/sound_taxonomy.md.
    run        S1: L-second scenes (drone + distracter + background + wind),
               scored by the detectors; one CSV row per scene and detector.

A scene. All stems sit at -26 dBFS active RMS (scripts/build_stems.py), so a
gain in dB is a level difference:

    drone       at --drone-dbfs (active RMS), present through the whole scene
    distracter  at drone - DDR    (DDR: drone-to-distracter ratio, --ddr)
    background  at drone - --dbr  (one background stem, fixed per scene)
    wind        at drone - --dwr  (off by default)

DDR = 0 dB means the distracter is as loud as the drone over their active
parts. Every scene has a control: the same audio without the drone, which
measures false alarms against that distracter at that level.

Masking. Broadband DDR says little about whether a drone can be heard: a
gunshot masks it for a moment, a truck in the drone's band masks it for the
whole scene. The glimpse proportion (Cooke 2006, "A glimpsing model of speech
perception in noise") counts the spectro-temporal cells where the drone
exceeds everything else by at least 3 dB, in 32 ERB-spaced bands from 100 Hz to
8 kHz and 10 ms frames, out of the cells where the drone carries energy (within
40 dB of its own loudest cell). 1 = the drone is unmasked everywhere it sounds,
0 = masked everywhere.

    uv run python scripts/scenes.py describe
    uv run python scripts/scenes.py run --out artifacts/scenes/sc1
    uv run python scripts/scenes.py analyze artifacts/scenes/sc1

The numbers are a stress test on rendered mixtures, not detection
performance (CLAUDE.md section 5). The drone stems come from the Glasgow set,
which was in iteration 6's training pool, so recall here is not recall on an
unseen drone.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import stress_test as st  # noqa: E402  (sets OMP_NUM_THREADS before torch loads)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import resample_poly, stft  # noqa: E402

from dad.wavio import Clip, Truth, counts, stream, window_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEMS = ROOT / "data" / "processed" / "stems"
FS = 44100
STEM_DBFS = -26.0


# ------------------------------------------------------------------ hearing

def erb_edges(n=32, lo=100.0, hi=8000.0):
    """n bands equally spaced on the ERB-number scale (Glasberg and Moore 1990)."""
    erb = lambda f: 21.4 * np.log10(1 + 0.00437 * f)          # noqa: E731
    inv = lambda e: (10 ** (e / 21.4) - 1) / 0.00437           # noqa: E731
    return inv(np.linspace(erb(lo), erb(hi), n + 1))


EDGES = erb_edges()


def band_power(x):
    """Power per ERB band per 10 ms frame. The 46 ms window gives every band at least 2 bins (a
    23 ms window leaves the lowest bands empty)."""
    f, _, Z = stft(x, FS, nperseg=2048, noverlap=2048 - 441)
    P = np.abs(Z) ** 2
    idx = np.digitize(f, EDGES) - 1
    return np.stack([P[idx == b].sum(axis=0) for b in range(len(EDGES) - 1)])


def glimpse_proportion(target, masker, local_snr_db=3.0, range_db=40.0):
    T, M = band_power(target), band_power(masker)
    active = T > T.max() * 10 ** (-range_db / 10)
    if not active.any():
        return float("nan")
    return float(np.mean((T > M * 10 ** (local_snr_db / 10))[active]))


# ------------------------------------------------------------------- stems

def load_stems():
    rows = list(csv.DictReader(open(STEMS / "manifest.csv", encoding="utf-8")))
    for r in rows:
        r["audio"], fs = sf.read(ROOT / r["file"])
        assert fs == FS
    return rows


def fit(x, n, loop):
    """Exactly n samples: a longer stem gives its most energetic n samples; a shorter one is looped
    (steady sounds) or centred in silence (events)."""
    if len(x) >= n:
        c = np.concatenate([[0.0], np.cumsum(x ** 2)])
        s = int(np.argmax(c[n:] - c[:-n]))
        return x[s:s + n]
    if loop:
        return np.tile(x, n // len(x) + 1)[:n]
    pad = n - len(x)
    return np.pad(x, (pad // 2, pad - pad // 2))


# --------------------------------------------------------------- describe

def describe(out_md: Path):
    rows = load_stems()
    drone = np.mean([band_power(r["audio"]).mean(axis=1) for r in rows if r["group"] == "drone"], axis=0)
    drone_db = 10 * np.log10(drone / drone.sum())
    lines = []
    for (group, cls) in dict.fromkeys((r["group"], r["cls"]) for r in rows):
        feats = []
        for r in (r for r in rows if (r["group"], r["cls"]) == (group, cls)):
            x = r["audio"]
            spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
            f = np.fft.rfftfreq(len(x), 1 / FS)
            centroid = (spec * f).sum() / spec.sum()
            low = spec[f < 300].sum() / spec.sum()
            fr = 10 * np.log10(np.mean(x[: len(x) // 441 * 441].reshape(-1, 441) ** 2, axis=1) + 1e-12)
            fr = fr[fr > fr.max() - 60]
            stationarity = np.std(fr)                          # dB spread of 10 ms frame levels
            crest = 20 * np.log10(np.abs(x).max() / np.sqrt(np.mean(x ** 2)))
            # harmonicity: height of the cepstral peak for a fundamental of 60-600 Hz (S5's cepstrogram)
            seg = fit(x, 1 << 14, loop=True)                 # its loudest 0.37 s, not a silent start
            ceps = np.fft.irfft(np.log(np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) + 1e-9))
            q = np.arange(len(ceps)) / FS
            band = (q > 1 / 600) & (q < 1 / 60)
            harmonic = ceps[band].max() / np.std(ceps[band])
            bp = band_power(x).mean(axis=1)
            overlap = float(np.sum(np.minimum(bp / bp.sum(), drone / drone.sum())))   # 1 = same spectrum
            feats.append([centroid, 100 * low, stationarity, crest, harmonic, overlap])
        m = np.mean(feats, axis=0)
        lines.append((group, cls, len(feats), *m))
    hdr = ("| Group | Class | Stems | Spectral centroid (Hz) | Energy below 300 Hz (%) | Frame-level spread (dB) "
           "| Crest factor (dB) | Cepstral peak (z) | Spectral overlap with drone |\n|---|---|---|---|---|---|---|---|---|\n")
    table = "".join(f"| {g} | {c} | {n} | {a:.0f} | {b:.0f} | {s:.1f} | {cr:.1f} | {h:.1f} | {o:.2f} |\n"
                    for g, c, n, a, b, s, cr, h, o in lines)
    erb_mid = np.sqrt(EDGES[:-1] * EDGES[1:])
    top = np.argsort(drone_db)[::-1][:6]
    out_md.write_text(TAXONOMY_HEAD + hdr + table + TAXONOMY_FOOT.format(
        bands=", ".join(f"{erb_mid[i]:.0f} Hz" for i in sorted(top))), encoding="utf-8")
    print(hdr + table)
    print(f"-> {out_md}")


TAXONOMY_HEAD = """# Sound taxonomy (supervisor note S10)

What each class of sound in the stem library (`data/processed/stems/`,
report entry 052) is like, measured from its stems with
`uv run python scripts/scenes.py describe`. The point of describing them
first: a class is defined by what it does to the signal, which is also what
decides whether it can mask or imitate a drone.

Columns (means over the class's stems):

- **Spectral centroid**: where the energy sits.
- **Energy below 300 Hz**: the low end, where wind and engines live.
- **Frame-level spread**: standard deviation of 10 ms frame levels (within
  60 dB of the loudest). Low = steady (wind, engines, drones); high = events.
- **Crest factor**: peak over RMS. High = impulsive (gunfire, explosions).
- **Cepstral peak**: height of the largest cepstral peak for a fundamental
  of 60-600 Hz, in standard deviations. High = harmonic, with a fundamental
  and its overtones, as a drone's blade-pass tone has (cepstrum: S5).
- **Spectral overlap with drone**: the shared area of the class's and the
  drones' mean spectra in 32 ERB bands (1 = same shape, 0 = disjoint). High
  overlap means it can mask the drone, and a harmonic, steady sound with
  high overlap can also be taken for one.

"""

TAXONOMY_FOOT = """
The drones' strongest ERB bands (mean of the 24 Glasgow stems, at 1 m):
{bands}.

How to read it as a classification, by the columns: steady (low spread) +
harmonic (high cepstral peak) + high overlap describes a drone look-alike,
the likely source of false alarms; steady + high overlap without harmonicity
masks without imitating; impulsive classes (high spread and crest factor)
mask only for moments. Which classes fall where is read from the table above,
and tested against the detectors in report entry 053.

Means over 1-3 stems per class are a description, not a classifier: a class
with one or two stems (traffic, idling) is described by those recordings.
"""


# --------------------------------------------------------------------- run

def run(a):
    rows = load_stems()
    n = int(a.length * FS)
    by = lambda g: [r for r in rows if r["group"] == g]           # noqa: E731
    drones = by("drone")[:: max(1, len(by("drone")) // a.drones)][: a.drones]
    distracters, backgrounds, winds = by("distracter"), by("background"), by("wind")
    cfg, scorers, thr = st.load_detectors(a.models)
    rng = np.random.default_rng(a.seed)
    g = lambda db: 10 ** ((db - STEM_DBFS) / 20)                  # noqa: E731  stem gain to reach db

    a.out.mkdir(parents=True, exist_ok=True)
    f = open(a.out / "scenes.csv", "w", newline="")
    w = None
    sid = 0
    for drone, dis, ddr in itertools.product(drones, distracters, a.ddr):
        bg = backgrounds[rng.integers(len(backgrounds))]
        wind = winds[rng.integers(len(winds))]
        d = fit(drone["audio"], n, loop=True) * g(a.drone_dbfs)
        m = (fit(dis["audio"], n, loop=False) * g(a.drone_dbfs - ddr)
             + fit(bg["audio"], n, loop=True) * g(a.drone_dbfs - a.dbr)
             + (fit(wind["audio"], n, loop=True) * g(a.drone_dbfs - a.dwr) if a.dwr is not None else 0))
        gp = glimpse_proportion(d, m)
        snr = 10 * np.log10(np.mean(d ** 2) / np.mean(m ** 2))
        for present in (True, False):
            sid += 1
            mix = d + m if present else m
            audio = resample_poly(np.clip(mix, -1, 1), 160, 441).astype(np.float32)
            truth = Truth([dict(start_s=0.0, end_s=a.length)] if present else [])
            row = dict(scene=sid, pair=(sid + 1) // 2, drone_present=present, drone=drone["id"],
                       distracter_class=dis["cls"], distracter=dis["id"], background=f"{bg['cls']}:{bg['id']}",
                       wind=f"{wind['cls']}:{wind['id']}" if a.dwr is not None else "off",
                       ddr_db=ddr, dbr_db=a.dbr, dwr_db=a.dwr, drone_dbfs=a.drone_dbfs,
                       broadband_snr_db=round(snr, 2), glimpse_proportion=round(gp, 4),
                       clipped_frac=round(float(np.mean(np.abs(mix) > 1)), 5))
            for mdl, sc in scorers.items():
                dec = stream(Clip(f"scene{sid}", audio, FS, 1, "memory"), sc, cfg, thr[mdl])
                c = counts(dec, window_labels(dec, truth, cfg.window_seconds))
                res = dict(row, model=mdl, threshold=round(thr[mdl], 4), max_score=round(float(dec.score.max()), 4),
                           alarm_share=round(float(dec.drone.mean()), 4), **c)
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(res))
                    w.writeheader()
                w.writerow(res)
        if sid % 100 == 0:
            print(f"  {sid} scenes")
    f.close()
    print(f"{sid} scenes ({sid // 2} with a drone, {sid // 2} controls) -> {a.out / 'scenes.csv'}")


def analyze(out: Path):
    """Recall by distracter class and DDR, false alarms by class, and whether recall follows the
    glimpse proportion or the broadband ratio (Spearman, over all scenes and within each class)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    rows = list(csv.DictReader(open(out / "scenes.csv")))
    models = list(dict.fromkeys(r["model"] for r in rows))
    classes = list(dict.fromkeys(r["distracter_class"] for r in rows))
    ddrs = sorted({float(r["ddr_db"]) for r in rows})
    rec = lambda r: int(r["tp"]) / (int(r["tp"]) + int(r["fn"]))  # noqa: E731
    lines = []
    for m in models:
        dr = [r for r in rows if r["model"] == m and r["drone_present"] == "True"]
        ct = [r for r in rows if r["model"] == m and r["drone_present"] == "False"]
        y = np.array([rec(r) for r in dr])
        gp = np.array([float(r["glimpse_proportion"]) for r in dr])
        ddr = np.array([float(r["ddr_db"]) for r in dr])
        rho_gp, rho_ddr = spearmanr(gp, y)[0], spearmanr(ddr, y)[0]
        within = [(spearmanr(gp[i], y[i])[0], spearmanr(ddr[i], y[i])[0])
                  for c in classes for i in [np.array([r["distracter_class"] == c for r in dr])]
                  if np.std(y[i]) > 0]
        wg, wd = np.nanmean([a for a, _ in within]), np.nanmean([b for _, b in within])
        lines.append(f"\n### {m}\n\nRecall over all {len(dr)} drone scenes: {y.mean():.3f}. "
                     f"Spearman with recall, all scenes: glimpse proportion {rho_gp:.2f}, DDR {rho_ddr:.2f}; "
                     f"mean within a class ({len(within)} classes with any spread): glimpse {wg:.2f}, DDR {wd:.2f}.\n")
        # Recall alone is inflated where the distracter itself raises alarms, so also per pair: alarm share
        # with the drone minus without it (net detection), and how well max score separates the two.
        from sklearn.metrics import roc_auc_score
        ctl = {r["pair"]: r for r in ct}
        net = np.array([float(r["alarm_share"]) - float(ctl[r["pair"]]["alarm_share"]) for r in dr])
        aucs = [roc_auc_score([1] * len(s) + [0] * len(s), [float(r["max_score"]) for r in s]
                              + [float(ctl[r["pair"]]["max_score"]) for r in s])
                for d in ddrs for s in [[r for r in dr if float(r["ddr_db"]) == d]]]
        lines.append(f"Net detection (alarm share with drone minus its control), mean {net.mean():+.2f}; Spearman "
                     f"with net: glimpse {spearmanr(gp, net)[0]:.2f}, DDR {spearmanr(ddr, net)[0]:.2f}. "
                     "Scene AUC, max score with drone against its control, by DDR "
                     + ", ".join(f"{d:+.0f}: {v:.2f}" for d, v in zip(ddrs, aucs)) + ".\n")
        lines.append("\n| Distracter | " + " | ".join(f"recall at DDR {d:+.0f}" for d in ddrs)
                     + " | glimpse prop. at DDR 0 | false-alarm share (controls), DDR "
                     + f"{ddrs[0]:+.0f} / {ddrs[-1]:+.0f} |\n|" + "---|" * (len(ddrs) + 3) + "\n")
        for c in classes:
            cell = lambda d, src=dr: [r for r in src if r["distracter_class"] == c and float(r["ddr_db"]) == d]  # noqa: E731
            recs = [np.mean([rec(r) for r in cell(d)]) for d in ddrs]
            g0 = np.mean([float(r["glimpse_proportion"]) for r in cell(0.0)]) if 0.0 in ddrs else float("nan")
            fa = [np.mean([float(r["alarm_share"]) for r in cell(d, ct)]) for d in (ddrs[0], ddrs[-1])]
            lines.append(f"| {c} | " + " | ".join(f"{v:.2f}" for v in recs) + f" | {g0:.2f} | {fa[0]:.2f} / {fa[1]:.2f} |\n")
    (out / "summary.md").write_text("".join(lines))
    print("".join(lines))

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models):
        dr = [r for r in rows if r["model"] == m and r["drone_present"] == "True"]
        sc = ax.scatter([float(r["glimpse_proportion"]) for r in dr], [rec(r) for r in dr],
                        c=[float(r["ddr_db"]) for r in dr], cmap="viridis", s=8, alpha=0.6)
        ax.set_title(m); ax.set_xlabel("glimpse proportion"); ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("window recall")
    fig.colorbar(sc, ax=axes, label="DDR (dB)")
    fig.savefig(out / "recall_vs_glimpses.png", dpi=110)
    print(f"-> {out / 'summary.md'}, {out / 'recall_vs_glimpses.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    an = sub.add_parser("analyze")
    an.add_argument("out", type=Path)
    d = sub.add_parser("describe")
    d.add_argument("--out", type=Path, default=ROOT / "docs" / "sound_taxonomy.md")
    r = sub.add_parser("run")
    r.add_argument("--ddr", nargs="+", type=float, default=[-20, -10, 0, 10, 20], help="drone-to-distracter, dB")
    r.add_argument("--dbr", type=float, default=20.0, help="drone-to-background, dB")
    r.add_argument("--dwr", type=float, default=None, help="drone-to-wind, dB (default: no wind)")
    r.add_argument("--drone-dbfs", type=float, default=-30.0, help="drone level in the scene (active RMS)")
    r.add_argument("--drones", type=int, default=4, help="how many of the 24 drone stems (evenly spaced)")
    r.add_argument("--length", type=float, default=5.0)
    r.add_argument("--models", nargs="+", default=list(st.MODELS), choices=list(st.MODELS))
    r.add_argument("--seed", type=int, default=0, help="which background (and wind) stem each scene gets")
    r.add_argument("--out", type=Path, default=ROOT / "artifacts" / "scenes" / "run")
    a = ap.parse_args()
    {"describe": lambda: describe(a.out), "run": lambda: run(a), "analyze": lambda: analyze(a.out)}[a.cmd]()


if __name__ == "__main__":
    main()
