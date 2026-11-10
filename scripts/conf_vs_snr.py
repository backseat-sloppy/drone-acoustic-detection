"""Confidence against measured SNR for two detectors, from a complete-test rounds.csv:

    .venv/Scripts/python scripts/conf_vs_snr.py ROUNDS.csv OUT_DIR [--models "it9 CRNN" "it16 CRNN 600k s1"]

Reads every row once (stdlib csv: the venv has no pandas). Drone rounds are binned by their measured SNR
(`snr_db`, drone over the whole background in the drone's stretch) in 2 dB bins and split into
"war soundscape only" and "with other noises" (wind or cars on). No-drone rounds give the confidence
without a drone per background combination. Writes conf_vs_snr.png and conf_vs_snr.csv (the table view).
The error band is 1.96 standard errors over rounds; rounds share drone files and starts, so it is too narrow.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BIN = 2.0
MIN_N = 100                                   # bins with fewer rounds are not drawn
COLORS = ["#2a78d6", "#eb6834"]               # categorical slots 1 and 2 of the dataviz reference palette
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COMBOS = ["war", "wind", "cars", "war+wind", "war+cars", "wind+cars", "war+wind+cars"]


def combo(r):
    on = [n for n, c in (("war", "war soundscape_db"), ("wind", "wind (Pd)_db"), ("cars", "cars_db")) if r[c] != "off"]
    return "+".join(on) or "silence"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rounds", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--models", nargs=2, default=["it9 CRNN", "it16 CRNN 600k s1"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    acc = defaultdict(lambda: [0, 0.0, 0.0])          # (group, model, bin) -> n, sum, sum of squares
    nd = defaultdict(lambda: [0, 0.0, 0.0])           # (combo, model) -> no-drone rounds
    pair = defaultdict(dict)                          # round -> model -> (group, bin, conf), for the paired difference
    diff = defaultdict(lambda: [0, 0.0, 0.0])
    with open(a.rounds, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m = r["detector"]
            if m not in a.models:
                continue
            c = combo(r)
            if r["drone"] == "none":
                if r["mean_conf_no_drone"] != "" and c != "silence":
                    s = nd[(c, m)]; v = float(r["mean_conf_no_drone"]); s[0] += 1; s[1] += v; s[2] += v * v
                continue
            if r["snr_db"] == "" or r["mean_conf_drone"] == "" or c == "silence":
                continue
            snr = float(r["snr_db"])
            if not math.isfinite(snr):
                continue
            g = "war only" if c == "war" else ("other noises" if ("wind" in c or "cars" in c) else None)
            if g is None:
                continue
            b = math.floor(snr / BIN) * BIN
            v = float(r["mean_conf_drone"])
            s = acc[(g, m, b)]; s[0] += 1; s[1] += v; s[2] += v * v
            p = pair[r["round"]]; p[m] = (g, b, v)
            if len(p) == 2:
                (g0, b0, v0), (_, _, v1) = p[a.models[0]], p[a.models[1]]
                d = diff[(g0, b0)]; dv = v1 - v0; d[0] += 1; d[1] += dv; d[2] += dv * dv
                del pair[r["round"]]

    def stats(s):
        n, sm, sq = s
        mean = sm / n
        sd = math.sqrt(max(sq / n - mean * mean, 0.0))
        return n, 100 * mean, 100 * 1.96 * sd / math.sqrt(n)

    rows = []
    for (g, m, b), s in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        n, mean, ci = stats(s)
        rows.append(dict(kind="drone", group=g, model=m, snr_bin_lo=b, n_rounds=n, mean_conf_pct=round(mean, 2), ci95_pct=round(ci, 2)))
    for (c, m), s in nd.items():
        n, mean, ci = stats(s)
        rows.append(dict(kind="no drone", group=c, model=m, snr_bin_lo="", n_rounds=n, mean_conf_pct=round(mean, 2), ci95_pct=round(ci, 2)))
    for (g, b), s in sorted(diff.items()):
        n, mean, ci = stats(s)
        rows.append(dict(kind=f"paired diff ({a.models[1]} - {a.models[0]})", group=g, model="", snr_bin_lo=b, n_rounds=n,
                         mean_conf_pct=round(mean, 2), ci95_pct=round(ci, 2)))
    with open(a.out / "conf_vs_snr.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader(); w.writerows(rows)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    ax = [axs[0, 0], axs[1, 0], axs[1, 1], axs[0, 1]]   # other noises, war only, no drone, paired difference
    min_n = {"other noises": MIN_N, "war only": 30}
    for k, g in enumerate(["other noises", "war only"]):
        x = ax[k]
        for m, col in zip(a.models, COLORS):
            pts = sorted((b, stats(s)) for (gg, mm, b), s in acc.items() if gg == g and mm == m and s[0] >= min_n[g])
            xs = [b + BIN / 2 for b, _ in pts]; ys = [st[1] for _, st in pts]; cs = [st[2] for _, st in pts]
            x.fill_between(xs, [y - c for y, c in zip(ys, cs)], [y + c for y, c in zip(ys, cs)], color=col, alpha=0.18, lw=0)
            x.plot(xs, ys, color=col, lw=2, label=m)
        n = sum(s[0] for (gg, mm, b), s in acc.items() if gg == g and mm == a.models[0])
        x.set_title(f"Drone present, {g}: {n:,} rounds per model\n2 dB bins with >= {min_n[g]} rounds; band = 95 % CI over rounds",
                    fontsize=10, loc="left")
        x.set_xlabel("measured SNR, drone over background (dB)")
        x.set_ylabel("mean confidence in drone windows (%)")
        x.set_ylim(0, 100); x.set_xlim(-24, 72); x.grid(True, color=GRID, lw=0.8); x.set_axisbelow(True)
        x.legend(loc="lower right", frameon=False, labelcolor=INK2)
        for sp in ("top", "right"):
            x.spines[sp].set_visible(False)
    x = ax[3]
    pts = sorted((b, stats(s)) for (g, b), s in diff.items() if g == "other noises" and s[0] >= MIN_N)
    xs = [b + BIN / 2 for b, _ in pts]; ys = [st[1] for _, st in pts]; cs = [st[2] for _, st in pts]
    x.axhline(0, color=INK2, lw=1)
    x.fill_between(xs, [y - c for y, c in zip(ys, cs)], [y + c for y, c in zip(ys, cs)], color=INK2, alpha=0.18, lw=0)
    x.plot(xs, ys, color=INK, lw=2)
    x.set_title(f"Drone present, other noises: {a.models[1]} minus {a.models[0]}, same rounds\n"
                f"above 0: {a.models[1]} more confident; below 0: {a.models[0]} more confident", fontsize=10, loc="left")
    x.set_xlabel("measured SNR, drone over background (dB)")
    x.set_ylabel("difference in mean confidence (points)")
    x.set_xlim(-24, 72); x.grid(True, color=GRID, lw=0.8); x.set_axisbelow(True)
    for sp in ("top", "right"):
        x.spines[sp].set_visible(False)
    x = ax[2]
    w = 0.38
    for j, (m, col) in enumerate(zip(a.models, COLORS)):
        ys = [stats(nd[(c, m)])[1] if nd[(c, m)][0] else 0 for c in COMBOS]
        x.barh([i + (j - 0.5) * w for i in range(len(COMBOS))], ys, height=w - 0.04, color=col, label=m)
        for i, y in enumerate(ys):
            x.text(y + 1, i + (j - 0.5) * w, f"{y:.0f}", va="center", fontsize=8, color=INK2)
    x.set_yticks(range(len(COMBOS)), [f"{c}  (n={nd[(c, a.models[0])][0]})" for c in COMBOS])
    x.invert_yaxis(); x.set_xlim(0, 100)
    x.legend(loc="lower right", frameon=False, labelcolor=INK2)
    x.set_title("No drone: mean confidence by background (lower is better)\nn = no-drone rounds per model", fontsize=10, loc="left")
    x.set_xlabel("mean confidence (%)")
    x.grid(True, axis="x", color=GRID, lw=0.8); x.set_axisbelow(True)
    for sp in ("top", "right"):
        x.spines[sp].set_visible(False)
    fig.suptitle(f"Test 1 (9_11 ParameterTest): confidence against SNR, {a.models[0]} vs {a.models[1]}  |  "
                 "10 ITU ARIS steady excerpts, drone 60 dB SPL, 40 s rounds, 1 s windows", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3, w_pad=3)
    fig.savefig(a.out / "conf_vs_snr.png", dpi=130)
    print("wrote", a.out / "conf_vs_snr.png", "and conf_vs_snr.csv;", len(rows), "table rows")


if __name__ == "__main__":
    main()
