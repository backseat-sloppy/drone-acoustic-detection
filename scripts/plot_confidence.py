"""Confidence over time: score WAV files with one or more detectors and plot the result.

For each WAV the script reads the audio, resamples it to the frontend rate, slides the
1 s window along it hop by hop (0.125 s), turns each window into a log-mel spectrogram
and lets each detector score it. That is the same code path as `dad wav` (dad.wavio).
One panel per file: confidence on the y axis, window-end time on the x axis, one line
per detector with its frozen threshold dashed. Hops labelled drone by a `dad` sidecar
(.json next to the WAV) are shaded. With a single file, each detector gets its own panel
and y scale, since `comb` scores are not on 0-1.

    uv run python scripts/plot_confidence.py "Pure-data+Script/EnviromentTest/out.wav"
    uv run python scripts/plot_confidence.py some/folder --detector comb --csv hops.csv
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dad.config import DEFAULT, FrontendConfig
from dad.wavio import evaluate_file, load_scorer, threshold_from_run

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
DEFAULT_DETECTORS = ["artifacts/models/it6_final_none_simdist_dscnn.pt",
                     "artifacts/models/it6_final_none_simdist_lgbm.pkl"]
DEFAULT_RUN = "artifacts/runs/it6_final.json"
NAMES = {"dscnn": "CNN", "lgbm": "LightGBM", "comb": "comb"}


def label_of(spec):
    return next((v for k, v in NAMES.items() if k in Path(spec).stem), Path(spec).stem)


def comb_threshold(run):
    """The comb has no checkpoint; take the 'comb (no ML)' threshold of the arm that held nothing out."""
    rows = json.loads(Path(run).read_text(encoding="utf-8"))["results"]
    hit = [r for r in rows if "comb" in r["scorer"] and r.get("integrate", 1) == 1 and r["arm"] == "hold out none"]
    return float(hit[0]["deploy_threshold"]) if hit else None


def score_all(files, detectors, cfg, run, device):
    """{detector label: (threshold, {file name: evaluate_file result})}"""
    out = {}
    for spec in detectors:
        if not run:
            thr = None
        elif spec == "comb":
            thr = comb_threshold(run)
        else:
            thr = threshold_from_run(run, spec, cfg.window_seconds)
        scorer = load_scorer(spec, cfg, device)
        out[label_of(spec)] = (thr, {f.name: evaluate_file(f, scorer, cfg, thr) for f in files})
        print(f"{label_of(spec)}: {len(files)} files scored, threshold {thr}")
    return out


def plot(scored, out_path, cols):
    files = sorted(next(iter(scored.values()))[1])
    single = len(files) == 1   # one file: one panel per detector, each on its own y scale (comb is not 0-1)
    panels = [(files[0], [k]) for k in scored] if single else [(f, list(scored)) for f in files]
    cols = 1 if single else min(cols, len(files))
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(9 if single else max(4.2, 3.2 * cols), 2.4 * rows + 0.6),
                             sharex=True, sharey=not single, squeeze=False)
    colors = dict(zip(scored, COLORS))
    handles = {}
    for ax, (name, dets) in zip(axes.flat, panels):
        first = next(iter(scored.values()))[1][name]
        dec, lab = first["decisions"], first.get("labels")
        if lab is not None:
            hop = DEFAULT.hop_seconds
            for t, y in zip(dec.t_end, lab):
                if y == 1:
                    ax.axvspan(t - hop, t, color="#000000", alpha=0.06, lw=0)
        for label in dets:
            thr, res = scored[label]
            d = res[name]["decisions"]
            handles[label] = ax.plot(d.t_end, d.score, color=colors[label], lw=1.5, label=label)[0]
            if thr is not None:
                ax.axhline(thr, color=colors[label], lw=1, ls="--")
        ax.set_title(name.removesuffix(".wav") + (f" - {dets[0]}" if single else ""), fontsize=8, color="#333333")
        top = max(max(float(scored[k][1][name]["decisions"].score.max()), scored[k][0] or 0) for k in dets)
        if top <= 1:
            ax.set_ylim(0, 1)
        else:   # comb: its threshold sits near 2 and peaks reach hundreds, so a linear axis hides it
            ax.set_yscale("symlog", linthresh=1)
            ax.set_ylim(0, top * 1.5)
        ax.grid(color="#e5e5e5", lw=0.6)
        ax.tick_params(labelsize=7, colors="#555555")
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("score", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s, window end)", fontsize=8)
    fig.legend(list(handles.values()), list(handles), loc="upper center", ncol=len(scored), frameon=False,
               title="dashed = threshold, grey = drone in truth", title_fontsize=8, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))
    fig.savefig(out_path, dpi=130)
    print(f"{len(files)} files -> {out_path}")


def write_csv(scored, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["detector", "file", "t_end_s", "score", "drone", "truth"])
        for label, (_, res) in scored.items():
            for name, r in sorted(res.items()):
                d, lab = r["decisions"], r.get("labels")
                for i in range(len(d.score)):
                    w.writerow([label, name, round(float(d.t_end[i]), 4), float(d.score[i]),
                                "" if d.drone is None else bool(d.drone[i]), "" if lab is None else int(lab[i])])
    print(f"per-hop scores -> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("paths", nargs="+", help="WAV files, or folders searched for *.wav")
    p.add_argument("--detector", action="append",
                   help=f"repeatable: 'comb', a .pt or a .pkl (default: {', '.join(DEFAULT_DETECTORS)})")
    p.add_argument("--threshold-from", default=DEFAULT_RUN, help="run JSON with each checkpoint's frozen threshold")
    p.add_argument("--window", type=float, default=DEFAULT.window_seconds)
    p.add_argument("--hop", type=float, default=DEFAULT.hop_seconds)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--out", default="confidence.png")
    p.add_argument("--csv", default=None, help="also write every hop's score")
    a = p.parse_args()
    detectors = a.detector or DEFAULT_DETECTORS
    if len(detectors) > len(COLORS):
        p.error(f"at most {len(COLORS)} detectors per plot")
    files = []
    for q in map(Path, a.paths):
        files.extend(sorted(q.rglob("*.wav")) if q.is_dir() else [q])
    if not files:
        p.error("no WAV files found")
    cfg = FrontendConfig(window_seconds=a.window, hop_seconds=a.hop)
    scored = score_all(files, detectors, cfg, a.threshold_from, a.device)
    plot(scored, a.out, a.cols)
    if a.csv:
        write_csv(scored, a.csv)


if __name__ == "__main__":
    main()
