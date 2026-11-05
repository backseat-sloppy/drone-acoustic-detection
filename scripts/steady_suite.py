"""Iteration 11, step 2: score checkpoints on real drones and on steady and real noise, as the GUI does.

    uv run python scripts/steady_suite.py                                  # the GUI's models
    uv run python scripts/steady_suite.py --model "it11 CRNN=artifacts/models/x.pt,artifacts/runs/x.json"

Every file is read and scored as scripts/detect_gui.py does (dad.wavio: channel 0, 16 kHz, a 1 s
window every 0.125 s, drone when the score reaches the model's threshold frozen on validation).
Sets:
  drone      data/extracted/sealed_peek/Drone_*.wav: ten 10 s ITU ARIS drone excerpts (entry 101)
  birds      data/extracted/sealed_peek/Fugle.wav: no drone
  stems      data/processed/stems: background, distracter and wind stems, subset "eval" (unseen) or
             "dev" (possibly a training negative, entry 105); aircraft classes apart
  war        the war stems per class, with their subset, and the 600 s war track
  colour     white, pink and brown noise, 20 s at -30 dBFS RMS, seed 0: synthetic tripwires only
For the drones it reports the share of windows alarmed and the mean and median score (the
"confidence"); for every no-drone set the share of windows alarmed (false alarms).

Writes artifacts/steady_suite/<tag>.csv (one row per model and file), <tag>_drones.png (each drone
file's and the birds' confidence over time, one line per model, dashed = its threshold, as in the
GUI) and prints a summary.

Because a frozen threshold mixes separation with operating point, the summary also gives, per model,
a threshold-free view on the same windows: the AUC of the ARIS drone windows against the unseen
no-drone windows (stems eval, birds, war clips from eval, war track), and the ARIS recall at the
threshold where those no-drone windows alarm 10% and 5% (matched false alarms).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import drone_war_grid as dwg  # noqa: E402

from dad.config import DEFAULT  # noqa: E402
from dad.steady import slope_noise  # noqa: E402
from dad.wavio import Clip, Decisions, load_scorer, read_wav, stream, threshold_from_run  # noqa: E402

ROOT = dwg.ROOT
PEEK = ROOT / "data/extracted/sealed_peek"
STEMS = ROOT / "data/processed/stems"
WAR_TRACK = ROOT / "data/processed/war/war_track.wav"
AIRCRAFT = {"aircraft", "helicopter", "prop_plane", "jet"}
# iteration 12's judged set: the eval stems of the classes its FSD50K confusers cover (entry 122)
CONFUSER_EVAL = {"aircraft", "buzzing_insect", "car", "idling", "motorcycle", "power_tool", "prop_plane", "truck",
                 "helicopter", "traffic", "rain", "stream", "insects"}
COLOURS = {"white": 0.0, "pink": -3.0, "brown": -6.0}
DEFAULT_MODELS = ["it6 CNN", "it9 CNN", "it9 CRNN"]


def items():
    """(set, cls, subset, path or Clip)."""
    for p in sorted(PEEK.glob("Drone_*.wav")):
        yield "drone", "aris", "sealed_peek", p
    if (PEEK / "Fugle.wav").exists():
        yield "birds", "birds", "sealed_peek", PEEK / "Fugle.wav"
    with open(STEMS / "manifest.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["group"] == "drone":
                continue
            subset = "eval" if r["source"].endswith("eval") else "dev"
            yield ("war" if r["group"] == "war" else "stems"), r["cls"], subset, ROOT / r["file"]
    if WAR_TRACK.exists():
        yield "war", "war_track", "track", WAR_TRACK
    n = 20 * DEFAULT.sample_rate
    for name, slope in COLOURS.items():
        x = slope_noise(n, slope, np.random.default_rng(0)) * 10 ** (-30 / 20)
        yield "colour", name, "synthetic", Clip(name, x, DEFAULT.sample_rate, 1, "FLOAT")


def parse_models(specs):
    out = {}
    for s in specs:
        if "=" in s:
            name, rest = s.split("=", 1)
            ckpt, run = rest.split(",")
            out[name] = (Path(run), Path(ckpt))
        else:
            run, ckpt = dwg.MODELS[s]
            out[s] = (run, ckpt)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=None,
                    help="a name from the GUI's list, or NAME=CHECKPOINT,RUN_JSON (repeatable)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="suite")
    ap.add_argument("--smooth", type=int, default=1,
                    help="average each score with the k-1 before it (causal); the frozen thresholds were set "
                         "for single windows, so with k > 1 only the matched-false-alarm figures are fair")
    a = ap.parse_args()
    models = parse_models(a.model or DEFAULT_MODELS)
    clips = []
    for st, cls, sub, src in items():
        clip = src if isinstance(src, Clip) else read_wav(src, DEFAULT)
        clips.append((st, cls, sub, clip))
    rows, traces, pooled_scores = [], {}, {}
    for name, (run, ckpt) in models.items():
        thr = threshold_from_run(run, str(ckpt), DEFAULT.window_seconds)
        scorer = load_scorer(str(ckpt), DEFAULT, a.device)
        for st, cls, sub, clip in clips:
            d = smooth(stream(clip, scorer, DEFAULT, thr), a.smooth)
            if not len(d.score):
                continue
            if st in ("drone", "birds"):
                traces[(name, Path(clip.path).name)] = (d.t_end, d.score, thr)
            if st == "drone" or (st != "colour" and sub in ("eval", "track", "sealed_peek")):
                pooled_scores.setdefault(name, {"pos": [], "neg": []})["pos" if st == "drone" else "neg"].append(d.score)
            rows.append(dict(model=name, threshold=round(thr, 4), set=st, cls=cls, subset=sub,
                             file=Path(clip.path).name, windows=len(d.score),
                             alarm=round(float(d.drone.mean()), 4), mean_score=round(float(d.score.mean()), 4),
                             median_score=round(float(np.median(d.score)), 4)))
        print(f"scored {name} (threshold {thr:.3f})", flush=True)
    out = ROOT / "artifacts/steady_suite" / f"{a.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(traces, out.with_name(f"{a.tag}_drones.png"))
    print(summary(rows))
    print(matched(pooled_scores))
    print(f"-> {out}")


def plot(traces, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = list(dict.fromkeys(f for _, f in traces))
    models = list(dict.fromkeys(m for m, _ in traces))
    fig, axes = plt.subplots(len(files), 1, figsize=(10, 1.6 * len(files)), sharex=True, squeeze=False)
    for ax, f in zip(axes[:, 0], files):
        for i, m in enumerate(models):
            t, s, thr = traces[(m, f)]
            line, = ax.plot(t, s, lw=1.2, label=m, color=f"C{i}")
            ax.axhline(thr, ls="--", lw=0.8, color=line.get_color())
        ax.set_ylim(0, 1)
        ax.set_ylabel(f.removesuffix(".wav"), rotation=0, ha="right", va="center", fontsize=8)
    axes[0, 0].legend(fontsize=7, ncol=len(models), loc="lower right")
    axes[-1, 0].set_xlabel("time (s), decision at the window's end")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def smooth(d: Decisions, k: int) -> Decisions:
    """Causal moving average over k decisions; the first k-1 average what exists so far."""
    if k <= 1 or not len(d.score):
        return d
    n = len(d.score)
    c = np.concatenate([[0.0], np.cumsum(d.score)])
    w = np.minimum(np.arange(1, n + 1), k)
    return Decisions(d.t_end, (c[1:] - c[np.arange(1, n + 1) - w]) / w, d.threshold)


def matched(scores):
    """AUC and recall at matched false alarms on unseen real no-drone windows, per model."""
    lines = [f"{'model':<22}{'AUC drone vs unseen':>21}{'recall @10% FA':>16}{'recall @5% FA':>15}{'neg windows':>13}"]
    for m, d in scores.items():
        pos, neg = np.concatenate(d["pos"]), np.concatenate(d["neg"])
        r = np.searchsorted(np.sort(neg), pos, side="left")         # negatives strictly below each positive
        ties = np.searchsorted(np.sort(neg), pos, side="right") - r
        auc = float((r + 0.5 * ties).mean() / len(neg))
        rec = [float((pos > np.quantile(neg, 1 - fa)).mean()) for fa in (0.10, 0.05)]
        lines.append(f"{m:<22}{auc:>21.3f}{rec[0]:>16.3f}{rec[1]:>15.3f}{len(neg):>13}")
    return "\n".join(lines)


def pooled(rows, keep):
    """Window-weighted alarm share over the rows `keep` selects."""
    r = [x for x in rows if keep(x)]
    n = sum(x["windows"] for x in r)
    return sum(x["alarm"] * x["windows"] for x in r) / n if n else float("nan")


def summary(rows):
    lines = []
    cols = [
        ("drone alarm", lambda x: x["set"] == "drone"),
        ("birds", lambda x: x["set"] == "birds"),
        ("stems eval", lambda x: x["set"] == "stems" and x["subset"] == "eval" and x["cls"] not in AIRCRAFT),
        ("aircraft eval", lambda x: x["set"] == "stems" and x["subset"] == "eval" and x["cls"] in AIRCRAFT),
        ("stems dev", lambda x: x["set"] == "stems" and x["subset"] == "dev"),
        ("confuser eval", lambda x: x["set"] == "stems" and x["subset"] == "eval" and x["cls"] in CONFUSER_EVAL),
        ("war clips", lambda x: x["set"] == "war" and x["subset"] != "track"),
        ("tank", lambda x: x["cls"] == "tank"),
        ("war track", lambda x: x["subset"] == "track"),
        ("white", lambda x: x["cls"] == "white"),
        ("pink", lambda x: x["cls"] == "pink"),
        ("brown", lambda x: x["cls"] == "brown"),
    ]
    lines.append(f"{'model':<22}{'drone mean score':>17}" + "".join(f"{c:>14}" for c, _ in cols))
    for m in dict.fromkeys(x["model"] for x in rows):
        r = [x for x in rows if x["model"] == m]
        dr = [x["mean_score"] for x in r if x["set"] == "drone"]
        lines.append(f"{m:<22}{np.mean(dr) if dr else float('nan'):>17.3f}"
                     + "".join(f"{100 * pooled(r, k):>13.1f}%" for _, k in cols))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
