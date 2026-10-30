"""Every combination of the environment's parameters, in memory: the full environment grid.

The axes of scripts/env_sim.py (iteration 7, no drone):

    N  war step 1-10 (1 = off, 10 = the war track as rendered, -5 dB a step)
    O  war offset: where the 20 s window starts in the 10 min war track
    W  wind step 1-10 (1 = off, 2..10 = 40..80 dB SPL)
    D  one distracter class at one distance step (2 = far .. 10 = near), or none

The default grid is 10 x 6 offsets x 10 x (1 + 11 classes x 9 steps) =
60,000 environments of --length (20) s. Each distracter spec is rendered once
with a fixed seed (--seed + its index), so a spec is the same audio
(stem, pitch, start) in every environment it appears in. Each environment is built in memory,
scored by the detectors (the same `stream` as `dad wav`) and thrown away;
nothing but the CSV is written.

There is no drone, so every alarm is a false alarm: the CSV says, for every
combination, how much of it each detector takes for a drone.

One row per environment: its id, the parameters, the mix's RMS in dB SPL and
peak in dBFS, and per detector the highest and mean score, the share of
windows at or above the frozen threshold (alarm_share), and the first alarm.

Speed. The CNN is the slow part: at a 0.125 s hop it takes about 1.1 s per
20 s environment on one core. The default hop is 0.5 s, the
hop iteration 6 was evaluated at (eval_hop in artifacts/runs/it6_final.json)
and within REQUIREMENTS (FREE, 0.05-0.5 s), and --workers processes share the
grid. The run resumes: environments already in the CSV are skipped.

    uv run python scripts/env_grid.py --out artifacts/environment/grid2
    uv run python scripts/env_grid.py --summary artifacts/environment/grid2
"""
from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import env_sim as es  # noqa: E402
import stress_test as st  # noqa: E402  (sets OMP_NUM_THREADS before torch loads)

import numpy as np  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

REF_DB = 60.0
G = {}                                      # per worker: tracks, detectors


def specs(steps):
    import distracters as dt
    return ["none"] + [f"{c}:{s}" for c in dt.CATALOGUE for s in steps if s > 1]


def init(length, hop, models, offsets, dsteps, seed):
    from dad.config import FrontendConfig
    n = int(length * es.FS)
    src = es.load()
    G["war"] = {o: es.war_window(src, o, n)[0] for o in offsets}
    G["wind"] = es.loop(src["wind"] / es.active_rms(src["wind"]) * es.P_REF * 10 ** (REF_DB / 20), n)
    G["dist"] = {}
    for i, sp in enumerate(specs(dsteps)):
        d = es.parse_distracters(sp)
        _, layers, info = es.build(1, 0, 1, d, n, seed + i, src)
        G["dist"][sp] = (sum(layers.values(), np.zeros(n)), info["distracters"])
    cfg = FrontendConfig(hop_seconds=hop)
    _, _, thr = st.load_detectors([])
    G["cfg"], G["thr"] = cfg, thr
    G["scorers"] = {m: st.load_scorer(st.MODELS[m] if m == "comb" else str(st.ROOT / st.MODELS[m]), cfg)
                    for m in models}


def score(job):
    from dad.wavio import Clip, stream
    k, (N, O, W, D) = job
    g, wdb = es.war_gain_db(N), es.step_db(W)
    mix = G["dist"][D][0].copy()
    if g is not None:
        mix += G["war"][O] * 10 ** (g / 20)
    if wdb is not None:
        mix += G["wind"] * 10 ** ((wdb - REF_DB) / 20)
    wav = mix / es.FULL
    audio = resample_poly(np.clip(wav, -1, 1), 160, 441).astype(np.float32)
    cls, step = (D.split(":") + [1])[:2] if D != "none" else ("none", 1)
    row = dict(env_id=k, N_war_step=N, N_war_gain_db="off" if g is None else g, O_offset_s=O,
               W_wind_step=W, W_wind_db="off" if wdb is None else wdb, D_class=cls, D_step=int(step),
               D_distance_m=G["dist"][D][1][0]["distance_m"] if G["dist"][D][1] else None,
               **es.describe_mix(mix))
    for m, sc in G["scorers"].items():
        d = stream(Clip(f"env{k}", audio, es.FS, 1, "memory"), sc, G["cfg"], G["thr"][m])
        hit = d.t_end[d.drone]
        row.update({f"{m}_max": round(float(d.score.max()), 4), f"{m}_mean": round(float(d.score.mean()), 4),
                    f"{m}_alarm_share": round(float(d.drone.mean()), 4),
                    f"{m}_first_alarm_s": round(float(hit[0]), 3) if len(hit) else None})
    return row


def summary(out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(out / "environments.csv")))
    models = [c[:-12] for c in rows[0] if c.endswith("_alarm_share")]
    lines = [f"{len(rows)} environments, no drone: every alarm is a false alarm.\n\n"
             "| Detector | Environments with any alarm | Mean alarm share | Environments under 5% alarm share (R3) |\n"
             "|---|---|---|---|\n"]
    for m in models:
        a = np.array([float(r[f"{m}_alarm_share"]) for r in rows])
        lines.append(f"| {m} | {np.mean(a > 0):.3f} | {a.mean():.3f} | {np.mean(a <= 0.05):.3f} |\n")
    axes_ = [("N_war_step", "war step"), ("O_offset_s", "war offset (s)"), ("W_wind_step", "wind step"),
             ("D_class", "distracter class"), ("D_step", "distracter step")]
    fig, axs = plt.subplots(len(axes_), 1, figsize=(10, 3 * len(axes_)))
    for ax, (key, title) in zip(axs, axes_):
        num = lambda v: v.replace(".", "", 1).isdigit()  # noqa: E731
        vals = sorted({r[key] for r in rows}, key=lambda v: (0, float(v), "") if num(v) else (1, 0.0, v))
        lines.append(f"\nMean alarm share by {title}, over everything else:\n\n| Detector | "
                     + " | ".join(vals) + " |\n|" + "---|" * (len(vals) + 1) + "\n")
        for m in models:
            ys = [np.mean([float(r[f"{m}_alarm_share"]) for r in rows if r[key] == v]) for v in vals]
            lines.append(f"| {m} | " + " | ".join(f"{y:.2f}" for y in ys) + " |\n")
            ax.plot(range(len(vals)), ys, marker="o", label=m)
        ax.set_xticks(range(len(vals)), vals, rotation=30 if key == "D_class" else 0, fontsize=8)
        ax.set_title(title); ax.set_ylabel("mean alarm share"); ax.grid(alpha=0.3)
    axs[0].legend()
    fig.tight_layout(); fig.savefig(out / "false_alarms_by_axis.png", dpi=110)
    (out / "summary.md").write_text("".join(lines))
    print("".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--war", nargs="+", type=int, default=es.STEPS, help="N steps 1-10 (1 = off)")
    ap.add_argument("--offsets", nargs="+", type=float, default=es.OFFSETS, help="O, s into the war track")
    ap.add_argument("--wind", nargs="+", type=int, default=es.STEPS, help="W steps 1-10 (1 = off)")
    ap.add_argument("--distracter-steps", nargs="+", type=int, default=es.STEPS[1:],
                    help="D: every class alone at each of these steps, plus none")
    ap.add_argument("--seed", type=int, default=0, help="distracter spec i uses seed + i")
    ap.add_argument("--length", type=float, default=20.0)
    ap.add_argument("--hop", type=float, default=0.5)
    ap.add_argument("--models", nargs="+", default=list(st.MODELS), choices=list(st.MODELS))
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=None, help="only the first N environments (a trial run)")
    ap.add_argument("--out", type=Path, default=st.ROOT / "artifacts" / "environment" / "grid")
    ap.add_argument("--summary", type=Path, default=None, help="summarise an existing run and stop")
    a = ap.parse_args()
    if a.summary:
        return summary(a.summary)

    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "environments.csv"
    done = {int(r["env_id"]) for r in csv.DictReader(open(path))} if path.exists() else set()
    grid = list(itertools.product(a.war, a.offsets, a.wind, specs(a.distracter_steps)))
    jobs = [(k, p) for k, p in enumerate(grid, start=1) if k not in done][: a.limit]
    total = len(grid)
    print(f"{total} environments in the grid, {len(done)} done, {len(jobs)} to go, {a.workers} workers, hop {a.hop} s")
    f = open(path, "a", newline="")
    w = None
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, init, (a.length, a.hop, a.models, a.offsets,
                                                         a.distracter_steps, a.seed)) as pool:
        for i, row in enumerate(pool.imap_unordered(score, jobs, chunksize=16), start=1):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(row))
                if not done:
                    w.writeheader()
            w.writerow(row)
            if i % 1000 == 0:
                f.flush()
                rate = i / (time.time() - t0)
                print(f"  {len(done) + i}/{total}  {rate:.1f}/s  about {(len(jobs) - i) / rate / 60:.0f} min left", flush=True)
    f.close()
    print(f"done: {path}")


if __name__ == "__main__":
    main()
