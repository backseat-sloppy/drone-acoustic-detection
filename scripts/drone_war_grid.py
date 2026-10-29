"""A Pure Data drone in the war track: every drone speed x every war clip x every war level.

For each war level (dB added to the war track), for each drone speed, for each war clip:

    drone   rendered by DroneSound.pd through Pure-data+Script/EnviromentTest/drone_sim.py,
            as drone_only.py does: one drone sound (--drone-seed, --pscale), flying straight at
            the microphone from speed x --flight-s metres, so it hits the mic --flight-s seconds
            after it starts. Rendered once per speed and cached in <out>/drones/.
    war     a --length s window of the 10 min war track (data/processed/war/war_track.wav,
            scripts/war_track.py), --wars windows spread evenly over the track, at
            --war-start-db + level x --war-step-db (0 dB = the track as rendered).
    wind    optional, env_sim's wind step (1 = off).

The drone starts at --onset s in the clip. Every mix is scored hop by hop by the CNN, lgbm
and comb at their frozen iteration 6 thresholds, against a sidecar that marks the drone from
its first sound reaching the mic to impact. Output in --out:

    results.csv            one row per mix and detector: recall, false-positive rate, lead time
    hops.csv               every hop's score
    grid_war<+N>dB.png     one figure per war level: rows = drone speeds, columns = war clips,
                           the three detectors in each panel

Synthetic scaffolding (CLAUDE.md section 5): never detection performance.

    uv run python scripts/drone_war_grid.py --speeds 10 --wars 10 --levels 3 --out artifacts/drone_war/run1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
PD_DIR = ROOT / "Pure-data+Script" / "EnviromentTest"
sys.path[:0] = [str(Path(__file__).parent), str(PD_DIR)]
import env_sim as es  # noqa: E402

from dad.config import DEFAULT  # noqa: E402
from dad.wavio import evaluate_file, load_scorer, rates, threshold_from_run  # noqa: E402

C = 343.0
RUN = ROOT / "artifacts/runs/it6_final.json"
MODELS = ROOT / "artifacts/models"
DETECTORS = {"CNN": str(MODELS / "it6_final_none_simdist_dscnn.pt"),
             "LightGBM": str(MODELS / "it6_final_none_simdist_lgbm.pkl"), "comb": "comb"}
COLORS = {"CNN": "#2a78d6", "LightGBM": "#eb6834", "comb": "#1baf7a"}


def comb_threshold():
    rows = json.loads(RUN.read_text(encoding="utf-8"))["results"]
    return next(float(r["deploy_threshold"]) for r in rows
                if "comb" in r["scorer"] and r.get("integrate", 1) == 1 and r["arm"] == "hold out none")


def render_drones(speeds_kmh, a, cache):
    """{km/h: (pascals, params)}. Pd writes out.wav and throttle.txt next to the patch; both are put back."""
    cache.mkdir(parents=True, exist_ok=True)
    keep = {p: p.read_bytes() for p in (PD_DIR / "out.wav", PD_DIR / "throttle.txt") if p.exists()}
    out = {}
    try:
        for v in speeds_kmh:
            wav, js = cache / f"drone_s{a.drone_seed:02d}_{v:.0f}kmh.wav", cache / f"drone_s{a.drone_seed:02d}_{v:.0f}kmh.json"
            if not wav.exists():
                from drone_sim import make_scene                      # imports render.py, which needs Pd
                ms = v / 3.6
                _, drone, fs, p, _ = make_scene(a.drone_seed, pscale=a.pscale, speed=ms, start_dist=ms * a.flight_s,
                                                elev_deg=a.elev_deg, az_deg=0.0)
                assert fs == es.FS, fs
                sf.write(wav, (drone / es.FULL).astype(np.float32), fs, subtype="FLOAT")
                js.write_text(json.dumps(p, indent=2))
                print(f"rendered drone {v:.0f} km/h")
            x, _ = sf.read(wav, dtype="float64")
            out[v] = (x * es.FULL, json.loads(js.read_text()))
    finally:
        for p, b in keep.items():
            p.write_bytes(b)
    return out


def squash(score, thr):
    """comb scores run 0 to hundreds; s/(s+thr) maps them onto 0-1 with the threshold at 0.5."""
    return score / (score + thr)


def plot_level(res, speeds, offsets, thr, gain_db, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = {k: (0.5 if k == "comb" else t) for k, t in thr.items()}
    fig, axes = plt.subplots(len(speeds), len(offsets), figsize=(2.3 * len(offsets), 1.5 * len(speeds) + 0.8),
                             sharex=True, sharey=True, squeeze=False)
    for i, v in enumerate(speeds):
        for j, off in enumerate(offsets):
            ax = axes[i, j]
            r = res[(v, off)]
            lab, t_end = r["labels"], r["CNN"].t_end
            hop = DEFAULT.hop_seconds
            for t, y in zip(t_end, lab):
                if y == 1:
                    ax.axvspan(t - hop, t, color="#000000", alpha=0.07, lw=0)
            for k in DETECTORS:
                d = r[k]
                s = squash(d.score, thr[k]) if k == "comb" else d.score
                ax.plot(d.t_end, s, color=COLORS[k], lw=1.0, label=k)
                ax.axhline(shown[k], color=COLORS[k], lw=0.7, ls="--")
            ax.set_ylim(0, 1)
            ax.grid(color="#e5e5e5", lw=0.5)
            ax.tick_params(labelsize=6, colors="#555555")
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(f"war @ {off:.0f} s", fontsize=7, color="#333333")
            if j == 0:
                ax.set_ylabel(f"{v:.0f} km/h", fontsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)", fontsize=7)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=9,
               title=f"war {gain_db:+.0f} dB  |  dashed = threshold, grey = drone in truth, comb shown as s/(s+{thr['comb']:.2f})",
               title_fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / fig.get_figheight()))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"-> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speeds", type=int, default=10, help="number of drone speeds")
    ap.add_argument("--speed-min", type=float, default=100.0, help="km/h")
    ap.add_argument("--speed-max", type=float, default=200.0, help="km/h")
    ap.add_argument("--wars", type=int, default=10, help="number of war clips (windows of the war track)")
    ap.add_argument("--levels", type=int, default=3, help="number of war levels")
    ap.add_argument("--war-start-db", type=float, default=-20.0, help="war gain of the first level, dB")
    ap.add_argument("--war-step-db", type=float, default=10.0, help="added to the war gain per level, dB")
    ap.add_argument("--wind", type=int, default=1, help="env_sim wind step 1-10 (1 = off)")
    ap.add_argument("--length", type=float, default=20.0, help="clip length, s")
    ap.add_argument("--onset", type=float, default=0.0, help="drone start in the clip, s")
    ap.add_argument("--flight-s", type=float, default=10.0, help="seconds from drone start to impact")
    ap.add_argument("--drone-seed", type=int, default=1)
    ap.add_argument("--pscale", type=float, default=0.85, help="pitch scale (drone_only.py seed 1: 0.85)")
    ap.add_argument("--elev-deg", type=float, default=40.0)
    ap.add_argument("--keep", action="store_true", help="keep the mixed WAVs")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/drone_war/run1")
    a = ap.parse_args()
    if a.onset + a.flight_s > a.length:
        ap.error("the drone must hit the mic inside the clip: onset + flight-s <= length")

    speeds = [float(v) for v in np.linspace(a.speed_min, a.speed_max, a.speeds)]
    gains = [a.war_start_db + k * a.war_step_db for k in range(a.levels)]
    src = es.load()
    n = int(a.length * es.FS)
    track_s = len(src["war"]) / es.FS
    offsets = [float(o) for o in np.round(np.linspace(0, track_s - a.length, a.wars), 1)]
    drones = render_drones(speeds, a, a.out / "drones")

    thr = {k: comb_threshold() if spec == "comb" else threshold_from_run(RUN, spec, DEFAULT.window_seconds)
           for k, spec in DETECTORS.items()}
    scorers = {k: load_scorer(spec, DEFAULT, "cpu") for k, spec in DETECTORS.items()}
    work = a.out / "mixes"
    work.mkdir(parents=True, exist_ok=True)
    war = {o: es.build(10, o, a.wind, {}, n, 0, src)[1] for o in offsets}   # layers: war (0 dB) and wind

    rows_f = open(a.out / "results.csv", "w", newline="", encoding="utf-8")
    hops_f = open(a.out / "hops.csv", "w", newline="", encoding="utf-8")
    rows = csv.DictWriter(rows_f, fieldnames=["war_gain_db", "speed_kmh", "war_offset_s", "detector", "threshold",
                                              "recall", "false_positive_rate", "first_alarm_s", "lead_s",
                                              "tp", "fn", "fp", "tn", "left_out"])
    hops = csv.writer(hops_f)
    rows.writeheader()
    hops.writerow(["war_gain_db", "speed_kmh", "war_offset_s", "detector", "t_end_s", "score", "drone", "truth"])
    for g in gains:
        res = {}
        for v in speeds:
            x, p = drones[v]
            j = int(a.onset * es.FS)
            start = a.onset + p["start_dist"] / C                  # first drone sound reaches the mic
            impact = a.onset + len(x) / es.FS
            truth = dict(drones=[dict(start_s=round(start, 4), end_s=round(impact, 4), impact_s=round(impact, 4))])
            for off in offsets:
                L = war[off]
                mix = L.get("war", 0) * 10 ** (g / 20) + L.get("wind", 0) + np.zeros(n)
                mix[j:j + len(x)] += x[: n - j]
                wav = work / f"war{g:+.0f}dB_{v:.0f}kmh_off{off:.0f}.wav"
                sf.write(wav, (mix / es.FULL).astype(np.float32), es.FS, subtype="FLOAT")
                wav.with_suffix(".json").write_text(json.dumps(truth))
                cell = {}
                for k, sc in scorers.items():
                    r = evaluate_file(wav, sc, DEFAULT, thr[k])
                    d, lab = r["decisions"], r["labels"]
                    cell[k], cell["labels"] = d, lab
                    q, lead = rates(r["counts"], r["leads"]), r["leads"][0]
                    rows.writerow(dict(war_gain_db=g, speed_kmh=round(v, 1), war_offset_s=off, detector=k,
                                       threshold=thr[k], recall=q["recall"], false_positive_rate=q["false_positive_rate"],
                                       first_alarm_s=lead["first_alarm_s"], lead_s=lead["lead_s"], **r["counts"]))
                    hops.writerows([g, round(v, 1), off, k, round(float(t), 4), float(s), bool(b), int(y)]
                                   for t, s, b, y in zip(d.t_end, d.score, d.drone, lab))
                res[(v, off)] = cell
                if not a.keep:
                    wav.unlink()
                    wav.with_suffix(".json").unlink()
            print(f"war {g:+.0f} dB, {v:.0f} km/h: {len(offsets)} clips scored")
        plot_level(res, speeds, offsets, thr, g, a.out / f"grid_war{g:+.0f}dB.png")
    rows_f.close()
    hops_f.close()


if __name__ == "__main__":
    main()
