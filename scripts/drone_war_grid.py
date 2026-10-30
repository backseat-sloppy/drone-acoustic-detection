"""A Pure Data drone in the war track: every drone speed x every war clip x every war level.

For each war level (dB added to the war track), for each drone speed, for each war clip:

    drone   rendered by DroneSound.pd through Pure-data+Script/EnviromentTest/drone_sim.py,
            as drone_only.py does: one drone sound (--drone-seed, --pscale), flying straight at
            the microphone from speed x --flight-s metres, so it hits the mic --flight-s seconds
            after it starts. Rendered once and cached in artifacts/drone_war/drones/.
    war     a --length s window of the 10 min war track (data/processed/war/war_track.wav,
            scripts/war_track.py), --wars windows spread evenly over the track, at
            --levels gains spread evenly from --war-min-db to --war-max-db (0 dB = the track as rendered).
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
import functools
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
            stem = f"drone_s{a.drone_seed:02d}_p{a.pscale:.2f}_e{a.elev_deg:g}_f{a.flight_s:g}_{v:.1f}kmh"
            wav, js = cache / f"{stem}.wav", cache / f"{stem}.json"
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




def draw_grid(fig, res, speeds, offsets, thr, gain_db, rect=(0.0, 0.0, 1.0, 0.93)):
    """The level's grid into `fig`, inside rect (left, bottom, right, top): rows = speeds, columns = war clips."""
    shown = {k: (0.5 if k == "comb" else t) for k, t in thr.items()}
    small = len(speeds) * len(offsets) > 16
    fs = 7 if small else 9
    gs = fig.add_gridspec(len(speeds), len(offsets), left=rect[0] + 0.04, bottom=rect[1] + 0.05,
                          right=rect[2] - 0.005, top=rect[3], hspace=0.35, wspace=0.08)
    axes = []
    for i, v in enumerate(speeds):
        for j, off in enumerate(offsets):
            ax = fig.add_subplot(gs[i, j], sharex=axes[0] if axes else None, sharey=axes[0] if axes else None)
            axes.append(ax)
            r = res[(v, off)]
            hop = DEFAULT.hop_seconds
            for t, y in zip(r["CNN"].t_end, r["labels"]):
                if y == 1:
                    ax.axvspan(t - hop, t, color="#000000", alpha=0.07, lw=0)
            for k in DETECTORS:
                d = r[k]
                ax.plot(d.t_end, squash(d.score, thr[k]) if k == "comb" else d.score, color=COLORS[k], lw=1.0, label=k)
                ax.axhline(shown[k], color=COLORS[k], lw=0.7, ls="--")
            ax.set_ylim(0, 1)
            ax.grid(color="#e5e5e5", lw=0.5)
            ax.tick_params(labelsize=fs - 1, colors="#555555")
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(f"war @ {off:.0f} s", fontsize=fs, color="#333333")
            if j == 0:
                ax.set_ylabel(f"{v:.0f} km/h", fontsize=fs)
            else:
                ax.tick_params(labelleft=False)
            if i == len(speeds) - 1:
                ax.set_xlabel("time (s)", fontsize=fs)
            else:
                ax.tick_params(labelbottom=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=((rect[0] + rect[2]) / 2, 1.0),
               title=f"war {gain_db:+.0f} dB  |  dashed = threshold, grey = drone in truth, "
                     f"comb shown as s/(s+{thr['comb']:.2f})", title_fontsize=8)
    return axes


def plot_level(res, speeds, offsets, thr, gain_db, path):
    from matplotlib.figure import Figure                     # no pyplot: leaves the GUI's backend alone
    fig = Figure(figsize=(2.3 * len(offsets) + 0.6, 1.5 * len(speeds) + 1.0))
    draw_grid(fig, res, speeds, offsets, thr, gain_db, rect=(0.0, 0.0, 1.0, 1 - 0.9 / fig.get_figheight()))
    fig.savefig(path, dpi=110)
    print(f"-> {path}")


def summarise(rows):
    """{(war gain, detector): recall, false-positive rate, alarms >= 3 s before impact, no alarm, mixes}"""
    out = {}
    for g, k in sorted({(r["war_gain_db"], r["detector"]) for r in rows}):
        v = [r for r in rows if r["war_gain_db"] == g and r["detector"] == k]
        tp, fn = sum(r["tp"] for r in v), sum(r["fn"] for r in v)
        fp, tn = sum(r["fp"] for r in v), sum(r["tn"] for r in v)
        leads = [r["lead_s"] for r in v if r["lead_s"] is not None]
        out[(g, k)] = dict(recall=tp / (tp + fn) if tp + fn else None,
                           false_positive_rate=fp / (fp + tn) if fp + tn else None,
                           alarm_3s=sum(x >= 3.0 for x in leads), no_alarm=len(v) - len(leads), mixes=len(v))
    return out


@functools.cache
def _sources():
    return es.load()


@functools.cache
def _detectors():
    thr = {k: comb_threshold() if spec == "comb" else threshold_from_run(RUN, spec, DEFAULT.window_seconds)
           for k, spec in DETECTORS.items()}
    return thr, {k: load_scorer(spec, DEFAULT, "cpu") for k, spec in DETECTORS.items()}


def defaults():
    return argparse.Namespace(speeds=10, speed_min=100.0, speed_max=200.0, wars=10, levels=3, war_min_db=-20.0,
                              war_max_db=0.0, wind=1, length=20.0, onset=0.0, flight_s=10.0, drone_seed=1,
                              pscale=0.85, elev_deg=40.0, keep=False, out=ROOT / "artifacts/drone_war/run1")


def run(a, progress=print, on_level=None, stop=lambda: False):
    """The whole grid for parameters `a` (see defaults()); writes results.csv and hops.csv in a.out.
    on_level(gain, res, rows so far) runs as each war level finishes. Returns speeds, offsets, gains, thr, res by gain, rows."""
    if a.onset + a.flight_s > a.length:
        raise ValueError("the drone must hit the mic inside the clip: onset + flight-s <= length")
    a.out = Path(a.out)
    speeds = [float(v) for v in np.linspace(a.speed_min, a.speed_max, a.speeds)]
    gains = [float(g) for g in np.linspace(a.war_min_db, a.war_max_db, a.levels)] if a.levels > 1 else [a.war_min_db]
    src = _sources()
    n = int(a.length * es.FS)
    offsets = [float(o) for o in np.round(np.linspace(0, len(src["war"]) / es.FS - a.length, a.wars), 1)]
    progress(f"rendering {len(speeds)} drones (cached after the first time)")
    drones = render_drones(speeds, a, ROOT / "artifacts/drone_war/drones")   # shared: names hold every drone parameter
    thr, scorers = _detectors()
    work = a.out / "mixes"
    work.mkdir(parents=True, exist_ok=True)
    war = {o: es.build(10, o, a.wind, {}, n, 0, src)[1] for o in offsets}   # layers: war (0 dB) and wind

    rows, by_gain = [], {}
    with open(a.out / "hops.csv", "w", newline="", encoding="utf-8") as hops_f:
        hops = csv.writer(hops_f)
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
                    if stop():
                        raise InterruptedError("stopped")
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
                        rows.append(dict(war_gain_db=g, speed_kmh=round(v, 1), war_offset_s=off, detector=k,
                                         threshold=thr[k], recall=q["recall"], false_positive_rate=q["false_positive_rate"],
                                         first_alarm_s=lead["first_alarm_s"], lead_s=lead["lead_s"], **r["counts"]))
                        hops.writerows([g, round(v, 1), off, k, round(float(t), 4), float(s), bool(b), int(y)]
                                       for t, s, b, y in zip(d.t_end, d.score, d.drone, lab))
                    res[(v, off)] = cell
                    if not a.keep:
                        wav.unlink()
                        wav.with_suffix(".json").unlink()
                progress(f"war {g:+.0f} dB, {v:.0f} km/h: {len(offsets)} clips scored")
            by_gain[g] = res
            if on_level:
                on_level(g, res, rows)
    if not any(work.iterdir()):
        work.rmdir()
    with open(a.out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return dict(speeds=speeds, offsets=offsets, gains=gains, thr=thr, res=by_gain, rows=rows)


def main():
    d = defaults()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speeds", type=int, default=d.speeds, help="number of drone speeds")
    ap.add_argument("--speed-min", type=float, default=d.speed_min, help="km/h")
    ap.add_argument("--speed-max", type=float, default=d.speed_max, help="km/h")
    ap.add_argument("--wars", type=int, default=d.wars, help="number of war clips (windows of the war track)")
    ap.add_argument("--levels", type=int, default=d.levels, help="number of war levels")
    ap.add_argument("--war-min-db", type=float, default=d.war_min_db, help="war gain of the quietest level, dB")
    ap.add_argument("--war-max-db", type=float, default=d.war_max_db, help="war gain of the loudest level, dB")
    ap.add_argument("--wind", type=int, default=d.wind, help="env_sim wind step 1-10 (1 = off)")
    ap.add_argument("--length", type=float, default=d.length, help="clip length, s")
    ap.add_argument("--onset", type=float, default=d.onset, help="drone start in the clip, s")
    ap.add_argument("--flight-s", type=float, default=d.flight_s, help="seconds from drone start to impact")
    ap.add_argument("--drone-seed", type=int, default=d.drone_seed)
    ap.add_argument("--pscale", type=float, default=d.pscale, help="pitch scale (drone_only.py seed 1: 0.85)")
    ap.add_argument("--elev-deg", type=float, default=d.elev_deg)
    ap.add_argument("--keep", action="store_true", help="keep the mixed WAVs")
    ap.add_argument("--out", type=Path, default=d.out)
    a = ap.parse_args()
    try:
        r = run(a)
    except ValueError as e:
        ap.error(str(e))
    for g in r["gains"]:
        plot_level(r["res"][g], r["speeds"], r["offsets"], r["thr"], g, a.out / f"grid_war{g:+.0f}dB.png")


if __name__ == "__main__":
    main()
