"""A Pure Data drone in the war track: every drone speed x every war clip x every war level.

For each war level (dB added to the war track), for each drone speed, for each war clip:

    drone   rendered by DroneSound.pd through Pure-data+Script/EnviromentTest/drone_sim.py,
            as drone_only.py does: one drone sound (--drone-seed, --pscale), flying straight at
            the microphone from speed x --flight-s metres, so it hits the mic --flight-s seconds
            after it starts. Rendered once and cached in artifacts/drone_war/drones/.
    war     war-shaped noise (--war-source shaped, the default; scripts/war_shaped_noise.py):
            steady noise with the war's long-term spectrum, seed 1000 + clip index, scaled to the
            RMS of the track window below; or (--war-source track) that window itself:
            a --length s window of the 10 min war track (data/processed/war/war_track.wav,
            scripts/war_track.py), --wars windows spread evenly over the track, at
            --levels gains spread evenly from --war-min-db to --war-max-db (0 dB = the track as rendered);
            --no-war adds a column without war (drone only); --no-drone adds a row without a
            drone (war only, for false alarms). The cell with neither is silence and is skipped.
    wind    optional, env_sim's wind step (1 = off).

The drone starts at --onset s in the clip. Every mix is scored hop by hop by the CNN
and comb at their frozen iteration 6 thresholds, against a sidecar that marks the drone from
its first sound reaching the mic to impact. Output in --out:

    results.csv            one row per mix and detector: recall, false-positive rate, lead time
    hops.csv               every hop's score
    grid_war<+N>dB.png     one figure per war level (grid_waroff.png with the war off): rows = drone
                           speeds, columns = war clips, both detectors in each panel

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
import war_shaped_noise as wsn  # noqa: E402

from dad.config import DEFAULT  # noqa: E402
from dad.wavio import evaluate_file, load_scorer, rates, threshold_from_run  # noqa: E402

C = 343.0
RUN = ROOT / "artifacts/runs/it6_final.json"
MODELS = ROOT / "artifacts/models"
WSN_GAIN = ROOT / "Pure-data+Script/WarShapedNoise/war_gain.txt"   # 44.1 kHz table (docs/war_shaped_noise.md)
WSN_SEED = 1000                                              # test-noise seeds start at 1000 (docs/war_shaped_noise.md)
DETECTORS = {"CNN": str(MODELS / "it6_final_none_simdist_dscnn.pt"), "comb": "comb"}   # lgbm dropped (Mads, 2026-10-30)
COLORS = {"CNN": "#2a78d6", "comb": "#1baf7a"}
OFF = float("-inf")                                          # war gain of "no war": 10 ** (OFF / 20) == 0
NO_DRONE = 0.0                                               # drone "speed" of the no-drone row (war only)


def war_tag(g):
    return "off" if g == OFF else f"{g:+g}dB"


def war_offsets(a, track_s):
    """Start of each war clip in the track; with no war levels one clip is enough (they would all be silent)."""
    return [0.0] if a.levels == 0 else [float(o) for o in np.round(np.linspace(0, track_s - a.length, a.wars), 1)]


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
            if v == NO_DRONE:
                continue
            stem = f"drone_s{a.drone_seed:02d}_p{a.pscale:.2f}_e{a.elev_deg:g}_f{a.flight_s:g}_{v:g}kmh"
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


def hop_snr(drone, noise, t_end, fs, window_s, band=(50.0, 8000.0)):
    """Per-hop SNR in dB: drone energy over noise energy in each window [t_end - window_s, t_end),
    both band-passed to the frontend's mel range. -inf with no drone, +inf with no noise."""
    from scipy.signal import butter, sosfilt
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    cum = [np.concatenate([[0.0], np.cumsum(sosfilt(sos, z) ** 2)]) for z in (drone, noise)]
    b = np.clip(np.round(np.asarray(t_end) * fs).astype(int), 0, len(drone))
    a = np.clip(b - int(round(window_s * fs)), 0, None)
    e_d, e_n = (c[b] - c[a] for c in cum)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(np.maximum(e_d, 0) / np.maximum(e_n, 0))


def squash(score, thr):
    """comb scores run 0 to hundreds; s/(s+thr) maps them onto 0-1 with the threshold at 0.5."""
    return score / (score + thr)

def draw_grid(fig, cells, rows, cols, row_label, col_label, thr, title, rect=(0.0, 0.0, 1.0, 0.93),
              row_axis="drone speed", col_axis="war clip", x="time"):
    """A grid of score panels into `fig`, inside rect (left, bottom, right, top), over time
    (x="time") or over the hop's SNR (x="snr"). Each panel's ax.cell is its (row, col) key.
    cells[(row, col)] holds each detector's decisions; row_label / col_label name one row / column.
    Returns the axes and the two headings it added (the legend is fig.legends)."""
    shown = {k: (0.5 if k == "comb" else t) for k, t in thr.items()}
    small = len(rows) * len(cols) > 16
    fs = 7 if small else 9
    left, top = rect[0] + 0.05, rect[3] - 0.04
    gs = fig.add_gridspec(len(rows), len(cols), left=left, bottom=rect[1] + 0.05, right=rect[2] - 0.005,
                          top=top, hspace=0.35, wspace=0.08)
    axes = []
    for i, rv in enumerate(rows):
        for j, cv in enumerate(cols):
            ax = fig.add_subplot(gs[i, j], sharex=axes[0] if axes else None, sharey=axes[0] if axes else None)
            axes.append(ax)
            ax.cell = (rv, cv)
            if (rv, cv) not in cells:                       # no drone and no war: silence, not run
                ax.text(0.5, 0.5, "silence:\nnot run", ha="center", va="center", fontsize=fs, color="#999999",
                        transform=ax.transAxes)
            elif x == "snr" and not np.isfinite(cells[(rv, cv)]["snr"]).any():
                ax.text(0.5, 0.5, "no SNR:\n" + ("no drone" if rv == NO_DRONE else "no noise"), ha="center",
                        va="center", fontsize=fs, color="#999999", transform=ax.transAxes)
            for k in DETECTORS if (rv, cv) in cells else ():
                d = cells[(rv, cv)][k]
                y = squash(d.score, thr[k]) if k == "comb" else d.score
                if x == "snr":
                    ok = np.isfinite(cells[(rv, cv)]["snr"])
                    ax.plot(cells[(rv, cv)]["snr"][ok], y[ok], color=COLORS[k], ls="", marker=".", ms=3, label=k)
                else:
                    ax.plot(d.t_end, y, color=COLORS[k], lw=1.0, label=k)
                ax.axhline(shown[k], color=COLORS[k], lw=0.7, ls="--")
            ax.set_ylim(0, 1)
            ax.grid(color="#e5e5e5", lw=0.5)
            ax.tick_params(labelsize=fs - 1, colors="#555555")
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(col_label(cv), fontsize=fs + 1, weight="bold", color="#333333")
            if j == 0:
                ax.set_ylabel(row_label(rv), fontsize=fs + 1, weight="bold", color="#333333")
            else:
                ax.tick_params(labelleft=False)
            if i == len(rows) - 1:
                ax.set_xlabel("SNR (dB, 50-8000 Hz, per 1 s window)" if x == "snr" else "time (s)", fontsize=fs)
            else:
                ax.tick_params(labelbottom=False)
    heads = [fig.text((left + rect[2]) / 2, top + 0.032, f"{col_axis} →", ha="center", fontsize=10, color="#555555"),
             fig.text(rect[0] + 0.005, (rect[1] + 0.05 + top) / 2, f"{row_axis} ↓", rotation=90, va="center",
                      fontsize=10, color="#555555")]
    handles = next((h for h in (ax.get_legend_handles_labels() for ax in axes) if h[0]), ([], []))  # skip blank panels
    fig.legend(*handles, loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=((rect[0] + rect[2]) / 2, 1.0),
               title=f"{title}  |  dashed = threshold, comb shown as s/(s+{thr['comb']:.2f})", title_fontsize=8)
    return axes + heads                                      # everything a caller must remove to redraw


def speed_label(v):
    return "no drone" if v == NO_DRONE else f"{v:g} km/h"


def draw_level(fig, res, speeds, offsets, thr, gain_db, rect=(0.0, 0.0, 1.0, 0.93)):
    """One war level: rows = drone speeds, columns = war clips."""
    return draw_grid(fig, res, speeds, offsets, speed_label, lambda off: f"war @ {off:g} s", thr,
                     f"war {war_tag(gain_db)}", rect, col_axis="war clip (start in the track)")


def draw_speed_war(fig, by_gain, speeds, gains, offset, thr, rect=(0.0, 0.0, 1.0, 0.93), title=None, x="time"):
    """One war clip: rows = drone speeds, columns = war levels (only the levels in by_gain)."""
    cells = {(v, g): by_gain[g][(v, offset)] for g in by_gain for v in speeds if (v, offset) in by_gain[g]}
    cols = [g for g in gains if g in by_gain]
    return draw_grid(fig, cells, speeds, cols, speed_label, lambda g: "no war" if g == OFF else f"war {g:+g} dB", thr,
                     title or f"war clip @ {offset:g} s", rect, col_axis="war noise level", x=x)


def plot_level(res, speeds, offsets, thr, gain_db, path):
    from matplotlib.figure import Figure                     # no pyplot: leaves the GUI's backend alone
    fig = Figure(figsize=(2.3 * len(offsets) + 0.9, 1.5 * len(speeds) + 1.3))
    draw_level(fig, res, speeds, offsets, thr, gain_db, rect=(0.0, 0.0, 1.0, 1 - 0.9 / fig.get_figheight()))
    fig.savefig(path, dpi=110)
    print(f"-> {path}")


def plot_speed_war(by_gain, speeds, gains, offset, thr, path, title=None, x="time"):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(2.3 * len(by_gain) + 0.9, 1.5 * len(speeds) + 1.3))
    draw_speed_war(fig, by_gain, speeds, gains, offset, thr, rect=(0.0, 0.0, 1.0, 1 - 0.9 / fig.get_figheight()),
                   title=title, x=x)
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
        with_drone = [r for r in v if r["tp"] + r["fn"] > 0]
        out[(g, k)] = dict(recall=tp / (tp + fn) if tp + fn else None,
                           false_positive_rate=fp / (fp + tn) if fp + tn else None,
                           alarm_3s=sum(x >= 3.0 for x in leads), no_alarm=len(with_drone) - len(leads), mixes=len(v))
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
                              pscale=0.85, elev_deg=40.0, keep=False, no_drone=False, no_war=False, war_source="shaped", out=ROOT / "artifacts/drone_war/run1")


def steps(lo, hi, n, what):
    """n values from lo to hi, both included and evenly spaced; n = 1 needs lo = hi (one value)."""
    if lo > hi:
        raise ValueError(f"{what}: min {lo:g} is above max {hi:g}")
    if n == 1 and lo != hi:
        raise ValueError(f"{what}: count 1 gives one value, so set min = max")
    if n > 1 and lo == hi:
        raise ValueError(f"{what}: min = max gives one value, so set the count to 1")
    return [float(v) for v in np.linspace(lo, hi, n)]


def speeds_and_gains(a):
    """The drone speeds (km/h) and war gains (dB) a run goes through; ValueError says what is wrong."""
    if a.speeds and not (0 < a.speed_min and a.speed_max <= 1000):
        raise ValueError("drone speed: above 0 and at most 1000 km/h (sound travels at 1235 km/h)")
    speeds = [NO_DRONE] * a.no_drone + (steps(a.speed_min, a.speed_max, a.speeds, "drone speed") if a.speeds else [])
    gains = [OFF] * a.no_war + (steps(a.war_min_db, a.war_max_db, a.levels, "war level") if a.levels else [])
    if not speeds:
        raise ValueError("drone: set a speed count above 0, or tick 'no drone'")
    if not gains:
        raise ValueError("war: set a level count above 0, or tick 'no war'")
    if speeds == [NO_DRONE] and gains == [OFF]:
        raise ValueError("no drone and no war only: nothing to test")
    return speeds, gains


def run(a, progress=print, on_level=None, stop=lambda: False):
    """The whole grid for parameters `a` (see defaults()); writes results.csv and hops.csv in a.out.
    on_level(gain, res, rows so far) runs as each war level finishes. Returns speeds, offsets, gains, thr, res by gain, rows."""
    if a.onset + a.flight_s > a.length:
        raise ValueError("the drone must hit the mic inside the clip: onset + flight-s <= length")
    a.out = Path(a.out)
    speeds, gains = speeds_and_gains(a)
    src = _sources()
    n = int(a.length * es.FS)
    offsets = war_offsets(a, len(src["war"]) / es.FS)
    progress(f"rendering {len(speeds)} drones (cached after the first time)")
    drones = render_drones(speeds, a, ROOT / "artifacts/drone_war/drones")   # shared: names hold every drone parameter
    thr, scorers = _detectors()
    work = a.out / "mixes"
    work.mkdir(parents=True, exist_ok=True)
    war = {o: es.build(10, o, a.wind, {}, n, 0, src)[1] for o in offsets}   # layers: war (0 dB) and wind
    if a.war_source == "shaped":
        # war-shaped noise in place of the war track: the war's long-term spectrum, no events,
        # at the RMS of the track window it replaces, so 0 dB is as loud as the track at 0 dB
        gain = np.loadtxt(WSN_GAIN)
        for k, o in enumerate(offsets):
            if "war" in war[o]:
                x = wsn.shaped_noise(gain, es.FS, a.length, WSN_SEED + k)[:n]
                war[o]["war"] = x * np.sqrt(np.mean(war[o]["war"] ** 2) / np.mean(x ** 2))

    rows, by_gain = [], {}
    with open(a.out / "hops.csv", "w", newline="", encoding="utf-8") as hops_f:
        hops = csv.writer(hops_f)
        hops.writerow(["war_gain_db", "speed_kmh", "war_offset_s", "detector", "t_end_s", "score", "drone", "truth", "snr_db"])
        for g in gains:
            res = {}
            for v in speeds:
                if v == NO_DRONE and g == OFF:
                    continue                                          # silence: nothing to score
                j = int(a.onset * es.FS)
                if v == NO_DRONE:
                    x, truth = np.zeros(0), dict(drones=[])           # every window is "no drone"
                else:
                    x, p = drones[v]
                    start = a.onset + p["start_dist"] / C              # first drone sound reaches the mic
                    impact = a.onset + len(x) / es.FS
                    truth = dict(drones=[dict(start_s=round(start, 4), end_s=round(impact, 4), impact_s=round(impact, 4))])
                for off in offsets:
                    if stop():
                        raise InterruptedError("stopped")
                    L = war[off]
                    noise = L.get("war", 0) * 10 ** (g / 20) + L.get("wind", 0) + np.zeros(n)
                    drone_part = np.zeros(n)
                    drone_part[j:j + len(x)] = x[: n - j]
                    mix = noise + drone_part
                    wav = work / f"war{war_tag(g)}_{v:g}kmh_off{off:g}.wav"
                    sf.write(wav, (mix / es.FULL).astype(np.float32), es.FS, subtype="FLOAT")
                    wav.with_suffix(".json").write_text(json.dumps(truth))
                    cell = {}
                    for k, sc in scorers.items():
                        r = evaluate_file(wav, sc, DEFAULT, thr[k])
                        d, lab = r["decisions"], r["labels"]
                        cell[k], cell["labels"] = d, lab
                        if "snr" not in cell:
                            cell["snr"] = hop_snr(drone_part, noise, d.t_end, es.FS, DEFAULT.window_seconds)
                        q = rates(r["counts"], r["leads"])
                        lead = r["leads"][0] if r["leads"] else dict(first_alarm_s=None, lead_s=None)
                        rows.append(dict(war_source=a.war_source, war_gain_db=g, speed_kmh=round(v, 1), war_offset_s=off, detector=k,
                                         threshold=thr[k], recall=q["recall"], false_positive_rate=q["false_positive_rate"],
                                         first_alarm_s=lead["first_alarm_s"], lead_s=lead["lead_s"], **r["counts"]))
                        hops.writerows([g, round(v, 1), off, k, round(float(t), 4), float(s), bool(b), int(y),
                                        round(float(q_), 2)]
                                       for t, s, b, y, q_ in zip(d.t_end, d.score, d.drone, lab, cell["snr"]))
                    res[(v, off)] = cell
                    if not a.keep:
                        wav.unlink()
                        wav.with_suffix(".json").unlink()
                progress(f"war {war_tag(g)}, {v:.0f} km/h: {len(offsets)} clips scored")
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
    ap.add_argument("--speeds", type=int, default=d.speeds, help="number of drone speeds (may be 0 with --no-drone)")
    ap.add_argument("--speed-min", type=float, default=d.speed_min, help="km/h")
    ap.add_argument("--speed-max", type=float, default=d.speed_max, help="km/h")
    ap.add_argument("--wars", type=int, default=d.wars, help="number of war clips (windows of the war track)")
    ap.add_argument("--levels", type=int, default=d.levels, help="number of war levels (may be 0 with --no-war)")
    ap.add_argument("--no-drone", action="store_true", help="add a row with no drone (war only: false alarms)")
    ap.add_argument("--no-war", action="store_true", help="add a column with no war (drone only)")
    ap.add_argument("--war-min-db", type=float, default=d.war_min_db, help="war gain of the quietest level, dB")
    ap.add_argument("--war-max-db", type=float, default=d.war_max_db, help="war gain of the loudest level, dB")
    ap.add_argument("--war-source", choices=["shaped", "track"], default=d.war_source,
                    help="shaped: war-shaped noise (stationary, the war's spectrum); track: the war track itself")
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
        plot_level(r["res"][g], r["speeds"], r["offsets"], r["thr"], g, a.out / f"grid_war{war_tag(g)}.png")


if __name__ == "__main__":
    main()
