"""The test engine behind scripts/Complete_Test_Gui.py: builds every test round, mixes it, scores it.

A test is a grid:

    starts       the drone starts at 0, step, 2 step, ... s into a soundscape of `length` s, as long as
                 the whole 10 s drone fits: length - 10 + 1 starts at a 1 s step (12 s -> 3, 20 s -> 11)
    drones       real drone files (their first 10 s) or synthetic Pure Data drones (one per speed,
                 rendered by drone_war_grid.render_drones, 10 s from start to impact)
    drone levels dB SPL, the RMS of the drone's 10 s; or "as rendered" for synthetic drones
    backgrounds  every combination of the enabled layers' levels (dB SPL each)

and optionally one no-drone round per background, for false alarms. Rounds = starts x drones x drone
levels x backgrounds (+ backgrounds).

Levels are real levels, not SNRs. Every WAV in this project has full scale = 110 dB SPL
(env_sim.FULL_SCALE_DB), so a layer at L dB SPL has RMS 10 ** ((L - 110) / 20) in WAV units. A
background layer's level is its long-term level, the RMS over the whole source (all 600 s of the war
track, the whole wind stem): a calm stretch of the war track stays calm and an explosion stays loud.

Each round is scored by every chosen detector hop by hop (dad.wavio.stream: 1 s window, 0.125 s
hop). Windows that lie inside the drone's sound are drone, windows that lie wholly outside are no
drone, and windows that straddle its start or end are left out, as in dad.wavio.window_labels
(requirement R2's reading, entry 039). Per round and detector: recall, false-positive rate,
precision, first alarm (from a window at least half drone) and its delay after the drone starts,
lead before impact (synthetic), mean confidence in drone and in no-drone windows.

Mixes are not stored: a round is rebuilt from its parameters (rebuild()), so 1000+ rounds cost only
the CSV and the per-hop scores. Synthetic scaffolding (CLAUDE.md section 5): numbers from these mixes
are not detection performance.
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
import drone_war_grid as dwg  # noqa: E402

from dad.config import DEFAULT  # noqa: E402
from dad.wavio import Clip, lead_times, load_scorer, read_wav, stream, window_labels, Truth  # noqa: E402

SR = DEFAULT.sample_rate
FULL_SCALE_DB = dwg.es.FULL_SCALE_DB                         # 1.0 in a WAV = 110 dB SPL
DRONE_S = 10.0                                               # every drone sound is 10 s
OUT_ROOT = dwg.ROOT / "artifacts/complete_test"
WAR_TRACK = dwg.es.WAR / "war_track.wav"
WIND_STEM = dwg.es.STEMS / dwg.es.WIND
WSN_LOOP_S = 60.0


def rms_for(db_spl: float) -> float:
    return 10 ** ((db_spl - FULL_SCALE_DB) / 20)


def db_spl(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0
    return FULL_SCALE_DB + 20 * np.log10(r) if r > 0 else float("-inf")


# ------------------------------------------------------------------ background layers

@dataclass
class Layer:
    """A background source: `load()` gives its whole signal at 16 kHz (WAV units). The level of a
    round sets the RMS of that whole signal; a round takes `length` s of it from the batch's start."""
    name: str
    note: str
    available: bool
    load: object = None                                      # () -> np.ndarray, called once
    _x: np.ndarray | None = None
    _rms: float = 0.0

    def signal(self) -> tuple[np.ndarray, float]:
        if self._x is None:
            self._x = np.asarray(self.load(), dtype=np.float64)
            self._rms = float(np.sqrt(np.mean(self._x ** 2)))
        return self._x, self._rms

    def window(self, n: int, start_s: float, level_db: float) -> np.ndarray:
        x, r = self.signal()
        i = int(round(start_s * SR)) % len(x)
        return np.take(x, np.arange(i, i + n), mode="wrap") * (rms_for(level_db) / r)


def _war_track():
    x, sr = sf.read(WAR_TRACK, dtype="float64")
    return resample_poly(x, SR, sr)


def _wsn():
    x = dwg.wsn.shaped_noise(np.loadtxt(dwg.WSN_GAIN), dwg.es.FS, WSN_LOOP_S, dwg.WSN_SEED)
    return resample_poly(x, SR, dwg.es.FS)


def _wind():
    x, sr = sf.read(WIND_STEM, dtype="float64", always_2d=True)
    return resample_poly(x[:, 0], SR, sr)


def make_layers() -> dict[str, Layer]:
    """Every background layer the GUI lists, available or not (cars, helicopter, insects: not yet)."""
    return {
        "war soundscape": Layer("war soundscape", "the 600 s war track (scripts/war_track.py), calm, normal and "
                                "intense phases", WAR_TRACK.exists(), _war_track),
        "war-shaped noise": Layer("war-shaped noise", "steady noise with the war's long-term spectrum, 60 s loop",
                                  dwg.WSN_GAIN.exists(), _wsn),
        "wind": Layer("wind", "wind on the capsule (FSD50K stem), looped" if WIND_STEM.exists() else
                      "stem missing on this machine: run scripts/build_stems.py", WIND_STEM.exists(), _wind),
        "cars": Layer("cars", "not set up yet", False),
        "helicopter": Layer("helicopter", "not set up yet", False),
        "insects": Layer("insects", "not set up yet", False),
    }


# ------------------------------------------------------------------ the test plan

@dataclass
class Plan:
    """Everything a test needs, as the GUI sets it. `drones` are file paths (real) or speeds in
    km/h (synthetic); `drone_levels` None means "as rendered" (synthetic only)."""
    kind: str                                                # "real" or "synthetic"
    length_s: float
    drones: list
    drone_levels: list | None
    layers: dict                                             # layer name -> list of dB SPL (enabled only)
    detectors: list                                          # model names (dwg.MODELS) and/or "comb"
    step_s: float = 1.0
    no_drone_rounds: bool = False
    soundscape_start_s: float | None = None                  # None: drawn from seed 1000 per batch
    calibrate_fa: float | None = None                        # site calibration (entry 148): thresholds at this FA
    calibrate_s: float = 120.0                               # on this many seconds of each background, no drone
    synth: dict = field(default_factory=lambda: dict(drone_seed=1, pscale=0.85, elev_deg=40.0))

    def starts(self) -> list[float]:
        last = self.length_s - DRONE_S
        if last < 0:
            return []
        return [float(round(s, 6)) for s in np.arange(0, last + 1e-9, self.step_s)]

    def backgrounds(self) -> list[dict]:
        names = list(self.layers)
        return [dict(zip(names, combo)) for combo in itertools.product(*(self.layers[n] for n in names))]

    def rounds(self) -> list[dict]:
        """Every round, in the order they run: drone rounds, then the no-drone rounds."""
        out = []
        levels = self.drone_levels if self.drone_levels is not None else [None]
        for bg in self.backgrounds():
            for d in self.drones:
                for lv in levels:
                    for s in self.starts():
                        out.append(dict(drone=d, drone_level_db=lv, start_s=s, background=bg))
        if self.no_drone_rounds:
            out += [dict(drone=None, drone_level_db=None, start_s=None, background=bg) for bg in self.backgrounds()]
        for i, r in enumerate(out, 1):
            r["round"] = i
        return out

    def count(self) -> dict:
        n_bg = len(self.backgrounds())
        n_lv = len(self.drone_levels) if self.drone_levels is not None else 1
        drone_rounds = len(self.starts()) * len(self.drones) * n_lv * n_bg
        total = drone_rounds + (n_bg if self.no_drone_rounds else 0)
        return dict(starts=len(self.starts()), drones=len(self.drones), drone_levels=n_lv, backgrounds=n_bg,
                    drone_rounds=drone_rounds, total=total, detector_runs=total * len(self.detectors))

    def check(self) -> list[str]:
        """What is wrong with the plan, in words; empty when it can run."""
        out = []
        if self.length_s < DRONE_S:
            out.append(f"soundscape length must be at least the drone's {DRONE_S:g} s")
        if self.step_s <= 0:
            out.append("start step must be above 0 s")
        if not self.drones:
            out.append("add at least one drone file" if self.kind == "real" else "give at least one drone speed")
        if self.kind == "real" and self.drone_levels is None:
            out.append("real drone files need a level in dB SPL")
        if not self.detectors:
            out.append("tick at least one detector")
        if not self.layers and not self.drones:
            out.append("nothing to test")
        if self.calibrate_fa is not None:
            if not self.layers:
                out.append("site calibration needs at least one background layer to listen to")
            if not 0 < self.calibrate_fa < 1:
                out.append("calibration false-alarm rate must be between 0 and 100 %")
            if self.calibrate_s < 10:
                out.append("calibrate on at least 10 s of background")
        return out


# ------------------------------------------------------------------ drones

class Drones:
    """The 10 s drone signals (WAV units, 16 kHz) and where their sound starts and ends within them."""

    def __init__(self, plan: Plan, progress=print):
        self.plan, self.sig, self.span = plan, {}, {}
        if plan.kind == "real":
            for p in plan.drones:
                x = read_wav(p, DEFAULT).audio.astype(np.float64)
                if len(x) < DRONE_S * SR:
                    raise ValueError(f"{Path(p).name} is {len(x) / SR:.1f} s; drone files must be at least {DRONE_S:g} s")
                self.sig[p] = x[: int(DRONE_S * SR)]
                self.span[p] = (0.0, DRONE_S, None)              # the whole file is drone; no impact
        else:
            a = dwg.defaults()
            a.flight_s = DRONE_S
            a.drone_seed, a.pscale, a.elev_deg = (plan.synth[k] for k in ("drone_seed", "pscale", "elev_deg"))
            progress(f"rendering {len(plan.drones)} synthetic drones with Pure Data (cached after the first time)")
            rendered = dwg.render_drones([float(v) for v in plan.drones], a, dwg.ROOT / "artifacts/drone_war/drones")
            for v in plan.drones:
                pa, prm = rendered[float(v)]
                x = resample_poly(pa / dwg.es.FULL, SR, dwg.es.FS)[: int(DRONE_S * SR)]
                self.sig[v] = np.pad(x, (0, max(0, int(DRONE_S * SR) - len(x))))
                arrive = prm["start_dist"] / dwg.C               # first sound reaches the microphone
                self.span[v] = (arrive, DRONE_S, DRONE_S)        # impact at the end of the 10 s

    def get(self, d, level_db) -> np.ndarray:
        x = self.sig[d]
        if level_db is None:                                     # synthetic, as rendered
            return x
        s0, s1, _ = self.span[d]
        part = x[int(s0 * SR): int(s1 * SR)]
        r = float(np.sqrt(np.mean(part ** 2))) or 1.0
        return x * (rms_for(level_db) / r)


def drone_label(d) -> str:
    return Path(d).name if isinstance(d, str) else f"{float(d):g} km/h"


# ------------------------------------------------------------------ detectors

def load_detectors(names: list[str]) -> tuple[dict, dict]:
    thr, sc = {}, {}
    for m in names:
        if m == "comb":
            thr[m], sc[m] = dwg.comb_threshold(), load_scorer("comb", DEFAULT, "cpu")
        else:
            t, s = dwg._detectors(m)
            thr[m], sc[m] = t["CNN"], s["CNN"]
    return thr, sc


def calibrate(plan: Plan, layers: dict, scorers: dict, frozen: dict, start_s: float) -> tuple[dict, dict]:
    """Site calibration (report entry 148): each detector's threshold set on the site's own background.

    For every background of the plan (each combination of the enabled layers' levels), `calibrate_s`
    seconds of the layers with no drone are mixed, from the stretch that follows the test's own stretch
    ([start_s + length_s, ...), so the calibration audio is never the audio the rounds are cut from unless
    a source is shorter than length_s + calibrate_s and wraps: `overlaps_test` says so per layer). Every
    detector streams it as in a round (1 s window, 0.125 s hop); the windows of all backgrounds are pooled
    and the threshold is rangerun.fa_threshold at `calibrate_fa`. This is what a fielded system does when
    it is commissioned on a recording of where it stands, instead of keeping a threshold frozen on
    another rig's validation (entries 141, 147)."""
    from dad.rangerun import fa_threshold
    n = int(round(plan.calibrate_s * SR))
    c0 = start_s + plan.length_s
    pooled = {k: [] for k in scorers}
    for bg in plan.backgrounds():
        mix = np.sum([layers[name].window(n, c0, lv) for name, lv in bg.items()], axis=0)
        clip = Clip("site calibration", mix.astype(np.float32), SR, 1, "FLOAT")
        for k, sc in scorers.items():
            pooled[k].append(np.asarray(stream(clip, sc, DEFAULT, frozen[k]).score, dtype=np.float64))
    thr = {k: float(fa_threshold(np.concatenate(v), plan.calibrate_fa)) for k, v in pooled.items()}
    info = dict(fa=plan.calibrate_fa, seconds=plan.calibrate_s, start_s=c0, backgrounds=len(plan.backgrounds()),
                windows={k: int(sum(len(x) for x in v)) for k, v in pooled.items()},
                fa_at_frozen={k: float((np.concatenate(v) >= frozen[k]).mean()) for k, v in pooled.items()},
                overlaps_test={name: bool((plan.length_s + plan.calibrate_s) * SR > len(layers[name].signal()[0]))
                               for name in plan.layers},
                frozen_thresholds=frozen)
    return thr, info


# ------------------------------------------------------------------ a round

def mix_round(r: dict, plan: Plan, layers: dict[str, Layer], drones: Drones, start_s: float) -> dict:
    """The parts of round r at 16 kHz: each background layer, the drone, the mix; and its truth."""
    n = int(round(plan.length_s * SR))
    parts = {name: layers[name].window(n, start_s, lv) for name, lv in r["background"].items()}
    truth = Truth([])
    if r["drone"] is not None:
        x = drones.get(r["drone"], r["drone_level_db"])
        j = int(round(r["start_s"] * SR))
        d = np.zeros(n)
        d[j:j + len(x)] = x[: n - j]
        parts["drone"] = d
        s0, s1, imp = drones.span[r["drone"]]
        k = dict(start_s=r["start_s"] + s0, end_s=r["start_s"] + s1)
        if imp is not None:
            k["impact_s"] = r["start_s"] + imp
        truth = Truth([k])
    mix = np.sum(list(parts.values()), axis=0) if parts else np.zeros(n)
    return dict(parts=parts, mix=mix, truth=truth)


def score_round(m: dict, thr: dict, scorers: dict) -> dict:
    """Every detector over a mixed round: decisions, labels and the round's numbers."""
    clip = Clip("round", m["mix"].astype(np.float32), SR, 1, "FLOAT")
    out = {}
    for k, sc in scorers.items():
        d = stream(clip, sc, DEFAULT, thr[k])
        lab = window_labels(d, m["truth"], DEFAULT.window_seconds)
        conf = dwg.squash(d.score, thr[k]) if k == "comb" else d.score
        tp, fn = int((d.drone & (lab == 1)).sum()), int((~d.drone & (lab == 1)).sum())
        fp, tn = int((d.drone & (lab == 0)).sum()), int((~d.drone & (lab == 0)).sum())
        ratio = lambda a, b: a / b if b else None                # noqa: E731
        first = delay = lead = None
        if m["truth"].drones:
            k0 = m["truth"].drones[0]
            hit = d.drone & (d.t_end - DEFAULT.window_seconds / 2 >= k0["start_s"]) & (d.t_end <= k0["end_s"])
            if hit.any():
                first = float(d.t_end[hit][0])
                delay = first - k0["start_s"]
            if "impact_s" in k0:
                lt = lead_times(d, m["truth"], DEFAULT.window_seconds)
                lead = lt[0]["lead_s"] if lt else None
        out[k] = dict(dec=d, labels=lab, conf=conf,
                      row=dict(tp=tp, fn=fn, fp=fp, tn=tn, left_out=int((lab == -1).sum()),
                               recall=ratio(tp, tp + fn), false_positive_rate=ratio(fp, fp + tn),
                               precision=ratio(tp, tp + fp), first_alarm_s=first, delay_s=delay, lead_s=lead,
                               any_alarm=bool(d.drone.any()), any_false_alarm=bool(fp),
                               mean_conf_drone=float(conf[lab == 1].mean()) if (lab == 1).any() else None,
                               mean_conf_no_drone=float(conf[lab == 0].mean()) if (lab == 0).any() else None))
    return out


def hop_levels(parts: dict, t_end: np.ndarray) -> dict:
    """dB SPL of every part (and the mix) in each 1 s window ending at t_end, as the detectors hear it."""
    n = int(DEFAULT.window_seconds * SR)
    out = {}
    for name, x in parts.items():
        c = np.concatenate([[0.0], np.cumsum(x ** 2)])
        b = np.clip(np.round(t_end * SR).astype(int), 0, len(x))
        a = np.clip(b - n, 0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[name] = FULL_SCALE_DB + 10 * np.log10((c[b] - c[a]) / np.maximum(b - a, 1))
    return out


# ------------------------------------------------------------------ a whole test

def row_fields(plan: Plan) -> list[str]:
    return (["round", "kind", "drone", "drone_level_db", "start_s", *[f"{n}_db" for n in plan.layers], "detector",
             "threshold", "tp", "fn", "fp", "tn", "left_out", "recall", "false_positive_rate", "precision",
             "first_alarm_s", "delay_s", "lead_s", "any_alarm", "any_false_alarm", "mean_conf_drone",
             "mean_conf_no_drone"])


class Test:
    """One run of a plan. `run()` goes round by round (stop() between rounds), appends to rounds.csv,
    keeps every round's per-hop scores in memory and in scores.npz, and writes config.json."""

    def __init__(self, plan: Plan, out_dir: Path | None = None):
        self.plan = plan
        self.out = Path(out_dir or OUT_ROOT / f"{time.strftime('%Y%m%d_%H%M%S')}_{plan.kind}")
        self.layers = make_layers()
        self.rounds = plan.rounds()
        self.results: dict[int, dict] = {}                   # round -> detector -> row (+ "scores")
        self.start_s = plan.soundscape_start_s
        self.drones = None
        self.thr = self.scorers = None
        self.calibration = None

    def prepare(self, progress=print):
        if self.start_s is None:                             # one start per batch, so the longest window fits
            longest = max((self.layers[n].signal()[0].size for n in self.plan.layers), default=0) / SR
            span = max(longest - self.plan.length_s, 0)
            self.start_s = float(np.random.default_rng(dwg.WSN_SEED).integers(0, int(span) + 1)) if span else 0.0
        self.drones = Drones(self.plan, progress)
        progress("loading detectors")
        self.thr, self.scorers = load_detectors(self.plan.detectors)
        if self.plan.calibrate_fa is not None:
            progress(f"calibrating thresholds on {self.plan.calibrate_s:g} s of each background "
                     f"({len(self.plan.backgrounds())} backgrounds) at {100 * self.plan.calibrate_fa:g}% false alarms")
            self.thr, self.calibration = calibrate(self.plan, self.layers, self.scorers, self.thr, self.start_s)
        self.out.mkdir(parents=True, exist_ok=True)
        cfg = dict(plan=asdict(self.plan), soundscape_start_s=self.start_s, thresholds=self.thr,
                   calibration=self.calibration,
                   count=self.plan.count(), drone_spans={drone_label(k): v for k, v in self.drones.span.items()})
        (self.out / "config.json").write_text(json.dumps(cfg, indent=1, default=str), encoding="utf-8")

    def rebuild(self, r: dict) -> dict:
        return mix_round(r, self.plan, self.layers, self.drones, self.start_s)

    def run(self, progress=print, on_round=None, stop=lambda: False):
        if self.drones is None:
            self.prepare(progress)
        fields = row_fields(self.plan)
        t0 = time.time()
        scores = {}
        with open(self.out / "rounds.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fields)
            w.writeheader()
            for i, r in enumerate(self.rounds, 1):
                if stop():
                    progress(f"stopped after {i - 1} of {len(self.rounds)} rounds")
                    break
                res = score_round(self.rebuild(r), self.thr, self.scorers)
                self.results[r["round"]] = {k: v["row"] for k, v in res.items()}
                for k, v in res.items():
                    scores[f"{r['round']}|{k}"] = v["dec"].score
                    w.writerow(dict(round=r["round"], kind=self.plan.kind,
                                    drone=drone_label(r["drone"]) if r["drone"] is not None else "none",
                                    drone_level_db="as rendered" if r["drone"] is not None and r["drone_level_db"] is None
                                    else r["drone_level_db"], start_s=r["start_s"],
                                    **{f"{n}_db": lv for n, lv in r["background"].items()},
                                    detector=k, threshold=self.thr[k], **v["row"]))
                fh.flush()
                el = time.time() - t0
                progress(f"round {i} of {len(self.rounds)}  ({el:.0f} s, about {el / i * (len(self.rounds) - i):.0f} s left)")
                if on_round:
                    on_round(i, r)
        np.savez_compressed(self.out / "scores.npz", **scores)
        write_summary(self)
        return self.results


def write_summary(test: Test) -> Path:
    """summary.csv: per detector, and per detector and each varied setting, the mean of the round numbers."""
    rows = []
    keys = ["drone", "drone_level_db", "start_s", *test.plan.layers]

    def setting(r, k):
        if k in r["background"]:
            return r["background"][k]
        return drone_label(r[k]) if k == "drone" and r[k] is not None else r[k]

    def agg(group):
        out = {}
        for m in ("recall", "false_positive_rate", "precision", "delay_s", "mean_conf_drone", "mean_conf_no_drone"):
            v = [g[m] for g in group if g[m] is not None]
            out[m] = float(np.mean(v)) if v else None
        out["rounds"] = len(group)
        out["share_detected"] = float(np.mean([g["first_alarm_s"] is not None for g in group])) if group else None
        out["share_with_false_alarm"] = float(np.mean([g["any_false_alarm"] for g in group])) if group else None
        return out

    by_round = {r["round"]: r for r in test.rounds}
    for det in test.plan.detectors:
        done = [(by_round[i], res[det]) for i, res in test.results.items() if det in res]
        drone = [(r, x) for r, x in done if r["drone"] is not None]
        rows.append(dict(detector=det, setting="all drone rounds", value="", **agg([x for _, x in drone])))
        nd = [x for r, x in done if r["drone"] is None]
        if nd:
            rows.append(dict(detector=det, setting="no-drone rounds", value="", **agg(nd)))
        for k in keys:
            vals = sorted({str(setting(r, k)) for r, _ in drone})
            if len(vals) > 1:
                for v in vals:
                    rows.append(dict(detector=det, setting=k, value=v,
                                     **agg([x for r, x in drone if str(setting(r, k)) == v])))
    p = test.out / "summary.csv"
    if rows:
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    test.summary = rows
    return p
