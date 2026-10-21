"""Run a detector over WAV files, hop by hop, as it would run live.

A WAV of any sample rate and bit depth is read, reduced to its first channel
and resampled to the frontend's 16 kHz. Every hop the detector sees the last
window of audio and returns a score; at or above the threshold the answer is
`True` (drone), else `False` (requirement R1). A decision is stamped with the
time its window ends, which is the earliest moment it could exist live.

If a sidecar `<name>.json` sits next to the WAV, the decisions are scored
against it (requirements R2 to R5). Its format:

    {"drones": [{"start_s": 7.0, "end_s": 12.0, "impact_s": 12.0}]}

`impact_s` is optional; without it R5 is not scored for that drone. Every
other key in the file is kept as metadata and copied into the result.

Window labels for R2-R4. A window lying wholly inside a drone interval is a
drone window; a window that touches no drone interval is a no-drone window;
a window that straddles a start or an end is left out, since it holds both.
With a 1 s window that leaves out exactly the windows within 1 s of a
drone's start or end.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import DEFAULT, FrontendConfig

REACTION_S = 3.0  # R5: the alarm must come at least this long before impact


@dataclass
class Clip:
    path: str
    audio: np.ndarray        # float32, mono, at cfg.sample_rate
    sr_orig: int
    channels_orig: int
    subtype: str


def read_wav(path, cfg: FrontendConfig = DEFAULT) -> Clip:
    """Read any soundfile-readable file as float32 mono at the frontend rate.

    Multi-channel files are reduced to channel 0, as the processed store was
    (entries 007 and 023); the channel count is kept so a caller can warn.
    """
    with sf.SoundFile(str(path)) as f:
        x, sr, subtype = f.read(dtype="float64", always_2d=True), f.samplerate, f.subtype
    y = x[:, 0]
    if sr != cfg.sample_rate:
        g = math.gcd(cfg.sample_rate, sr)
        y = resample_poly(y, cfg.sample_rate // g, sr // g)
    return Clip(str(path), y.astype(np.float32), sr, x.shape[1], subtype)


@dataclass
class Decisions:
    t_end: np.ndarray        # (N,) seconds; the window covers [t_end - window, t_end)
    score: np.ndarray        # (N,)
    threshold: float | None

    @property
    def drone(self) -> np.ndarray | None:
        return None if self.threshold is None else self.score >= self.threshold


def stream(clip: Clip, scorer, cfg: FrontendConfig = DEFAULT, threshold: float | None = None,
           batch: int = 256) -> Decisions:
    """Score every full window that starts on a hop. The first decision comes
    one window in; a clip shorter than a window yields none."""
    n, hop = cfg.samples_per_window, cfg.samples_per_hop
    starts = np.arange(0, len(clip.audio) - n + 1, hop) if len(clip.audio) >= n else np.empty(0, int)
    out = []
    for i in range(0, len(starts), batch):
        a = np.stack([clip.audio[s:s + n] for s in starts[i:i + batch]])
        out.append(np.asarray(scorer.score(a), dtype=np.float64))
    score = np.concatenate(out) if out else np.empty(0)
    return Decisions((starts + n) / cfg.sample_rate, score, threshold)


# -------------------------------------------------------------- ground truth

@dataclass
class Truth:
    drones: list[dict]
    meta: dict = field(default_factory=dict)


def read_truth(wav_path) -> Truth | None:
    p = Path(wav_path).with_suffix(".json")
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    drones = d.pop("drones", [])
    for k in drones:
        if not 0 <= k["start_s"] < k["end_s"] or k.get("impact_s", k["start_s"]) < k["start_s"]:
            raise ValueError(f"{p}: need 0 <= start_s < end_s and impact_s >= start_s, got {k}")
    return Truth(drones, d)


def window_labels(dec: Decisions, truth: Truth, window_s: float) -> np.ndarray:
    """1 drone, 0 no drone, -1 left out (straddles a start or an end)."""
    a, b = dec.t_end - window_s, dec.t_end
    lab = np.zeros(len(a), int)
    for k in truth.drones:
        inside = (a >= k["start_s"]) & (b <= k["end_s"])
        touch = (b > k["start_s"]) & (a < k["end_s"])
        lab[touch & ~inside & (lab != 1)] = -1
        lab[inside] = 1
    return lab


def lead_times(dec: Decisions, truth: Truth, window_s: float) -> list[dict]:
    """R5, per drone with an impact time: the first alarm whose window is at
    least half drone, no later than impact, and how long before impact it came.
    An alarm from a window that is mostly pre-drone audio (an explosion just
    before the drone appears) is not credited to the drone."""
    out = []
    if dec.threshold is None:
        return out
    for k in truth.drones:
        if "impact_s" not in k:
            continue
        hit = dec.drone & (dec.t_end - window_s / 2 >= k["start_s"]) & (dec.t_end <= k["impact_s"])
        t1 = float(dec.t_end[hit][0]) if hit.any() else None
        lead = None if t1 is None else k["impact_s"] - t1
        out.append(dict(start_s=k["start_s"], impact_s=k["impact_s"], first_alarm_s=t1, lead_s=lead,
                        met=lead is not None and lead >= REACTION_S))
    return out


# -------------------------------------------------------------------- metrics

def counts(dec: Decisions, labels: np.ndarray) -> dict:
    d = dec.drone
    return dict(tp=int((d & (labels == 1)).sum()), fn=int((~d & (labels == 1)).sum()),
                fp=int((d & (labels == 0)).sum()), tn=int((~d & (labels == 0)).sum()),
                left_out=int((labels == -1).sum()))


def rates(c: dict, leads: list[dict]) -> dict:
    """Recall (R2), false-positive rate (R3), precision (R4), drone share, R5."""
    def ratio(a, b):
        return a / b if b else None
    pos, neg = c["tp"] + c["fn"], c["fp"] + c["tn"]
    met = [x["met"] for x in leads]
    return dict(recall=ratio(c["tp"], pos), false_positive_rate=ratio(c["fp"], neg),
                precision=ratio(c["tp"], c["tp"] + c["fp"]), drone_share=ratio(pos, pos + neg),
                drone_windows=pos, no_drone_windows=neg,
                approaches=len(met), approaches_alarm_3s_before_impact=ratio(sum(met), len(met)))


def evaluate_file(path, scorer, cfg: FrontendConfig = DEFAULT, threshold: float | None = None) -> dict:
    """Everything known about one file. `decisions` holds the per-hop answers."""
    clip = read_wav(path, cfg)
    dec = stream(clip, scorer, cfg, threshold)
    res = dict(file=str(path), seconds=round(len(clip.audio) / cfg.sample_rate, 3), sr_orig=clip.sr_orig,
               channels_orig=clip.channels_orig, subtype=clip.subtype, decisions=dec)
    truth = read_truth(path)
    if truth is not None:
        lab = window_labels(dec, truth, cfg.window_seconds)
        res.update(truth=truth, labels=lab)
        if threshold is not None:
            res.update(counts=counts(dec, lab), leads=lead_times(dec, truth, cfg.window_seconds))
    return res


def summarise(results: list[dict]) -> dict:
    """Pool the windows and approaches of every scored file."""
    scored = [r for r in results if "counts" in r]
    total = {k: sum(r["counts"][k] for r in scored) for k in ("tp", "fn", "fp", "tn", "left_out")}
    leads = [x for r in scored for x in r["leads"]]
    return dict(files=len(results), files_with_truth=len(scored), counts=total, **rates(total, leads))


# ------------------------------------------------------------------ detectors

def load_scorer(spec: str, cfg: FrontendConfig = DEFAULT, device: str = "cpu"):
    """`comb` for the harmonic-comb baseline (no training), a `.pt` network
    checkpoint or a `.pkl` LightGBM model written by `scripts/run_range.py`."""
    from . import realrun as rr
    if spec == "comb":
        return rr.CombScorer(cfg)
    if spec.endswith(".pt"):
        return rr.NetScorer.load(spec, cfg, device)
    if spec.endswith(".pkl"):
        return rr.LgbmScorer.load(spec, cfg)
    raise ValueError(f"unknown detector {spec!r}: give 'comb', a .pt or a .pkl checkpoint")


def threshold_from_run(run_json, checkpoint, window_s: float) -> float:
    """The deployment threshold a run froze on validation for this checkpoint.

    Checkpoints are named `<run>_<held-out unit>_<recipe>_<arch>.pt`, with `:`
    in the unit written as `_`. Only the single-window result (integrate = 1)
    is used, because that is what `stream` produces. The run's window must
    match `window_s`: a checkpoint does not record it, and a network fed the
    wrong window length runs without error and scores nonsense.
    """
    from .rangerun import RECIPES
    run = json.loads(Path(run_json).read_text(encoding="utf-8"))
    if checkpoint == "comb":
        raise ValueError("the comb's threshold depends on the held-out arm: pass --threshold with the "
                         "deploy_threshold of the 'comb (no ML)' result for that arm in the run file")
    trained = run["args"].get("window_seconds", 1.0)
    if trained != window_s:
        raise ValueError(f"{run_json} was trained with a {trained} s window; pass --window {trained}")
    name = Path(checkpoint).name
    rows = [r for r in run["results"] if r.get("integrate", 1) == 1]
    loaded = [r for r in rows if r.get("loaded") == name]
    if loaded:
        return float(loaded[0]["deploy_threshold"])
    stem, prefix = Path(checkpoint).stem, run["args"]["name"] + "_"
    if not stem.startswith(prefix):
        raise ValueError(f"{name} was not written by run {run['args']['name']!r}")
    rest = stem[len(prefix):]
    arch = rest.rsplit("_", 1)[-1]
    for recipe in RECIPES:
        tail = f"_{recipe}_{arch}"
        if rest.endswith(tail):
            held = rest[: -len(tail)]
            hit = [r for r in rows if r["scorer"] == arch and r["recipe"] == recipe
                   and r["arm"].replace(":", "_") == f"hold out {held}"]
            if hit:
                return float(hit[0]["deploy_threshold"])
    raise ValueError(f"no single-window result for {name} in {run_json}")
