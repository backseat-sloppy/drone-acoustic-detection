"""Iteration 9: synthetic training windows (report entries 086-089).

Drone: the Pure Data drone, `DroneSound.pd`, through its numpy port
`dad.pdsynth` (Mads, entry 089), so it runs where Pd is not installed. Every
window draws its own drone and flight, so no single frequency can be learned
(the supervisor's point, entry 086):

    pscale        the patch's pitch scale, log-uniform 0.4-3.0; with the
                  throttle this puts the motor fundamental at about
                  80-900 Hz (real drone peaks reach about 950 Hz, entry 086)
    throttle      render.py's throttle model: cruise 30-110 on the 0-127
                  slider, wobble 2-10, a dive ramp in 30% of flights
    motion        straight at the microphone (approach) or past it (fly-by),
                  0-60 m/s, through `dad.propagation`: 1/r, Doppler, ISO
                  9613-1 air absorption

The patch renders at 44.1 kHz and is resampled to the frontend rate. Two
seconds are emitted and a window of `n` samples is cut from the part that
has fully arrived at the microphone.

Soundscape: a bed (a real no-drone window from a training rig, or a
background or wind stem), then with probability `p_confuser` one distracter
or war stem at -10 to +15 dB against the bed, then the drone at a
signal-to-noise ratio of -20 to +10 dB in band (above 50 Hz). About a quarter
of windows hold no drone. No real drone audio is used.

    s = SynthSampler(rig_recs, seed=0)
    L, y = s.batch(0, 64)          # level-normalised log-mel, labels
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from . import chains as ch
from . import pdsynth, propagation
from . import realrun as rr
from .config import DEFAULT, FrontendConfig
from .features import LogMel
from .realdata import Recording
from .wavio import read_wav

STEMS = Path(__file__).resolve().parents[2] / "data" / "processed" / "stems"
BED_GROUPS = ("background", "wind")
CONFUSER_GROUPS = ("distracter", "war")


def drone(rng: np.random.Generator, n: int, fs: int = 16_000) -> tuple[np.ndarray, dict]:
    """One Pd drone heard at the microphone, `n` samples at `fs`, unit RMS, and its parameters."""
    dive = rng.random() < 0.3
    p = dict(pscale=float(math.exp(rng.uniform(math.log(0.4), math.log(3.0)))),
             base=float(rng.uniform(30.0, 110.0)), wobble=float(rng.uniform(2.0, 10.0)),
             dive_start=float(rng.uniform(0.0, 0.9)) if dive else 0.7,
             dive_gain=float(rng.uniform(10.0, 60.0)) if dive else 0.0,
             speed=float(rng.uniform(0.0, 60.0)), motion="approach" if rng.random() < 0.5 else "flyby")
    m = 2 * n                                                # emitted samples at fs
    steps = int(math.ceil(m / fs / 0.02)) + 1
    thr = pdsynth.make_throttle(rng, steps, p["base"], p["wobble"], p["dive_start"], p["dive_gain"])
    x = pdsynth.render(thr, p["pscale"], seed=int(rng.integers(2 ** 31)))
    g = math.gcd(fs, pdsynth.FS)
    x = resample_poly(x, fs // g, pdsynth.FS // g)[:m]
    t = np.arange(len(x)) / fs
    if p["motion"] == "approach":
        p["closest_m"] = float(rng.uniform(10.0, 400.0))   # where the emitted stretch ends
        r = p["closest_m"] + p["speed"] * (t[-1] - t)
    else:
        p["closest_m"] = float(rng.uniform(5.0, 200.0))
        r = propagation.pass_by(len(x), p["speed"], p["closest_m"], rng.uniform(0.0, t[-1]), fs)
    y = propagation.propagate(x, r, fs=fs)
    # Before the first emitted sample arrives the output is silent: start the window after it.
    first = int(math.ceil((r[0] - r.min()) / propagation.speed_of_sound() * fs))
    s = int(rng.integers(first, max(first, len(y) - n) + 1))
    y = y[s:s + n]
    return y / (y.std() + 1e-12), p


def _load_stems(groups: tuple[str, ...], fs: int) -> list[np.ndarray]:
    out = []
    with open(STEMS / "manifest.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["group"] in groups:
                out.append(read_wav(STEMS.parent.parent.parent / row["file"]).audio.astype(np.float64))
    if not out:
        raise FileNotFoundError(f"no stems of {groups} in {STEMS} (scripts/build_stems.py)")
    return out


class SynthSampler:
    """Training windows: synthetic drones in soundscapes. Same interface as
    rangerun.RangeSampler (`batch`, `seed`), so NetScorer and LgbmScorer
    train on it unchanged."""

    def __init__(self, rig_recs: list[Recording], cfg: FrontendConfig = DEFAULT, seed: int = 0,
                 p_drone: float = 0.75, p_confuser: float = 0.4, snr_db: tuple[float, float] = (-20.0, 10.0),
                 confuser_db: tuple[float, float] = (-10.0, 15.0), p_rig_bed: float = 0.5) -> None:
        self.cfg, self.seed, self.n = cfg, seed, cfg.samples_per_window
        self.p_drone, self.p_confuser, self.snr_db, self.confuser_db = p_drone, p_confuser, snr_db, confuser_db
        self.p_rig_bed = p_rig_bed if rig_recs else 0.0
        self.fe = LogMel(cfg)
        self.beds = _load_stems(BED_GROUPS, cfg.sample_rate)
        self.confusers = _load_stems(CONFUSER_GROUPS, cfg.sample_rate)
        grid = cfg.sample_rate // 10
        self.cw = ch.chain_windows(rig_recs, self.n, grid) if rig_recs else None
        self.rig_neg = np.flatnonzero(self.cw.y == 0) if rig_recs else np.empty(0, int)
        if rig_recs and not len(self.rig_neg):
            raise ValueError("the rig recordings hold no no-drone windows")

    def describe(self) -> str:
        rig_min = len(self.rig_neg) * 0.1 / 60
        return (f"synthetic drones in {len(self.beds)} bed stems + {rig_min:.1f} min of rig background "
                f"(bed share {self.p_rig_bed:.2f}), {len(self.confusers)} confuser stems at p={self.p_confuser}, "
                f"drone share {self.p_drone}, SNR {self.snr_db[0]:g} to {self.snr_db[1]:g} dB")

    def _crop(self, rng: np.random.Generator, x: np.ndarray) -> np.ndarray:
        if len(x) < self.n:
            x = np.tile(x, self.n // len(x) + 1)
        s = int(rng.integers(0, len(x) - self.n + 1))
        return x[s:s + self.n]

    def _bed(self, rng: np.random.Generator) -> np.ndarray:
        for _ in range(50):
            if rng.random() < self.p_rig_bed:
                i = self.rig_neg[int(rng.integers(0, len(self.rig_neg)))]
                x = np.asarray(self.cw.audio(i, self.n), dtype=np.float64)
            else:
                x = self._crop(rng, self.beds[int(rng.integers(0, len(self.beds)))])
            if rr.inband_power(x) > rr.SILENCE_POWER:
                return x
        raise RuntimeError("could not draw a non-silent bed")

    def window(self, rng: np.random.Generator, with_drone: bool) -> np.ndarray:
        x = self._bed(rng)
        if rng.random() < self.p_confuser:
            c = self._crop(rng, self.confusers[int(rng.integers(0, len(self.confusers)))])
            pc = rr.inband_power(c)
            if pc > rr.SILENCE_POWER:     # a crop can land in a stem's quiet stretch
                x = x + c * math.sqrt(rr.inband_power(x) * 10.0 ** (rng.uniform(*self.confuser_db) / 10.0) / pc)
        if with_drone:
            d, _ = drone(rng, self.n, self.cfg.sample_rate)
            x = x + d * math.sqrt(rr.inband_power(x) * 10.0 ** (rng.uniform(*self.snr_db) / 10.0)
                                  / max(rr.inband_power(d), 1e-20))
        return x

    def batch(self, index: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng([self.seed, index])
        y = (rng.random(batch_size) < self.p_drone).astype(np.int64)
        L = np.stack([self.fe.frames(self.window(rng, bool(t))) for t in y])
        L = L + np.stack([rr._eq_curve(rng, L.shape[1]) for _ in y])[:, :, None]
        return rr.level_normalise(L), y
