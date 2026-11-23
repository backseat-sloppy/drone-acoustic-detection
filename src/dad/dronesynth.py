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

Iteration 25 (the clean-drone world simulator): `drones=glasgow_drones(ids)`
replaces the Pd drone with real close-range Glasgow hover recordings, each
sent through `real_drone`'s simulated outdoor path; `ladder=k` emits families
of one scene alone plus the same scene with the same drone at k levels;
`stem_split="train"|"val"` keeps only FSD50K-dev stems, split by clip id.
"""
from __future__ import annotations

import csv
import math
import re
import zlib
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from . import chains as ch
from . import pdsynth, propagation
from . import realrun as rr
from .config import DEFAULT, FrontendConfig
from .features import LogMel
from .realdata import Recording
from .sealed import forbid_in_training
from .wavio import read_wav

STEMS = Path(__file__).resolve().parents[2] / "data" / "processed" / "stems"
GLASGOW = Path(__file__).resolve().parents[2] / "data" / "extracted" / "Glasgow"
UAV16 = Path(__file__).resolve().parents[2] / "data" / "extracted" / "UAV16" / "dataset"
U16_HELD = frozenset({"batch-20260522", "batch-20260701"})   # = scripts/prepare_new_data.py: never train
U16_PLAIN = frozenset(f"rec-undated-000{i}" for i in (1, 2, 3, 4))   # = rangerun: "pure environment"
_GLASGOW_NAME = re.compile(r"_d(\d+)_hover_(\d+)m_", re.IGNORECASE)
MIC_HEIGHT_M = 1.5
QUIET_FLOOR_DB = (-90.0, -50.0)    # iteration 27: a near-empty scene's noise floor (absolute; only ratios survive level_normalise)
QUIET_CONFUSER_DB = (10.0, 50.0)   # a confuser alone over that floor: well above it, as a faint car over silence
# iteration 28: a road vehicle driving past the microphone, through real_drone's outdoor path (entry 249)
VEHICLE_CLS = ("car", "truck", "motorcycle", "idling", "scooter")
VEHICLE_PATH = dict(height=(0.3, 2.0), speed=(3.0, 30.0), flyby_closest=(2.0, 40.0), p_flyby=1.0)
VEHICLE_R_REF = 5.0   # FSD50K gives no recording distance; a guess, it only sets the air-absorption start
# iteration 29: steady noises as no-drone (council, entry 257). Generic acoustics, not shaped to any test layer
STEADY_SLOPE_DB_OCT = (-6.0, 3.0)      # power slope of the coloured noise (white 0, pink -3, brown -6)
STEADY_CONFUSER_SHARE = 0.25           # share of confusers that are a steady noise when p_steady > 0
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


def glasgow_drones(drone_ids, root: Path = GLASGOW) -> list[tuple[np.ndarray, float]]:
    """The Glasgow hover recordings of drones `drone_ids` (1-24) at 16 kHz as int16 (half the memory of
    float32, the source is 16-bit), each with its recording distance in metres (1 or 5)."""
    out = []
    for f in sorted(root.glob("*to*/*")):
        m = _GLASGOW_NAME.search(f.name)
        if f.suffix.lower() == ".wav" and m and int(m[1]) in set(drone_ids):
            forbid_in_training(f, where="glasgow drones")
            x = read_wav(f).audio
            out.append((np.round(np.clip(x, -1, 1) * 32767).astype(np.int16), float(m[2])))
    if not out:
        raise FileNotFoundError(f"no Glasgow recordings of drones {sorted(drone_ids)} under {root}")
    return out


def real_drone(rng: np.random.Generator, src: tuple[np.ndarray, float], n: int, fs: int = 16_000,
               reflection: tuple[float, float] = (0.3, 0.9), height: tuple[float, float] = (5.0, 120.0),
               speed: tuple[float, float] = (0.0, 60.0), flyby_closest: tuple[float, float] = (5.0, 200.0),
               p_flyby: float = 0.5) -> tuple[np.ndarray, dict]:
    """A real close-range recording heard outdoors: `n` samples at the microphone, unit RMS, and the draw.

    Rotor speed perturbed by resampling (0.8-1.25); a straight approach (ending 10-400 m away) or a
    fly-by (closest 5-200 m) at 0-60 m/s and 5-120 m height over a microphone at 1.5 m; direct path
    plus one ground reflection (image source, pressure coefficient `reflection`, a calibration knob,
    not a measured ground), each with 1/r, Doppler and ISO 9613-1 air absorption from the recording
    distance. The room the recording was made in stays in it (Glasgow is probably indoors).
    The geometry ranges are arguments so a ground vehicle can take the same path (`VEHICLE_PATH`).
    """
    x, r_ref = src
    k = Fraction(math.exp(rng.uniform(math.log(0.8), math.log(1.25)))).limit_denominator(50)
    m = 2 * n
    take = int(m * k) + 64
    s0 = int(rng.integers(0, len(x) - take))
    c = resample_poly(x[s0:s0 + take].astype(np.float64) / 32767.0, k.denominator, k.numerator)[:m]
    t = np.arange(m) / fs
    p = dict(rate=float(k), r_ref=r_ref, height_m=float(rng.uniform(*height)),
             speed=float(rng.uniform(*speed)), motion="approach" if rng.random() < 1.0 - p_flyby else "flyby",
             reflection=float(rng.uniform(*reflection)))
    if p["motion"] == "approach":
        p["closest_m"] = float(rng.uniform(10.0, 400.0))     # horizontal, where the emitted stretch ends
        dh = p["closest_m"] + p["speed"] * (t[-1] - t)
    else:
        p["closest_m"] = float(rng.uniform(*flyby_closest))
        dh = propagation.pass_by(m, p["speed"], p["closest_m"], rng.uniform(0.0, t[-1]), fs)
    r1 = np.hypot(dh, p["height_m"] - MIC_HEIGHT_M)
    r2 = np.hypot(dh, p["height_m"] + MIC_HEIGHT_M)
    y = propagation.propagate(c, r1, r_ref=r_ref, fs=fs)
    z = p["reflection"] * propagation.propagate(c, r2, r_ref=r_ref, fs=fs)
    # propagate drops each path's own shortest delay; put the reflection back at its extra delay.
    lag = int(round((r2.min() - r1.min()) / propagation.speed_of_sound() * fs))
    if lag < len(y):
        y[lag:lag + len(z)] += z[:len(y) - lag]
    first = int(math.ceil((r1[0] - r1.min()) / propagation.speed_of_sound() * fs))
    s = int(rng.integers(first, max(first, len(y) - n) + 1))
    y = y[s:s + n]
    return y / (y.std() + 1e-12), p


def _load_stems(groups: tuple[str, ...], fs: int, split: str | None = None,
                classes: tuple[str, ...] | None = None) -> list[tuple[str, np.ndarray]]:
    """(group, audio) of every stem in `groups`. With `split`, FSD50K-dev stems only (FSD50K eval is the
    unseen test material of earlier iterations), one in five by clip id for "val", the rest for "train"."""
    out = []
    with open(STEMS / "manifest.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["group"] not in groups or (classes is not None and row["cls"] not in classes):
                continue
            if split is not None and (row["source"] != "fsd50k_dev"
                                      or (zlib.crc32(row["id"].encode()) % 5 == 0) != (split == "val")):
                continue
            forbid_in_training(row["source"], row["file"], where="dronesynth stems")
            out.append((row["group"], read_wav(STEMS.parent.parent.parent / row["file"]).audio.astype(np.float64)))
    if not out:
        raise FileNotFoundError(f"no stems of {groups} in {STEMS} (scripts/build_stems.py)")
    return out


def glasgow_noise(split: str, root: Path = GLASGOW) -> tuple[np.ndarray, float]:
    """Glasgow's own no-drone recording (60.5 min, same dataset and microphone as the drones) as an int16
    source for `real_drone`: the first 80 % for "train", the last 20 % for "val"."""
    f = root / "noise" / "_uav_2022noise_n_noise_1m_.WAV"
    forbid_in_training(f, where="glasgow noise")
    x = read_wav(f).audio
    cut = int(0.8 * len(x))
    x = x[:cut] if split == "train" else x[cut:]
    return np.round(np.clip(x, -1, 1) * 32767).astype(np.int16), 1.0


def _u16_rows(root: Path) -> list[dict]:
    with open(root / "metadata" / "recordings.csv", newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if r["data_origin"] == "self_collected"]


def _u16_audio(r: dict, root: Path) -> np.ndarray:
    """Channel 0 of a UAV16 recording at 16 kHz as int16 (the 16 microphones are 42.5 mm apart: one take)."""
    f = (root / "data" / ("real_uav" if r["has_uav"] == "true" else "real_background") / r["session_id"]
         / r["recording_id"] / f"{r['recording_id']}__ch-00.wav")
    forbid_in_training(f, where="uav16")
    return np.round(np.clip(read_wav(f).audio, -1, 1) * 32767).astype(np.int16)


def uav16_drones(split: str, max_range_m: float = 15.0, root: Path = UAV16) -> list[tuple[np.ndarray, float]]:
    """UAV16 DJI Flip hovers within `max_range_m` (slant), each with its slant range as r_ref: the four
    closest grid points (5-10 m out, 5-10 m up). "val" is the two held batches, "train" the other six.
    Unlike Glasgow these are outdoors, so the recording's own field noise travels with the drone."""
    out = [(_u16_audio(r, root), math.hypot(float(r["horizontal_distance_m"]), float(r["height_m"])))
           for r in _u16_rows(root) if r["has_uav"] == "true"
           and (r["acquisition_batch_id"] in U16_HELD) == (split == "val")
           and math.hypot(float(r["horizontal_distance_m"]), float(r["height_m"])) <= max_range_m]
    if not out:
        raise FileNotFoundError(f"no UAV16 drones for {split!r} under {root}")
    return out


def uav16_noise(split: str, root: Path = UAV16) -> tuple[np.ndarray, float]:
    """UAV16's four plain-environment recordings, one after another, as a chain-twin source: the first
    80 % of each for "train", the last 20 % for "val" (as glasgow_noise)."""
    parts = []
    for r in _u16_rows(root):
        if r["recording_id"] in U16_PLAIN:
            x = _u16_audio(r, root)
            cut = int(0.8 * len(x))
            parts.append(x[:cut] if split == "train" else x[cut:])
    if len(parts) != len(U16_PLAIN):
        raise FileNotFoundError(f"UAV16 plain-environment recordings missing under {root}")
    return np.concatenate(parts), 1.0


def colour_noise(rng: np.random.Generator, n: int, fs: int) -> np.ndarray:
    """Stationary Gaussian noise, unit RMS: a power slope drawn from STEADY_SLOPE_DB_OCT, and in half the
    draws a low-pass of order 1-4 with its corner at 150-3000 Hz (the dark, rumbling noises)."""
    f = np.fft.rfftfreq(n, 1 / fs)
    g = np.ones_like(f)
    g[1:] = (f[1:] / 1000.0) ** (rng.uniform(*STEADY_SLOPE_DB_OCT) / (20 * math.log10(2)))
    g[0] = 0.0
    if rng.random() < 0.5:
        g /= np.sqrt(1 + (f / rng.uniform(150.0, 3000.0)) ** (2 * int(rng.integers(1, 5))))
    x = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * g, n)
    return x / (x.std() + 1e-12)


def synth_wind(rng: np.random.Generator, n: int, fs: int) -> np.ndarray:
    """Wind as a microphone hears it, from generic acoustics: low-pass noise (corner 60-800 Hz, order 1-3)
    under a gust envelope of three slow sines (0.1-2 Hz, depth 0.2-1.0 in log amplitude). Unit RMS."""
    f = np.fft.rfftfreq(n, 1 / fs)
    g = 1 / np.sqrt(1 + (f / rng.uniform(60.0, 800.0)) ** (2 * int(rng.integers(1, 4))))
    g[0] = 0.0
    x = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * g, n)
    t = np.arange(n) / fs
    gust = sum(np.sin(2 * np.pi * rng.uniform(0.1, 2.0) * t + rng.uniform(0, 2 * np.pi)) for _ in range(3)) / 3
    x = x * np.exp(rng.uniform(0.2, 1.0) * gust)
    return x / (x.std() + 1e-12)


def steady(rng: np.random.Generator, n: int, fs: int) -> tuple[str, np.ndarray]:
    """One steady no-drone sound: coloured noise or synthetic wind, equally likely."""
    return ("colour_noise", colour_noise(rng, n, fs)) if rng.random() < 0.5 else ("synth_wind", synth_wind(rng, n, fs))


class SynthSampler:
    """Training windows: synthetic drones in soundscapes. Same interface as
    rangerun.RangeSampler (`batch`, `seed`), so NetScorer and LgbmScorer
    train on it unchanged.

    Iteration 25 options (all off by default):
      drones      Glasgow recordings (`glasgow_drones`) in place of the Pd drone, through `real_drone`
      twin        a no-drone source (`glasgow_noise`) sent through the same `real_drone` path and added
                  to a share `p_twin` of the no-drone windows at the drone's SNR range, so the recording
                  chain and the simulator's processing are in both classes (council, iteration 25)
      ladder      k > 0: families of one scene alone plus the same scene with one drone at k SNRs
      stem_split  "train" or "val": FSD50K-dev stems only, split by clip id (`_load_stems`)
    Beds and confusers are chosen by group first (each group equally likely), then by stem.
    """

    def __init__(self, rig_recs: list[Recording], cfg: FrontendConfig = DEFAULT, seed: int = 0,
                 p_drone: float = 0.75, p_confuser: float = 0.4, snr_db: tuple[float, float] = (-20.0, 10.0),
                 confuser_db: tuple[float, float] = (-10.0, 15.0), p_rig_bed: float = 0.5, p_quiet: float = 0.0, p_quiet_confuser: float = 0.7,
                 p_vehicle: float = 0.0, p_steady: float = 0.0, p_steady_confuser: float = STEADY_CONFUSER_SHARE,
                 drones: list[tuple[np.ndarray, float]] | None = None, twin: tuple[np.ndarray, float] | list | None = None,
                 p_twin: float = 0.5, ladder: int = 0, stem_split: str | None = None,
                 bed_groups: tuple[str, ...] = BED_GROUPS, confuser_groups: tuple[str, ...] = CONFUSER_GROUPS) -> None:
        self.cfg, self.seed, self.n = cfg, seed, cfg.samples_per_window
        self.p_drone, self.p_confuser, self.snr_db, self.confuser_db = p_drone, p_confuser, snr_db, confuser_db
        self.p_rig_bed = p_rig_bed if rig_recs else 0.0
        self.p_quiet, self.p_quiet_confuser = p_quiet, p_quiet_confuser
        # twin: one source or a list (one per rig, picked evenly)
        self.twin = [twin] if isinstance(twin, tuple) else (twin or None)
        self.drones, self.p_twin, self.ladder = drones, p_twin, ladder
        self.fe = LogMel(cfg)
        self.beds = _by_group(_load_stems(bed_groups, cfg.sample_rate, stem_split))
        self.confusers = _by_group(_load_stems(confuser_groups, cfg.sample_rate, stem_split))
        self.p_steady, self.p_steady_confuser = p_steady, p_steady_confuser   # iteration 29
        self.p_vehicle = p_vehicle   # share of confusers that are a road vehicle driving past (iteration 28)
        self.vehicles = [(np.round(np.clip(x, -1, 1) * 32767).astype(np.int16), VEHICLE_R_REF) for _, x in
                         _load_stems(("distracter",), cfg.sample_rate, stem_split, VEHICLE_CLS)] if p_vehicle else []
        grid = cfg.sample_rate // 10
        self.cw = ch.chain_windows(rig_recs, self.n, grid) if rig_recs else None
        self.rig_neg = np.flatnonzero(self.cw.y == 0) if rig_recs else np.empty(0, int)
        if rig_recs and not len(self.rig_neg):
            raise ValueError("the rig recordings hold no no-drone windows")

    def describe(self) -> str:
        rig_min = len(self.rig_neg) * 0.1 / 60
        count = lambda g: ", ".join(f"{k} {len(v)}" for k, v in g.items())
        src = f"{len(self.drones)} real drone recordings" if self.drones else "Pd drones"
        return (f"{src} in bed stems ({count(self.beds)}) + {rig_min:.1f} min of rig background "
                f"(bed share {self.p_rig_bed:.2f}), confusers ({count(self.confusers)}) at p={self.p_confuser}, "
                f"twin negatives {'p=%g' % self.p_twin if self.twin else 'none'}, "
                + (f"ladder families of 1 + {self.ladder}" if self.ladder else f"drone share {self.p_drone}")
                + f", SNR {self.snr_db[0]:g} to {self.snr_db[1]:g} dB"
                + (f", quiet scenes p={self.p_quiet:g} (confuser alone p={self.p_quiet_confuser:g})" if self.p_quiet else "")
                + (f", vehicle pass-bys ({len(self.vehicles)} stems) as p={self.p_vehicle:g} of confusers" if self.p_vehicle else "")
                + (f", steady beds p={self.p_steady:g} and steady confusers p={self.p_steady_confuser:g}" if self.p_steady else ""))

    def _crop(self, rng: np.random.Generator, x: np.ndarray) -> np.ndarray:
        if len(x) < self.n:
            x = np.tile(x, self.n // len(x) + 1)
        s = int(rng.integers(0, len(x) - self.n + 1))
        return x[s:s + self.n]

    @staticmethod
    def _pick(rng: np.random.Generator, groups: dict[str, list[np.ndarray]]) -> tuple[str, np.ndarray]:
        g = list(groups)[int(rng.integers(0, len(groups)))]
        return g, groups[g][int(rng.integers(0, len(groups[g])))]

    def _quiet(self, rng: np.random.Generator) -> tuple[str, np.ndarray]:
        """A near-empty bed: digital silence (a quarter), else a white or pink noise floor at a tiny level.
        The Complete Test GUI's empty layers are exact zeros, and level_normalise blows any floor up to
        full level, so the network must see these as no drone (entry 246)."""
        if rng.random() < 0.25:
            return "silence", np.zeros(self.n)
        w = rng.standard_normal(self.n)
        if rng.random() < 0.5:   # pink: 1/sqrt(f) amplitude
            f = np.fft.rfft(w)
            f[1:] /= np.sqrt(np.arange(1, len(f)))
            f[0] = 0.0
            w = np.fft.irfft(f, self.n)
        return "quiet", w / (w.std() + 1e-12) * 10.0 ** (rng.uniform(*QUIET_FLOOR_DB) / 20.0)

    def _bed(self, rng: np.random.Generator) -> tuple[str, np.ndarray]:
        if self.p_quiet and rng.random() < self.p_quiet:
            return self._quiet(rng)
        if self.p_steady and rng.random() < self.p_steady:
            return steady(rng, self.n, self.cfg.sample_rate)
        for _ in range(50):
            if rng.random() < self.p_rig_bed:
                i = self.rig_neg[int(rng.integers(0, len(self.rig_neg)))]
                g, x = "rig", np.asarray(self.cw.audio(i, self.n), dtype=np.float64)
            else:
                g, stem = self._pick(rng, self.beds)
                x = self._crop(rng, stem)
            if rr.inband_power(x) > rr.SILENCE_POWER:
                return g, x
        raise RuntimeError("could not draw a non-silent bed")

    def scene(self, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        """A no-drone scene: a bed and, with p_confuser, one confuser stem."""
        g, x = self._bed(rng)
        info = dict(bed=g, confuser="none")
        quiet = g in ("quiet", "silence")
        if rng.random() < (self.p_quiet_confuser if quiet else self.p_confuser):
            if self.p_steady and rng.random() < self.p_steady_confuser:
                cg, c = steady(rng, self.n, self.cfg.sample_rate)
            elif self.vehicles and rng.random() < self.p_vehicle:
                cg = "vehicle"
                c = real_drone(rng, self.vehicles[int(rng.integers(0, len(self.vehicles)))], self.n,
                               self.cfg.sample_rate, **VEHICLE_PATH)[0]
            else:
                cg, stem = self._pick(rng, self.confusers)
                c = self._crop(rng, stem)
            pc = rr.inband_power(c)
            if pc > rr.SILENCE_POWER:     # a crop can land in a stem's quiet stretch
                if g == "silence":        # alone: any level, level_normalise removes it
                    x = x + c / math.sqrt(pc) * 1e-2
                else:
                    db = rng.uniform(*(QUIET_CONFUSER_DB if quiet else self.confuser_db))
                    x = x + c * math.sqrt(rr.inband_power(x) * 10.0 ** (db / 10.0) / pc)
                info["confuser"] = cg
        return x, info

    def drone(self, rng: np.random.Generator) -> np.ndarray:
        if self.drones:
            return real_drone(rng, self.drones[int(rng.integers(0, len(self.drones)))], self.n, self.cfg.sample_rate)[0]
        return drone(rng, self.n, self.cfg.sample_rate)[0]

    @staticmethod
    def mix(x: np.ndarray, d: np.ndarray, snr_db: float) -> np.ndarray:
        """x plus d at an in-band SNR of d over x."""
        px = rr.inband_power(x)
        if px <= rr.SILENCE_POWER:   # digital silence: the drone alone (any level, level_normalise removes it)
            return x + d * 1e-2 / math.sqrt(max(rr.inband_power(d), 1e-20))
        return x + d * math.sqrt(px * 10.0 ** (snr_db / 10.0) / max(rr.inband_power(d), 1e-20))

    def negative(self, rng: np.random.Generator, x: np.ndarray, info: dict) -> np.ndarray:
        """The scene as a no-drone window; with a twin source, a share p_twin carries it like a drone."""
        if self.twin and rng.random() < self.p_twin:
            info["twin"] = True
            src = self.twin[int(rng.integers(0, len(self.twin)))]
            return self.mix(x, real_drone(rng, src, self.n, self.cfg.sample_rate)[0], rng.uniform(*self.snr_db))
        return x

    def window(self, rng: np.random.Generator, with_drone: bool) -> np.ndarray:
        x, info = self.scene(rng)
        if with_drone:
            return self.mix(x, self.drone(rng), rng.uniform(*self.snr_db))
        return self.negative(rng, x, info)

    def family(self, rng: np.random.Generator) -> list[tuple[np.ndarray, int, float, dict]]:
        """(audio, label, snr_db, scene info): the scene alone, then with one drone at `ladder` SNRs."""
        x, info = self.scene(rng)
        d = self.drone(rng)
        neg_info = dict(info)
        out = [(self.negative(rng, x, neg_info), 0, float("nan"), neg_info)]
        for snr in np.sort(rng.uniform(*self.snr_db, self.ladder))[::-1]:
            out.append((self.mix(x, d, float(snr)), 1, float(snr), info))
        return out

    def batch(self, index: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng([self.seed, index])
        if self.ladder:
            w = []
            while len(w) < batch_size:
                w += self.family(rng)
            w = w[:batch_size]
            audio, y = [a for a, *_ in w], np.array([t for _, t, *_ in w], dtype=np.int64)
        else:
            y = (rng.random(batch_size) < self.p_drone).astype(np.int64)
            audio = [self.window(rng, bool(t)) for t in y]
        L = np.stack([self.fe.frames(a) for a in audio])
        L = L + np.stack([rr._eq_curve(rng, L.shape[1]) for _ in y])[:, :, None]
        return rr.level_normalise(L), y


def _by_group(stems: list[tuple[str, np.ndarray]]) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {}
    for g, x in stems:
        out.setdefault(g, []).append(x)
    if not out:
        raise FileNotFoundError(f"no stems in {STEMS} (scripts/build_stems.py)")
    return out
