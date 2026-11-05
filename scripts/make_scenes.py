"""Labelled scenes: a Pure Data drone in a random soundscape (supervisor notes S12-S14, 2026-11-03).

Each scene is --seconds of 16 kHz mono audio plus a JSON file of time-stamped labels:

    drone     with probability --p-drone. DroneSoundHarm.pd through drone_sim.make_scene
              (random flight, Doppler, 1/r). Cruise fundamental f0 ~ U(100, 700) Hz, set
              through pscale; harmonic falloff ~ log-U(1, 16), the motor low-pass at
              falloff x f0 (two poles), so the harmonic ratio is random but moves with pitch.
    bed       one ambience stem, always: wind, rain, stream or traffic, at -10..0 dB.
    events    Poisson(--mean-events) quiet stems: insects, buzzing insect, mosquito, birds,
              at --bug-db (default -24..-12 dB).
    transient Poisson(--mean-transients) loud short stems: gunfire, machine gun, small arms,
              mortar and artillery, explosions near and far, thunder, shouting, dog, at
              -6..+12 dB. They mask the frames they fall on (the case the CRNN is for).
    confuser  with probability --p-confuser, exactly one engine or tonal stem: helicopter,
              truck, car, motorcycle, scooter, aircraft, prop plane, jet, tank, generator,
              idling engine, mower, chainsaw, power tool, siren, air-raid siren, at -6..+6 dB.
              (Supervisor's list, report entry 097; was wind, bugs and helicopters only.)

Non-drone stems come from data/processed/stems/ (scripts/build_stems.py), all at the
same active RMS, mixed at random gains. The drone is scaled so its RMS over its own
active samples is --snr dB (uniform) against the bed, events and confuser over the same
samples; transients are added after, so a boom masks the drone without turning it down.
The mix is then scaled to -26 dBFS RMS, never above -0.1 dBFS peak.

Labels are one event per source: {label, role, onset, offset}, seconds in the scene;
the label is the stem class. A drone event runs from its first arriving sound to the
end of its flight (impact) or of the scene.

--drone-only: every scene is one Pd drone alone, labelled with its event and parameters.

--realism (report: the synthetic-to-real gap): each drone gets its own fingerprint, and each
scene its own recording chain. Drone, drawn per drone:
    detune    each of the 4 motors offset by U(-5, +5) % of the cruise pitch (Pd det0..det3;
              without it every drone has the patch's fixed 0, +2.3, -3.7, +5.1 Hz)
    noise     extra prop noise at U(-25, -3) dB against the motor tone, band-passed at
              log-U(300, 4000) Hz with Q U(0.5, 2), following the throttle (drone_sim.prop_noise)
    ground    a reflection off flat ground, the microphone U(0.5, 3) m up, U(-8, -1) dB
Recording chain, drawn per seed and applied to the whole mix, so a drone scene and the no-drone
scene of the same seed get the same chain and the chain says nothing about the label:
    mic       high-pass at log-U(30, 200) Hz (2 poles) and a smooth random EQ: U(-6, +6) dB at
              6 points spread log-evenly over 50-8000 Hz, interpolated in log frequency
    room      a short reverb: exponentially decaying noise, RT60 U(0.05, 0.5) s, U(-25, -8) dB
The mix is levelled again after the chain (-26 dBFS RMS, peak at most -0.1 dBFS).

--real-drones DIR (REQUIREMENTS.md: Glasgow drones d1-d16 train, d17-d20 validate, d21-d24 test;
report entries 106-107): with probability --p-real a drone scene flies a real Glasgow hover
recording (1 m or 5 m, a random start, looped) instead of the Pd drone, through the same flight,
Doppler, 1/r and, with --realism, ground reflection and chain. The drone number comes from
--drone-ids, or from --val-drone-ids for seeds >= --val-seed, so validation drones never train.
The choice uses its own rng: a Pd scene is the same with or without --real-drones.
--steady P (report entry 116; master's iteration 11, entry 100, uses the same family): with
probability P a scene gets a steady broadband noise bed: Gaussian noise with a random spectral slope
U(-12, +3) dB/octave and a smooth random EQ (U(-6, +6) dB at 6 log-even points, 50-8000 Hz), with a
slow level wobble (0.2-2 Hz, +-3 dB), at --steady-db against a stem. It is never the war spectrum
(war_gain.txt is the test background). Drawn from rng [seed, 6], so the drone and no-drone scenes of
one seed get the same noise; it is part of the SNR reference.

--room-noise FILE adds an excerpt of that recording (the Glasgow room without a drone) to every
scene, drone or not, at --room-db against a stem (random start; drone and no-drone scenes of one
seed get the same excerpt), so the room is never a sign of the drone.

    uv run python scripts/make_scenes.py --realism --real-drones data/extracted/Glasgow --p-real 0.5 \\
        --room-noise data/extracted/Glasgow/noise/_uav_2022noise_n_noise_1m_.WAV --n 2160 --p-drone 1 ...

Stems from fsd50k_eval belong to a held-out test source (docs/datasets.md): a scene that
uses one is marked heldout=1. --sources fsd50k_dev keeps them out entirely.

    uv run python scripts/make_scenes.py --n 100 --out data/processed/scenes/run1
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
PD_DIR = ROOT / "Pure-data+Script" / "EnviromentTest"
STEMS = ROOT / "data" / "processed" / "stems" / "manifest.csv"
FS = 16000

ROLES = {
    "bed": ["wind", "rain", "stream", "traffic"],
    "event": ["insects", "buzzing_insect", "mosquito", "birds"],
    "transient": ["gunfire", "machine_gun", "small_arms", "mortar_artillery", "explosion",
                  "explosion_far", "thunder", "shout", "dog"],
    "confuser": ["helicopter", "truck", "car", "motorcycle", "scooter", "aircraft", "prop_plane",
                 "jet", "tank", "generator", "idling", "mower", "chainsaw", "power_tool", "siren",
                 "air_raid_siren"],
}
GAIN_DB = {"bed": (-10, 0), "event": (-6, 6), "transient": (-6, 12), "confuser": (-6, 6)}


def load_stems(sources: set[str] | None) -> dict[str, list[dict]]:
    rows = [r for r in csv.DictReader(STEMS.open()) if sources is None or r["source"] in sources]
    pools = {role: [r for r in rows if r["cls"] in classes] for role, classes in ROLES.items()}
    for role, pool in pools.items():
        if not pool:
            raise SystemExit(f"no {role} stems left after --sources {sorted(sources)}")
    return pools


@functools.lru_cache(maxsize=None)
def read_16k(path: str) -> np.ndarray:
    x, fs = sf.read(ROOT / path, dtype="float64", always_2d=True)
    assert fs == 44100, (path, fs)
    return resample_poly(x.mean(axis=1), 160, 441)


def place(x: np.ndarray, n: int, rng, fill: bool) -> tuple[np.ndarray, int, int]:
    """x in an n-sample track: tiled from a random start (fill), or at a random onset."""
    if fill:
        # ponytail: tiling clicks at the seams of short beds; crossfade if it is audible
        start = int(rng.integers(len(x)))
        return np.resize(np.roll(x, -start), n), 0, n
    if len(x) >= n:
        start = int(rng.integers(len(x) - n + 1))
        return x[start:start + n].copy(), 0, n
    onset = int(rng.integers(n - len(x) + 1))
    track = np.zeros(n)
    track[onset:onset + len(x)] = x
    return track, onset, onset + len(x)


def mix(noise: list[np.ndarray], drone: np.ndarray | None, snr_db: float | None, extra=()):
    """Sum the noise tracks, add the drone at snr_db over its active samples, add the extra
    tracks (transients: not part of the SNR reference), level the mix.
    No noise tracks: the drone alone (snr_db unused)."""
    total = np.sum(noise, axis=0) if noise else np.zeros(len(drone))
    if drone is not None:
        g = 1.0
        if noise:
            active = drone != 0
            g = np.sqrt(np.mean(total[active] ** 2) / np.mean(drone[active] ** 2)) * 10 ** (snr_db / 20)
        total = total + g * drone
    for x in extra:
        total = total + x
    return level(total)


def realism_params(f0: float, rng) -> dict:
    """--realism: this drone's own motor detune, prop-noise band and ground reflection."""
    return dict(detune=[round(float(d), 3) for d in f0 * rng.uniform(-0.05, 0.05, 4)],
                noise_db=float(rng.uniform(-25, -3)), noise_fc=float(np.exp(rng.uniform(np.log(300), np.log(4000)))),
                noise_q=float(rng.uniform(0.5, 2)), mic_h=float(rng.uniform(0.5, 3)), refl_db=float(rng.uniform(-8, -1)))


def drone_ids(text: str) -> set[int]:
    """'1-16' or '17,18,20' or '1-4,9' -> the drone numbers."""
    out = set()
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out |= set(range(int(lo), int(hi or lo) + 1))
    return out


@functools.lru_cache(maxsize=None)
def glasgow_files(root: str) -> dict[int, list[str]]:
    """Drone number -> its hover recordings (1 m and 5 m) under the Glasgow folder."""
    out: dict[int, list[str]] = {}
    for f in sorted(Path(root).rglob("*.[wW][aA][vV]")):
        m = re.search(r"_d(\d+)_hover_\d+m_", f.name)
        if m:
            out.setdefault(int(m.group(1)), []).append(str(f))
    return out


@functools.lru_cache(maxsize=64)
def read_44k(path: str) -> np.ndarray:
    x, fs = sf.read(path, dtype="float64", always_2d=True)
    assert fs == 44100, (path, fs)
    return x[:, 0]


def real_drone(seed: int, args) -> str | None:
    """--real-drones: the recording this seed's drone flies, or None for the Pd drone."""
    if not args.real_drones:
        return None
    r = np.random.default_rng([seed, 4])
    if r.random() >= args.p_real:
        return None
    ids = drone_ids(args.val_drone_ids if seed >= args.val_seed else args.drone_ids)
    files = [f for d, fs in glasgow_files(str(args.real_drones)).items() if d in ids for f in fs]
    if not files:
        raise SystemExit(f"no Glasgow hover recordings for drones {sorted(ids)} under {args.real_drones}")
    return files[int(r.integers(len(files)))]


def render_drone(seed: int, rng, n: int, f0_range=(100, 700), falloff_range=(1, 16), realism=False, real=None):
    """Pd drone flight (or, with real, that recording flown) at 16 kHz, cropped or placed in n
    samples; params and (onset, offset)."""
    from drone_sim import make_scene   # imports render.py, which looks for Pd

    f0 = rng.uniform(*f0_range)
    base = rng.uniform(50, 70)                       # cruise throttle, as make_scene draws it
    falloff = float(np.exp(rng.uniform(*np.log(falloff_range))))
    pscale = f0 / (150 + 180 * base / 127)           # the patch's motor pitch at that throttle
    extra = realism_params(f0, rng) if realism else {}
    if real is not None:                             # a recorded drone has its own pitch, rotors and noise
        extra = {k: v for k, v in extra.items() if k in ("mic_h", "refl_db")}
        extra["dry"] = read_44k(real)
    _, y, fs, p, _ = make_scene(seed, base=base, pscale=pscale, falloff=falloff, **extra)
    assert fs == 44100, fs
    y = resample_poly(y, 160, 441)
    first = int(np.flatnonzero(np.abs(y) > 1e-12 * np.abs(y).max())[0])
    y[:first] = 0.0                                  # resampling ringing before the first sound
    track, lo, hi = place(y, n, rng, fill=False)
    on = np.flatnonzero(track)
    params = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()}
    params.update(f0_cruise=round(f0, 1), falloff=round(falloff, 3))
    if real is not None:                             # f0 / falloff / pscale did not reach the sound
        params.update(real=str(Path(real).name), f0_cruise=None, falloff=None)
    return track, params, (int(on[0]), int(on[-1]) + 1)


@functools.lru_cache(maxsize=1)
def room_16k(path: str) -> np.ndarray:
    x, fs = sf.read(path, dtype="float64", always_2d=True)
    return resample_poly(x[:, 0], 160, 441) if fs == 44100 else resample_poly(x[:, 0], FS, fs)


def steady_noise(seed: int, n: int, fs: int = FS) -> tuple[np.ndarray, dict]:
    """--steady: unit-RMS broadband noise with a random slope, smooth EQ and slow wobble; its params."""
    r = np.random.default_rng([seed, 6])
    slope, eq = float(r.uniform(-12, 3)), [round(float(g), 2) for g in r.uniform(-6, 6, 6)]
    rate, phase = float(r.uniform(0.2, 2.0)), float(r.uniform(0, 2 * np.pi))
    f = np.fft.rfftfreq(n, 1 / fs)
    lf = np.log2(np.maximum(f, 50.0) / 1000.0)                         # octaves from 1 kHz
    g_db = slope * lf + np.interp(np.log(np.maximum(f, 50.0)), np.log(np.geomspace(50, 8000, 6)), eq)
    x = np.fft.irfft(np.fft.rfft(r.standard_normal(n)) * 10 ** (g_db / 20), n)
    x *= 10 ** (3 / 20 * np.sin(2 * np.pi * rate * np.arange(n) / fs + phase))
    return x / np.sqrt(np.mean(x ** 2)), dict(slope_db_oct=round(slope, 2), eq_db=eq, wobble_hz=round(rate, 3))


def room_excerpt(seed: int, n: int, args) -> tuple[np.ndarray, float]:
    """--room-noise: n samples from a random start, scaled to a stem's level (-26 dBFS) + room_db."""
    r = np.random.default_rng([seed, 3])
    x = room_16k(str(args.room_noise))
    s = int(r.integers(len(x) - n))
    g = float(r.uniform(*args.room_db))
    seg = x[s:s + n]
    return seg / np.sqrt(np.mean(seg ** 2)) * 10 ** ((-26 + g) / 20), g


def level(x: np.ndarray) -> np.ndarray:
    """-26 dBFS RMS, never above -0.1 dBFS peak."""
    scale = min(10 ** (-26 / 20) / np.sqrt(np.mean(x ** 2)), 10 ** (-0.1 / 20) / np.abs(x).max())
    return (x * scale).astype(np.float32)


def chain_params(rng) -> dict:
    """--realism: one recording chain (microphone and room), drawn from the scene's seed."""
    return dict(hp_hz=float(np.exp(rng.uniform(np.log(30), np.log(200)))),
                eq_db=[round(float(g), 2) for g in rng.uniform(-6, 6, 6)],
                rt60_s=float(rng.uniform(0.05, 0.5)), room_db=float(rng.uniform(-25, -8)),
                room_seed=int(rng.integers(2 ** 31)))


def apply_chain(x: np.ndarray, c: dict, fs: int = FS) -> np.ndarray:
    """High-pass, smooth EQ (c["eq_db"] at log-even points 50-8000 Hz), then a decaying-noise reverb."""
    from scipy.signal import butter, fftconvolve, sosfilt
    y = sosfilt(butter(2, c["hp_hz"], btype="high", fs=fs, output="sos"), x)
    f = np.fft.rfftfreq(2 * len(y), 1 / fs)                       # zero-padded: no wrap-around
    pts = np.log(np.geomspace(50, 8000, len(c["eq_db"])))
    g = 10 ** (np.interp(np.log(np.maximum(f, 1.0)), pts, c["eq_db"]) / 20)
    y = np.fft.irfft(np.fft.rfft(y, 2 * len(y)) * g)[:len(y)]
    t = np.arange(int(c["rt60_s"] * fs)) / fs
    ir = np.random.default_rng(c["room_seed"]).standard_normal(len(t)) * 10 ** (-3 * t / c["rt60_s"])
    wet = fftconvolve(y, ir)[:len(y)]
    return y + wet * np.sqrt(np.mean(y ** 2) / max(np.mean(wet ** 2), 1e-30)) * 10 ** (c["room_db"] / 20)


def make_one(i: int, seed: int, pools, args) -> dict:
    rng = np.random.default_rng(seed)
    n = int(args.seconds * FS)
    noise, transients, events = [], [], []

    def add(role: str, stem: dict):
        x = read_16k(stem["file"]) * 10 ** (rng.uniform(*(args.bug_db if role == "event" else GAIN_DB[role])) / 20)
        track, lo, hi = place(x, n, rng, fill=(role == "bed"))
        (transients if role == "transient" else noise).append(track)
        events.append(dict(label=stem["cls"], role=role, onset=round(lo / FS, 3), offset=round(hi / FS, 3),
                           stem=stem["file"], source=stem["source"], license=stem["license"]))

    pick = lambda role: pools[role][int(rng.integers(len(pools[role])))]  # noqa: E731
    if args.room_noise:
        seg, g = room_excerpt(seed, n, args)
        noise.append(seg)
        events.append(dict(label="glasgow_room", role="bed", onset=0.0, offset=round(n / FS, 3),
                           source="glasgow:noise", gain_db=round(g, 2)))
    if args.steady and np.random.default_rng([seed, 7]).random() < args.steady:
        x, sp = steady_noise(seed, n)
        g = float(np.random.default_rng([seed, 8]).uniform(*args.steady_db))
        noise.append(x * 10 ** ((-26 + g) / 20))
        events.append(dict(label="steady_noise", role="bed", onset=0.0, offset=round(n / FS, 3),
                           source="synthetic", gain_db=round(g, 2), **sp))
    if not args.drone_only:
        add("bed", pick("bed"))
        for _ in range(rng.poisson(args.mean_events)):
            add("event", pick("event"))
        for _ in range(rng.poisson(args.mean_transients)):
            add("transient", pick("transient"))
        if rng.random() < args.p_confuser:
            add("confuser", pick("confuser"))

    drone, drone_params, snr = None, None, None
    if args.drone_only or rng.random() < args.p_drone:
        snr = None if args.drone_only else float(rng.uniform(*args.snr))
        real = real_drone(seed, args)
        drone, drone_params, (lo, hi) = render_drone(seed, rng, n, args.f0, args.falloff, args.realism, real)
        events.insert(0, dict(label="drone", role="drone", onset=round(lo / FS, 3), offset=round(hi / FS, 3),
                              source="pd:DroneSoundHarm.pd" if real is None else
                              f"glasgow:{Path(real).parent.name}/{Path(real).name}"))

    name = f"scene_{i:06d}"
    out, chain = mix(noise, drone, snr, transients), None
    if args.realism:                       # its own rng: drone and no-drone scenes of one seed share it
        chain = chain_params(np.random.default_rng([seed, 2]))
        out = level(apply_chain(out.astype(np.float64), chain))
    sf.write(args.out / "audio" / f"{name}.wav", out, FS)
    label = dict(scene=name, seed=seed, fs=FS, seconds=args.seconds, snr_db=None if snr is None else round(snr, 2),
                 heldout=any(e["source"] == "fsd50k_eval" for e in events), events=events, drone=drone_params,
                 chain=chain)
    (args.out / "labels" / f"{name}.json").write_text(json.dumps(label, indent=1))
    return dict(scene=name, seed=seed, has_drone=int(drone is not None),
                has_confuser=int(any(e["role"] == "confuser" for e in events)),
                labels=";".join(sorted({e["label"] for e in events})), snr_db=label["snr_db"],
                f0_cruise=drone_params and drone_params["f0_cruise"],
                falloff=drone_params and drone_params["falloff"], heldout=int(label["heldout"]))


def write_manifest(out: Path, rows: list[dict]) -> None:
    with (out / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "scenes" / "run1")
    ap.add_argument("--seed", type=int, default=0, help="scene i uses seed + i")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--p-drone", type=float, default=0.75)
    ap.add_argument("--p-confuser", type=float, default=0.4)
    ap.add_argument("--mean-events", type=float, default=2.0)
    ap.add_argument("--mean-transients", type=float, default=1.0,
                    help="mean number of gunfire / explosion / shout ... stems per scene (Poisson)")
    ap.add_argument("--snr", type=float, nargs=2, default=(-20.0, 10.0))
    ap.add_argument("--f0", type=float, nargs=2, default=(100.0, 700.0), help="cruise fundamental range, Hz")
    ap.add_argument("--falloff", type=float, nargs=2, default=(1.0, 16.0), help="harmonic falloff range (log-uniform)")
    ap.add_argument("--bug-db", type=float, nargs=2, default=(-24.0, -12.0),
                    help="gain of the quiet event stems, dB (others: bed -10..0, transient -6..+12, confuser -6..+6)")
    ap.add_argument("--drone-only", action="store_true",
                    help="every scene is one Pd drone and nothing else (no bed, bugs or helicopter)")
    ap.add_argument("--realism", action="store_true",
                    help="per-drone motor detune, prop-noise band and ground reflection; per-seed mic and room chain")
    ap.add_argument("--real-drones", type=Path, default=None, help="the Glasgow folder (data/extracted/Glasgow)")
    ap.add_argument("--p-real", type=float, default=0.5, help="share of drone scenes with a real drone")
    ap.add_argument("--drone-ids", default="1-16", help="Glasgow drones for seeds < --val-seed")
    ap.add_argument("--val-drone-ids", default="17-20", help="Glasgow drones for seeds >= --val-seed")
    ap.add_argument("--val-seed", type=int, default=1800, help="as train_clean.py: seeds from here validate")
    ap.add_argument("--steady", type=float, default=0.0, help="probability of a steady broadband noise bed")
    ap.add_argument("--steady-db", type=float, nargs=2, default=(-6.0, 12.0), help="its level against a stem, dB")
    ap.add_argument("--room-noise", type=Path, default=None, help="a drone-free room recording added to every scene")
    ap.add_argument("--room-db", type=float, nargs=2, default=(-20.0, -5.0), help="its level against a stem, dB")
    ap.add_argument("--sources", default=None, help="comma list of stem sources, e.g. fsd50k_dev (default: all)")
    return ap


def check(args) -> None:
    """Raise ValueError for ranges the notes or the patch rule out."""
    if not -40 <= args.snr[0] <= args.snr[1]:
        raise ValueError("SNR needs -40 <= min <= max (notes: never below -40 dB)")
    if not 20 <= args.f0[0] <= args.f0[1] <= 2000:
        raise ValueError("f0 needs 20 <= min <= max <= 2000 Hz")
    if not args.bug_db[0] <= args.bug_db[1]:
        raise ValueError("bug level needs min <= max")
    if not 0 < args.falloff[0] <= args.falloff[1]:
        raise ValueError("falloff needs 0 < min <= max")
    if args.mean_events < 0 or args.mean_transients < 0:
        raise ValueError("mean event counts are >= 0")
    if not 0 <= args.steady <= 1:
        raise ValueError("--steady is 0 to 1")
    if not 0 <= args.p_real <= 1:
        raise ValueError("--p-real is 0 to 1")
    if args.real_drones and drone_ids(args.drone_ids) & drone_ids(args.val_drone_ids):
        raise ValueError("--drone-ids and --val-drone-ids overlap: a validation drone would train")
    if not (0 <= args.p_drone <= 1 and 0 <= args.p_confuser <= 1):
        raise ValueError("probabilities are 0 to 1")


sys.path.insert(0, str(PD_DIR))


def main() -> None:
    args = parser().parse_args()
    try:
        check(args)
    except ValueError as e:
        raise SystemExit(str(e))

    pools = None if args.drone_only else load_stems(set(args.sources.split(",")) if args.sources else None)
    for sub in ("audio", "labels"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.n):
        rows.append(make_one(i, args.seed + i, pools, args))
        print(rows[-1])
    write_manifest(args.out, rows)


if __name__ == "__main__":
    main()
