"""Labelled scenes: a Pure Data drone in a random soundscape (supervisor notes S12-S14, 2026-11-03).

Each scene is --seconds of 16 kHz mono audio plus a JSON file of time-stamped labels:

    drone     with probability --p-drone. DroneSoundHarm.pd through drone_sim.make_scene
              (random flight, Doppler, 1/r). Cruise fundamental f0 ~ U(100, 700) Hz, set
              through pscale; harmonic falloff ~ log-U(1, 16), the motor low-pass at
              falloff x f0 (two poles), so the harmonic ratio is random but moves with pitch.
    bed       one wind stem, always.
    events    Poisson(--mean-events) bug stems: insects, buzzing insect, mosquito, at
              --bug-db (default -24..-12 dB, against wind at -10..0 dB).
    confuser  with probability --p-confuser, exactly one helicopter stem.
              (Only wind, bugs and helicopters, at Marius's request, report entry 082.)

Non-drone stems come from data/processed/stems/ (scripts/build_stems.py), all at the
same active RMS, mixed at random gains. The drone is scaled so its RMS over its own
active samples is --snr dB (uniform) against everything else over the same samples.
The mix is then scaled to -26 dBFS RMS, never above -0.1 dBFS peak.

Labels are one event per source: {label, role, onset, offset}, seconds in the scene;
the label is the stem class. A drone event runs from its first arriving sound to the
end of its flight (impact) or of the scene.

--drone-only: every scene is one Pd drone alone, labelled with its event and parameters.

Stems from fsd50k_eval belong to a held-out test source (docs/datasets.md): a scene that
uses one is marked heldout=1. --sources fsd50k_dev keeps them out entirely.

    uv run python scripts/make_scenes.py --n 100 --out data/processed/scenes/run1
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
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
PD_DIR = ROOT / "Pure-data+Script" / "EnviromentTest"
STEMS = ROOT / "data" / "processed" / "stems" / "manifest.csv"
FS = 16000

ROLES = {
    "bed": ["wind"],
    "event": ["insects", "buzzing_insect", "mosquito"],
    "confuser": ["helicopter"],
}
GAIN_DB = {"bed": (-10, 0), "event": (-6, 6), "confuser": (-6, 6)}


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


def mix(noise: list[np.ndarray], drone: np.ndarray | None, snr_db: float | None):
    """Sum the noise tracks, add the drone at snr_db over its active samples, level the mix.
    No noise tracks: the drone alone (snr_db unused)."""
    total = np.sum(noise, axis=0) if noise else np.zeros(len(drone))
    if drone is not None:
        g = 1.0
        if noise:
            active = drone != 0
            g = np.sqrt(np.mean(total[active] ** 2) / np.mean(drone[active] ** 2)) * 10 ** (snr_db / 20)
        total = total + g * drone
    scale = min(10 ** (-26 / 20) / np.sqrt(np.mean(total ** 2)), 10 ** (-0.1 / 20) / np.abs(total).max())
    return (total * scale).astype(np.float32)


def render_drone(seed: int, rng, n: int, f0_range=(100, 700), falloff_range=(1, 16)):
    """Pd drone flight at 16 kHz, cropped or placed in n samples; params and (onset, offset)."""
    from drone_sim import make_scene   # imports render.py, which looks for Pd

    f0 = rng.uniform(*f0_range)
    base = rng.uniform(50, 70)                       # cruise throttle, as make_scene draws it
    falloff = float(np.exp(rng.uniform(*np.log(falloff_range))))
    pscale = f0 / (150 + 180 * base / 127)           # the patch's motor pitch at that throttle
    _, y, fs, p, _ = make_scene(seed, base=base, pscale=pscale, falloff=falloff)
    assert fs == 44100, fs
    y = resample_poly(y, 160, 441)
    first = int(np.flatnonzero(np.abs(y) > 1e-12 * np.abs(y).max())[0])
    y[:first] = 0.0                                  # resampling ringing before the first sound
    track, lo, hi = place(y, n, rng, fill=False)
    on = np.flatnonzero(track)
    params = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()}
    params.update(f0_cruise=round(f0, 1), falloff=round(falloff, 3))
    return track, params, (int(on[0]), int(on[-1]) + 1)


def make_one(i: int, seed: int, pools, args) -> dict:
    rng = np.random.default_rng(seed)
    n = int(args.seconds * FS)
    noise, events = [], []

    def add(role: str, stem: dict):
        x = read_16k(stem["file"]) * 10 ** (rng.uniform(*(args.bug_db if role == "event" else GAIN_DB[role])) / 20)
        track, lo, hi = place(x, n, rng, fill=(role == "bed"))
        noise.append(track)
        events.append(dict(label=stem["cls"], role=role, onset=round(lo / FS, 3), offset=round(hi / FS, 3),
                           stem=stem["file"], source=stem["source"], license=stem["license"]))

    pick = lambda role: pools[role][int(rng.integers(len(pools[role])))]  # noqa: E731
    if not args.drone_only:
        add("bed", pick("bed"))
        for _ in range(rng.poisson(args.mean_events)):
            add("event", pick("event"))
        if rng.random() < args.p_confuser:
            add("confuser", pick("confuser"))

    drone, drone_params, snr = None, None, None
    if args.drone_only or rng.random() < args.p_drone:
        snr = None if args.drone_only else float(rng.uniform(*args.snr))
        drone, drone_params, (lo, hi) = render_drone(seed, rng, n, args.f0, args.falloff)
        events.insert(0, dict(label="drone", role="drone", onset=round(lo / FS, 3), offset=round(hi / FS, 3),
                              source="pd:DroneSoundHarm.pd"))

    name = f"scene_{i:06d}"
    sf.write(args.out / "audio" / f"{name}.wav", mix(noise, drone, snr), FS)
    label = dict(scene=name, seed=seed, fs=FS, seconds=args.seconds, snr_db=None if snr is None else round(snr, 2),
                 heldout=any(e["source"] == "fsd50k_eval" for e in events), events=events, drone=drone_params)
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
    ap.add_argument("--snr", type=float, nargs=2, default=(-20.0, 10.0))
    ap.add_argument("--f0", type=float, nargs=2, default=(100.0, 700.0), help="cruise fundamental range, Hz")
    ap.add_argument("--falloff", type=float, nargs=2, default=(1.0, 16.0), help="harmonic falloff range (log-uniform)")
    ap.add_argument("--bug-db", type=float, nargs=2, default=(-24.0, -12.0),
                    help="gain of insects, buzzing insect and mosquito stems, dB (others: bed -10..0, helicopter -6..+6)")
    ap.add_argument("--drone-only", action="store_true",
                    help="every scene is one Pd drone and nothing else (no bed, bugs or helicopter)")
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
