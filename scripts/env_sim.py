"""Sound-environment simulator: 20 s battlefield environments without a drone (iteration 7).

Three layers, each set on the 1-10 scale (1 = off):

    N  war, step 1-10        a 20 s window of the fixed 10 min war track
                             (scripts/war_track.py, entry 062), at a gain:
                             10 = as rendered (a microphone near the front),
                             each step down -5 dB, so 2 = -40 dB    --war
    O  war offset, s         where the window starts in the track   --offset
    W  wind, step 1-10       microphone wind, looped; 2..10 = 40..80 dB SPL
                             active RMS (step_db)                    --wind
    D  distracters           one distance step 1-10 per class (1 = off,
                             2 = far, 10 = near; scripts/distracters.py,
                             entry 063), e.g. "car:6" or "mower:4,birds:8";
                             "none" for no distracter                --distracters
    L  length, s (20)                                                --length

Each environment has its own seed (--seed + its id), which draws each
distracter's stem, pitch and start; the CSV records it, so an environment can
be rebuilt from its row.

Output, ready for a drone to be added: mono, 44.1 kHz, 32-bit float WAV,
1.0 = 110 dB SPL, the calibration of the drone-only renders
(Pure-data+Script/EnviromentTest/drone_only.py). A drone clip adds to an
environment sample for sample, at any offset, and the levels stay in dB SPL.
Near weapons peak above 1.0, which the float file keeps (`clipped` in the CSV
says so). No drone is added here.

    uv run python scripts/env_sim.py --out artifacts/environment/env1

Writes env_NNN.wav and environments.csv: one row per file with its
parameters and seed, the war events inside the window (times relative to the
window), every distracter's stem, distance, pitch, start and level, and the
measured level and peak of the mix.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
STEMS = ROOT / "data" / "processed" / "stems"
WAR = ROOT / "data" / "processed" / "war"
FS = 44100
P_REF = 20e-6
FULL_SCALE_DB = 110.0                    # as drone_sim.py: 1.0 in the WAV = 110 dB SPL
FULL = P_REF * 10 ** (FULL_SCALE_DB / 20)
# "Strong wind directly on a microphone", CC0: wind on the capsule, 90% of its energy below 100 Hz (entry 057)
WIND = "wind/wind/fsd50k_eval_382591.wav"
STEPS = list(range(1, 11))
# Grid defaults. Offsets every 100 s cover the calm, intense and normal phases of the seed-0 track.
OFFSETS = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]


def step_db(step) -> float | None:
    """Wind: 1 = off, 2..10 = 40, 45, ..., 80 dB SPL (Mads, 2026-10-26)."""
    s = int(step)
    if not 1 <= s <= 10:
        raise ValueError(f"level step must be 1-10, got {step}")
    return None if s == 1 else 40.0 + 5.0 * (s - 2)


def war_gain_db(step) -> float | None:
    """War: 1 = off, 10 = the track as rendered, -5 dB per step down."""
    s = int(step)
    if not 1 <= s <= 10:
        raise ValueError(f"war step must be 1-10, got {step}")
    return None if s == 1 else 5.0 * (s - 10)


def active_rms(x):
    n = 1764                                                    # 40 ms frames
    e = np.mean(x[: len(x) // n * n].reshape(-1, n) ** 2, axis=1)
    return np.sqrt(np.mean(e[e > e.max() * 0.01]))             # frames within 20 dB of the loudest


def loop(x, n, fade=int(0.05 * FS)):
    """x repeated to n samples, each seam a linear crossfade."""
    out = x.copy()
    ramp = np.linspace(0, 1, fade)
    while len(out) < n:
        out = np.concatenate([out[:-fade], out[-fade:] * (1 - ramp) + x[:fade] * ramp, x[fade:]])
    return out[:n]


def parse_distracters(spec: str) -> dict:
    """'none' -> {}; 'car:6,mower:4' -> {'car': 6, 'mower': 4}."""
    if spec in ("", "none"):
        return {}
    return {c: int(s) for c, s in (p.split(":") for p in spec.split(","))}


def load(war_dir=WAR):
    """The war track (pascals), its events and phases, the wind stem and the distracter stems."""
    import distracters as dt                                     # here: distracters imports this module
    x, fs = sf.read(war_dir / "war_track.wav")
    if fs != FS:
        raise SystemExit(f"war track at {fs} Hz, want {FS}")
    rows = lambda name: list(csv.DictReader(open(war_dir / name, encoding="utf-8")))  # noqa: E731
    return dict(war=x * FULL, war_events=rows("events.csv"), phases=rows("phases.csv"),
                wind=sf.read(STEMS / WIND)[0], distracters=dt.load_stems())


def war_window(src, offset, n):
    """The n samples of the war track from `offset` s (pascals), and the events and phases inside them."""
    i = int(round(offset * FS))
    if not 0 <= i <= len(src["war"]) - n:
        raise ValueError(f"offset {offset} s: the window must fit in the {len(src['war']) / FS:g} s track")
    t0, t1 = i / FS, (i + n) / FS
    ev = [dict(cls=e["cls"], start_s=round(max(float(e["start_s"]), t0) - t0, 3),
               end_s=round(min(float(e["end_s"]), t1) - t0, 3), distance_m=float(e["distance_m"]),
               peak_at_mic_db=e["peak_at_mic_db"])
          for e in src["war_events"] if float(e["start_s"]) < t1 and float(e["end_s"]) > t0]
    ph = sorted({p["phase"] for p in src["phases"] if float(p["start_s"]) < t1 and float(p["end_s"]) > t0})
    return src["war"][i:i + n], ev, ph


def build(war_step, offset, wind_step, distracters: dict, n, seed, src):
    """(mix in pascals, layers, info). layers: name -> pascals, so a GUI can rescale them."""
    import distracters as dt
    rng = np.random.default_rng(seed)
    layers, info = {}, dict(war_events=[], war_phases=[], distracters=[])
    g = war_gain_db(war_step)
    if g is not None:
        w, info["war_events"], info["war_phases"] = war_window(src, offset, n)
        layers["war"] = w * 10 ** (g / 20)
        for e in info["war_events"]:
            e["peak_at_mic_db"] = round(float(e["peak_at_mic_db"]) + g, 1) if e["peak_at_mic_db"] else None
    db = step_db(wind_step)
    if db is not None:
        layers["wind"] = loop(src["wind"] / active_rms(src["wind"]) * P_REF * 10 ** (db / 20), n)
    for cls in sorted(distracters):
        x, row = dt.render(cls, distracters[cls], n / FS, rng, src["distracters"])
        if row:
            layers[cls] = x
            info["distracters"].append(row)
    mix = sum(layers.values(), np.zeros(n))
    return mix, layers, info


def describe_mix(mix):
    rms = np.sqrt(np.mean(mix ** 2))
    return dict(mix_rms_db_spl=round(20 * np.log10(rms / P_REF), 1) if rms > 0 else None,
                peak_dbfs=round(20 * np.log10(np.abs(mix).max() / FULL), 1) if rms > 0 else None,
                clipped=bool(np.abs(mix).max() > FULL))


def row_for(k, name, L, seed, war_step, offset, wind_step, distracters, mix, info):
    import distracters as dt
    g, db = war_gain_db(war_step), step_db(wind_step)
    return dict(env_id=k, file=name, L_s=L, fs=FS, seed=seed, N_war_step=war_step,
                N_war_gain_db="off" if g is None else g, O_offset_s=offset, war_phases="+".join(info["war_phases"]),
                W_wind_step=wind_step, W_wind_db="off" if db is None else db,
                **{f"D_{c}_step": distracters.get(c, 1) for c in dt.CATALOGUE},
                **describe_mix(mix), n_war_events=len(info["war_events"]),
                war_events=json.dumps(info["war_events"]), distracters=json.dumps(info["distracters"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--war", nargs="+", type=int, default=[1, 6, 10], help="N, step 1-10 (1 = off)")
    ap.add_argument("--offset", nargs="+", type=float, default=[150.0, 450.0], help="O, s into the war track")
    ap.add_argument("--wind", nargs="+", type=int, default=[1, 5], help="W, step 1-10 (1 = off)")
    ap.add_argument("--distracters", nargs="+", default=None,
                    help='D, one spec per environment axis value: "none", "car:6", "mower:4,birds:8" '
                         "(default: none, and each class alone at step 6)")
    ap.add_argument("--length", type=float, default=20.0, help="L, s")
    ap.add_argument("--seed", type=int, default=0, help="environment k uses seed + k")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "environment" / "run")
    a = ap.parse_args()

    import distracters as dt
    specs = a.distracters or ["none"] + [f"{c}:6" for c in dt.CATALOGUE]
    src = load()
    n = int(a.length * FS)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, (N, O, W, D) in enumerate(itertools.product(a.war, a.offset, a.wind, specs), start=1):
        seed = a.seed + k
        mix, _, info = build(N, O, W, parse_distracters(D), n, seed, src)
        name = f"env_{k:03d}.wav"
        sf.write(a.out / name, (mix / FULL).astype(np.float32), FS, subtype="FLOAT")
        rows.append(row_for(k, name, a.length, seed, N, O, W, parse_distracters(D), mix, info))
    with open(a.out / "environments.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} environments of {a.length:g} s -> {a.out}")


if __name__ == "__main__":
    main()
