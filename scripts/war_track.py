"""The war track (iteration 7): one fixed 10 min recording of a war zone, no drone.

Everything heard in a war zone, from the war stems (data/processed/stems/war/,
report entry 060), at random times, distances and pitches:

    * each event class has a base rate per minute, a distance range
      (log-uniform), a pitch range and a level at 1 m (CATALOGUE);
    * the rates follow phases of 30-120 s: calm (x0.3), normal (x1) and
      intense (x2.5), so the track has lulls and firefights;
    * times are a Poisson process, thinned by the phase (one seed fixes all);
    * each event is a piece of one of its class's stems, pitched, set to its
      level at 1 m, and carried to the microphone by dad.propagation
      (spreading and air absorption, ISO 9613-1).

Levels at 1 m are calibration knobs. Impulsive classes are set by their
peak (weapon noise is published as peak SPL), steady ones by active RMS
(frames within 20 dB of the loudest). The source of each number is next to
it; "estimate" means none was found. Linear spreading from 1 m overstates a
blast's level far away (a shock front loses energy faster than 1/r near the
source): the far levels of the loudest classes are a ceiling.

Output (derived data, CLAUDE.md section 6), mono 44.1 kHz 32-bit float,
1.0 = 110 dB SPL as the drone renders: peaks of near weapons exceed 1.0,
which the float file keeps.

    data/processed/war/war_track.wav
    data/processed/war/events.csv    one row per event: class, stem, start,
                                     end, distance, pitch, level at 1 m and at
                                     the microphone (peak and active RMS)
    data/processed/war/phases.csv    start, end, phase, rate factor

    uv run python scripts/war_track.py                  # seed 0, 10 min
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import env_sim as es  # noqa: E402

from dad import propagation as pr  # noqa: E402

ROOT = es.ROOT
STEMS = es.STEMS / "war"
OUT = ROOT / "data" / "processed" / "war"
FS = es.FS
FADE = int(0.01 * FS)
# Peak levels: US Army Public Health Command, "Noise levels of common Army equipment" (Fort Campbell
# hearing conservation table, archived 2012-10-14), as quoted in search results 2026-10-29, not opened.
# ANSI S3.5-1997 speech levels at 1 m. FEMA CPG 1-17 / ANSI S12.14: sirens rated in dB(C) at 100 ft.
CATALOGUE = {
    # class: rate per min, distance m (lo, hi), piece length s (None = whole stem), pitch +- semitones,
    #        level at 1 m (dB SPL), "peak" or "rms", source; stems = another class's stems
    "small_arms": dict(rate=6.0, dist=(200, 1500), length=(0.3, 3.0), pitch=1.0, level=157.0, ref="peak",
                       src="M16A2 at the shooter, 157 dB peak (Army table)"),
    "machine_gun": dict(rate=3.0, dist=(300, 2000), length=(1.0, 6.0), pitch=1.0, level=156.0, ref="peak",
                        src="mean of M249 159.5, M60 155, M2 153 dB peak at the gunner (Army table)"),
    "mortar_artillery": dict(rate=3.0, dist=(500, 6000), length=None, pitch=2.0, level=178.8, ref="peak",
                             src="81 mm mortar, 178.8 dB peak at 1 m from the muzzle (Army table); "
                                 "M198 155 mm howitzer 178 at the gunner"),
    "explosion_far": dict(rate=4.0, dist=(2000, 15000), length=None, pitch=2.0, level=180.0, ref="peak",
                          src="estimate: shell impacts taken as loud as the howitzer muzzle"),
    # the far front: many impacts 8-20 km away at a steady rate (not phased), so the track never falls silent
    "distant_front": dict(stems="explosion_far", phased=False, rate=30.0, dist=(8000, 20000), length=None, pitch=3.0, level=180.0,
                          ref="peak", src="as explosion_far"),
    "tank": dict(rate=0.5, dist=(200, 2000), length=(8.0, 20.0), pitch=1.0, level=115.0, ref="rms",
                 src="estimate: heavy truck (80-90 dB(A) at 15 m) plus 10 dB"),
    "jet": dict(rate=0.1, dist=(2000, 8000), length=None, pitch=1.0, level=156.7, ref="rms",
                src="F-16 Lmax 103-110 dB at 1000 ft (mid 107), US Navy/FAA comparison table"),
    "air_raid_siren": dict(rate=0.1, dist=(1000, 5000), length=(40.0, 90.0), pitch=0.0, level=154.7, ref="rms",
                           src="outdoor warning siren 115-130 dB(C) at 100 ft (mid 125); FEMA CPG 1-17"),
    "shout": dict(rate=1.0, dist=(30, 200), length=None, pitch=2.0, level=82.3, ref="rms",
                  src="ANSI S3.5-1997, shouted speech at 1 m"),
}
# MLRS salvo (Grad): 20-40 launches 0.5 s apart, the mortar firing stem standing in for one launch
SALVO = dict(rate=0.15, dist=(5000, 15000), count=(20, 40), gap=0.5, stem="fsd50k_dev_426161.wav",
             cls="mortar_artillery")
PHASES = {"calm": 0.3, "normal": 1.0, "intense": 2.5}
PHASE_P = (0.35, 0.45, 0.2)


def load_stems():
    stems = {}
    for cls in {c.get("stems", k) for k, c in CATALOGUE.items()}:
        files = sorted((STEMS / cls).glob("*.wav"))
        if not files:
            raise SystemExit(f"no stems in {STEMS / cls}; run scripts/build_stems.py --picked")
        stems[cls] = {p.name: sf.read(p)[0] for p in files}
    return stems


def phases(seconds, rng):
    out, t = [], 0.0
    while t < seconds:
        d = rng.uniform(30, 120)
        name = rng.choice(list(PHASES), p=PHASE_P)
        out.append(dict(start_s=round(t, 2), end_s=round(min(t + d, seconds), 2), phase=name,
                        factor=PHASES[name]))
        t += d
    return out


def times(rate_per_min, seconds, ph, rng):
    """Poisson event times at rate x phase factor, by thinning; ph None = steady rate."""
    top = max(PHASES.values()) if ph else 1.0
    t = np.cumsum(rng.exponential(60 / (rate_per_min * top), int(rate_per_min / 60 * top * seconds * 2) + 10))
    t = t[t < seconds]
    if ph is None:
        return t
    factor = np.array([next(p["factor"] for p in ph if p["start_s"] <= s < p["end_s"]) for s in t])
    return t[rng.random(len(t)) < factor / top]


def piece(x, length, rng):
    if length is None:
        return x.copy()
    n = int(rng.uniform(*length) * FS)
    if n >= len(x):
        return es.loop(x, n)
    i = rng.integers(0, len(x) - n)
    y = x[i:i + n].copy()
    ramp = np.linspace(0, 1, FADE)
    y[:FADE] *= ramp
    y[-FADE:] *= ramp[::-1]
    return y


def at_1m(x, level, ref):
    """x scaled to pascals at 1 m: its peak, or its active RMS, at `level` dB SPL."""
    now = np.abs(x).max() if ref == "peak" else es.active_rms(x)
    return x / now * es.P_REF * 10 ** (level / 20)


def db(v):
    return round(20 * np.log10(v / es.P_REF), 1) if v > 0 else None


def render(seconds, seed, stems):
    rng = np.random.default_rng(seed)
    ph = phases(seconds, rng)
    n = int(seconds * FS)
    track, events = np.zeros(n), []

    def add(cls, stem, t, x, dist, semis, cat):
        x = at_1m(pr.pitch(x, semis), cat["level"], cat["ref"])
        y = pr.propagate(x, dist)
        i = int(t * FS)
        y = y[: max(0, n - i)]
        track[i:i + len(y)] += y
        events.append(dict(cls=cls, stem=stem, start_s=round(t, 3), end_s=round((i + len(y)) / FS, 3),
                           distance_m=round(dist, 1), pitch_semitones=round(semis, 2), level_1m_db=cat["level"],
                           level_ref=cat["ref"], peak_at_mic_db=db(np.abs(y).max()) if len(y) else None,
                           rms_at_mic_db=db(es.active_rms(y)) if len(y) > 1764 else None))

    def dist(lo_hi):
        return float(np.exp(rng.uniform(*np.log(lo_hi))))

    for cls, cat in CATALOGUE.items():
        pool = stems[cat.get("stems", cls)]
        for t in times(cat["rate"], seconds, ph if cat.get("phased", True) else None, rng):
            stem = rng.choice(sorted(pool))
            semis = rng.uniform(-cat["pitch"], cat["pitch"])
            add(cls, stem, t, piece(pool[stem], cat["length"], rng), dist(cat["dist"]), semis, cat)
    for t in times(SALVO["rate"], seconds, ph, rng):
        d, x = dist(SALVO["dist"]), stems[SALVO["cls"]][SALVO["stem"]]
        for k in range(rng.integers(SALVO["count"][0], SALVO["count"][1] + 1)):
            tk = t + k * SALVO["gap"] + rng.uniform(-0.05, 0.05)
            if tk < seconds:
                add("mlrs_salvo", SALVO["stem"], tk, x.copy(), d, rng.uniform(-0.5, 0.5), CATALOGUE[SALVO["cls"]])
    events.sort(key=lambda e: e["start_s"])
    return track, events, ph


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    track, events, ph = render(a.minutes * 60, a.seed, load_stems())
    a.out.mkdir(parents=True, exist_ok=True)
    full = es.P_REF * 10 ** (es.FULL_SCALE_DB / 20)
    sf.write(a.out / "war_track.wav", (track / full).astype(np.float32), FS, subtype="FLOAT")
    for name, rows in (("events.csv", events), ("phases.csv", ph)):
        with open(a.out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(events)} events, {len(ph)} phases, {a.minutes:g} min, seed {a.seed} -> {a.out}")


if __name__ == "__main__":
    main()
