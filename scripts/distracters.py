"""Distracters (iteration 7): sounds that can be taken for a drone, placed by distance and pitch.

Each class (CATALOGUE) has a level at 1 m, a distance range and a pitch range,
and one of three motions:

    loop     a static source looped through the environment (mower, chainsaw,
             generator, birds)
    wander   looped, its distance wandering around the set one (log-normal,
             0.3 Hz), as an insect circling the microphone; the propagation
             model's travel time gives it its own small Doppler
    once     played once at a random time: the stem is itself a pass-by or
             flyover recording (car, motorcycle, scooter, helicopter, prop
             plane), so its Doppler and level sweep are the recording's own,
             and the set distance is the closest approach

A distance step 1-10 (the 1-10 scale of entry 058): 1 = off, 2 = the far end
of the class's range, 10 = the near end, log-spaced. Pitch is drawn from
+-pitch semitones per environment and logged.

Levels at 1 m are active RMS (frames within 20 dB of the loudest), the same
as a pass-by maximum. The source of each number is next to it; "estimate"
means no source was found, "memory" that it was not re-checked.

    x, row = render("mower", 6, 20.0, np.random.default_rng(0), load_stems())
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import env_sim as es  # noqa: E402

from dad import propagation as pr  # noqa: E402

STEMS = es.STEMS / "distracter"
FS = es.FS
HEMI = 10 * np.log10(2 * np.pi)          # sound power (LwA) to pressure at 1 m over hard ground: -8 dB
CATALOGUE = {
    "car": dict(motion="once", dist=(5, 300), pitch=1.0, level=87.5,
                src="memory: UN R51.03 pass-by limits 68-72 dB(A) at 7.5 m (mid 70) + 17.5 dB"),
    "motorcycle": dict(motion="once", dist=(5, 300), pitch=1.0, level=94.5,
                       src="memory: UN R41.04 limit 77 dB(A) at 7.5 m (class 3) + 17.5 dB"),
    "scooter": dict(motion="once", dist=(5, 300), pitch=1.5, level=88.5,
                    src="memory: UN R63.02 moped limit 71 dB(A) at 7.5 m + 17.5 dB"),
    "helicopter": dict(motion="once", dist=(100, 3000), pitch=1.0, level=115.5,
                       src="Robinson R44 II flight manual: flyover SEL 79-82 dB(A) at 492 ft; Lmax ~ SEL - 8 "
                           "(a 6 s pass) = 72 at 150 m, + 43.5 dB"),
    "prop_plane": dict(motion="once", dist=(150, 3000), pitch=1.0, level=126.0,
                       src="Cessna 172R type certificate: 76.3 dB(A) flyover (ICAO Annex 16 ch. 10); "
                           "estimate: taken at 300 m, + 49.5 dB"),
    "mower": dict(motion="loop", dist=(3, 200), pitch=1.0, level=96.0 - HEMI,
                  src="memory: EU 2000/14/EC limit LwA 96 dB (cutting width <= 50 cm), hemispherical"),
    "chainsaw": dict(motion="loop", dist=(5, 300), pitch=1.0, level=115.0 - HEMI,
                     src="declared LwA of petrol saws 113-117 dB (Husqvarna 353/372XP 115), hemispherical"),
    "generator": dict(motion="loop", dist=(3, 200), pitch=0.5, level=95.0 - HEMI,
                      src="memory: EU 2000/14/EC limit LwA 95 + log P (P < 2 kVA), hemispherical"),
    "birds": dict(motion="loop", dist=(2, 100), pitch=1.0, level=85.0,
                  src="songbirds 80-93 dB SPL at 1 m (Brumm et al.; red-winged blackbird 88.5-93.5)"),
    "mosquito": dict(motion="wander", dist=(0.03, 0.5), pitch=2.0, level=0.0,
                     src="Belton 1994: about 40 dB SPL at 1 cm (OCR'd snippet, not checked) - 40 dB"),
    "buzzing_insect": dict(motion="wander", dist=(0.1, 3.0), pitch=2.0, level=30.0,
                           src="estimate: bee or fly 50 dB at 10 cm"),
}


def load_stems():
    out = {}
    for cls in CATALOGUE:
        files = sorted((STEMS / cls).glob("*.wav"))
        if not files:
            raise SystemExit(f"no stems in {STEMS / cls}; run scripts/build_stems.py (--picked)")
        out[cls] = {p.name: sf.read(p)[0] for p in files}
    return out


def step_distance(cls, step):
    """Distance step 1-10 -> metres (None = off): 2 = far end, 10 = near end, log-spaced."""
    s = int(step)
    if not 1 <= s <= 10:
        raise ValueError(f"distance step must be 1-10, got {step}")
    if s == 1:
        return None
    near, far = CATALOGUE[cls]["dist"]
    return float(far * (near / far) ** ((s - 2) / 8))


def wander(n, d, rng, depth=0.5, hz=0.3):
    """Distance per sample wandering log-normally around d: a new target every 1 / (2 hz) s, linear between."""
    k = int(n / FS * hz * 2) + 2
    return d * np.exp(np.interp(np.arange(n), np.linspace(0, n - 1, k), rng.normal(0, depth, k)))


def render(cls, step, seconds, rng, stems, semitones=None):
    """(pascals, row) for one distracter in an environment of `seconds`; silence and None if step is 1."""
    n = int(seconds * FS)
    d = step_distance(cls, step)
    if d is None:
        return np.zeros(n), None
    cat = CATALOGUE[cls]
    stem = str(rng.choice(sorted(stems[cls])))
    semis = rng.uniform(-cat["pitch"], cat["pitch"]) if semitones is None else semitones
    x = pr.pitch(stems[cls][stem], semis)
    x = x / es.active_rms(x) * es.P_REF * 10 ** (cat["level"] / 20)
    if cat["motion"] == "once":
        y = pr.propagate(x, d)[:n]
        start = int(rng.integers(0, n - len(y) + 1))
    else:
        x = es.loop(x, n)
        y = pr.propagate(x, d if cat["motion"] == "loop" else wander(n, d, rng))[:n]
        start = 0
    out = np.zeros(n)
    out[start:start + len(y)] = y
    return out, dict(cls=cls, stem=stem, motion=cat["motion"], step=int(step), distance_m=round(d, 3),
                     pitch_semitones=round(float(semis), 2), level_1m_db=round(cat["level"], 1),
                     start_s=round(start / FS, 3), end_s=round((start + len(y)) / FS, 3),
                     rms_at_mic_db=round(20 * np.log10(es.active_rms(y) / es.P_REF), 1))
