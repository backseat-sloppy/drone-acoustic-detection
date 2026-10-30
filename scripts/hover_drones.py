"""A hovering Pure Data drone (0 km/h) heard at fixed distances.

The drone sound is DroneSound.pd, rendered through render.py with the same throttle model
and the same drone parameters as the flights of drone_war_grid.py (seed 1: base throttle,
wobble, pitch scale 0.85, L_1m), but with no dive: a hover holds its throttle. The dry render
is calibrated so its RMS at 1 m is L_1m, as drone_sim.make_scene does, then carried to each
distance by dad.propagation (1/r spreading and ISO 9613-1 air absorption).

Output in --out, 1.0 = 110 dB SPL as every drone render here:
    hover_s01_<D>m.wav          calibrated, for the detectors
    hover_s01_<D>m_listen.wav   the same audio, unscaled: what you hear is the calibrated level

    uv run python scripts/hover_drones.py --distances 50 100 200 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
PD_DIR = ROOT / "Pure-data+Script" / "EnviromentTest"
sys.path.insert(0, str(PD_DIR))
from drone_sim import FULL_SCALE_DB, P_REF  # noqa: E402

from dad.propagation import propagate  # noqa: E402

FS = 44100
FULL = P_REF * 10 ** (FULL_SCALE_DB / 20)
SEED1 = dict(L_1m=78.49465161608389, pscale=0.85, base=66.55405187640883, wobble=5.6367965454766455)  # drone_s01_*.json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distances", nargs="+", type=float, default=[50, 100, 200, 300], help="m")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/drone_war/hover")
    a = ap.parse_args()
    if min(a.distances) <= 0:
        ap.error("distances must be > 0 m")

    from render import make_throttle, render_dry                  # needs Pd
    keep = {p: p.read_bytes() for p in (PD_DIR / "out.wav", PD_DIR / "throttle.txt") if p.exists()}
    try:
        n = int(round(a.seconds / 0.02))
        thr = make_throttle(np.random.default_rng(1), n, SEED1["base"], SEED1["wobble"], dive_start=1.0, dive_gain=0.0)
        x, fs = render_dry(thr, SEED1["pscale"])
    finally:
        for p, b in keep.items():                                  # Pd writes next to the patch; put it back
            p.write_bytes(b)
    assert fs == FS, fs
    x = x[: int(a.seconds * FS)]
    x = x / np.sqrt(np.mean(x ** 2)) * P_REF * 10 ** (SEED1["L_1m"] / 20)   # pascals at 1 m

    a.out.mkdir(parents=True, exist_ok=True)
    for d in a.distances:
        y = propagate(x, d)
        stem = a.out / f"hover_s01_{d:g}m"
        for name in (f"{stem}.wav", f"{stem}_listen.wav"):      # listening copy: never rescaled (Mads)
            sf.write(name, (y / FULL).astype(np.float32), FS, subtype="FLOAT")
        spl = 20 * np.log10(np.sqrt(np.mean(y ** 2)) / P_REF)
        Path(f"{stem}.json").write_text(json.dumps(dict(seed=1, speed_kmh=0, distance_m=d, seconds=a.seconds,
                                                        rms_db_spl=round(spl, 1), **SEED1), indent=2))
        print(f"{d:g} m: {spl:.1f} dB SPL -> {stem}.wav")


if __name__ == "__main__":
    main()
