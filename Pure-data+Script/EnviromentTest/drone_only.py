"""Render drone-only flights (no background, wind or events) to drone_only/.

Nothing is random between runs. For each fixed seed in SEEDS (one drone sound: its pitch
scale, throttle curve and loudness come from the seed), N_DRONES flights at speeds that rise
in even steps, each from its own start point around the microphone. Start distance =
speed x 10 s, so every file is 10 s long and ends at impact.
Files: drone_s<seed>_<i>.wav (calibrated), _listen.wav (normalised), .json (parameters).
"""
import json
import numpy as np
import soundfile as sf
from drone_sim import make_scene, HERE, P_REF, FULL_SCALE_DB

SEEDS = [1, 2, 3, 4, 5]                    # one drone sound per seed: the seed sets throttle curve and level
PSCALES = np.linspace(0.85, 1.25, len(SEEDS))   # pitch scale per seed, evenly over make_scene's range
                                                # (the drawn values bunched at 1.00-1.02 for seeds 1, 3, 4, 5)
N_DRONES = 10                              # flights per seed
SPEEDS = np.linspace(100 / 3.6, 200 / 3.6, N_DRONES)   # m/s: 27.778 (100 km/h) to 55.556 (200 km/h), even steps
DURATION_S, ELEV_DEG = 10.0, 40.0          # every flight reaches the mic after exactly 10 s
AZIMUTHS = np.arange(N_DRONES) * 360.0 / N_DRONES      # deg: start points evenly around the mic

OUT = HERE / "drone_only"
OUT.mkdir(exist_ok=True)
scale = P_REF * 10 ** (FULL_SCALE_DB / 20)

for seed, pscale in zip(SEEDS, PSCALES):
    for i, (speed, az) in enumerate(zip(SPEEDS, AZIMUTHS), start=1):
        _, drone, fs, p, start = make_scene(seed, pscale=float(pscale), speed=float(speed), start_dist=float(speed) * DURATION_S,
                                            elev_deg=ELEV_DEG, az_deg=float(az))   # 2nd value: drone alone
        p["start_xyz"] = [round(float(v), 1) for v in start]
        name = f"drone_s{seed:02d}_{i:02d}"
        sf.write(OUT / f"{name}.wav", (drone / scale).astype(np.float32), fs, subtype="FLOAT")  # 1.0 = 110 dB SPL
        sf.write(OUT / f"{name}_listen.wav", (drone / np.abs(drone).max()).astype(np.float32), fs, subtype="FLOAT")
        (OUT / f"{name}.json").write_text(json.dumps(p, indent=2))
        print(name, f"pscale {p['pscale']:.2f}  L_1m {p['L_1m']:.1f} dB  speed {p['speed']:.1f} m/s  "
                    f"az {az:.0f} deg  {len(drone)/fs:.1f} s")
