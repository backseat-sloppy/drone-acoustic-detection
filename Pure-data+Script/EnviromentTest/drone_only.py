"""Render 10 drone-only flights (no background, wind or events) to drone_only/.

Nothing is random between files: the same drone (one seed) flies at speeds that rise
in even steps, each from its own start point around the microphone. Start distance =
speed x 10 s, so every file is 10 s long and ends at impact.
"""
import json
import numpy as np
import soundfile as sf
from drone_sim import make_scene, HERE, P_REF, FULL_SCALE_DB

SEED = 1                                   # same drone sound (pitch, throttle) in every file
SPEEDS = np.linspace(28, 45, 10)           # m/s, rising
DURATION_S, ELEV_DEG = 10.0, 40.0          # every flight reaches the mic after exactly 10 s
AZIMUTHS = np.arange(10) * 36.0            # deg: 10 start points, 36 deg apart

OUT = HERE / "drone_only"
OUT.mkdir(exist_ok=True)
scale = P_REF * 10 ** (FULL_SCALE_DB / 20)

for i, (speed, az) in enumerate(zip(SPEEDS, AZIMUTHS), start=1):
    _, drone, fs, p, start = make_scene(SEED, speed=float(speed), start_dist=float(speed) * DURATION_S,
                                        elev_deg=ELEV_DEG, az_deg=float(az))   # 2nd value: drone alone
    p["start_xyz"] = [round(float(v), 1) for v in start]
    sf.write(OUT / f"drone_{i:02d}.wav", (drone / scale).astype(np.float32), fs, subtype="FLOAT")  # 1.0 = 110 dB SPL
    sf.write(OUT / f"drone_{i:02d}_listen.wav", (drone / np.abs(drone).max()).astype(np.float32), fs, subtype="FLOAT")
    (OUT / f"drone_{i:02d}.json").write_text(json.dumps(p, indent=2))
    print(i, f"speed {p['speed']:.1f} m/s  az {az:.0f} deg  xyz {p['start_xyz']}  {len(drone)/fs:.1f} s")
