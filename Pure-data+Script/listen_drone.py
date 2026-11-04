import numpy as np
import soundfile as sf
from pathlib import Path
from drone_sim import make_scene

HERE = Path(__file__).parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

SEEDS = range(1, 4)
DIST_RANGE = (300, 500)     # start distance (m)
EXPONENT = 0.5              # 1.0 = real physics (1/r). 0.5 = gentler. 0 = constant loudness
PEAK = 0.9                  # loudest sample in the file (0..1)
C = 343.0

for seed in SEEDS:
    _, drone, fs, p, _ = make_scene(seed, dist_range=DIST_RANGE)   # drone alone, as heard at the mic
    D, v = p["start_dist"], p["speed"]

    # true drone distance for every sample (the sound heard at time t left the drone earlier)
    t = np.arange(len(drone)) / fs
    te = np.clip((t - D / C) / (1 - v / C), 0, None)
    r = np.maximum(D - v * te, 1.0)

    y = drone * r ** (1 - EXPONENT)        # turns 1/r into 1/r^EXPONENT
    y = y[int(D / C * fs):]                # cut the silence before the sound arrives
    y = y / np.abs(y).max() * PEAK         # peak-normalize: no clipping, no distortion
    fade = int(0.05 * fs)
    y[:fade] *= np.linspace(0, 1, fade)

    sf.write(OUT / f"drone_listen_{seed}.wav", y.astype(np.float32), fs)
    print(f"seed {seed}: {len(y)/fs:.1f} s, speed {v:.1f} m/s")