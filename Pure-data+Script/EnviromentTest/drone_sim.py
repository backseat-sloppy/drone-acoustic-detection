import json
import numpy as np
from scipy.signal import butter, sosfilt
import soundfile as sf
from pathlib import Path
from render import render_dry, make_throttle

HERE = Path(__file__).parent
C, P_REF = 343.0, 20e-6
FULL_SCALE_DB = 110.0                      # dB SPL that maps to +-1.0 in the calibrated WAV

def prop_noise(rng, n, fs, thr, fc, q, step_s=0.02):
    """Broadband prop noise: white noise band-passed at fc (Q q, two poles each side), following
    the throttle (louder at full throttle), unit RMS. thr: slider values 0..127, one per step_s."""
    lo, hi = fc / (1 + 1 / (2 * q)), min(fc * (1 + 1 / (2 * q)), 0.45 * fs)
    nz = sosfilt(butter(2, [lo, hi], btype="band", fs=fs, output="sos"), rng.standard_normal(n))
    env = 0.3 + np.interp(np.arange(n) / fs, np.arange(len(thr)) * step_s, np.asarray(thr) / 127)
    nz = nz * env
    return nz / np.sqrt(np.mean(nz ** 2))


def make_scene(seed, bg_db=40.0, dist_range=(300, 500), **fixed):
    """fixed: set any drawn parameter (speed, start_dist, elev_deg, az_deg, ...) instead of drawing it.
    Optional extras (all off unless given in fixed; make_scenes.py --realism draws them):
      detune     4 motor offsets in Hz (render_dry)
      noise_db, noise_fc, noise_q
                 extra prop noise (prop_noise) at noise_db against the motor tone, added at 1 m
      mic_h, refl_db
                 ground reflection: the image source mirrored in the ground mic_h metres below the
                 microphone, refl_db below the direct path (flat ground, no phase change)
      dry        a real recording at 44.1 kHz (a drone near the microphone) that replaces the Pd render:
                 looped from a random start to the flight time, then flown like the Pd drone"""
    rng = np.random.default_rng(seed)

    # random parameters of this drone and flight
    p = dict(
        seed=seed,
        speed=rng.uniform(28, 45),         # m/s
        start_dist=rng.uniform(*dist_range),  # m
        elev_deg=rng.uniform(20, 60),
        az_deg=rng.uniform(0, 360),
        L_1m=rng.uniform(76, 84),          # dB SPL at 1 m (loud drone)
        pscale=rng.uniform(0.85, 1.25),    # pitch scale in the patch
        base=rng.uniform(50, 70),
        wobble=rng.uniform(4, 8),
        dive_start=rng.uniform(0.6, 0.8),
        dive_gain=rng.uniform(60, 90),
        bg_db=bg_db,
    )
    p.update(fixed)                        # after the draws, so the rng sequence stays the same

    T = p["start_dist"] / p["speed"]       # flight time = time to impact
    n = int(T / 0.02)
    thr = make_throttle(rng, n, p["base"], p["wobble"], p["dive_start"], p["dive_gain"])
    if p.get("dry") is not None:           # a real drone in place of Pd (make_scenes.py --real-drones)
        fs = 44100
        dry = np.asarray(p.pop("dry"), dtype=np.float64)
        x = np.resize(np.roll(dry, -int(np.random.default_rng([seed, 5]).integers(len(dry)))), int(T * fs))
    else:
        x, fs = render_dry(thr, p["pscale"], falloff=p.get("falloff"), detune=p.get("detune"))  # dry, 1 m

    N = int(T * fs)                        # trim or pad to exactly the flight time
    x = np.pad(x, (0, max(0, N - len(x))))[:N]
    if p.get("noise_db") is not None:      # its own rng: the flight's draws stay as they were
        nz = prop_noise(np.random.default_rng([seed, 1]), N, fs, thr, p["noise_fc"], p["noise_q"])
        x = x + nz * np.sqrt(np.mean(x ** 2)) * 10 ** (p["noise_db"] / 20)

    # calibrate: RMS of the cruise part equals L_1m in pascals
    cruise = x[: int(N * p["dive_start"])]
    x = x / np.sqrt(np.mean(cruise**2)) * P_REF * 10 ** (p["L_1m"] / 20)

    # 3D trajectory (emission time t), straight line to the listener at the origin
    t = np.arange(N) / fs
    el, az = np.radians(p["elev_deg"]), np.radians(p["az_deg"])
    start = p["start_dist"] * np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    pos = start[None, :] * (1 - t[:, None] / T)

    # propagation: delay + Doppler + 1/r
    r = np.maximum(np.linalg.norm(pos, axis=1), 1.0)
    y = np.interp(t, t + r / C, x / r, left=0.0, right=0.0)
    if p.get("mic_h") is not None:         # ground at z = -mic_h: image source at z' = -2 mic_h - z
        img = pos * np.array([1.0, 1.0, -1.0]) - np.array([0.0, 0.0, 2 * p["mic_h"]])
        r2 = np.maximum(np.linalg.norm(img, axis=1), 1.0)
        y = y + 10 ** (p["refl_db"] / 20) * np.interp(t, t + r2 / C, x / r2, left=0.0, right=0.0)

    # background
    bg = rng.standard_normal(N)
    bg = bg / np.sqrt(np.mean(bg**2)) * P_REF * 10 ** (bg_db / 20)
    return y + bg, y, fs, p, pos[0]

if __name__ == "__main__":
    for seed in (1, 2, 3):
        mix, drone, fs, p, start = make_scene(seed)
        p["start_xyz"] = [round(float(v), 1) for v in start]
        scale = P_REF * 10 ** (FULL_SCALE_DB / 20)
        sf.write(HERE / f"scene_{seed}.wav", (mix / scale).astype(np.float32), fs)           # calibrated
        sf.write(HERE / f"scene_{seed}_listen.wav", (mix / np.abs(mix).max()).astype(np.float32), fs)
        (HERE / f"scene_{seed}.json").write_text(json.dumps(p, indent=2))
        print(seed, {k: round(v, 2) if isinstance(v, float) else v for k, v in p.items()})