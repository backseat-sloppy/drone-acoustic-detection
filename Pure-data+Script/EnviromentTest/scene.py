import numpy as np
from scipy.signal import lfilter
from drone_sim import make_scene

C, P_REF = 343.0, 20e-6


def lowpass(x, fc, fs):
    a = np.exp(-2 * np.pi * fc / fs)
    return lfilter([1 - a], [1, -a], x)


def to_spl(x, L):
    """Scale so RMS = L dB SPL, in pascals."""
    return x / np.sqrt(np.mean(x**2)) * P_REF * 10 ** (L / 20)


def unit(x):
    return x / np.sqrt(np.mean(x**2))


def propagate(x, t, pos):
    r = np.maximum(np.linalg.norm(pos, axis=1), 1.0)
    return np.interp(t, t + r / C, x / r, left=0.0, right=0.0), r


def wind(rng, N, fs, level_db):
    """Diffuse low-frequency noise with slow gusts (no position)."""
    w = lowpass(rng.standard_normal(N), 250, fs)
    gust = lowpass(rng.standard_normal(N), 0.3, fs)
    env = np.clip(1 + 0.5 * gust / gust.std(), 0.1, None)
    return to_spl(w * env, level_db)


def render_event(kind, rng, fs):
    """Returns the dry sound at 1 m and its 3D positions over time."""
    if kind == "bang":                              # door slam, impact, etc.
        n = int(0.6 * fs); t = np.arange(n) / fs
        x = lowpass(rng.standard_normal(n) * np.exp(-t / 0.08), rng.uniform(1500, 5000), fs)
        x = to_spl(x, rng.uniform(90, 105))
        az, dist = rng.uniform(0, 2 * np.pi), rng.uniform(20, 200)
        p0 = np.array([dist * np.cos(az), dist * np.sin(az), rng.uniform(0, 10)])
        return x, np.tile(p0, (n, 1))
    # vehicle driving past: harmonic engine + road noise, moving, with Doppler
    n = int(14 * fs); t = np.arange(n) / fs
    f0 = rng.uniform(70, 130)
    tone = unit(sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6)))
    noise = unit(lowpass(rng.standard_normal(n), 800, fs))
    x = to_spl(tone + noise, rng.uniform(80, 90))
    a, closest, v = rng.uniform(0, 2 * np.pi), rng.uniform(15, 80), rng.uniform(8, 20)
    pc = np.array([closest * np.cos(a), closest * np.sin(a), 1.0])
    direction = np.array([-np.sin(a), np.cos(a), 0.0])
    return x, pc + direction * (v * (t - 7))[:, None]


def add_events(rng, N, fs, rate_per_min):
    mix, log = np.zeros(N), []
    for _ in range(rng.poisson(rate_per_min * N / fs / 60)):
        kind = str(rng.choice(["bang", "vehicle"], p=[0.6, 0.4]))
        x, pos = render_event(kind, rng, fs)
        n, pad = len(x), int(1.2 * fs)             # room for the propagation delay
        x = np.pad(x, (0, pad)); pos = np.pad(pos, ((0, pad), (0, 0)), mode="edge")
        t = np.arange(len(x)) / fs
        y, r = propagate(x, t, pos)
        i0 = int(rng.integers(0, N)); m = min(len(y), N - i0)
        mix[i0:i0 + m] += y[:m]
        log.append(dict(kind=kind, t_start=round(i0 / fs, 2), duration=round(n / fs, 1),
                        start_xyz=[round(float(v), 1) for v in pos[0]],
                        min_dist=round(float(r.min()), 1)))
    return mix, log


def make_full_scene(seed, bg_db=40.0, wind_db=45.0, event_rate=6.0, with_drone=True, dist_range=(300, 500)):
    mix, drone, fs, p, _ = make_scene(seed, bg_db, dist_range)
    rng = np.random.default_rng(seed + 1000)
    N = len(mix)
    noise_floor = mix - drone                      # white background from make_scene
    ev, log = add_events(rng, N, fs, event_rate)
    full = noise_floor + wind(rng, N, fs, wind_db) + ev + (drone if with_drone else 0)
    p.update(wind_db=wind_db, with_drone=with_drone, events=log)
    return full, drone, fs, p