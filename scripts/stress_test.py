"""Parametric stress test: a Pure Data drone in a fixed war environment.

Every test clip is L seconds long (5 s) and is built in memory, scored by each
detector hop by hop (the same `stream` that `dad wav` uses) and thrown away.
Only the parameters and the results are kept, in two CSV files: one row per
clip and detector (`clips.csv`) and one row per hop (`hops.csv`). Any clip can
be rebuilt from its row, because the audio is deterministic.

Parameters (each takes a list; the run is the full grid):

    V  drone speed, km/h                     --speed
    R0 drone distance when it is first heard --start-dist
    D  seconds of drone at the end of the clip (0 = no drone, a control)
                                              --drone-len
    N  war noise (gunfire and explosion), dB SPL RMS over the clip; "off" mutes it
                                              --war
    W  wind, dB SPL RMS                       --wind
    B  birds, dB SPL RMS                      --birds
    C  car passing by, dB SPL RMS             --car
    L  clip length, s (fixed at 5)            --length

The drone flies straight at the microphone from R0 at V for D seconds, so it
ends at R1 = R0 - V*D (impact if that reaches 0 inside the clip). Its sound is
rendered once per speed by `dad.pdsynth`, the numpy port of
`Pure-data+Script/EnviromentTest/DroneSound.pd` (or Pd itself with --synth pd)
(throttle rises with speed), then distance (1/r), propagation delay and Doppler
are applied in numpy. The environment is always the same five FSD50K clips
(`scripts/fetch_env_clips.py`), only their levels change.

These numbers come from rendered drone audio and are a stress test, not
detection performance (CLAUDE.md section 5).

    uv run python scripts/stress_test.py --out artifacts/stress/run1
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from glob import glob
from pathlib import Path

# torch and lightgbm each bring an OpenMP runtime; with both threaded the first conv2d deadlocks on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PD_DIR = ROOT / "Pure-data+Script" / "EnviromentTest"
ENV_DIR = ROOT / "data" / "extracted" / "fsd50k_env"
RUN = ROOT / "artifacts" / "runs" / "it6_final.json"
MODELS = {"comb": "comb", "lgbm": "artifacts/models/it6_final_none_simdist_lgbm.pkl",
          "dscnn": "artifacts/models/it6_final_none_simdist_dscnn.pt"}
sys.path.insert(0, str(ROOT / "src"))
from dad import pdsynth                           # noqa: E402
from dad.config import FrontendConfig             # noqa: E402
from dad.wavio import Clip, Truth, counts, lead_times, load_scorer, stream, window_labels  # noqa: E402

FS = 44100
C_SOUND, P_REF = 343.0, 20e-6
FULL_SCALE_DB = 110.0      # dB SPL that maps to +-1.0, as in drone_sim.py
STEP_MS = 20


def pa(db: float) -> float:
    return P_REF * 10 ** (db / 20)


def rms_db(x: np.ndarray) -> float:
    return 20 * math.log10(max(np.sqrt(np.mean(x ** 2)), 1e-12) / P_REF)


# ------------------------------------------------------------------ drone

def make_throttle(rng, n, base=60, wobble=6.0, dive_start=0.7, dive_gain=80):
    """Copied from Pure-data+Script/EnviromentTest/render.py, whose import exits when Pd is not
    in /Applications. n steps of 20 ms: steady cruise, small corrections, optional dive."""
    slow = np.convolve(rng.standard_normal(n), np.ones(25) / 25, mode="same")
    slow = slow / slow.std() * wobble
    fast = rng.standard_normal(n) * 0.8
    ramp = np.clip((np.arange(n) / n - dive_start) / (1 - dive_start), 0, 1) * dive_gain
    return np.clip(base + slow + fast + ramp, 0, 127)


def throttle_for(v_kmh: float) -> float:
    # ponytail: linear guess, 50 km/h -> 40, 200 km/h -> 100 on the 0..127 slider; replace with a measured curve
    return 40 + 60 * (v_kmh - 50) / 150


def throttle_curve(v_kmh: float, seconds: float, seed: int) -> np.ndarray:
    return make_throttle(np.random.default_rng(seed), int(seconds * 1000 / STEP_MS) + 10,
                         base=throttle_for(v_kmh), wobble=4.0, dive_start=0.7, dive_gain=0.0)


def render_drone(v_kmh: float, seconds: float, seed: int, synth: str = "python") -> np.ndarray:
    """Dry drone at 1 m: the numpy port of DroneSound.pd (dad.pdsynth), or Pd itself."""
    thr = throttle_curve(v_kmh, seconds, seed)
    x = pdsynth.render(thr, fs=FS, step_ms=STEP_MS, seed=seed) if synth == "python" else render_pd_curve(thr)
    if len(x) < seconds * FS:
        raise SystemExit(f"{synth} rendered {len(x) / FS:.1f} s, wanted {seconds} s")
    return x[: int(seconds * FS)]


def render_pd_curve(thr: np.ndarray) -> np.ndarray:
    """DroneSound.pd run headless on a throttle curve, in a temp dir so the repo copy stays untouched.
    Pd applies the first value 20 ms in (qlist waits before each message), so that delay is cut."""
    pd = glob("/Applications/Pd*.app/Contents/Resources/bin/pd") or [shutil.which("pd")]
    if not pd[0]:
        raise SystemExit("Pure Data not found (looked in /Applications and on PATH); use --synth python")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(PD_DIR / "DroneSound.pd", tmp)
        (tmp / "throttle.txt").write_text("\n".join(["0;", "pscale 1.0;"] + [f"{STEP_MS};\nthrottle {v:.2f};" for v in thr]) + "\n")
        (tmp / "run.pd").write_text("#N canvas 0 0 300 200 12;\n#X obj 10 10 loadbang;\n"
                                    "#X msg 10 50 \\; autorun bang;\n#X connect 0 0 1 0;\n")
        subprocess.run([pd[0], "-nogui", "-nosound", "-batch", "-r", str(FS), "-open", "DroneSound.pd",
                        "-open", "run.pd"], cwd=tmp, timeout=120, capture_output=True)
        x, fs = sf.read(tmp / "out.wav")
    if fs != FS:
        raise SystemExit(f"Pd rendered at {fs} Hz, wanted {FS}")
    return x[int(STEP_MS / 1000 * FS):]


def drone_at_mic(dry: np.ndarray, v_ms: float, r0: float, d: float, L: float, l1m: float):
    """Drone heard at the microphone from t0 = L - D. r(t) is the distance the
    drone had when it emitted the sound heard at t, and that sound left it at
    t - r(t)/c. A drone flying at v covers distance at v / (1 - v/c) per second
    of listening time, which gives the moving-source Doppler c / (c - v).
    Silent before t0 and after impact."""
    t = np.arange(int(L * FS)) / FS
    t0 = L - d
    v_rx = v_ms / (1 - v_ms / C_SOUND)
    r = r0 - v_rx * (t - t0)
    impact = t0 + r0 / v_rx
    on = (t >= t0) & (r > 0)
    margin = len(dry) / FS - L                        # dry audio starts `margin` s before t = 0
    src = dry / np.sqrt(np.mean(dry ** 2)) * pa(l1m)  # RMS at 1 m = L_1m
    te = t - np.maximum(r, 1.0) / C_SOUND + margin
    y = np.interp(te, np.arange(len(src)) / FS, src) / np.maximum(r, 1.0)
    y[~on] = 0.0
    fade = min(int(0.05 * FS), on.sum())             # 50 ms fade-in so the onset is not a click
    if fade:
        i0 = np.argmax(on)
        y[i0:i0 + fade] *= np.linspace(0, 1, fade)
        if impact <= L:                              # 20 ms fade-out: a hard stop at impact clicks
            i1 = i0 + on.sum()
            n = min(int(0.02 * FS), on.sum())
            y[i1 - n:i1] *= np.linspace(1, 0, n)
    return y, t0, (impact if impact <= L else None), max(r0 - v_rx * d, 0.0)


# ------------------------------------------------------------ environment

def load_env(L: float) -> dict[str, np.ndarray]:
    """The loudest L-second stretch of each clip, scaled to 0 dB SPL RMS (1 unit = P_REF)."""
    out = {}
    for f in sorted(ENV_DIR.glob("*.wav")):
        x, fs = sf.read(f, always_2d=True)
        x = x[:, 0] if fs == FS else resample_poly(x[:, 0], FS, fs)
        n = int(L * FS)
        x = np.pad(x, (0, max(0, n - len(x))))
        c = np.concatenate([[0.0], np.cumsum(x ** 2)])
        s = int(np.argmax(c[n:] - c[:-n]))           # start of the most energetic n samples
        seg = x[s:s + n]
        out[f.stem] = seg / np.sqrt(np.mean(seg ** 2)) * P_REF
    if not out:
        raise SystemExit(f"no clips in {ENV_DIR}; run scripts/fetch_env_clips.py first")
    return out


def level(v: str) -> float | None:
    return None if v.lower() == "off" else float(v)


def build_clip(env, dry, v, r0, d, levels, L, l1m):
    """One clip in pascals: (background, drone, t0, impact, r1). `levels` maps an
    environment clip name to dB SPL, or None to leave it out; d = 0 means no drone."""
    background = sum((env[k] * 10 ** (db / 20) for k, db in levels.items() if db is not None and k in env),
                     np.zeros(int(L * FS)))
    if d > 0:
        return (background, *drone_at_mic(dry, v / 3.6, r0, d, L, l1m))
    return background, np.zeros_like(background), None, None, None


def to_model(mix_pa: np.ndarray) -> np.ndarray:
    """Pascals -> what a WAV at 110 dB SPL full scale holds, clipped, at the frontend's 16 kHz."""
    return resample_poly(np.clip(mix_pa / pa(FULL_SCALE_DB), -1, 1), 160, 441).astype(np.float32)


def load_detectors(models):
    cfg = FrontendConfig()
    run = json.loads(RUN.read_text())
    thr = {r["scorer"].split()[0]: r["deploy_threshold"] for r in run["results"] if r.get("integrate", 1) == 1}
    scorers = {m: load_scorer(MODELS[m], cfg) if m == "comb" else load_scorer(str(ROOT / MODELS[m]), cfg)
               for m in models}
    return cfg, scorers, thr


# -------------------------------------------------------------------- run

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", nargs="+", type=float, default=[50, 100, 150, 200], help="V, km/h")
    ap.add_argument("--start-dist", nargs="+", type=float, default=[50, 100, 200, 400], help="R0, m")
    ap.add_argument("--drone-len", nargs="+", type=float, default=[0, 2, 3.5, 5], help="D, s (0 = no drone)")
    ap.add_argument("--war", nargs="+", default=["off", "50", "60", "70"], help="N, dB SPL or off")
    ap.add_argument("--wind", nargs="+", default=["off", "50", "60", "70"], help="W, dB SPL or off")
    ap.add_argument("--birds", nargs="+", default=["45"], help="B, dB SPL or off")
    ap.add_argument("--car", nargs="+", default=["50"], help="C, dB SPL or off")
    ap.add_argument("--length", type=float, default=5.0, help="L, s")
    ap.add_argument("--l1m", type=float, default=80.0, help="drone level at 1 m, dB SPL")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=0, help="throttle wobble and noise of the drone")
    ap.add_argument("--synth", choices=["python", "pd"], default="python",
                    help="drone synth: the numpy port of DroneSound.pd (default) or Pd itself")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "stress" / "run")
    a = ap.parse_args()
    L = a.length
    if any(d > L or d < 0 for d in a.drone_len):
        raise SystemExit("--drone-len must lie in [0, L]")

    cfg, scorers, thr = load_detectors(a.models)
    env = load_env(L)
    print("environment clips:", ", ".join(env), "| thresholds:", {m: round(thr[m], 4) for m in a.models})

    max_r = max(a.start_dist)
    dry = {v: render_drone(v, L + max_r / C_SOUND + 1, a.seed, a.synth) for v in a.speed if any(a.drone_len)}
    print(f"drone renders ({a.synth}):", {v: f"{len(x) / FS:.1f} s" for v, x in dry.items()})

    # D = 0 holds no drone, so speed and distance do not apply: one control clip per environment
    drone_grid = [(v, r, d) for v, r, d in itertools.product(a.speed, a.start_dist, a.drone_len) if d > 0]
    if 0 in a.drone_len:
        drone_grid.append((None, None, 0.0))
    env_grid = list(itertools.product(a.war, a.wind, a.birds, a.car))

    a.out.mkdir(parents=True, exist_ok=True)
    fc = open(a.out / "clips.csv", "w", newline="")
    fh = open(a.out / "hops.csv", "w", newline="")
    clips = hops = None
    n = 0
    for (v, r0, d), (N, W, B, C) in itertools.product(drone_grid, env_grid):
        n += 1
        levels = dict(gunfire=level(N), explosion=level(N), wind=level(W), birds=level(B), car=level(C))
        background, drone, t0, impact, r1 = build_clip(env, dry.get(v), v, r0, d, levels, L, a.l1m)
        clipped = float(np.mean(np.abs(background + drone) > pa(FULL_SCALE_DB)))
        clip = Clip(f"clip{n}", to_model(background + drone), FS, 1, "memory")

        on = drone != 0
        truth = Truth([dict(start_s=t0, end_s=impact or L, **({"impact_s": impact} if impact else {}))] if d > 0 else [])
        row = dict(clip_id=n, L_s=L, V_kmh=v, R0_m=r0, R1_m=None if r1 is None else round(r1, 1), D_s=d,
                   drone_start_s=t0, impact_s=None if impact is None else round(impact, 3),
                   N_war_db=N, W_wind_db=W, B_birds_db=B, C_car_db=C, L1m_db=a.l1m,
                   drone_db_end=round(rms_db(drone[on][-FS // 4:]), 1) if d > 0 else None,  # last 0.25 s heard
                   env_db=round(rms_db(background), 1),
                   snr_db=round(rms_db(drone[on]) - rms_db(background[on]), 1) if d > 0 else None,
                   clipped_frac=round(clipped, 5))
        for m, sc in scorers.items():
            dec = stream(clip, sc, cfg, thr[m])
            lab = window_labels(dec, truth, cfg.window_seconds)
            c = counts(dec, lab)
            lead = lead_times(dec, truth, cfg.window_seconds)
            alarm = dec.t_end[dec.drone]
            res = dict(row, model=m, threshold=round(thr[m], 4), max_score=round(float(dec.score.max()), 4),
                       mean_score=round(float(dec.score.mean()), 4), clip_alarm=bool(dec.drone.any()),
                       first_alarm_s=round(float(alarm[0]), 3) if len(alarm) else None, **c,
                       recall=round(c["tp"] / (c["tp"] + c["fn"]), 4) if c["tp"] + c["fn"] else None,
                       fpr=round(c["fp"] / (c["fp"] + c["tn"]), 4) if c["fp"] + c["tn"] else None,
                       lead_s=None if not lead or lead[0]["lead_s"] is None else round(lead[0]["lead_s"], 3),
                       met_3s=lead[0]["met"] if lead else None)
            if clips is None:
                clips = csv.DictWriter(fc, fieldnames=list(res))
                clips.writeheader()
                hops = csv.DictWriter(fh, fieldnames=["clip_id", "model", "t_end_s", "score", "drone", "label"])
                hops.writeheader()
            clips.writerow(res)
            hops.writerows(dict(clip_id=n, model=m, t_end_s=round(float(t), 3), score=round(float(s), 5),
                                drone=bool(b), label=int(l)) for t, s, b, l in zip(dec.t_end, dec.score, dec.drone, lab))
        if n % 50 == 0:
            print(f"  {n} / {len(drone_grid) * len(env_grid)} clips")
    fc.close()
    fh.close()
    (a.out / "config.json").write_text(json.dumps(dict(vars(a), out=str(a.out), full_scale_db=FULL_SCALE_DB,
                                                       fs=FS, frontend=cfg.describe()), indent=2))
    print(f"{n} clips x {len(scorers)} detectors -> {a.out / 'clips.csv'} and hops.csv")
    plot(a.out)


def plot(out: Path) -> None:
    """Per parameter and detector. Drone clips: window recall (the environment alone can raise an
    alarm, so 'any alarm in the clip' would flatter); D = 0 clips: window false-positive rate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(out / "clips.csv")))
    models = sorted({r["model"] for r in rows})
    params = [("V_kmh", "V speed (km/h)"), ("R0_m", "R0 start distance (m)"), ("D_s", "D drone seconds"),
              ("N_war_db", "N war noise (dB SPL)"), ("W_wind_db", "W wind (dB SPL)")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (p, name) in zip(axes.flat, params + [("N_war_db", "no-drone clips: N war noise (dB SPL)")]):
        control = name.startswith("no-drone")
        col = "fpr" if control else "recall"
        sub = [r for r in rows if (r["D_s"] == "0.0") == control and r[p] != "" and r[col] != ""]
        keys = sorted({r[p] for r in sub}, key=lambda k: -1 if k == "off" else float(k))
        for m in models:
            ys = [np.mean([float(r[col]) for r in sub if r["model"] == m and r[p] == k]) for k in keys]
            ax.plot(keys, ys, marker="o", label=m)
        ax.set_xlabel(name)
        ax.set_ylabel("window false-positive rate" if control else "window recall")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes.flat[0].legend()
    fig.suptitle("Stress test (rendered Pd drone + FSD50K environment) - not detection performance")
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=110)
    print(f"plot -> {out / 'summary.png'}")


if __name__ == "__main__":
    main()
