"""REAPER projects for the 39-speaker array (test 3, natural environment simulation), built from the Live demo test's
parts: its drone files, its drone level, its speeds, its background layers at their levels, and its propagation
(dad.propagation.propagate: 1 / r spreading, ISO 9613-1 air absorption, Doppler).

    uv run python scripts/reaper_array.py [--hours 5] [--per-hour 12] [--seed 0] [--out artifacts/reaper_array] [--verify]

Writes into --out:

    allday.RPP          --hours of background, drones at random times (one per slot of 3600 / per-hour s)
    demo.RPP            a background-only region to loop, then one region per drone; marker N jumps to drone N
    media/              background_loop.wav (60 s), event_NNN.wav (the drone pool), speaker_check.wav; 39 ch, 48 kHz
    speakers.csv        the layout (row N = hardware output N) and the VBAP triangles; speakers.png draws it
    background.csv      every layer on every speaker: source, loop offset, gain, level
    pool.csv            every rendered drone event: path, speed, file, geometry
    events.csv          every drone on the all-day timeline (and demo.csv for the demo)
    trajectories/       per pool event, every 10 ms: position, distance, direction, level, 39 speaker gains
    project.json        every setting, the commit, the command, the md5 of every WAV
    drone_array_speaker_view.jsfx   a live view on the master: the array from above, each speaker lit by the
                        level on its output as it plays (also copied into REAPER's Effects folder, where REAPER finds it)
    verify.csv          (--verify) REAPER's render of the speaker check and of the first drone, against the WAVs

Routing: the master bus has 40 channels and sends channel N to hardware output N (N = 1..39). Three tracks of 40
channels feed it: Background (the loop), Drones, Speaker check (muted; 1 s of pink noise per speaker in turn).

Conventions (ASSUMED until Mads/Marius confirm): azimuth counter-clockwise from the front (90 = left), elevation up;
the listener (microphone) at the array's centre. Levels in WAV units: full scale = 110 dB SPL; the speakers are not
calibrated, so project.json carries calibration_db = null until someone measures the speaker check at the centre.
Left out: the engine's ground reflection and outdoor reverb (written for 16 kHz; the room adds its own).
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).parent))
import complete_test_engine as E  # noqa: E402
from dad import propagation as prop  # noqa: E402

SR = 48000
ROOT = Path(__file__).resolve().parents[1]
LAYOUT = [(0, 90), (0, 60), (60, 60), (120, 60), (180, 60), (240, 60), (300, 60),
          *[(a, 30) for a in range(0, 360, 36)], *[(a, 0) for a in range(0, 360, 30)], *[(a, -30) for a in range(0, 360, 36)]]
N = len(LAYOUT)                                              # 39
DRONE_DIR = ROOT / "data/recordedDroneClipped"                # the live test's drone files (not ITU ARIS)
DRONE_DB = 80.0                                              # dB SPL at 1 m, the live test's level
SPEEDS = (50.0, 100.0, 150.0)                                # km/h, the live test's speeds
TOWARDS_S, ACROSS_S = 10.0, 10.0                             # 10 s each, as the live test (Marius, 2026-11-23)
CLOSEST_M = 1.0                                              # towards: the path passes this far above the centre
FRAME_S = 0.01                                               # VBAP gains and trajectory rows every 10 ms
LOOP_S, FADE_S = 60.0, 0.1                                   # background loop and its seam crossfade
# The live test's layers and levels (its latest run's config, 2026-11-22). spread "all": every speaker, each from its
# own offset in the loop (diffuse), only for steady noise: a layer with events (shots, a pass-by) played that way repeats
# each event once per speaker (entry 254). A list: the outputs (1-based) that play its one copy, the same signal on each.
LAYERS = {
    "war soundscape": (E.WAR_SOUNDSCAPE, 60.0, [20, 28]),    # (60, 0), (300, 0): front left and front right
    "wind (Pd)": (E._wind_file(E.WIND_PD), 50.0, "all"),
    "cars": (E._cars_file(E.CAR_PD), 5.0, [21]),             # (90, 0)
    "helicopter": (E._heli_file(), 50.0, [11]),              # (108, 30)
    "insects": (E._bees_file(), 30.0, [33, 35]),             # (108, -30), (180, -30): near the ground
}
CHECK_DB, CHECK_S, CHECK_GAP_S = 70.0, 1.0, 0.25            # speaker check: pink noise per speaker


def unit(az, el):
    az, el = np.radians(az), np.radians(el)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)


SPK = unit(*np.array(LAYOUT, float).T)                       # 39 x 3
TRI = ConvexHull(SPK).simplices                              # VBAP triangles
TRI_INV = np.linalg.inv(SPK[TRI])                            # g @ SPK[tri] = p  ->  g = p @ inv


def vbap(p: np.ndarray) -> np.ndarray:
    """Power-normalised VBAP gains (frames x 39) for directions p (frames x 3, any length)."""
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    g = np.einsum("fi,tij->ftj", p, TRI_INV)                 # frames x triangles x 3
    best = g.min(axis=2).argmax(axis=1)                      # the triangle the direction falls in
    gb = np.clip(g[np.arange(len(p)), best], 0, None)
    out = np.zeros((len(p), N))
    np.put_along_axis(out, TRI[best], gb / np.linalg.norm(gb, axis=1, keepdims=True), axis=1)
    return out


def read_mono(f: Path) -> np.ndarray:
    x, sr = sf.read(f, dtype="float64", always_2d=True)
    return resample_poly(x.mean(axis=1), SR, sr) if sr != SR else x.mean(axis=1)


def loop_to(x: np.ndarray, n: int) -> np.ndarray:
    """x repeated to n samples, each seam an equal-power crossfade."""
    f = int(FADE_S * SR)
    w = np.linspace(0, np.pi / 2, f)
    out = x.copy()
    while len(out) < n:
        out = np.concatenate([out[:-f], out[-f:] * np.cos(w) + x[:f] * np.sin(w), x[f:]])
    return out[:n]


def seamless(x: np.ndarray, start: int, n: int) -> np.ndarray:
    """n samples of x (wrapping) from start, whose end runs on into its beginning: a loop REAPER can repeat."""
    f = int(FADE_S * SR)
    y = np.take(x, np.arange(start, start + n + f), mode="wrap")
    w = np.linspace(0, np.pi / 2, f)
    y[:f] = y[:f] * np.sin(w) + y[n:n + f] * np.cos(w)
    return y[:n]


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def db(x):
    return E.FULL_SCALE_DB + 20 * np.log10(max(x, 1e-12))


def write39(f: Path, y: np.ndarray):
    """A 39-channel 16-bit WAV (frames x 39). 16 bit: the floor sits at 14 dB SPL, below any layer."""
    peak = float(np.abs(y).max())
    if peak >= 1.0:
        raise ValueError(f"{f.name} clips (peak {peak:.2f}): lower a level")
    sf.write(f, y.astype(np.float32), SR, subtype="PCM_16")


# ------------------------------------------------------------------ the drone events

def path(kind: str, v: float, T: float, rng) -> tuple[dict, callable]:
    """A straight flight at v m/s: its parameters and p(t) (emission time, s -> positions, m)."""
    az = float(rng.uniform(0, 360))
    if kind == "towards":                                    # dives at `climb` degrees onto the centre, impact at T
        climb = float(rng.uniform(5, 40))
        u = unit(az, climb)
        geo = dict(azimuth_deg=az, climb_deg=climb, height_m=None, closest_m=CLOSEST_M)
        return geo, lambda t: u[None] * (v * (T - t))[:, None] + np.array([0, 0, CLOSEST_M])
    h, d = float(rng.uniform(10, 60)), float(rng.uniform(5, 50))   # across: passes d m to one side, h m up, at T / 2
    c = np.array([d * np.cos(np.radians(az)), d * np.sin(np.radians(az)), h])
    side = rng.choice([-1, 1])
    u = side * np.array([-np.sin(np.radians(az)), np.cos(np.radians(az)), 0])
    geo = dict(azimuth_deg=az, climb_deg=None, height_m=h, closest_m=float(np.hypot(d, h)))
    return geo, lambda t: c[None] + u[None] * (v * (t - T / 2))[:, None]


def render_event(k: int, kind: str, speed: float, src: Path, rng, media: Path, traj: Path) -> dict:
    T = TOWARDS_S if kind == "towards" else ACROSS_S
    x = read_mono(src)
    x = x[-int(T * SR):] if kind == "towards" else loop_to(x, int(T * SR))   # towards: the file's end is the impact
    x *= E.rms_for(DRONE_DB) / rms(x)                         # RMS = the level at 1 m
    v = speed / 3.6
    geo, pos = path(kind, v, T, rng)
    t = np.arange(len(x)) / SR
    p = pos(t)
    r = np.linalg.norm(p, axis=1)
    y = prop.propagate(x, r, fs=SR)                           # heard at the centre, Doppler included
    arrive = t + (r - r.min()) / prop.speed_of_sound()        # propagate()'s timeline: the common delay dropped
    tf = np.arange(0, len(y) / SR, FRAME_S)                   # heard time of each frame
    pf = pos(np.interp(tf, arrive, t))                        # where the drone was when that sound left it
    gf = vbap(pf)
    n = len(y)
    idx = np.arange(n) / SR / FRAME_S
    g = np.stack([np.interp(idx, np.arange(len(tf)), gf[:, s]) for s in range(N)], 1)
    write39(media / f"event_{k:03d}.wav", y[:, None] * g)
    hop = int(FRAME_S * SR)
    lvl = [db(rms(y[i:i + 10 * hop])) for i in range(0, n, hop)]   # 100 ms windows
    rf = np.linalg.norm(pf, axis=1)
    with open(traj / f"event_{k:03d}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["heard_s", "emitted_s", "x_m", "y_m", "z_m", "distance_m", "azimuth_deg", "elevation_deg",
                    "level_db_spl_100ms"] + [f"g{s + 1:02d}" for s in range(N)])
        te = np.interp(tf, arrive, t)
        for i in range(len(tf)):
            w.writerow([f"{tf[i]:.3f}", f"{te[i]:.3f}", *(f"{c:.2f}" for c in pf[i]), f"{rf[i]:.2f}",
                        f"{np.degrees(np.arctan2(pf[i, 1], pf[i, 0])) % 360:.1f}", f"{np.degrees(np.arcsin(pf[i, 2] / rf[i])):.1f}",
                        f"{lvl[min(i, len(lvl) - 1)]:.1f}"] + [f"{gf[i, s]:.4f}" for s in range(N)])
    energy = (g ** 2 * (y ** 2)[:, None]).sum(0)
    used = [s + 1 for s in range(N) if gf[:, s].max() > 0.01]
    i_close = int(np.argmin(r))
    return dict(event=k, file=f"media/event_{k:03d}.wav", path=kind, speed_kmh=speed, level_db_spl_1m=DRONE_DB,
                source=str(src.relative_to(ROOT)), source_part=("last" if kind == "towards" else "looped") + f" {T:g} s",
                duration_s=round(n / SR, 6), closest_or_impact_s=round(float(arrive[i_close]), 6),
                start_distance_m=round(float(r[0]), 1), **{k2: (round(v2, 2) if v2 is not None else "") for k2, v2 in geo.items()},
                start_azimuth_deg=round(float(np.degrees(np.arctan2(p[0, 1], p[0, 0])) % 360), 1),
                start_elevation_deg=round(float(np.degrees(np.arcsin(p[0, 2] / r[0]))), 1),
                end_azimuth_deg=round(float(np.degrees(np.arctan2(p[-1, 1], p[-1, 0])) % 360), 1),
                end_elevation_deg=round(float(np.degrees(np.arcsin(p[-1, 2] / r[-1]))), 1),
                doppler=True, peak_db_spl_centre=round(max(lvl), 1), speakers=" ".join(map(str, used)),
                loudest_speaker_whole_event=int(energy.argmax()) + 1)


# ------------------------------------------------------------------ background and speaker check

def background(rng, media: Path) -> list[dict]:
    n = int(LOOP_S * SR)
    out, rows = np.zeros((n, N)), []
    for name, (f, level, spread) in LAYERS.items():
        if not f.exists():
            raise FileNotFoundError(f"{name}: {f} missing (render it once with the Complete test GUI)")
        x = read_mono(f)
        spk = list(range(1, N + 1)) if spread == "all" else spread
        # the sum at the centre is `level`: incoherent copies add in power, one copy on several speakers in amplitude
        gain = E.rms_for(level) / rms(x) / (np.sqrt(len(spk)) if spread == "all" else len(spk))
        one = int(rng.integers(len(x)))
        for s in spk:
            off = int(rng.integers(len(x))) if spread == "all" else one
            out[:, s - 1] += gain * seamless(x, off, n)
            rows.append(dict(layer=name, speaker=s, source=str(f.relative_to(ROOT)), offset_s=round(off / SR, 3),
                             gain=f"{gain:.6g}", level_db_spl_speaker=round(level - (10 if spread == "all" else 20) * np.log10(len(spk)), 2),
                             level_db_spl_layer=level, spread="all" if spread == "all" else "fixed"))
    write39(media / "background_loop.wav", out)
    return rows


def speaker_check(rng, media: Path):
    k, g = int(CHECK_S * SR), int(CHECK_GAP_S * SR)
    out = np.zeros(((k + g) * N, N))
    for s in range(N):
        X = np.fft.rfft(rng.standard_normal(k))
        X[1:] /= np.sqrt(np.arange(1, len(X)))               # pink
        x = np.fft.irfft(X, k)
        x *= E.rms_for(CHECK_DB) / rms(x) * np.hanning(k) ** 0.1
        out[s * (k + g): s * (k + g) + k, s] = x
    write39(media / "speaker_check.wav", out)


# ------------------------------------------------------------------ the speaker view (a live meter in REAPER)

JSFX = "drone_array_speaker_view.jsfx"                       # REAPER/Effects/<this>, on the master of both projects


def map_xy(p: np.ndarray) -> np.ndarray:
    """Directions (n x 3) on the top view: centre = straight up, radius 90 - elevation; front up, left left."""
    p = p / np.linalg.norm(p, axis=-1, keepdims=True)
    az, rho = np.arctan2(p[..., 1], p[..., 0]), 90 - np.degrees(np.arcsin(np.clip(p[..., 2], -1, 1)))
    return np.stack([-rho * np.sin(az), rho * np.cos(az)], -1)


def jsfx() -> str:
    """A JSFX that passes the 39 channels through untouched and draws the array from above (as speakers.png), each
    speaker lit by the level on its output over the last 50 ms (dB SPL at full scale = 110), with the loudest named."""
    xy = map_xy(SPK)
    el = np.array(LAYOUT, float)[:, 1]
    edges = sorted({tuple(sorted((int(tri[i]), int(tri[j])))) for tri in TRI if not (el[tri] == -30).all()
                    for i, j in ((0, 1), (1, 2), (2, 0))})
    pins = "".join(f"in_pin:{s + 1}\n" for s in range(N)) + "".join(f"out_pin:{s + 1}\n" for s in range(N))
    pos = " ".join(f"X[{s}]={xy[s, 0]:.2f}; Y[{s}]={xy[s, 1]:.2f};" for s in range(N))
    edg = " ".join(f"E[{2 * i}]={a}; E[{2 * i + 1}]={b};" for i, (a, b) in enumerate(edges))
    return f"""desc:39-speaker array view (drone-acoustic-detection)
//tags: analysis meter
// Generated by scripts/reaper_array.py. Audio passes through unchanged.
slider1:30<0,110,1>Dark below (dB SPL)
slider2:80<0,110,1>Full brightness at (dB SPL)
{pins}options:no_meter

@init
N={N}; NE={len(edges)}; X=0; Y=100; ACC=200; LV=300; E=1000;
{pos}
{edg}
cnt=0;

@sample
i=0; loop(N, ACC[i] += sqr(spl(i)); i += 1;);
(cnt += 1) >= srate * 0.05 ? (
  i=0; loop(N, LV[i] = max(sqrt(ACC[i] / cnt), LV[i] * 0.6); ACC[i] = 0; i += 1;);
  cnt = 0;
);

@gfx 640 680
gfx_clear = 0;
s = min(gfx_w, gfx_h - 40) / 300; cx = gfx_w / 2; cy = 40 + (gfx_h - 40) / 2;
gfx_setfont(1, "Arial", max(10, 6 * s));
gfx_set(0.3, 0.3, 0.3, 1);
i=0; loop(NE, a=E[2*i]; b=E[2*i+1]; gfx_line(cx + X[a]*s, cy - Y[a]*s, cx + X[b]*s, cy - Y[b]*s); i += 1;);
best=0; bdb=-999; tot=0;
i=0; loop(N,
  tot += sqr(LV[i]);
  d = LV[i] > 0 ? 110 + 20 * log10(LV[i]) : -999;
  d > bdb ? (bdb = d; best = i;);
  k = min(1, max(0, (d - slider1) / (slider2 - slider1)));
  k > 0 ? gfx_set(0.35 + 0.65 * k, 0.25 + 0.7 * k * k, 0.2 * (1 - k), 1) : gfx_set(0.22, 0.22, 0.22, 1);
  gfx_circle(cx + X[i]*s, cy - Y[i]*s, (8 + 10 * k) * s, 1, 1);
  gfx_set(1, 1, 1, 1);
  sprintf(#n, "%d", i + 1); gfx_measurestr(#n, w, h);
  gfx_x = cx + X[i]*s - w / 2; gfx_y = cy - Y[i]*s - h / 2; gfx_drawstr(#n);
  i += 1;
);
gfx_set(0.8, 0.8, 0.8, 1);
#l = "front"; gfx_measurestr(#l, w, h); gfx_x = cx - w / 2; gfx_y = cy - 135 * s; gfx_drawstr(#l);
#l = "back"; gfx_measurestr(#l, w, h); gfx_x = cx - w / 2; gfx_y = cy + 128 * s; gfx_drawstr(#l);
gfx_x = cx - 148 * s; gfx_y = cy; gfx_drawstr("left");
#l = "right"; gfx_measurestr(#l, w, h); gfx_x = cx + 148 * s - w; gfx_y = cy; gfx_drawstr(#l);
gfx_set(1, 1, 1, 1); gfx_x = 10; gfx_y = 10;
tot > 0.000000000001 ? gfx_printf("loudest: speaker %d (%.0f dB SPL)    all 39: %.0f dB SPL", best + 1, bdb, 110 + 10 * log10(tot))
        : gfx_drawstr("silence");
"""


def install_jsfx(out: Path):
    """The view beside the projects, and in this machine's REAPER Effects folder, where REAPER looks for it."""
    (out / JSFX).write_text(jsfx())
    fx = Path.home() / "Library/Application Support/REAPER/Effects"
    if fx.is_dir():
        (fx / JSFX).write_text(jsfx())
        print(f"speaker view installed: {fx / JSFX}")
    else:
        print(f"REAPER's Effects folder not found: copy {out / JSFX} into it (REAPER: Options > Show resource path)")


# ------------------------------------------------------------------ REAPER projects

def item(pos, length, f, name, loop=0):
    return (f'    <ITEM\n      POSITION {pos:.6f}\n      LENGTH {length:.6f}\n      LOOP {loop}\n      NAME "{name}"\n'
            f'      <SOURCE WAVE\n        FILE "{f}"\n      >\n    >\n')


def track(name, items, mute=False):
    return (f'  <TRACK\n    NAME "{name}"\n    VOLPAN 1 0 -1 -1 1\n    MUTESOLO {int(mute)} 0 0\n    NCHAN 40\n'
            f'    MAINSEND 1 0\n{"".join(items)}  >\n')


def project(tracks: str, markers: list[str], extra: str = "") -> str:
    """A REAPER project: master of 40 channels, channel N to hardware output N, the speaker view on the master."""
    hw = "".join(f"  MASTERHWOUT {i | 1024} 0 1 0 0 0 {i | 1024} -1\n" for i in range(N))   # master ch N -> output N
    view = ("  <MASTERFXLIST\n    SHOW 0\n    LASTSEL 0\n    DOCKED 0\n    BYPASS 0 0 0\n"
            f'    <JS "{JSFX}" ""\n      30 80' + " -" * 62 + "\n    >\n    FLOAT 120 80 640 680\n    WAK 0 0\n  >\n")
    return ('<REAPER_PROJECT 0.1 "7.84/macOS-arm64" 0\n  SAMPLERATE 48000 1 0\n  MASTER_NCH 40 2\n'
            '  MASTER_VOLUME 1 0 -1 -1 1\n' + hw + view + extra +
            '  RENDER_FILE "render"\n  RENDER_PATTERN ""\n  RENDER_FMT 0 40 48000\n  RENDER_RANGE 1 0 0 0 1000\n'
            '  <RENDER_CFG\n    ZXZhdxgAAQ==\n  >\n' + "".join(markers) + tracks + ">\n")


def rpp(length_s, drone_items, markers, check_len):
    return project(track("Background", [item(0, length_s, "media/background_loop.wav", "background", loop=1)]) +
                   track("Drones", drone_items) +
                   track("Speaker check (1 s per speaker, 1..39)", [item(0, check_len, "media/speaker_check.wav", "speaker check")],
                         mute=True), markers)


def region(i, a, b, name):
    return f'  MARKER {i} {a:.6f} "{name}" 1\n  MARKER {i} {b:.6f} "" 1\n'


def label(e):
    return f"{e['path']} {e['speed_kmh']:g} km/h az {e['azimuth_deg']:.0f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hours", type=float, default=5.0)
    ap.add_argument("--per-hour", type=float, default=12.0)
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/reaper_array")
    ap.add_argument("--verify", action="store_true", help="render with REAPER and check (verify.csv)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out, media, traj = a.out, a.out / "media", a.out / "trajectories"
    for d in (media, traj):
        d.mkdir(parents=True, exist_ok=True)

    with open(out / "speakers.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["output", "azimuth_deg", "elevation_deg", "x", "y", "z", "triangles"])
        for s, (az, el) in enumerate(LAYOUT):
            tris = [i for i, t in enumerate(TRI) if s in t]
            w.writerow([s + 1, az, el, *(f"{c:.4f}" for c in SPK[s]), " ".join(map(str, tris))])
    draw_array(out / "speakers.png")
    install_jsfx(out)

    print("background loop ...")
    bg = background(rng, media)
    write_csv(out / "background.csv", bg)
    speaker_check(rng, media)
    check_len = N * (CHECK_S + CHECK_GAP_S)

    files = sorted(Path(f) for f in glob.glob(str(DRONE_DIR / "*.wav")))
    pool = []
    for k in range(1, a.pool + 1):
        kind = "towards" if k % 2 else "across"
        speed = SPEEDS[(k - 1) // 2 % len(SPEEDS)]
        src = files[int(rng.integers(len(files)))]
        print(f"event {k}/{a.pool}: {kind} {speed:g} km/h, {src.name}")
        pool.append(render_event(k, kind, speed, src, rng, media, traj))
    write_csv(out / "pool.csv", pool)

    # all day: one drone per slot, at a random time inside it, never overlapping the next slot
    length = a.hours * 3600
    slot = 3600 / a.per_hour
    events, items, marks = [], [], []
    for i in range(int(length // slot)):
        e = pool[int(rng.integers(len(pool)))]
        t0 = round((i * slot + float(rng.uniform(5, slot - e["duration_s"] - 5))) * SR) / SR   # on a sample
        events.append(dict(n=i + 1, start_s=round(t0, 6), end_s=round(t0 + e["duration_s"], 6),
                           closest_or_impact_s=round(t0 + e["closest_or_impact_s"], 6),
                           **{k: v for k, v in e.items() if k not in ("duration_s", "closest_or_impact_s")}))
        items.append(item(t0, e["duration_s"], e["file"], f"{i + 1}: event {e['event']}"))
        marks.append(region(i + 1, t0, t0 + e["duration_s"], f"{i + 1}: event {e['event']} {label(e)}"))
    write_csv(out / "events.csv", events)
    (out / "allday.RPP").write_text(rpp(length, items, marks, check_len))

    # demo: region 1 is background only (loop it); then each pool event once, marker N one second before drone N
    t, demo, items, marks = LOOP_S, [], [], [region(100, 0, LOOP_S, "background only: loop me")]
    for e in pool:
        t0 = t + 2
        items.append(item(t0, e["duration_s"], e["file"], f"drone {e['event']}"))
        marks.append(f'  MARKER {e["event"]} {t0 - 1:.6f} "drone {e["event"]}: {label(e)}" 0\n')
        marks.append(region(100 + e["event"], t0, t0 + e["duration_s"], f"drone {e['event']}"))
        demo.append(dict(marker=e["event"], start_s=round(t0, 6), closest_or_impact_s=round(t0 + e["closest_or_impact_s"], 6),
                         end_s=round(t0 + e["duration_s"], 6), event=e["event"], label=label(e)))
        t = round((t0 + e["duration_s"] + 3) * SR) / SR
    write_csv(out / "demo.csv", demo)
    (out / "demo.RPP").write_text(rpp(t, items, marks, check_len))

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    meta = dict(command="uv run python scripts/reaper_array.py " + " ".join(sys.argv[1:]), commit=commit, seed=a.seed,
                hours=a.hours, per_hour=a.per_hour, pool=a.pool, sample_rate=SR, wav="PCM_16, 39 channels",
                full_scale_db_spl=E.FULL_SCALE_DB, calibration_db=None,
                conventions="azimuth counter-clockwise from the front (90 = left), elevation up; row N = output N (ASSUMED)",
                drone=dict(dir=str(DRONE_DIR.relative_to(ROOT)), level_db_spl_1m=DRONE_DB, speeds_kmh=SPEEDS,
                           towards_s=TOWARDS_S, across_s=ACROSS_S, closest_m=CLOSEST_M,
                           towards_climb_deg=[5, 40], across_height_m=[10, 60], across_offset_m=[5, 50]),
                propagation=dict(model="dad.propagation.propagate", doppler=True, temp_c=prop.TEMP_C, rh_pct=prop.RH_PCT,
                                 pa_kpa=prop.PA_KPA, speed_of_sound=prop.speed_of_sound(), left_out="ground reflection, outdoor reverb"),
                panning=dict(method="VBAP over the convex hull, power-normalised", frame_s=FRAME_S, triangles=len(TRI)),
                background=dict(loop_s=LOOP_S, seam_fade_s=FADE_S,
                                layers={k: dict(source=str(f.relative_to(ROOT)), level_db_spl=lv, spread=sp) for k, (f, lv, sp) in LAYERS.items()}),
                speaker_check=dict(level_db_spl=CHECK_DB, burst_s=CHECK_S, gap_s=CHECK_GAP_S, noise="pink"),
                md5={p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(media.glob("*.wav"))})
    (out / "project.json").write_text(json.dumps(meta, indent=2))
    print(f"done: {out}")
    if a.verify and not verify(out):
        sys.exit(1)


# ------------------------------------------------------------------ check: REAPER's own render against what we wrote

REAPER = "/Applications/REAPER.app/Contents/MacOS/REAPER"


def verify(out: Path):
    """Renders two stretches of allday.RPP with REAPER and writes verify.csv: (1) the speaker check alone (background
    muted): each burst must land on its own output and nowhere else, bit-exact; (2) background + the first drone:
    the render must equal background_loop.wav repeated plus the event at its events.csv start (16-bit steps only)."""
    t = (out / "allday.RPP").read_text()
    bg_on = 'MUTESOLO 0 0 0\n    NCHAN 40\n    MAINSEND 1 0\n    <ITEM\n      POSITION 0.000000\n      LENGTH ' + f"{float(t.split('LENGTH ')[1].split()[0]):.6f}"
    chk_off = 'MUTESOLO 1 0 0\n    NCHAN 40\n    MAINSEND 1 0\n    <ITEM\n      POSITION 0.000000\n      LENGTH ' + f"{N * (CHECK_S + CHECK_GAP_S):.6f}"
    assert bg_on in t and chk_off in t
    e = next(csv.DictReader(open(out / "events.csv")))
    end = float(e["end_s"]) + 2
    jobs = {"check": (N * (CHECK_S + CHECK_GAP_S), t.replace(bg_on, bg_on.replace("MUTESOLO 0", "MUTESOLO 1")).replace(chk_off, chk_off.replace("MUTESOLO 1", "MUTESOLO 0"))),
            "mix": (end, t)}
    rows, ev = [], sf.read(out / e["file"])[0]
    bg = sf.read(out / "media/background_loop.wav")[0]
    for name, (length, text) in jobs.items():
        wav = out / f"verify_{name}.wav"
        text = text.replace("RENDER_RANGE 1 0 0 0 1000", f"RENDER_RANGE 0 0 {length:.6f} 0 1000").replace('RENDER_FILE "render"', f'RENDER_FILE "{wav}"')
        (out / f"verify_{name}.RPP").write_text(text)
        subprocess.run([REAPER, "-nosplash", "-renderproject", str(out / f"verify_{name}.RPP")], capture_output=True, timeout=600)
        x = sf.read(wav)[0]
        if name == "check":
            ref = sf.read(out / "media/speaker_check.wav")[0]
            step = int((CHECK_S + CHECK_GAP_S) * SR)
            for s in range(N):
                lv = np.sqrt((x[s * step:s * step + int(CHECK_S * SR), :N] ** 2).mean(0))
                rows.append(dict(test="speaker check", item=s + 1, expected_output=s + 1, loudest_output=int(lv.argmax()) + 1,
                                 level_db_spl=round(db(lv.max()), 2), other_outputs_max_db_spl=round(db(np.sort(lv)[-2]), 2),
                                 max_abs_diff=float(np.abs(x[s * step:(s + 1) * step, :N] - ref[s * step:(s + 1) * step]).max())))
        else:
            exp = np.take(bg, np.arange(len(x)), axis=0, mode="wrap")
            j = int(round(float(e["start_s"]) * SR))
            exp[j:j + len(ev)] += ev[:len(x) - j]
            dr = x[j:j + len(ev), :N] - np.take(bg, np.arange(j, j + len(ev)), axis=0, mode="wrap")   # the drone in the render
            rows.append(dict(test="background + drone", item=f"event {e['event']} at {e['start_s']} s", expected_output=e["loudest_speaker_whole_event"],
                             loudest_output=int((dr ** 2).sum(0).argmax()) + 1, level_db_spl=round(db(np.sqrt((x[:, :N] ** 2).sum(1).mean())), 2),
                             other_outputs_max_db_spl="", max_abs_diff=float(np.abs(x[:, :N] - exp).max())))
        wav.unlink()
        (out / f"verify_{name}.RPP").unlink()
    write_csv(out / "verify.csv", rows)
    bad = [r for r in rows if r["max_abs_diff"] > 2 ** -15 or (r["test"] == "speaker check" and r["loudest_output"] != r["expected_output"])]
    print(f"verify: {len(rows)} checks, {len(bad)} failed -> {out / 'verify.csv'}")
    return not bad


def draw_array(f: Path):
    """speakers.png: the 39 speakers seen from above (centre = straight up, outer ring = -30 degrees; front at the
    top, left on the left) with the VBAP triangles the drone is panned over, and the same on an azimuth-elevation map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    az, el = np.array(LAYOUT, float).T
    rho = 90 - el
    px, py = -rho * np.sin(np.radians(az)), rho * np.cos(np.radians(az))
    col = {90: "tab:purple", 60: "tab:red", 30: "tab:orange", 0: "tab:green", -30: "tab:blue"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw=dict(width_ratios=[1, 1.3]))
    for tri in TRI:
        if (el[tri] == -30).all():                             # the floor under the lowest ring: no drone goes there
            continue
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a.plot(px[[tri[i], tri[j]]], py[[tri[i], tri[j]]], color="0.85", lw=0.7, zorder=0)
    for e, c in col.items():
        a.add_patch(plt.Circle((0, 0), 90 - e, fill=False, ls=":", color=c, alpha=0.5))
    for s in range(N):
        for ax, x, y in ((a, px[s], py[s]), (b, az[s], el[s])):
            ax.scatter(x, y, s=330, color=col[int(el[s])], zorder=2)
            ax.text(x, y, str(s + 1), ha="center", va="center", color="white", fontsize=8, weight="bold", zorder=3)
    a.annotate("front (0)", (0, 128), ha="center"), a.annotate("left (90)", (-128, 0), ha="center", rotation=90, va="center")
    a.annotate("right (270)", (128, 0), ha="center", rotation=-90, va="center"), a.annotate("back (180)", (0, -132), ha="center")
    a.set(xlim=(-140, 140), ylim=(-140, 140), aspect="equal", xticks=[], yticks=[],
          title="From above: centre = straight up, rings at 60, 30, 0, -30 degrees elevation\n(numbers = hardware outputs; grey = VBAP triangles)")
    b.set(xlim=(-15, 345), ylim=(-45, 100), xticks=range(0, 360, 30), yticks=[-30, 0, 30, 60, 90],
          xlabel="azimuth (degrees, counter-clockwise from the front: 90 = left, ASSUMED)", ylabel="elevation (degrees)",
          title="Azimuth and elevation of each output")
    b.grid(alpha=0.3)
    for s in (a, b):
        for sp in s.spines.values():
            sp.set_visible(s is b)
    fig.suptitle("39-speaker array (row N of the layout = output N, ASSUMED)", y=0.995)
    fig.tight_layout()
    fig.savefig(f, dpi=120)
    plt.close(fig)


def write_csv(f: Path, rows: list[dict]):
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
