"""A window for the stress test: set the parameters with sliders, build one
clip, see what each detector makes of it, and listen to it.

    uv run python scripts/stress_gui.py

Same clip construction as `scripts/stress_test.py` (Pd drone, five FSD50K
environment clips, 110 dB SPL full scale); nothing is written to disk except
a temporary WAV while playing on macOS. A level slider at 0 turns that sound
off. The drone is rendered by dad.pdsynth (the numpy port of the Pd patch), once per
speed (steps of 10 km/h), and cached.
"""
from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import stress_test as st  # noqa: E402  (sets OMP_NUM_THREADS before torch loads)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from matplotlib.widgets import Button, Slider  # noqa: E402

from dad.wavio import Clip, Truth, counts, lead_times, stream, window_labels  # noqa: E402

L, R0_MAX = 5.0, 600.0
SLIDERS = [  # key, label, min, max, start, step
    ("V", "V speed (km/h)", 50, 200, 100, 10),
    ("R0", "R0 start distance (m)", 10, R0_MAX, 150, 10),
    ("D", "D drone seconds (0 = none)", 0, L, 3, 0.5),
    ("N", "N war noise (dB SPL, 0 = off)", 0, 90, 50, 1),
    ("W", "W wind (dB SPL, 0 = off)", 0, 90, 50, 1),
    ("B", "B birds (dB SPL, 0 = off)", 0, 90, 45, 1),
    ("C", "C car (dB SPL, 0 = off)", 0, 90, 50, 1),
    ("L1m", "drone at 1 m (dB SPL)", 60, 100, 80, 1),
]
COLORS = {"comb": "tab:blue", "lgbm": "tab:green", "dscnn": "tab:orange"}

cfg, scorers, thr = st.load_detectors(list(st.MODELS))
env = st.load_env(L)
DRY_S = L + R0_MAX / st.C_SOUND + 1
# Render the starting speed now: generate() flushes events while Pd runs, and on the macOS backend a
# flush before plt.show() leaves the window deaf to the mouse (every slider and button dead)
dry_cache: dict[float, np.ndarray] = {SLIDERS[0][4]: st.render_drone(SLIDERS[0][4], DRY_S, seed=0)}
last = {}

fig = plt.figure(figsize=(13, 8))
fig.canvas.manager.set_window_title("Drone stress test")
ax_spec = fig.add_axes([0.06, 0.62, 0.62, 0.33])
ax_score = fig.add_axes([0.06, 0.30, 0.62, 0.26], sharex=ax_spec)
ax_info = fig.add_axes([0.72, 0.30, 0.26, 0.65])
ax_info.axis("off")
sliders = {}
for i, (key, label, lo, hi, v0, step) in enumerate(SLIDERS):
    sliders[key] = Slider(fig.add_axes([0.20, 0.22 - i * 0.025, 0.45, 0.018]), label, lo, hi, valinit=v0, valstep=step)
btn_run = Button(fig.add_axes([0.72, 0.17, 0.12, 0.05]), "Generate")
btn_play = Button(fig.add_axes([0.86, 0.17, 0.12, 0.05]), "Play")


def off(db: float) -> float | None:
    return None if db <= 0 else db


def generate(_=None) -> None:
    p = {k: s.val for k, s in sliders.items()}
    if p["D"] > 0 and p["V"] not in dry_cache:
        ax_info.clear(); ax_info.axis("off")
        ax_info.text(0, 1, f"Rendering the drone at {p['V']:.0f} km/h ...", va="top")
        fig.canvas.draw(); fig.canvas.flush_events()
        dry_cache[p["V"]] = st.render_drone(p["V"], DRY_S, seed=0)
    levels = dict(gunfire=off(p["N"]), explosion=off(p["N"]), wind=off(p["W"]), birds=off(p["B"]), car=off(p["C"]))
    bg, drone, t0, impact, r1 = st.build_clip(env, dry_cache.get(p["V"]), p["V"], p["R0"], p["D"], levels, L, p["L1m"])
    audio = st.to_model(bg + drone)
    last["audio"] = audio
    clip = Clip("gui", audio, st.FS, 1, "memory")
    d = p["D"]
    truth = Truth([dict(start_s=t0, end_s=impact or L, **({"impact_s": impact} if impact else {}))] if d > 0 else [])

    ax_spec.clear()
    ax_spec.specgram(audio, NFFT=1024, Fs=cfg.sample_rate, noverlap=768, cmap="magma", vmin=-160)
    ax_spec.set_ylabel("Hz")
    ax_spec.set_title("What the detectors hear (16 kHz, 110 dB SPL full scale)")
    ax_score.clear()
    lines = [f"V {p['V']:.0f} km/h   R0 {p['R0']:.0f} m   D {d:g} s",
             f"N {p['N']:.0f}  W {p['W']:.0f}  B {p['B']:.0f}  C {p['C']:.0f} dB SPL", ""]
    if d > 0:
        on = drone != 0
        ax_score.axvspan(t0, impact or L, color="grey", alpha=0.15, label="drone heard")
        lines += [f"drone heard   {t0:.2f} s -> {(impact or L):.2f} s",
                  f"distance      {p['R0']:.0f} m -> {r1:.0f} m" + ("  (impact)" if impact else ""),
                  f"SNR           {st.rms_db(drone[on]) - st.rms_db(bg[on]):.1f} dB",
                  f"environment   {st.rms_db(bg):.1f} dB SPL", ""]
    else:
        lines += ["no drone (control)", f"environment   {st.rms_db(bg):.1f} dB SPL", ""]
    for m, sc in scorers.items():
        dec = stream(clip, sc, cfg, thr[m])
        lab = window_labels(dec, truth, cfg.window_seconds)
        c = counts(dec, lab)
        ld = lead_times(dec, truth, cfg.window_seconds)
        ax_score.plot(dec.t_end, dec.score / thr[m], color=COLORS[m], label=m)
        rec = f"{c['tp'] / (c['tp'] + c['fn']):.2f}" if c["tp"] + c["fn"] else "  - "
        fpr = f"{c['fp'] / (c['fp'] + c['tn']):.2f}" if c["fp"] + c["tn"] else "  - "
        lead = "" if not ld else ("  no alarm before impact" if ld[0]["lead_s"] is None else f"  lead {ld[0]['lead_s']:.2f} s")
        first = dec.t_end[dec.drone]
        lines.append(f"{m:6} recall {rec}  fp {fpr}  first alarm "
                     f"{first[0]:.2f} s{lead}" if len(first) else f"{m:6} recall {rec}  fp {fpr}  no alarm")
    ax_score.axhline(1, color="red", lw=1, ls="--", label="threshold")
    ax_score.set_ylabel("score / threshold")
    ax_score.set_xlabel("time (s), stamped at each window's end")
    ax_score.set_xlim(0, L)
    ax_score.legend(loc="upper left", fontsize=8)
    ax_info.clear(); ax_info.axis("off")
    ax_info.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)
    fig.canvas.draw_idle()


def play(_=None) -> None:
    """Normalised to the 99.9th percentile, not the peak: a drone that reaches the microphone is
    ~40 dB louder in its last moments than at 150 m, and a peak scale would leave the rest inaudible.
    The few louder samples are clipped. The detectors get the calibrated level."""
    if "audio" not in last:
        return
    x = np.clip(last["audio"] / max(np.percentile(np.abs(last["audio"]), 99.9), 1e-9) * 0.9, -1, 1)
    if sys.platform == "win32":
        import winsound
        buf = io.BytesIO()
        sf.write(buf, x, cfg.sample_rate, format="WAV", subtype="PCM_16")
        winsound.PlaySound(buf.getvalue(), winsound.SND_MEMORY | winsound.SND_ASYNC)
        return
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(f.name, x, cfg.sample_rate, subtype="PCM_16")
    player = ["afplay"] if sys.platform == "darwin" else ["aplay", "-q"]
    threading.Thread(target=lambda: (subprocess.run(player + [f.name]), Path(f.name).unlink()), daemon=True).start()


btn_run.on_clicked(generate)
btn_play.on_clicked(play)
generate()
plt.show()
