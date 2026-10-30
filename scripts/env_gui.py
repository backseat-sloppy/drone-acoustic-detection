"""One environment clip, set live (iteration 7).

    uv run python scripts/env_gui.py

The 20 s environment of scripts/env_sim.py (no drone) plays in a loop. The
sliders are env_sim's parameters, each on the 1-10 scale (1 = off):

    war        N: gain on the war track window (10 = as rendered, -5 dB a step)
    offset     O: where the 20 s window starts in the 10 min war track
    wind       W: 2..10 = 40..80 dB SPL
    one per distracter class: distance step, 2 = far .. 10 = near

The plot shows each layer's level over the 20 s and the total, and a line
marks where playback is. "New seed" draws other distracter stems, pitches and
start times.

What plays is env_sim.build with these parameters and the seed, so Save
writes exactly that (mono 44.1 kHz float, 1.0 = 110 dB SPL) to --out and adds
the env_sim row to --out/environments.csv; the clip can be rebuilt from its
row. Nothing is written until Save is pressed.

"Listen +dB" is only how loud you hear it: a 60 dB SPL scene is -50 dBFS in
the calibrated file, too quiet to hear, so playback is raised. It does not
change what Save writes.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import distracters as dt  # noqa: E402
import env_sim as es  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402
import soundfile as sf  # noqa: E402
from matplotlib.widgets import Button, Slider  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--length", type=float, default=20.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", type=Path, default=es.ROOT / "artifacts" / "environment" / "saved")
a = ap.parse_args()

n = int(a.length * es.FS)
src = es.load()
HOP = int(0.05 * es.FS)
t_env = (np.arange(n // HOP) + 0.5) * HOP / es.FS
CLASSES = list(dt.CATALOGUE)
state = dict(pos=0, seed=a.seed, mix=np.zeros(n), layers={}, info={}, listen=1.0)
COLORS = dict(war="tab:red", wind="tab:blue",
              **{c: plt.get_cmap("tab20")(i) for i, c in enumerate(CLASSES)})


def level_curve(x):
    return 10 * np.log10(np.mean(x[: n // HOP * HOP].reshape(-1, HOP) ** 2, axis=1) / es.P_REF ** 2 + 1e-30)


def params():
    return (int(s_war.val), float(s_off.val), int(s_wind.val),
            {c: int(s.val) for c, s in s_dist.items() if int(s.val) > 1})


def callback(out, frames, time_info, status):
    i = state["pos"]
    y = state["mix"][np.arange(i, i + frames) % n]
    out[:, 0] = np.clip(y / es.FULL * state["listen"], -1, 1)
    state["pos"] = (i + frames) % n


fig = plt.figure(figsize=(13, 9))
fig.canvas.manager.set_window_title("Environment mixer")
ax = fig.add_axes([0.06, 0.62, 0.91, 0.34])
row_h = 0.032


def slider(k, label, lo, hi, v0, step, color, col=0):
    x0 = 0.12 if col == 0 else 0.55
    return Slider(fig.add_axes([x0, 0.53 - k * row_h, 0.28, 0.022]), label, lo, hi, valinit=v0, valstep=step,
                  color=color)


s_war = slider(0, "war (N)", 1, 10, 8, 1, COLORS["war"])
s_off = slider(1, "offset (O), s", 0, len(src["war"]) / es.FS - a.length, 150, 10, "grey")
s_wind = slider(2, "wind (W)", 1, 10, 4, 1, COLORS["wind"])
listen = slider(4, "listen +dB", 0, 60, 30, 1, "lightgrey")
s_dist = {c: slider(k, c, 1, 10, 1, 1, COLORS[c], col=1) for k, c in enumerate(CLASSES)}
b_play = Button(fig.add_axes([0.12, 0.08, 0.12, 0.05]), "Play / Stop")
b_seed = Button(fig.add_axes([0.26, 0.08, 0.12, 0.05]), "New seed")
b_save = Button(fig.add_axes([0.40, 0.08, 0.12, 0.05]), "Save clip")
info = fig.text(0.12, 0.17, "", fontsize=9, family="monospace", va="bottom")
cursor = ax.axvline(0, color="k", lw=1)
stream = sd.OutputStream(samplerate=es.FS, channels=1, blocksize=1024, callback=callback)


def rebuild(_=None):
    N, O, W, D = params()
    state["mix"], state["layers"], state["info"] = es.build(N, O, W, D, n, state["seed"], src)
    s_war.valtext.set_text(f"{N}  ({'off' if es.war_gain_db(N) is None else f'{es.war_gain_db(N):+g} dB'})")
    s_wind.valtext.set_text(f"{W}  ({'off' if es.step_db(W) is None else f'{es.step_db(W):g} dB SPL'})")
    for c, s in s_dist.items():
        d = dt.step_distance(c, s.val)
        s.valtext.set_text(f"{int(s.val)}  ({'off' if d is None else f'{d:.3g} m'})")
    state["listen"] = 10 ** (listen.val / 20)
    redraw()


def redraw():
    global cursor
    ax.clear()
    for name, x in state["layers"].items():
        ax.plot(t_env, level_curve(x), color=COLORS[name], lw=1, label=name)
    if state["layers"]:
        ax.plot(t_env, level_curve(state["mix"]), color="k", lw=1.6, label="total")
    ax.set_xlim(0, a.length); ax.set_ylim(0, 120)
    ax.set_xlabel("time (s)"); ax.set_ylabel("level, 50 ms RMS (dB SPL)")
    ph = "+".join(state["info"].get("war_phases", [])) or "-"
    ax.set_title(f"Environment, no drone - seed {state['seed']}, war phase: {ph}")
    ax.grid(alpha=0.3)
    if state["layers"]:                                   # nothing plotted, no legend (and no warning)
        ax.legend(loc="upper right", fontsize=8, ncol=7)
    cursor = ax.axvline(state["pos"] / es.FS, color="k", lw=1)
    m = es.describe_mix(state["mix"])
    info.set_text("all layers off" if m["mix_rms_db_spl"] is None else
                  f"mix {m['mix_rms_db_spl']} dB SPL, peak {m['peak_dbfs']} dBFS"
                  f"{' (above 1.0)' if m['clipped'] else ''}, {len(state['info']['war_events'])} war events")
    fig.canvas.draw_idle()


def play(_=None):
    stream.stop() if stream.active else stream.start()


def new_seed(_=None):
    state["seed"] += 1
    rebuild()


def save(_=None):
    a.out.mkdir(parents=True, exist_ok=True)
    csv_path = a.out / "environments.csv"
    old = list(csv.DictReader(open(csv_path, encoding="utf-8"))) if csv_path.exists() else []
    k = len(old) + 1
    name = f"env_{k:03d}.wav"
    sf.write(a.out / name, (state["mix"] / es.FULL).astype(np.float32), es.FS, subtype="FLOAT")
    N, O, W, D = params()
    row = es.row_for(k, name, a.length, state["seed"], N, O, W, D, state["mix"], state["info"])
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not old:
            w.writeheader()
        w.writerow(row)
    info.set_text(info.get_text().split("\nsaved")[0] + f"\nsaved {name}")
    fig.canvas.draw_idle()


def tick():
    cursor.set_xdata([state["pos"] / es.FS])
    fig.canvas.draw_idle()


for s in [s_war, s_off, s_wind, listen, *s_dist.values()]:
    s.on_changed(rebuild)
b_play.on_clicked(play)
b_seed.on_clicked(new_seed)
b_save.on_clicked(save)
timer = fig.canvas.new_timer(interval=100)
timer.add_callback(tick)
timer.start()
rebuild()
plt.show()
stream.close()
