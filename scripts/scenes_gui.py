"""A window for the labelled scene generator (scripts/make_scenes.py): type the recipe, press
Generate, and watch the scenes come in. Each scene is shown as a linear spectrogram (0-8 kHz,
1024-sample window, hop 256) above its labels on the same time axis: one row per source, the
drone in red. Prev and Next step through the run, Play plays the shown scene.

    uv run python scripts/scenes_gui.py

Runs in the background, so the window stays live; Stop ends a run after the current scene.
Each run goes to data/processed/scenes/gui_<date>_<time>/ (audio/, labels/, manifest.csv).
Tick 'drone only' for scenes that hold one Pd drone and nothing else.
Tick 'dev stems only' to leave out FSD50K eval stems (a held-out test source); today that stops
with "no bed stems", as no bed class has dev stems (report entry 080).
Synthetic scaffolding (CLAUDE.md section 5): never detection performance.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import make_scenes as ms  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from matplotlib.widgets import Button, CheckButtons, TextBox  # noqa: E402

D = ms.parser().parse_args([])
FIELDS = [  # key, label, start, type; a (key, i) key is one end of a range
    ("n", "scenes", D.n, int),
    ("seed", "first seed", D.seed, int),
    ("seconds", "scene length (s)", D.seconds, float),
    ("p_drone", "P(drone)", D.p_drone, float),
    ("p_confuser", "P(confuser)", D.p_confuser, float),
    ("mean_events", "mean events per scene", D.mean_events, float),
    (("snr", 0), "SNR min (dB)", D.snr[0], float),
    (("snr", 1), "SNR max (dB)", D.snr[1], float),
    (("f0", 0), "f0 min (Hz)", D.f0[0], float),
    (("f0", 1), "f0 max (Hz)", D.f0[1], float),
    (("falloff", 0), "falloff min", D.falloff[0], float),
    (("falloff", 1), "falloff max", D.falloff[1], float),
    (("bug_db", 0), "bug level min (dB)", D.bug_db[0], float),
    (("bug_db", 1), "bug level max (dB)", D.bug_db[1], float),
]
TEXT, GREY, DRONE = "#333333", "#777777", "#d62728"

fig = plt.figure(figsize=(16, 9))
fig.canvas.manager.set_window_title("Scene generator")
fig.text(0.015, 0.975, "Scene generator", fontsize=13, weight="bold", color=TEXT, va="top")
boxes = {}
for i, (key, label, v0, _) in enumerate(FIELDS):
    y = 0.90 - i * 0.037
    fig.text(0.015, y + 0.008, label, fontsize=9, color=TEXT)
    boxes[key] = TextBox(fig.add_axes([0.13, y, 0.06, 0.03]), "", initial=f"{v0:g}")
ax_drone = fig.add_axes([0.012, 0.375, 0.2, 0.03])
ax_drone.axis("off")
drone_only = CheckButtons(ax_drone, ["drone only (nothing else in the scene)"], [False])
ax_dev = fig.add_axes([0.012, 0.345, 0.2, 0.03])
ax_dev.axis("off")
dev_only = CheckButtons(ax_dev, ["dev stems only (no held-out audio)"], [False])
b_gen = Button(fig.add_axes([0.015, 0.295, 0.08, 0.04]), "Generate")
b_stop = Button(fig.add_axes([0.11, 0.295, 0.08, 0.04]), "Stop")
status = fig.text(0.015, 0.28, "Type the recipe and press Generate.", fontsize=8, color=TEXT, va="top", wrap=True)
b_prev = Button(fig.add_axes([0.015, 0.06, 0.055, 0.04]), "Prev")
b_next = Button(fig.add_axes([0.075, 0.06, 0.055, 0.04]), "Next")
b_play = Button(fig.add_axes([0.135, 0.06, 0.055, 0.04]), "Play")
info = fig.text(0.015, 0.23, "", fontsize=8, family="monospace", color=TEXT, va="top")
ax_spec = fig.add_axes([0.28, 0.40, 0.70, 0.53])
ax_lab = fig.add_axes([0.28, 0.06, 0.70, 0.28], sharex=ax_spec)

state = dict(thread=None, stop=threading.Event(), out=None, rows=[], shown=None)
events: queue.Queue = queue.Queue()


def params():
    """Generator args from the boxes; ValueError names a bad box or range."""
    a = ms.parser().parse_args([])
    a.snr, a.f0, a.falloff, a.bug_db = list(a.snr), list(a.f0), list(a.falloff), list(a.bug_db)
    for key, label, _, kind in FIELDS:
        try:
            v = kind(boxes[key].text.strip().replace(",", "."))
        except ValueError:
            raise ValueError(f"{label}: type a {'whole ' if kind is int else ''}number") from None
        if isinstance(key, tuple):
            getattr(a, key[0])[key[1]] = v
        else:
            setattr(a, key, v)
    if a.n < 1 or a.seconds <= 0 or a.mean_events < 0:
        raise ValueError("scenes >= 1, length > 0, mean events >= 0")
    ms.check(a)
    a.sources = "fsd50k_dev" if dev_only.get_status()[0] else None
    a.drone_only = drone_only.get_status()[0]
    a.out = ms.ROOT / "data/processed/scenes" / time.strftime("gui_%Y%m%d_%H%M%S")
    return a


def worker(a):
    rows = []
    try:
        pools = None if a.drone_only else ms.load_stems(set(a.sources.split(",")) if a.sources else None)
        for sub in ("audio", "labels"):
            (a.out / sub).mkdir(parents=True, exist_ok=True)
        for i in range(a.n):
            if state["stop"].is_set():
                events.put(("status", f"stopped after {i} scenes"))
                break
            events.put(("status", f"rendering scene {i + 1} of {a.n}"))
            rows.append(ms.make_one(i, a.seed + i, pools, a))
            events.put(("scene", rows[-1]))
        else:
            events.put(("status", f"done: {a.n} scenes in {a.out}"))
    except (Exception, SystemExit) as e:                      # shown in the window, not lost in a thread
        events.put(("status", f"error: {e}"))
    if rows:
        ms.write_manifest(a.out, rows)
    events.put(("done", None))


def show(k):
    """Draw scene k of the current run."""
    state["shown"] = k
    name = state["rows"][k]["scene"]
    x, fs = sf.read(state["out"] / "audio" / f"{name}.wav")
    lab = json.loads((state["out"] / "labels" / f"{name}.json").read_text())
    ax_spec.clear()
    ax_spec.specgram(x, NFFT=1024, Fs=fs, noverlap=1024 - 256, cmap="magma", vmin=-140)
    ax_spec.set_ylabel("Hz")
    ax_spec.set_title(f"{name}  ({k + 1} of {len(state['rows'])})", fontsize=10, loc="left")
    ax_spec.tick_params(labelbottom=False)
    ax_lab.clear()
    evs = lab["events"]
    for j, e in enumerate(evs):
        c = DRONE if e["role"] == "drone" else {"bed": "#9ecae1", "event": "#fdae6b", "confuser": "#756bb1"}[e["role"]]
        ax_lab.barh(j, e["offset"] - e["onset"], left=e["onset"], height=0.6, color=c)
    ax_lab.set_yticks(range(len(evs)), [f"{e['label']} ({e['role']})" for e in evs], fontsize=8)
    ax_lab.invert_yaxis()
    ax_lab.set_xlim(0, lab["seconds"])
    ax_lab.set_xlabel("s")
    d = lab["drone"]
    info.set_text("\n".join([
        f"drone      {'yes' if d else 'no'}",
        f"SNR        {lab['snr_db']} dB" if d else "",
        f"f0 cruise  {d['f0_cruise']} Hz" if d else "",
        f"falloff    {d['falloff']}" if d else "",
        f"speed      {d['speed']} m/s" if d else "",
        f"held out   {'yes' if lab['heldout'] else 'no'}",
    ]))
    fig.canvas.draw_idle()


def poll():
    changed = False
    while not events.empty():
        kind, x = events.get()
        if kind == "status":
            status.set_text(x)
        elif kind == "scene":
            state["rows"].append(x)
            if state["shown"] in (None, len(state["rows"]) - 2):   # follow the newest unless browsing
                show(len(state["rows"]) - 1)
        elif kind == "done":
            state["thread"] = None
        changed = True
    if changed:
        fig.canvas.draw_idle()


def on_generate(_):
    if state["thread"]:
        return
    try:
        a = params()
    except ValueError as e:
        status.set_text(str(e))
        fig.canvas.draw_idle()
        return
    state.update(out=a.out, rows=[], shown=None)
    state["stop"].clear()
    status.set_text("starting")
    fig.canvas.draw_idle()
    state["thread"] = threading.Thread(target=worker, args=(a,), daemon=True)
    state["thread"].start()


def step(d):
    if state["rows"]:
        show(min(max((state["shown"] or 0) + d, 0), len(state["rows"]) - 1))


def on_play(_):
    if state["shown"] is None:
        return
    try:
        import sounddevice as sd
        x, fs = sf.read(state["out"] / "audio" / f"{state['rows'][state['shown']]['scene']}.wav")
        sd.play(x, fs)
    except Exception as e:                                    # no audio device, for example
        status.set_text(f"play failed: {e}")
        fig.canvas.draw_idle()


b_gen.on_clicked(on_generate)
b_stop.on_clicked(lambda _: state["stop"].set())
b_prev.on_clicked(lambda _: step(-1))
b_next.on_clicked(lambda _: step(1))
b_play.on_clicked(on_play)
timer = fig.canvas.new_timer(interval=300)
timer.add_callback(poll)
timer.start()
if __name__ == "__main__":
    plt.show()
