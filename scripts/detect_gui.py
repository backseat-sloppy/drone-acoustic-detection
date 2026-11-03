"""Find drones in your own audio files: open one or more files and the CNN and comb listen to them.

    uv run python scripts/detect_gui.py

"Open audio files..." picks any number of WAV, FLAC, OGG or MP3 files. Each file is read as
dad.wavio reads it (channel 0, resampled to 16 kHz) and every detector scores it hop by hop as it
would run live: a 1 s window every 0.125 s, a decision stamped at the window's end, drone when the
score reaches the frozen iteration 6 threshold (the same detectors and thresholds as
scripts/drone_war_gui.py). Nothing else is set: no sliders, no added noise.

The list on the left gives each file's verdict per detector: DRONE if any window alarmed, with the
share of windows that alarmed, the first alarm and the longest unbroken alarm. Click a file in the
list, or use Previous / Next, to see its spectrogram and each detector's confidence over time
(dashed = threshold; comb is shown as s/(s+threshold), so its threshold sits at 0.5). Play plays the
file as the detectors hear it, normalised to be audible.

Clicking a red file (one where a detector alarmed) also opens its stats window: for the CNN and
the DSP detector (comb), the number of peaks above threshold (unbroken stretches of alarmed
decisions), the time above threshold (alarmed decisions x the 0.125 s hop), the longest peak and
the first alarm, with each peak shaded and numbered on the confidence curve and listed in a table.
Play / Stop sound there play that file.

Levels: the detectors were tuned on WAVs whose full scale is 110 dB SPL. A recording made at
another level is heard louder or softer than it was, and the CNN's scores depend on level, so a
verdict on an uncalibrated recording is a hint, not a measurement.

The table is written to artifacts/detect_gui/results.csv after every batch.
"""
from __future__ import annotations

import csv
import io
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import drone_war_grid as dwg  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from matplotlib.widgets import Button  # noqa: E402

from dad.config import DEFAULT  # noqa: E402
from dad.wavio import read_wav, stream  # noqa: E402

OUT = dwg.ROOT / "artifacts/detect_gui/results.csv"
AUDIO_TYPES = [("Audio files", "*.wav *.flac *.ogg *.mp3"), ("All files", "*.*")]
HOP_S = DEFAULT.samples_per_hop / DEFAULT.sample_rate
TEXT, GREY, RED = "#333333", "#777777", "#c0392b"
NAMES = {"CNN": "CNN", "comb": "DSP (comb)"}                 # detector -> name in the stats window
LIST_ROWS = 22                                               # files shown in the list at once

fig = plt.figure(figsize=(15, 9))
fig.canvas.manager.set_window_title("Find drones in audio files")
fig.text(0.015, 0.975, "Find drones in audio files", fontsize=13, weight="bold", color=TEXT, va="top")
b_open = Button(fig.add_axes([0.015, 0.875, 0.25, 0.045]), "Open audio files...")
b_prev = Button(fig.add_axes([0.015, 0.82, 0.12, 0.04]), "< Previous")
b_next = Button(fig.add_axes([0.145, 0.82, 0.12, 0.04]), "Next >")
b_play = Button(fig.add_axes([0.015, 0.77, 0.12, 0.04]), "Play")
b_hush = Button(fig.add_axes([0.145, 0.77, 0.12, 0.04]), "Stop sound")
status = fig.text(0.015, 0.75, "Open one or more audio files.", fontsize=8, color=TEXT, va="top", wrap=True)
ax_list = fig.add_axes([0.015, 0.03, 0.26, 0.68])
ax_list.axis("off")
ax_spec = fig.add_axes([0.33, 0.56, 0.65, 0.36])
ax_conf = fig.add_axes([0.33, 0.08, 0.65, 0.40], sharex=ax_spec)

state = dict(thread=None, files=[], cur=0, top=0, detectors=None)
events: queue.Queue = queue.Queue()


def detectors():
    """(thresholds, scorers), loaded on first use: the CNN takes a few seconds."""
    if state["detectors"] is None:
        state["detectors"] = dwg._detectors()
    return state["detectors"]


def peaks(drone: np.ndarray) -> list[tuple[int, int]]:
    """The unbroken stretches of alarmed decisions, as (first, last) decision index. One stretch
    is one peak above threshold: the score rises to the threshold, stays, and falls below it."""
    edges = np.diff(np.concatenate([[0], drone.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def detect(path: str) -> dict:
    """Every detector over one file. A file that cannot be read, or is shorter than a window,
    gets an 'error' instead of results."""
    f = dict(path=path, name=Path(path).name)
    try:
        clip = read_wav(path, DEFAULT)
    except Exception as e:                                   # noqa: BLE001  (shown in the list, the batch goes on)
        return {**f, "error": f"cannot read: {type(e).__name__}"}
    f.update(audio=clip.audio, seconds=len(clip.audio) / DEFAULT.sample_rate)
    if len(clip.audio) < DEFAULT.samples_per_window:
        return {**f, "error": f"shorter than the {DEFAULT.window_seconds:g} s window"}
    thr, scorers = detectors()
    f["dec"], f["summary"] = {}, {}
    for k, sc in scorers.items():
        d = stream(clip, sc, DEFAULT, thr[k])
        first = d.t_end[d.drone]
        p = peaks(d.drone)
        f["dec"][k] = d
        f["summary"][k] = dict(drone=bool(d.drone.any()), flagged=float(d.drone.mean()),
                               first_alarm_s=float(first[0]) if len(first) else None,
                               longest_s=max((b - a + 1 for a, b in p), default=0) * HOP_S,
                               peaks=p, above_s=int(d.drone.sum()) * HOP_S)
    return f


def worker(paths):
    try:
        for i, p in enumerate(paths):
            events.put(("status", f"listening to file {i + 1} of {len(paths)}: {Path(p).name}"
                                  + (" (the first file also loads the detectors)" if state["detectors"] is None else "")))
            events.put(("file", detect(p)))
        events.put(("status", f"done: {len(paths)} file(s). Table in {OUT.relative_to(dwg.ROOT).as_posix()}"))
    except Exception as e:                                   # shown in the window, not lost in a thread
        events.put(("status", f"error: {e}"))
    events.put(("done", None))


def verdict(f, k):
    if "error" in f:
        return "-"
    s = f["summary"][k]
    return "DRONE" if s["drone"] else "no drone"


def draw_list():
    """One line per file: number, name, each detector's verdict and share of windows flagged."""
    ax_list.clear()
    ax_list.axis("off")
    ax_list.set_xlim(0, 1)
    ax_list.set_ylim(0, 1)
    files = state["files"]
    if not files:
        return
    ax_list.text(0, 1, f"{'':3s}{'file':16s}{'CNN':>11s}{'comb':>11s}", family="monospace", fontsize=8,
                 weight="bold", color=TEXT, va="top")
    ax_list.cell_rows = {}
    for row, i in enumerate(range(state["top"], min(state["top"] + LIST_ROWS, len(files)))):
        f = files[i]
        name = f["name"] if len(f["name"]) <= 15 else f["name"][:14] + "~"
        if "error" in f:
            cols = f"  {f['error']}"
        else:
            cols = "".join(f"{('DRONE' if f['summary'][k]['drone'] else 'none'):>6s}{100 * f['summary'][k]['flagged']:4.0f}%"
                           for k in dwg.DETECTORS)
        y = 1 - (row + 1.3) / (LIST_ROWS + 3)
        hit = is_hit(f)
        ax_list.text(0, y, f"{'>' if i == state['cur'] else ' '}{i + 1:<2d}{name:16s}{cols}", family="monospace",
                     fontsize=8, color=RED if hit else TEXT, weight="bold" if i == state["cur"] else "normal",
                     va="top", picker=True).file_index = i
    more = len(files) - state["top"] - LIST_ROWS
    note = "click a file to show it, a red one also opens its stats; % = windows that alarmed"
    if state["top"] or more > 0:
        note = f"files {state['top'] + 1}-{min(state['top'] + LIST_ROWS, len(files))} of {len(files)}; scroll here. " + note
    ax_list.text(0, 1 - (LIST_ROWS + 2) / (LIST_ROWS + 3), note, fontsize=7, color=GREY, va="top", wrap=True)


def draw_file():
    """Spectrogram and confidence over time of the current file."""
    ax_spec.clear()
    ax_conf.clear()
    files = state["files"]
    if not files:
        return
    f = files[state["cur"]]
    if "audio" in f:
        ax_spec.specgram(f["audio"], NFFT=1024, Fs=DEFAULT.sample_rate, noverlap=768, cmap="magma", vmin=-160)
    ax_spec.set_ylabel("Hz")
    ax_spec.set_title(f"{state['cur'] + 1}/{len(files)}  {f['name']}" + (f"  ({f['seconds']:.1f} s)" if "seconds" in f else ""),
                      fontsize=10, color=TEXT, loc="left")
    if "error" in f:
        ax_conf.text(0.5, 0.5, f["error"], ha="center", va="center", color=GREY, transform=ax_conf.transAxes)
        return
    thr = detectors()[0]
    lines = []
    for k, d in f["dec"].items():
        y = dwg.squash(d.score, thr[k]) if k == "comb" else d.score
        ax_conf.plot(d.t_end, y, color=dwg.COLORS[k], lw=1.2, label=k)
        ax_conf.axhline(0.5 if k == "comb" else thr[k], color=dwg.COLORS[k], lw=0.8, ls="--")
        ax_conf.fill_between(d.t_end, 0, 1, where=d.drone, color=dwg.COLORS[k], alpha=0.10, step="pre", lw=0)
        s = f["summary"][k]
        lines.append(f"{k}: " + (f"DRONE  first alarm {s['first_alarm_s']:.2f} s, longest alarm {s['longest_s']:.2f} s, "
                                 f"{100 * s['flagged']:.0f}% of windows" if s["drone"] else "no drone heard"))
    ax_conf.set(ylim=(0, 1), xlim=(0, f["seconds"]), ylabel="confidence",
                xlabel="time (s), each decision stamped at the end of its 1 s window")
    ax_conf.set_title("   |   ".join(lines), fontsize=9, color=TEXT, loc="left")
    ax_conf.text(0.005, 0.02, f"dashed = threshold, shaded = alarm, comb shown as s/(s+{thr['comb']:.2f})",
                 fontsize=7, color=GREY, transform=ax_conf.transAxes)
    ax_conf.legend(loc="upper right", fontsize=8, frameon=False)
    ax_conf.grid(color="#e5e5e5", lw=0.5)


def redraw():
    draw_list()
    draw_file()
    fig.canvas.draw_idle()


def write_csv():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "seconds", "detector", "threshold", "verdict", "flagged_share", "first_alarm_s",
                    "longest_alarm_s", "peaks_above_threshold", "time_above_threshold_s", "error"])
        thr = state["detectors"][0] if state["detectors"] else {}
        for f in state["files"]:
            for k in dwg.DETECTORS:
                s = f.get("summary", {}).get(k, {})
                w.writerow([f["path"], round(f["seconds"], 3) if "seconds" in f else "", k, thr.get(k, ""),
                            verdict(f, k), s.get("flagged", ""), s.get("first_alarm_s", ""), s.get("longest_s", ""),
                            len(s["peaks"]) if s else "", s.get("above_s", ""), f.get("error", "")])


def poll():
    changed = False
    while not events.empty():
        kind, x = events.get()
        if kind == "status":
            status.set_text(x)
        elif kind == "file":
            state["files"].append(x)
            if len(state["files"]) == 1:
                state["cur"] = 0
            draw_list()
            if len(state["files"]) - 1 == state["cur"]:
                draw_file()
        elif kind == "done":
            state["thread"] = None
            write_csv()
            draw_list()
            draw_file()                                      # the title's file count is final now
        changed = True
    if changed:
        fig.canvas.draw_idle()


def pick_files():
    """A multi-select file dialog on top of the window (the Tk backend's own root when there is one)."""
    from tkinter import Tk, filedialog
    get_widget = getattr(fig.canvas, "get_tk_widget", None)
    root = None if get_widget else Tk()
    if root is not None:
        root.withdraw()
    try:
        return list(filedialog.askopenfilenames(parent=get_widget() if get_widget else root,
                                                title="Audio files to search for drones", filetypes=AUDIO_TYPES))
    finally:
        if root is not None:
            root.destroy()


def on_open(_):
    if state["thread"]:
        status.set_text("still listening: wait for this batch to finish")
        fig.canvas.draw_idle()
        return
    paths = pick_files()
    if not paths:
        return
    state.update(files=[], cur=0, top=0)
    redraw()
    state["thread"] = threading.Thread(target=worker, args=(paths,), daemon=True)
    state["thread"].start()


def show(i):
    if not state["files"]:
        return
    state["cur"] = i % len(state["files"])
    if not state["top"] <= state["cur"] < state["top"] + LIST_ROWS:   # keep the current file in the list
        state["top"] = max(0, state["cur"] - LIST_ROWS + 1) if state["cur"] >= state["top"] else state["cur"]
    redraw()


def is_hit(f) -> bool:
    return "error" not in f and any(f["summary"][k]["drone"] for k in dwg.DETECTORS)


def open_stats(i):
    """A window with one file's peaks above threshold and time above threshold, per detector:
    confidence over time with each peak shaded and numbered, and a table of the peaks."""
    f = state["files"][i]
    thr = detectors()[0]
    win, axs = plt.subplots(len(f["dec"]), 1, figsize=(14, 3.6 * len(f["dec"])), sharex=True, squeeze=False)
    win.canvas.manager.set_window_title(f"Stats: {f['name']}")
    win.suptitle(f"{f['name']}  ({f['seconds']:.1f} s)", fontsize=11, weight="bold", color=TEXT, x=0.01, ha="left")
    win.subplots_adjust(left=0.05, right=0.62, top=0.9, bottom=0.12, hspace=0.3)
    win.buttons = (Button(win.add_axes([0.64, 0.925, 0.08, 0.045]), "Play"),     # kept on win: a widget
                   Button(win.add_axes([0.73, 0.925, 0.08, 0.045]), "Stop sound"))  # nobody holds stops working
    win.buttons[0].on_clicked(lambda _: play(i=i))
    win.buttons[1].on_clicked(hush)
    for ax, (k, d) in zip(axs[:, 0], f["dec"].items()):
        s, line = f["summary"][k], (0.5 if k == "comb" else thr[k])
        y = dwg.squash(d.score, thr[k]) if k == "comb" else d.score
        ax.plot(d.t_end, y, color=dwg.COLORS[k], lw=1.2)
        ax.axhline(line, color=dwg.COLORS[k], lw=0.8, ls="--")
        rows = []
        for n, (a, b) in enumerate(s["peaks"], 1):
            t0, t1 = d.t_end[a], min(d.t_end[b] + HOP_S, f["seconds"])
            j = a + int(np.argmax(y[a:b + 1]))
            ax.axvspan(t0, t1, color=dwg.COLORS[k], alpha=0.12, lw=0)
            ax.annotate(str(n), (d.t_end[j], y[j]), textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=7, color=TEXT)
            rows.append(f"{n:3d} {t0:8.2f} {t1:8.2f} {(b - a + 1) * HOP_S:7.2f} {y[j]:6.2f}")
        ax.set(ylim=(0, 1.08), xlim=(0, f["seconds"]), ylabel="confidence")
        ax.set_title(NAMES[k], fontsize=10, weight="bold", color=dwg.COLORS[k], loc="left")
        ax.grid(color="#e5e5e5", lw=0.5)
        scored = len(d.score) * HOP_S
        lines = [NAMES[k],
                 f"peaks above threshold   {len(s['peaks'])}",
                 f"time above threshold    {s['above_s']:.2f} s ({100 * s['above_s'] / scored:.1f} % of {scored:.1f} s scored)",
                 f"longest peak            {s['longest_s']:.2f} s",
                 "first alarm             " + ("-" if s["first_alarm_s"] is None else f"{s['first_alarm_s']:.2f} s"), ""]
        if rows:
            lines += ["  # start(s)   end(s)  dur(s)    max"] + rows[:12]
            if len(rows) > 12:
                lines.append(f"  ... and {len(rows) - 12} more")
        bb = ax.get_position()
        win.text(0.64, bb.y1, "\n".join(lines), family="monospace", fontsize=8, color=TEXT, va="top")
    axs[-1, 0].set_xlabel("time (s), each decision stamped at the end of its 1 s window")
    win.text(0.01, 0.02, f"A peak = an unbroken stretch of decisions at or above the threshold (dashed). Each decision "
                         f"holds for one hop ({HOP_S:g} s), so time above threshold = alarmed decisions x {HOP_S:g} s. "
                         f"max = the highest confidence in the peak; comb (DSP) shown as s/(s+{thr['comb']:.2f}).",
             fontsize=7, color=GREY, wrap=True)
    win.show()


def on_pick(event):
    i = getattr(event.artist, "file_index", None)
    if i is not None:
        show(i)
        if is_hit(state["files"][i]):
            open_stats(i)


def on_scroll(event):
    if event.inaxes is not ax_list or len(state["files"]) <= LIST_ROWS:
        return
    step = -3 if event.button == "up" else 3
    state["top"] = int(np.clip(state["top"] + step, 0, len(state["files"]) - LIST_ROWS))
    draw_list()
    fig.canvas.draw_idle()


def play(_=None, i=None):
    """File i (default: the current one). Normalised to the 99.9th percentile, not the peak, so a
    quiet recording with a few loud clicks is still audible; the few louder samples are clipped.
    The detectors get the file as is."""
    i = state["cur"] if i is None else i
    if not state["files"] or "audio" not in state["files"][i]:
        return
    a = state["files"][i]["audio"]
    x = np.clip(a / max(np.percentile(np.abs(a), 99.9), 1e-9) * 0.9, -1, 1)
    if sys.platform == "win32":
        import winsound
        buf = io.BytesIO()
        sf.write(buf, x, DEFAULT.sample_rate, format="WAV", subtype="PCM_16")
        winsound.PlaySound(buf.getvalue(), winsound.SND_MEMORY | winsound.SND_ASYNC)
        return
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(f.name, x, DEFAULT.sample_rate, subtype="PCM_16")
    player = ["afplay"] if sys.platform == "darwin" else ["aplay", "-q"]
    state["player"] = subprocess.Popen(player + [f.name])
    threading.Thread(target=lambda: (state["player"].wait(), Path(f.name).unlink()), daemon=True).start()


def hush(_=None):
    if sys.platform == "win32":
        import winsound
        winsound.PlaySound(None, 0)
    elif state.get("player"):
        state["player"].terminate()


b_open.on_clicked(on_open)
b_prev.on_clicked(lambda _: show(state["cur"] - 1))
b_next.on_clicked(lambda _: show(state["cur"] + 1))
b_play.on_clicked(play)
b_hush.on_clicked(hush)
fig.canvas.mpl_connect("pick_event", on_pick)
fig.canvas.mpl_connect("scroll_event", on_scroll)
timer = fig.canvas.new_timer(interval=300)
timer.add_callback(poll)
timer.start()
plt.show()
