"""Find drones in your own audio files: open one or more files and a CNN and comb listen to them.

    uv run python scripts/detect_gui.py

"Open audio files..." picks any number of WAV, FLAC, OGG or MP3 files. Each file is read as
dad.wavio reads it (channel 0, resampled to 16 kHz) and every detector scores it hop by hop as it
would run live: a 1 s window every 0.125 s, a decision stamped at the window's end, drone when the
score reaches the model's frozen threshold. Nothing else is set: no sliders.

Model: the radio buttons at the bottom left pick the network that runs next to comb, for the next
batch opened. Each uses the threshold its own run froze at 5% false alarms on validation:
  it6 CNN   artifacts/models/it6_final_none_simdist_dscnn.pt (model 6, as scripts/drone_war_gui.py)
  it9 CNN   artifacts/models/final_dscnn_none_simdist_dscnn.pt (report entry 096)
  it9 CRNN  artifacts/models/final_crnn_none_simdist_crnn.pt (report entry 096)
  it10 CNN  artifacts/it10_clean/clean_s0_dscnn.pt (seed 0, trained on synthetic drones only)
comb keeps model 6's frozen threshold whichever model is picked.

War-shaped noise (optional): tick "add war-shaped noise" before opening files and type one or
more SNRs in dB, separated by spaces, commas or semicolons (each -20 to +30, at most 6, default
"0 10 30"; lower = more noise; decimals with a point). Each file then gets steady noise with the
war's long-term spectrum (scripts/war_shaped_noise.py, seed 1000 + the file's place in the batch,
the same noise at every SNR), scaled so the file's broadband RMS is SNR dB above the noise's
(docs/war_shaped_noise.md). Every SNR is its own row in the list; the detectors score the mix, Play
plays the mix, and the mix is saved as a 16 kHz float WAV in artifacts/detect_gui/mixes/. Below the
spectrogram the file gets one confidence graph per SNR, the selected one in bold; click a graph to
select that SNR (and hear it with Play). The setting applies to the next batch opened.

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
from matplotlib.widgets import Button, CheckButtons, RadioButtons, TextBox  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from dad.config import DEFAULT  # noqa: E402
from dad.wavio import read_wav, stream  # noqa: E402

OUT = dwg.ROOT / "artifacts/detect_gui/results.csv"
MIXES = OUT.parent / "mixes"
SNR_MIN, SNR_MAX, SNR_DEFAULT = -20.0, 30.0, "0 10 30"      # dB, file RMS over noise RMS
SNR_MAX_COUNT = 6                                            # graphs under the spectrogram
NOISE_LOOP_S = 60.0                                          # longer files repeat it (the noise is periodic)
AUDIO_TYPES = [("Audio files", "*.wav *.flac *.ogg *.mp3"), ("All files", "*.*")]
HOP_S = DEFAULT.samples_per_hop / DEFAULT.sample_rate
TEXT, GREY, RED = "#333333", "#777777", "#c0392b"
NAMES = {"CNN": "CNN", "comb": "DSP (comb)"}                 # detector -> name in the stats window
LIST_ROWS = 18                                               # files shown in the list at once
MODELS = dwg.MODELS

fig = plt.figure(figsize=(15, 9))
fig.canvas.manager.set_window_title("Find drones in audio files")
fig.text(0.015, 0.975, "Find drones in audio files", fontsize=13, weight="bold", color=TEXT, va="top")
ax_noise = fig.add_axes([0.015, 0.905, 0.15, 0.04], frameon=False)
c_noise = CheckButtons(ax_noise, ["add war-shaped noise"], [False])
for lbl in c_noise.labels:
    lbl.set_fontsize(8)
t_snr = TextBox(fig.add_axes([0.2, 0.91, 0.065, 0.03]), "SNR dB ", initial=SNR_DEFAULT)
t_snr.label.set_fontsize(8)
fig.text(0.135, 0.9, f"one or more, {SNR_MIN:+g} to {SNR_MAX:+g}, lower = more noise", fontsize=6.5, color=GREY,
         va="top")
b_open = Button(fig.add_axes([0.015, 0.84, 0.25, 0.045]), "Open audio files...")
b_prev = Button(fig.add_axes([0.015, 0.79, 0.12, 0.04]), "< Previous")
b_next = Button(fig.add_axes([0.145, 0.79, 0.12, 0.04]), "Next >")
b_play = Button(fig.add_axes([0.015, 0.745, 0.12, 0.04]), "Play")
b_hush = Button(fig.add_axes([0.145, 0.745, 0.12, 0.04]), "Stop sound")
status = fig.text(0.015, 0.73, "Open one or more audio files.", fontsize=8, color=TEXT, va="top", wrap=True)
ax_list = fig.add_axes([0.015, 0.15, 0.26, 0.54])
ax_model = fig.add_axes([0.015, 0.01, 0.25, 0.12], frameon=False)
ax_model.set_title("model, for the next batch opened", fontsize=8, color=TEXT, loc="left")
r_model = RadioButtons(ax_model, list(MODELS), active=0)
for lbl in r_model.labels:
    lbl.set_fontsize(8)
ax_list.axis("off")
ax_spec = fig.add_axes([0.33, 0.62, 0.65, 0.30])
CONF_BOX = (0.33, 0.07, 0.65, 0.50)                          # left, bottom, width, height of the graphs

state = dict(thread=None, files=[], cur=0, top=0, model=next(iter(MODELS)), loaded={}, conf_axes=[])
events: queue.Queue = queue.Queue()


def detectors():
    """(thresholds, scorers) of the batch's model and comb, loaded on first use: a network takes a few seconds."""
    m = state["model"]
    if m not in state["loaded"]:
        state["loaded"][m] = dwg._detectors(m)
    return state["loaded"][m]


def shown_as(k):
    """Detector k as shown: the picked model's name for the network."""
    return state["model"] if k == "CNN" else NAMES[k]


def peaks(drone: np.ndarray) -> list[tuple[int, int]]:
    """The unbroken stretches of alarmed decisions, as (first, last) decision index. One stretch
    is one peak above threshold: the score rises to the threshold, stays, and falls below it."""
    edges = np.diff(np.concatenate([[0], drone.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def war_noise(n: int, seed: int) -> np.ndarray:
    """n samples of war-shaped noise at the frontend's rate. Made at 44.1 kHz, the rate of the
    gain table, then resampled; at most NOISE_LOOP_S long, repeated for longer files."""
    sr = dwg.es.FS
    m = min(n, int(NOISE_LOOP_S * DEFAULT.sample_rate))
    x = dwg.wsn.shaped_noise(np.loadtxt(dwg.WSN_GAIN), sr, m / DEFAULT.sample_rate + 0.1, seed)
    return np.resize(resample_poly(x, DEFAULT.sample_rate, sr)[:m], n)


def add_noise(audio: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """audio plus war-shaped noise, the noise scaled so audio's broadband RMS is snr_db above it."""
    noise = war_noise(len(audio), seed)
    rms = lambda v: np.sqrt(np.mean(v ** 2))                  # noqa: E731
    return (audio + noise * rms(audio) / rms(noise) / 10 ** (snr_db / 20)).astype(audio.dtype)


def detect(path: str, noise: tuple[float, int] | None = None, group: int = 0) -> dict:
    """Every detector over one file, with war-shaped noise added first when noise = (SNR dB, seed).
    group = the file's place in the batch, shared by its rows at every SNR. A file that cannot be
    read, or is shorter than a window, gets an 'error' instead of results."""
    f = dict(path=path, name=Path(path).name, group=group)
    try:
        clip = read_wav(path, DEFAULT)
    except Exception as e:                                   # noqa: BLE001  (shown in the list, the batch goes on)
        return {**f, "error": f"cannot read: {type(e).__name__}"}
    f.update(audio=clip.audio, seconds=len(clip.audio) / DEFAULT.sample_rate)
    if len(clip.audio) < DEFAULT.samples_per_window:
        return {**f, "error": f"shorter than the {DEFAULT.window_seconds:g} s window"}
    if noise is not None:
        snr, seed = noise
        if not np.any(clip.audio):
            return {**f, "error": "silent: no level to set the noise against"}
        clip.audio = add_noise(clip.audio, snr, seed)
        f.update(audio=clip.audio, snr_db=snr, noise_seed=seed)
        MIXES.mkdir(parents=True, exist_ok=True)
        f["mix_path"] = str(MIXES / f"{Path(path).stem}_wsn_snr{snr:+g}dB_seed{seed}.wav")
        sf.write(f["mix_path"], clip.audio, DEFAULT.sample_rate, subtype="FLOAT")
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


def worker(paths, snrs=None):
    """Every file, at every SNR in snrs (None: no noise). One row per file and SNR."""
    try:
        for i, p in enumerate(paths):
            for snr in snrs or [None]:
                events.put(("status", f"listening to file {i + 1} of {len(paths)}: {Path(p).name}"
                                      + ("" if snr is None else f" at SNR {snr:+g} dB")
                                      + (" (the first file also loads the detectors)" if state["model"] not in state["loaded"] else "")))
                events.put(("file", detect(p, None if snr is None else (snr, dwg.WSN_SEED + i), group=i)))
        events.put(("status", f"done: {len(paths)} file(s). " + (f"Table in {OUT.relative_to(dwg.ROOT).as_posix()}"
                              if snrs is None else f"Table, mixes in {OUT.parent.relative_to(dwg.ROOT).as_posix()}")))
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
    ax_list.text(0, 1, f"{'':3s}{'file':16s}{shown_as('CNN'):>11s}{'comb':>11s}", family="monospace", fontsize=8,
                 weight="bold", color=TEXT, va="top")
    ax_list.cell_rows = {}
    for row, i in enumerate(range(state["top"], min(state["top"] + LIST_ROWS, len(files)))):
        f = files[i]
        tag = f" {f['snr_db']:+g}" if "snr_db" in f else ""      # the SNR stays visible, the name is cut
        name = f["name"] if len(f["name"]) + len(tag) <= 15 else f["name"][:14 - len(tag)] + "~"
        name += tag
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


def siblings(i: int) -> list[int]:
    """Rows of the same file as row i (one per SNR), in list order."""
    g = state["files"][i]["group"]
    return [j for j, f in enumerate(state["files"]) if f["group"] == g]


def conf_axes(n: int) -> list:
    """n confidence graphs stacked in CONF_BOX, sharing the spectrogram's time axis."""
    for ax in state["conf_axes"]:
        fig.delaxes(ax)
    state["conf_axes"] = []
    if not n:
        return []
    left, bottom, width, height = CONF_BOX
    gap = 0.035 if n > 1 else 0.0
    h = (height - gap * (n - 1)) / n
    state["conf_axes"] = [fig.add_axes([left, bottom + (n - 1 - r) * (h + gap), width, h], sharex=ax_spec)
                          for r in range(n)]
    return state["conf_axes"]


def plot_score(ax, d, y, k):
    """Confidence over time with alarms shaded. A short file has few decisions (a 1 s file has
    one, at 1.00 s), and a line through one point draws nothing, so few decisions get dots and
    each alarmed one shades the 1 s window it heard."""
    few = len(d.t_end) < 20
    ax.plot(d.t_end, y, color=dwg.COLORS[k], lw=1.2, label=shown_as(k), marker="o" if few else None, ms=5,
            clip_on=not few)                                 # the last dot sits on the right edge
    if few:
        for t in d.t_end[d.drone]:
            ax.axvspan(max(t - DEFAULT.window_seconds, 0), t, color=dwg.COLORS[k], alpha=0.10, lw=0)
    else:
        ax.fill_between(d.t_end, 0, 1, where=d.drone, color=dwg.COLORS[k], alpha=0.10, step="pre", lw=0)


def draw_conf(ax, f, thr, selected, legend):
    """One row's confidence over time: both detectors, thresholds dashed, alarms shaded."""
    head = f"SNR {f['snr_db']:+g} dB:  " if "snr_db" in f else ""
    if "error" in f:
        ax.text(0.5, 0.5, f["error"], ha="center", va="center", color=GREY, transform=ax.transAxes)
        ax.set_title(head, fontsize=9, color=TEXT, loc="left", weight="bold" if selected else "normal")
        return
    lines = []
    for k, d in f["dec"].items():
        y = dwg.squash(d.score, thr[k]) if k == "comb" else d.score
        plot_score(ax, d, y, k)
        ax.axhline(0.5 if k == "comb" else thr[k], color=dwg.COLORS[k], lw=0.8, ls="--")
        s = f["summary"][k]
        lines.append(f"{shown_as(k)}: " + (f"DRONE  first alarm {s['first_alarm_s']:.2f} s, longest alarm {s['longest_s']:.2f} s, "
                                 f"{100 * s['flagged']:.0f}% of windows" if s["drone"] else "no drone heard"))
    ax.set(ylim=(0, 1), xlim=(0, f["seconds"]), ylabel="confidence")
    ax.set_title(head + "   |   ".join(lines), fontsize=9, color=TEXT, loc="left",
                 weight="bold" if selected else "normal")
    if legend:
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(color="#e5e5e5", lw=0.5)
    if selected and "snr_db" in f:
        for side in ax.spines.values():
            side.set(linewidth=1.6, color=TEXT)


def draw_file():
    """Spectrogram of the current row, and one confidence graph per row of the same file
    (one per SNR; just one without noise)."""
    ax_spec.clear()
    files = state["files"]
    if not files:
        conf_axes(0)
        return
    f = files[state["cur"]]
    if "audio" in f:
        ax_spec.specgram(f["audio"], NFFT=1024, Fs=DEFAULT.sample_rate, noverlap=768, cmap="magma", vmin=-160)
    ax_spec.set_ylabel("Hz")
    ax_spec.set_title(f"{state['cur'] + 1}/{len(files)}  {f['name']}" + (f"  ({f['seconds']:.1f} s)" if "seconds" in f else "")
                      + (f"  + war-shaped noise, SNR {f['snr_db']:+g} dB (seed {f['noise_seed']})" if "snr_db" in f else ""),
                      fontsize=10, color=TEXT, loc="left")
    rows = siblings(state["cur"])
    axes = conf_axes(len(rows))
    thr = detectors()[0] if any("error" not in files[j] for j in rows) else {}
    for r, (ax, j) in enumerate(zip(axes, rows)):
        ax.row_index = j                                     # a click on the graph selects this row
        draw_conf(ax, files[j], thr, selected=j == state["cur"], legend=r == 0)
        if r < len(rows) - 1:
            ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("time (s), each decision stamped at the end of its 1 s window")
    if thr:
        note = f"dashed = threshold, shaded = alarm, comb shown as s/(s+{thr['comb']:.2f})"
        if len(rows) > 1:
            note += "; bold = the row shown above and played; click a graph to select it"
        axes[-1].text(0.005, 0.03, note, fontsize=7, color=GREY, transform=axes[-1].transAxes)


def redraw():
    draw_list()
    draw_file()
    fig.canvas.draw_idle()


def write_csv():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "seconds", "detector", "threshold", "verdict", "flagged_share", "first_alarm_s",
                    "longest_alarm_s", "peaks_above_threshold", "time_above_threshold_s", "noise_snr_db",
                    "noise_seed", "mix_file", "error"])
        thr = state["loaded"].get(state["model"], ({},))[0]
        for f in state["files"]:
            for k in dwg.DETECTORS:
                s = f.get("summary", {}).get(k, {})
                w.writerow([f["path"], round(f["seconds"], 3) if "seconds" in f else "", shown_as(k), thr.get(k, ""),
                            verdict(f, k), s.get("flagged", ""), s.get("first_alarm_s", ""), s.get("longest_s", ""),
                            len(s["peaks"]) if s else "", s.get("above_s", ""), f.get("snr_db", ""),
                            f.get("noise_seed", ""), f.get("mix_path", ""), f.get("error", "")])


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
            if x["group"] == state["files"][state["cur"]]["group"]:   # the shown file got another SNR graph
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


def noise_snrs(text: str) -> list[float]:
    """The SNRs typed in the box, in the order typed, repeats dropped; ValueError says what is wrong."""
    parts = text.replace(",", " ").replace(";", " ").split()
    if not parts:
        raise ValueError(f"SNR: type one or more numbers of dB from {SNR_MIN:+g} to {SNR_MAX:+g}")
    snrs = []
    for p in parts:
        try:
            snr = float(p)
        except ValueError:
            raise ValueError(f"SNR: '{p}' is not a number (decimals with a point, values apart with spaces)") from None
        if not SNR_MIN <= snr <= SNR_MAX:
            raise ValueError(f"SNR: {snr:g} dB is outside {SNR_MIN:+g} to {SNR_MAX:+g}")
        if snr not in snrs:
            snrs.append(snr)
    if len(snrs) > SNR_MAX_COUNT:
        raise ValueError(f"SNR: at most {SNR_MAX_COUNT} values, got {len(snrs)}")
    return snrs


def on_open(_):
    if state["thread"]:
        status.set_text("still listening: wait for this batch to finish")
        fig.canvas.draw_idle()
        return
    snr = None
    if c_noise.get_status()[0]:
        try:
            snr = noise_snrs(t_snr.text)
        except ValueError as e:
            status.set_text(str(e))
            fig.canvas.draw_idle()
            return
    paths = pick_files()
    if not paths:
        return
    state.update(files=[], cur=0, top=0, model=r_model.value_selected)
    redraw()
    state["thread"] = threading.Thread(target=worker, args=(paths, snr), daemon=True)
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
        few = len(d.t_end) < 20                              # see plot_score
        ax.plot(d.t_end, y, color=dwg.COLORS[k], lw=1.2, marker="o" if few else None, ms=5, clip_on=not few)
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
        ax.set_title(shown_as(k), fontsize=10, weight="bold", color=dwg.COLORS[k], loc="left")
        ax.grid(color="#e5e5e5", lw=0.5)
        scored = len(d.score) * HOP_S
        lines = [shown_as(k),
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


def on_click(event):
    """A click on one of the SNR graphs selects that row: its spectrogram, and Play plays it."""
    j = getattr(event.inaxes, "row_index", None)
    if j is not None and j != state["cur"] and event.inaxes in state["conf_axes"]:
        show(j)


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
        # winsound cannot play from memory asynchronously (RuntimeError), so play a temp file;
        # the previous one is stopped first and deleted
        import winsound
        winsound.PlaySound(None, 0)
        old, f = state.get("wav"), tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        f.close()
        sf.write(f.name, x, DEFAULT.sample_rate, subtype="PCM_16")
        state["wav"] = f.name
        if old:
            Path(old).unlink(missing_ok=True)
        winsound.PlaySound(f.name, winsound.SND_FILENAME | winsound.SND_ASYNC)
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
fig.canvas.mpl_connect("button_press_event", on_click)
timer = fig.canvas.new_timer(interval=300)
timer.add_callback(poll)
timer.start()
plt.show()
