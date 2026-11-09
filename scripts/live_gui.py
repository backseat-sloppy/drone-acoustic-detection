"""Listen for drones on a live microphone: rolling spectrogram, each detector's score, and an alarm light.

    uv run python scripts/live_gui.py --device 2
    uv run python scripts/live_gui.py --device 2 --detector "it10 CNN" --detector comb
    uv run python scripts/live_gui.py --from-wav PeterDrones/02_ddl_310321-T015_P4Pro_12m.wav

`uv run dad live --list-devices` lists the inputs; --device takes an index or part of a name.

The audio path is `dad live`'s (dad.live.LivePipeline): channel 0 (or --channel) at 16 kHz,
resampled in software when the device will not open at 16 kHz, a 1 s window every 0.125 s, each
decision stamped with the time its window ends. Detectors are named as in `dad live`: a preset
("it6 CNN", "it9 CNN", "it9 CRNN", "it10 CNN"), "comb", or a .pt / .pkl checkpoint, each optionally
"@THRESHOLD"; presets and comb bring the threshold their run froze on validation. Default: it6 CNN
and comb, as detect_gui.

Nothing runs until Start. Set up first, on the left (locked while detecting, as a safety; Stop
unlocks them):
    save sound (WAV)        the raw input, every channel, at the capture rate
    save decisions (CSV)    every decision of every detector
    CSV for Excel           on by default: `;` between fields, decimal commas and a first line
                            `sep=;`, so Excel with a decimal comma (Danish settings) opens it in
                            columns with numbers as numbers. Off (or --plain-csv): commas and
                            decimal dots, for pandas and other tools
    sections                every N s: cut the session into sections of "section length" seconds
                            (default 20): one WAV per section, and in the CSV one group of columns
                            per section, side by side (dad.live.SectionRecorder has the layout)
                            on start/end calls: save only between a start and an end call from
                            another script (dad.live_control: start_section(), end_section(); or
                            `uv run dad section start` / `end`), each pair its own WAV and its own
                            columns in the same CSV (dad.live.CalledSectionRecorder); detection
                            and the display run all the time, shaded spans mark the sections
                            none: one WAV and one group of columns for the whole session
    Choose folder...        where the WAVs and the CSV go (default artifacts/live_gui/); every
                            session's files are named live_<date>_<time>..., so nothing is overwritten
Start begins a session at t = 0; Stop ends it and finishes the files. Clear log empties the alarm
log, which lists alarm onsets and ends. Replay a saved WAV with `dad wav` or --from-wav.

The window, from the top:
    banner        DRONE (red) while any detector is at or above its threshold, else listening
                  (green); one line per detector with its latest score
    spectrogram   the log-mel the networks get (64 bands 50-8000 Hz, dB, 80 dB range), last --span s
    scores        each detector's confidence; dashed = its threshold, shaded = alarmed, dotted
                  vertical lines = section boundaries. comb's score has no upper bound and it10
                  CNN's threshold is 6e-6, so a detector whose threshold is above 1 or below 0.01 is
                  drawn as s/(s+threshold), as detect_gui draws comb (threshold at 0.5; the legend
                  says so)
    level         RMS of each window, in dB re full scale

Levels: the networks were trained with 110 dB SPL at full scale and their scores depend on level.
On an uncalibrated microphone the scores are hints, not measurements; --gain-db shifts the input.

The window listens for start/end calls on 127.0.0.1:47800 (--control-port; 0 turns it off),
from this computer only. A call is refused, with the reason, unless the window is detecting with
sections set to "on start/end calls".

--folder, --section-s, --sections and --plain-csv set the starting values of those settings;
--start starts at once. --snapshot S PATH starts at once, saves the window as an image after S seconds of audio and
quits (for checks without a person at the screen).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import math
import queue
import sys
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, CheckButtons, RadioButtons, TextBox

from dad.config import DEFAULT, FrontendConfig
from dad.features import LogMel, hz_to_mel, mel_to_hz
from dad.live import CalledSectionRecorder, LivePipeline, SectionRecorder, load_detector, mic_blocks, wav_blocks
from dad.live_control import DEFAULT_PORT, ControlServer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts/live_gui"
TEXT, GREY, RED, GREEN, LOCKED = "#333333", "#777777", "#c0392b", "#1e8449", "#b0b0b0"
START_COLORS, STOP_COLORS = ("#cfe8d5", "#b5dcbf"), ("#f4c7c3", "#eeaaa4")   # (face, hover)
COLORS = ["#2a78d6", "#e8710a", "#9334e6", "#d93025", "#6b4e16", "#0b8a8a"]
COMB_COLOR = "#1baf7a"
LOG_ROWS = 13
SECTION_MODES = {"every N s (length below)": "fixed", "on start/end calls (from a script)": "calls",
                 "none: one file per session": "none"}


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--device", default=None, help="input device: index or part of its name")
    p.add_argument("--samplerate", type=int, default=None)
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--detector", action="append", default=None, metavar="SPEC[@THRESHOLD]")
    p.add_argument("--window", type=float, default=DEFAULT.window_seconds)
    p.add_argument("--hop", type=float, default=DEFAULT.hop_seconds)
    p.add_argument("--gain-db", type=float, default=0.0)
    p.add_argument("--span", type=float, default=20.0, help="seconds shown")
    p.add_argument("--from-wav", default=None, help="replay a file, paced to real time, instead of the mic")
    p.add_argument("--torch-device", default="cpu")
    p.add_argument("--folder", default=str(OUT_DIR), help="where the WAVs and the CSV go (changeable in the window)")
    p.add_argument("--section-s", type=float, default=20.0, help="section length in seconds")
    p.add_argument("--sections", choices=["fixed", "calls", "none"], default="fixed",
                   help="starting choice of sections: every --section-s seconds, on start/end calls, or none")
    p.add_argument("--no-sections", action="store_true", help="the same as --sections none")
    p.add_argument("--control-port", type=int, default=DEFAULT_PORT,
                   help=f"port for start/end calls on 127.0.0.1 (default {DEFAULT_PORT}; 0: no calls)")
    p.add_argument("--plain-csv", action="store_true",
                   help="start with 'CSV for Excel' off: a plain comma-separated CSV with decimal dots")
    p.add_argument("--start", action="store_true", help="start detecting at once")
    p.add_argument("--snapshot", nargs=2, metavar=("SECONDS", "PATH"), default=None)
    a = p.parse_args()
    if a.device is not None and a.device.isdigit():
        a.device = int(a.device)
    if a.no_sections:
        a.sections = "none"
    return a


args = parse()
cfg = FrontendConfig(window_seconds=args.window, hop_seconds=args.hop)
print(f"frontend: {cfg.describe()}")
detectors = []
for spec in args.detector or ["it6 CNN", "comb"]:
    print(f"loading {spec} ...")
    detectors.append(load_detector(spec, cfg, args.torch_device))
names = [d.name for d in detectors]
thr = {d.name: d.threshold for d in detectors}
color = {k: COMB_COLOR if k == "comb" else COLORS[i % len(COLORS)] for i, k in enumerate(names)}
logmel = LogMel(cfg)
FPS = cfg.sample_rate / cfg.hop_length                      # spectrogram columns per second
N_COLS = int(round(args.span * FPS))
BAND_HZ = mel_to_hz(np.linspace(hz_to_mel(cfg.fmin), hz_to_mel(cfg.fmax), cfg.n_mels + 2))[1:-1]


def squashed(k: str) -> bool:
    return thr[k] is not None and (thr[k] > 1 or 0 < thr[k] < 0.01)


def shown(k: str, s):
    """What the score axis draws: most networks as they are; see squashed()."""
    s = np.asarray(s, dtype=np.float64)
    if squashed(k):
        return np.where(s > 0, s / (s + thr[k]), 0.0)
    return s


def shown_thr(k: str):
    return None if thr[k] is None else float(shown(k, thr[k]))


# ------------------------------------------------------------------ worker

events: queue.Queue = queue.Queue()
calls: queue.Queue = queue.Queue()                          # (cmd, Future) from the control server
live: dict = dict(rec=None)                                 # the running session's recorder, for status


def answer_calls(rec) -> None:
    """Apply waiting start/end calls here, between two blocks, where the audio stands."""
    while True:
        try:
            cmd, fut = calls.get_nowait()
        except queue.Empty:
            return
        try:
            k = rec.begin() if cmd == "start" else rec.end()
            fut.set_result(dict(ok=True, section=k, t_s=round(rec.now_s, 3)))
            events.put(("section", (cmd, k, rec.now_s)))
        except Exception as e:
            fut.set_result(dict(ok=False, error=str(e)))


def refuse_calls(reason: str) -> None:
    while True:
        try:
            _, fut = calls.get_nowait()
        except queue.Empty:
            return
        if not fut.done():
            fut.set_result(dict(ok=False, error=reason))


def worker(stop: threading.Event, session: dict) -> None:
    """Capture, score and save in this thread; hand decisions and status to the GUI through
    `events`. `session` holds the settings as they were at Start."""
    rec = None
    try:
        if args.from_wav:
            sr, ch, blocks = wav_blocks(args.from_wav, realtime=True, stop=stop.is_set)
            name = Path(args.from_wav).name
        else:
            import sounddevice as sd
            sr, ch, blocks = mic_blocks(args.device, args.samplerate, args.channels, stop=stop.is_set)
            name = sd.query_devices(args.device, "input")["name"]
        if not 0 <= args.channel < ch:
            raise ValueError(f"--channel {args.channel}: the input has {ch} channel(s)")
        pipe = LivePipeline(detectors, sr, cfg, args.gain_db)
        prefix = time.strftime("live_%Y%m%d_%H%M%S")
        if session["mode"] == "calls":
            rec = CalledSectionRecorder(session["folder"], prefix, sr, ch, detectors, audio=session["audio"],
                                        table=session["table"], excel=session["excel"])
            events.put(("rec", f"saving between start/end calls: {prefix}*"))
        elif session["audio"] or session["table"]:
            rec = SectionRecorder(session["folder"], prefix, sr, ch, detectors, session["section_s"],
                                  audio=session["audio"], table=session["table"], excel=session["excel"])
            events.put(("rec", f"saving: {prefix}*  (folder below)"))
        live["rec"] = rec
        events.put(("start", f"{name}, {sr} Hz, {ch} ch, channel {args.channel}"
                             + ("" if sr == cfg.sample_rate else f" (resampled to {cfg.sample_rate} Hz)")))
        reported = 0
        for b in blocks:
            if session["mode"] == "calls":
                answer_calls(rec)
            if rec is not None:
                rec.audio(b.audio)
            if b.status:
                events.put(("overflow", b.status))
            for d in pipe.feed(b.audio[:, args.channel]):
                if rec is not None:
                    rec.decision(d)
                events.put(("dec", d))
            if rec is not None and len(rec.errors) > reported:
                events.put(("warn", rec.errors[-1]))
                reported = len(rec.errors)
        events.put(("end", "stopped" if stop.is_set() else "input ended"))
    except Exception as e:                                   # shown in the window, not lost in a thread
        events.put(("error", f"{type(e).__name__}: {e}"))
    finally:
        live["rec"] = None
        refuse_calls("the detector stopped")
        if rec is not None:
            files = rec.close()
            wavs = sum(p.suffix == ".wav" for p in files)
            events.put(("rec", f"saved {wavs} WAV{'' if wavs == 1 else 's'}"
                               + (" + CSV" if rec.csv_path in files else "") + f": {rec.prefix}*"
                               + (f"  (CSV: {rec.errors[-1]})" if rec.errors and rec.csv_path not in files else "")))
        events.put(("done", None))


# ------------------------------------------------------------------ figure

fig = plt.figure(figsize=(15, 9))
fig.canvas.manager.set_window_title("Live drone detection")
fig.text(0.015, 0.975, "Live drone detection", fontsize=13, weight="bold", color=TEXT, va="top")
info = fig.text(0.015, 0.94, "", fontsize=7.5, color=GREY, va="top")
b_run = Button(fig.add_axes([0.015, 0.81, 0.12, 0.045]), "Start", color=START_COLORS[0], hovercolor=START_COLORS[1])
b_clear = Button(fig.add_axes([0.145, 0.81, 0.12, 0.045]), "Clear log")
rec_note = fig.text(0.015, 0.8, "", fontsize=7, color=GREY, va="top")
set_head = fig.text(0.015, 0.755, "settings", fontsize=9, weight="bold", color=TEXT, va="top")
c_save = CheckButtons(fig.add_axes([0.012, 0.665, 0.26, 0.085], frameon=False),
                      ["save sound (WAV)", "save decisions (CSV)", "CSV for Excel (; and decimal comma)"],
                      [True, True, not args.plain_csv])
sec_head = fig.text(0.015, 0.66, "sections", fontsize=8, color=TEXT, va="top")
r_sections = RadioButtons(fig.add_axes([0.012, 0.575, 0.26, 0.08], frameon=False), list(SECTION_MODES),
                          active=list(SECTION_MODES.values()).index(args.sections))
t_section = TextBox(fig.add_axes([0.2, 0.54, 0.06, 0.03]), "section length (s)  ", initial=f"{args.section_s:g}")
b_folder = Button(fig.add_axes([0.015, 0.485, 0.12, 0.04]), "Choose folder...")
folder_text = fig.text(0.015, 0.475, "", fontsize=7, color=TEXT, va="top", family="monospace")
fig.text(0.015, 0.43, "alarm log", fontsize=9, weight="bold", color=TEXT, va="top")
log_text = fig.text(0.015, 0.405, "", fontsize=8, family="monospace", color=TEXT, va="top", linespacing=1.5)
fig.text(0.015, 0.06, "levels: detectors trained with 110 dB SPL at full scale;\n"
                      "on an uncalibrated mic the scores are hints", fontsize=7, color=GREY, va="bottom")
for lbl in [*c_save.labels, *r_sections.labels]:
    lbl.set_fontsize(8)
t_section.label.set_fontsize(8)
SETTINGS = [c_save, r_sections, t_section, b_folder]        # locked while detecting

ax_ban = fig.add_axes([0.3, 0.84, 0.68, 0.12])
ax_ban.set_axis_off()
ban_box = ax_ban.add_patch(Rectangle((0, 0), 1, 1, transform=ax_ban.transAxes, color="#dddddd"))
ban_text = ax_ban.text(0.02, 0.5, "press Start", fontsize=26, weight="bold", color="white", va="center",
                       transform=ax_ban.transAxes)
det_text = ax_ban.text(0.98, 0.5, "", fontsize=10, family="monospace", color="white", va="center", ha="right",
                       transform=ax_ban.transAxes, linespacing=1.4)

ax_spec = fig.add_axes([0.3, 0.47, 0.68, 0.34])
ax_score = fig.add_axes([0.3, 0.19, 0.68, 0.25], sharex=ax_spec)
ax_level = fig.add_axes([0.3, 0.06, 0.68, 0.1], sharex=ax_spec)

spec = np.full((cfg.n_mels, N_COLS), np.nan, np.float32)
cmap = plt.get_cmap("magma").with_extremes(bad="#000004")
img = ax_spec.imshow(spec, origin="lower", aspect="auto", cmap=cmap, interpolation="nearest",
                     extent=(-args.span, 0, 0, cfg.n_mels), vmin=-100, vmax=-20)
ticks = [0, 8, 16, 24, 32, 40, 48, 56, 63]
ax_spec.set_yticks([t + 0.5 for t in ticks], [f"{BAND_HZ[t]:.0f}" for t in ticks])
ax_spec.set_ylabel("Hz (log-mel band centre)", fontsize=8)
ax_spec.tick_params(labelbottom=False, labelsize=7)

lines = {}
for k in names:
    (lines[k],) = ax_score.plot([], [], color=color[k], lw=1.3, label=k + (" (s/(s+thr))" if squashed(k) else ""))
    if shown_thr(k) is not None:
        ax_score.axhline(shown_thr(k), color=color[k], lw=0.8, ls="--")
ax_score.set(ylim=(0, 1.02))
ax_score.set_ylabel("confidence", fontsize=8)
ax_score.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=len(names))
ax_score.grid(color="#e5e5e5", lw=0.5)
ax_score.tick_params(labelbottom=False, labelsize=7)
(level_line,) = ax_level.plot([], [], color=GREY, lw=1.0)
ax_level.set(ylim=(-100, 0), xlim=(0, args.span))
ax_level.set_ylabel("dBFS", fontsize=8)
ax_level.set_xlabel("seconds since start", fontsize=8)
ax_level.grid(color="#e5e5e5", lw=0.5)
ax_level.tick_params(labelsize=7)

state: dict = dict(log=[], folder=Path(args.folder), session=None, thread=None, stop=None, locked=False, bounds=[])


def reset() -> None:
    spec[:] = np.nan
    state.update(t=[], level=[], score={k: [] for k in names}, drone={k: [] for k in names}, cols_shown=0,
                 t_now=0.0, wall0=time.perf_counter(), compute=0.0, overflows=0, on={k: False for k in names},
                 source="", dirty=True, called=[])


reset()
state["status"] = "press Start"


def log(msg: str) -> None:
    state["log"].append(msg)
    del state["log"][:-LOG_ROWS]
    log_text.set_text("\n".join(state["log"]))


def short(p: Path, width: int = 44) -> str:
    """A folder as shown: relative to the repo when inside it, cut from the left when long."""
    try:
        p = p.resolve().relative_to(ROOT)
    except ValueError:
        pass
    text = str(p)
    return text if len(text) <= width else "..." + text[-(width - 3):]


def wrap(text: str, width: int = 60) -> str:
    return "\n".join(text[i:i + width] for i in range(0, len(text), width))


def add_columns(d) -> None:
    """The spectrogram columns this decision adds: the frames that end after the last shown one."""
    target = int(math.floor(d.t_end * FPS))
    c = min(target - state["cols_shown"], int(cfg.window_seconds * FPS) - 3, N_COLS)
    if c <= 0:
        return
    tail = d.window[-((c - 1) * cfg.hop_length + cfg.n_fft):]
    lm = 10 / np.log(10) * logmel.frames(tail)[:, -c:]
    c = lm.shape[1]
    spec[:, :-c] = spec[:, c:]
    spec[:, -c:] = lm
    state["cols_shown"] = target


def on_decision(d) -> None:
    add_columns(d)
    state["t"].append(d.t_end)
    state["level"].append(d.level_dbfs)
    state["compute"] = d.compute_ms
    for k in names:
        state["score"][k].append(d.scores[k])
        on = bool(d.drone[k])
        state["drone"][k].append(on)
        if on != state["on"][k] and d.drone[k] is not None:
            log(f"{time.strftime('%H:%M:%S')} {d.t_end:7.2f}s  " + (f"DRONE {k} {d.scores[k]:.3g}" if on else f"clear {k}"))
        state["on"][k] = on
    keep = int(args.span / cfg.hop_seconds) + 4
    for key in ("t", "level"):
        del state[key][:-keep]
    for k in names:
        del state["score"][k][:-keep], state["drone"][k][:-keep]
    state["t_now"] = d.t_end
    state["dirty"] = True


def redraw() -> None:
    t_now = state["t_now"]
    x0, x1 = max(t_now, args.span) - args.span, max(t_now, args.span)
    img.set_extent((state["cols_shown"] / FPS - args.span, state["cols_shown"] / FPS, 0, cfg.n_mels))
    if np.isfinite(spec).any():
        lo, top = (float(v) for v in np.nanpercentile(spec, [2, 99.5]))
        img.set_clim(max(lo, top - 80), top)               # at most 80 dB, as detect_gui; less when quieter
    img.set_data(spec)
    t = np.asarray(state["t"])
    for coll in list(ax_score.collections):
        coll.remove()
    for k in names:
        y = shown(k, state["score"][k])
        lines[k].set_data(t, np.clip(y, 0, 1.02))
        if len(t):
            ax_score.fill_between(t, 0, 1.02, where=np.asarray(state["drone"][k]), step="pre", color=color[k],
                                  alpha=0.12, lw=0)
    level_line.set_data(t, state["level"])
    ax_spec.set_xlim(x0, x1)
    for a in state["bounds"]:
        a.remove()
    mode = state["session"]["mode"] if state["session"] else None
    sec = state["session"]["section_s"] if mode == "fixed" else None
    state["bounds"] = [] if not sec else [
        ax.axvline(b, color=TEXT, lw=0.9, ls=":", alpha=0.7) for ax in (ax_score, ax_level)
        for b in np.arange(math.ceil(x0 / sec) * sec, x1 + 1e-9, sec) if b > 0]
    for k, a, b in state["called"]:                          # sections from calls: shaded, numbered
        b = t_now if b is None else b
        if b >= x0 and a <= x1:
            state["bounds"] += [ax.axvspan(a, b, color=TEXT, alpha=0.07, lw=0) for ax in (ax_score, ax_level)]
            state["bounds"].append(ax_score.text(max(a, x0), 0.03, f" {k}", fontsize=7, color=TEXT))
    running = state["status"] == "listening"
    any_on = running and any(state["on"].values())         # after Stop the banner says so, not the last alarm
    ban_box.set_color(RED if any_on else GREEN if running else "#999999")
    ban_text.set_text("DRONE" if any_on else state["status"])
    det_text.set_text("\n".join(
        f"{k:<10} {state['score'][k][-1]:>8.3g}" + ("" if thr[k] is None else f" / {thr[k]:.3g}")
        + ("  DRONE" if state["on"][k] else "        ") for k in names) if len(t) else "")
    lag = 1000 * ((time.perf_counter() - state["wall0"]) - t_now) if len(t) and running else 0.0
    where = "" if not sec or not len(t) else f",  section {max(1, math.ceil(round(t_now / sec, 9)))} of {sec:g} s"
    if mode == "calls" and running:
        open_k = [k for k, _, b in state["called"] if b is None]
        where = f",  section {open_k[0]} saving" if open_k else ",  not saving (no section open)"
    info.set_text(f"{state['source']}\n" + cfg.describe().replace(" | 64", "\n64", 1)
                  + f"\nt = {t_now:.2f} s,  scoring {state['compute']:.1f} ms/window,  lag {lag:.0f} ms" + where
                  + (f",  {state['overflows']} input overflows" if state["overflows"] else ""))
    folder_text.set_text(short(state["folder"]) + ("" if control is None else f"\ncalls: 127.0.0.1:{control.port}"))
    fig.canvas.draw_idle()
    state["dirty"] = False


def lock(on: bool) -> None:
    """While detecting, the settings are locked: they belong to the session that is running."""
    state["locked"] = on
    for w in SETTINGS:
        w.active = not on                                    # matplotlib: an inactive widget ignores clicks and keys
    for lbl in [*c_save.labels, *r_sections.labels, sec_head, t_section.label, b_folder.label]:
        lbl.set_color(LOCKED if on else TEXT)
    t_section.text_disp.set_color(LOCKED if on else "black")
    folder_text.set_color(LOCKED if on else TEXT)
    set_head.set_text("settings  (locked while detecting)" if on else "settings")
    face, hover = STOP_COLORS if on else START_COLORS
    b_run.label.set_text("Stop" if on else "Start")
    b_run.color, b_run.hovercolor = face, hover
    b_run.ax.set_facecolor(face)
    if not on:
        on_toggle(None)                                      # the section length stays grey with sections off
    state["dirty"] = True


def settings() -> dict | None:
    """The settings as they stand, checked; None (and the reason in the log) if they will not do."""
    save_audio, save_table, excel = c_save.get_status()
    mode = SECTION_MODES[r_sections.value_selected]
    section_s = None
    if mode == "calls" and not (save_audio or save_table):
        log("start/end calls save sections: tick WAV or CSV")
        return None
    if mode == "fixed":
        try:
            section_s = float(t_section.text.replace(",", "."))
        except ValueError:
            section_s = float("nan")
        if not cfg.hop_seconds <= section_s <= 24 * 3600:
            log(f"section length: {cfg.hop_seconds:g} s or more, not {t_section.text!r}")
            return None
    if save_audio or save_table:
        try:
            state["folder"].mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log(f"folder: {e}"[:60])
            return None
    return dict(audio=save_audio, table=save_table, section_s=section_s, folder=state["folder"], excel=excel,
                mode=mode)


def poll() -> None:
    while True:
        try:
            kind, payload = events.get_nowait()
        except queue.Empty:
            break
        if kind == "dec":
            on_decision(payload)
        elif kind == "start":
            state.update(source=payload, status="listening", dirty=True)
        elif kind == "overflow":
            state["overflows"] += 1
        elif kind == "rec":
            rec_note.set_text(wrap(payload))
            state["dirty"] = True
        elif kind == "section":
            cmd, k, t_s = payload
            if cmd == "start":
                state["called"].append([k, t_s, None])
            else:
                state["called"][-1][2] = t_s
            log(f"{time.strftime('%H:%M:%S')} {t_s:7.2f}s  section {k} " + ("started" if cmd == "start" else "ended"))
            state["dirty"] = True
        elif kind == "warn":
            log("WARN " + payload[:55])
            print(payload, file=sys.stderr)
        elif kind in ("end", "error"):
            state.update(status=payload if kind == "end" else "error", dirty=True)
            if kind == "error":
                log("ERROR " + payload[:54])
                print(payload, file=sys.stderr)
        elif kind == "done":                                 # the worker has closed its files
            lock(False)
    if state["dirty"]:
        redraw()
    if args.snapshot and state["t_now"] >= float(args.snapshot[0]) and not state.get("snapped"):
        state["snapped"] = True
        fig.savefig(args.snapshot[1], dpi=90)
        print(f"snapshot -> {args.snapshot[1]}")
        stop_worker()
        poll()
        plt.close(fig)


def start_worker() -> None:
    t_section.stop_typing()                                  # a value still being typed counts
    session = settings()
    if session is None:
        state["dirty"] = True
        return
    reset()
    refuse_calls("a new session started")                    # nothing from before Start carries over
    state["session"], state["status"] = session, "starting"
    rec_note.set_text("" if session["audio"] or session["table"] else "not saving: both save boxes are off")
    state["stop"] = threading.Event()
    state["thread"] = threading.Thread(target=worker, args=(state["stop"], session), daemon=True)
    lock(True)
    state["thread"].start()


def stop_worker() -> None:
    if state["stop"] is not None:
        state["stop"].set()
        state["thread"].join(timeout=5)


def on_run(_) -> None:
    if state["locked"]:
        stop_worker()
    else:
        start_worker()


def pick_folder() -> str:
    """A folder dialog on top of the window (the Tk backend's own root when there is one)."""
    from tkinter import Tk, filedialog
    get_widget = getattr(fig.canvas, "get_tk_widget", None)
    root = None if get_widget else Tk()
    if root is not None:
        root.withdraw()
    try:
        return filedialog.askdirectory(parent=get_widget() if get_widget else root, mustexist=False,
                                       initialdir=str(state["folder"]), title="Folder for the WAVs and the CSV")
    finally:
        if root is not None:
            root.destroy()


def on_folder(_) -> None:
    if state["locked"]:
        return
    chosen = pick_folder()
    if chosen:
        state["folder"] = Path(chosen)
        state["dirty"] = True


def on_toggle(_) -> None:
    """The section length only matters with sections "every N s", the Excel format
    only with "save decisions (CSV)"; each is greyed when it does not apply."""
    _, table, _ = c_save.get_status()
    divide = SECTION_MODES[r_sections.value_selected] == "fixed"
    c_save.labels[2].set_color(TEXT if table else LOCKED)
    t_section.label.set_color(TEXT if divide else LOCKED)
    t_section.text_disp.set_color("black" if divide else LOCKED)
    fig.canvas.draw_idle()


def on_clear(_) -> None:
    state["log"] = []
    log_text.set_text("")
    fig.canvas.draw_idle()


def on_call(req: dict) -> dict:
    """A start/end/status call from another script (control server thread)."""
    cmd, sess = req.get("cmd"), state["session"]
    detecting = state["locked"]
    if cmd == "status":
        rec = live["rec"]
        return dict(ok=True, detecting=detecting, sections=None if not detecting else sess["mode"],
                    open=getattr(rec, "open", None), t_s=None if rec is None else round(rec.samples / rec.sr, 3))
    if cmd not in ("start", "end"):
        return dict(ok=False, error=f"unknown command {cmd!r}: start, end or status")
    if not detecting:
        return dict(ok=False, error="not detecting: press Start in the window first")
    if sess["mode"] != "calls":
        return dict(ok=False, error="this session's sections are not 'on start/end calls'; choose that before Start")
    fut: concurrent.futures.Future = concurrent.futures.Future()
    calls.put((cmd, fut))
    try:
        return fut.result(timeout=3)
    except concurrent.futures.TimeoutError:
        return dict(ok=False, error="the detector did not take the call within 3 s (no audio arriving?)")


control = None
if args.control_port:
    try:
        control = ControlServer(on_call, args.control_port).start()
    except OSError as e:
        log(f"no calls: port {args.control_port} in use")
        print(f"control port {args.control_port}: {e}", file=sys.stderr)


def on_close(_) -> None:
    stop_worker()
    if control is not None:
        control.close()


b_run.on_clicked(on_run)
b_folder.on_clicked(on_folder)
b_clear.on_clicked(on_clear)
c_save.on_clicked(on_toggle)
r_sections.on_clicked(on_toggle)
on_toggle(None)
fig.canvas.mpl_connect("close_event", on_close)
timer = fig.canvas.new_timer(interval=100)
timer.add_callback(poll)
redraw()
if args.start or args.snapshot:
    start_worker()
timer.start()
plt.show()
