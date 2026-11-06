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

The window, from the top:
    banner        DRONE (red) while any detector is at or above its threshold, else listening
                  (green); one line per detector with its latest score
    spectrogram   the log-mel the networks get (64 bands 50-8000 Hz, dB, 80 dB range), last --span s
    scores        each detector's confidence; dashed = its threshold, shaded = alarmed. comb's score
                  has no upper bound and it10 CNN's threshold is 6e-6, so a detector whose threshold
                  is above 1 or below 0.01 is drawn as s/(s+threshold), as detect_gui draws comb
                  (threshold at 0.5; the legend says so)
    level         RMS of each window, in dB re full scale

Buttons: Stop / Start (Start begins a new session at t = 0), Record (raw input, every channel, at the
capture rate, plus every decision as CSV, in artifacts/live_gui/; replay the WAV with `dad wav` or
--from-wav), Clear (empties the alarm log). The log on the left lists alarm onsets and ends.

Levels: the networks were trained with 110 dB SPL at full scale and their scores depend on level.
On an uncalibrated microphone the scores are hints, not measurements; --gain-db shifts the input.

--snapshot S PATH saves the window as an image after S seconds of audio and quits (for checks
without a person at the screen).
"""
from __future__ import annotations

import argparse
import csv
import math
import queue
import sys
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button

from dad.config import DEFAULT, FrontendConfig
from dad.features import LogMel, hz_to_mel, mel_to_hz
from dad.live import LivePipeline, load_detector, mic_blocks, wav_blocks

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts/live_gui"
TEXT, GREY, RED, GREEN = "#333333", "#777777", "#c0392b", "#1e8449"
COLORS = ["#2a78d6", "#e8710a", "#9334e6", "#d93025", "#6b4e16", "#0b8a8a"]
COMB_COLOR = "#1baf7a"
LOG_ROWS = 18


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
    p.add_argument("--snapshot", nargs=2, metavar=("SECONDS", "PATH"), default=None)
    a = p.parse_args()
    if a.device is not None and a.device.isdigit():
        a.device = int(a.device)
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
    """What the score axis draws: networks as they are, comb (unbounded) as s/(s+threshold)."""
    s = np.asarray(s, dtype=np.float64)
    if squashed(k):
        return np.where(s > 0, s / (s + thr[k]), 0.0)
    return s


def shown_thr(k: str):
    return None if thr[k] is None else float(shown(k, thr[k]))


# ------------------------------------------------------------------ worker

events: queue.Queue = queue.Queue()
ctl = dict(record=False)


def worker(stop: threading.Event) -> None:
    """Capture and score in this thread; hand decisions and status to the GUI through `events`."""
    rec = fh = writer = None
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
        events.put(("start", f"{name}, {sr} Hz, {ch} ch, channel {args.channel}"
                             + ("" if sr == cfg.sample_rate else f" (resampled to {cfg.sample_rate} Hz)")))
        for b in blocks:
            if ctl["record"] and rec is None:
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                stem = OUT_DIR / time.strftime("live_%Y%m%d_%H%M%S")
                rec = sf.SoundFile(str(stem.with_suffix(".wav")), "w", samplerate=sr, channels=ch, subtype="FLOAT")
                fh = open(stem.with_suffix(".csv"), "w", newline="", encoding="utf-8")
                writer = csv.writer(fh)
                writer.writerow(["t_rec_s", "t_end_s", "wall_clock", "level_dbfs", "detector", "score", "threshold",
                                 "drone"])
                rec_t0 = pipe.windows.received / sr
                events.put(("rec", f"recording -> {stem.with_suffix('.wav').relative_to(ROOT)}"))
            elif not ctl["record"] and rec is not None:
                rec.close(), fh.close()
                rec = fh = writer = None
                events.put(("rec", "recording saved"))
            if rec is not None:
                rec.write(b.audio)
            if b.status:
                events.put(("overflow", b.status))
            for d in pipe.feed(b.audio[:, args.channel]):
                if writer:
                    for k in names:
                        on = d.drone[k]
                        writer.writerow([f"{d.t_end - rec_t0:.4f}", f"{d.t_end:.4f}", time.strftime("%H:%M:%S"),
                                         f"{d.level_dbfs:.1f}", k, f"{d.scores[k]:.6g}",
                                         "" if thr[k] is None else f"{thr[k]:.6g}", "" if on is None else int(on)])
                events.put(("dec", d))
        events.put(("end", "input ended" if not stop.is_set() else "stopped"))
    except Exception as e:                                   # shown in the window, not lost in a thread
        events.put(("error", f"{type(e).__name__}: {e}"))
    finally:
        if rec is not None:
            rec.close(), fh.close()
            events.put(("rec", "recording saved"))


# ------------------------------------------------------------------ figure

fig = plt.figure(figsize=(15, 9))
fig.canvas.manager.set_window_title("Live drone detection")
fig.text(0.015, 0.975, "Live drone detection", fontsize=13, weight="bold", color=TEXT, va="top")
info = fig.text(0.015, 0.94, "", fontsize=7.5, color=GREY, va="top")
b_run = Button(fig.add_axes([0.015, 0.81, 0.075, 0.045]), "Stop")
b_rec = Button(fig.add_axes([0.1, 0.81, 0.075, 0.045]), "Record")
b_clear = Button(fig.add_axes([0.185, 0.81, 0.075, 0.045]), "Clear log")
rec_note = fig.text(0.015, 0.795, "", fontsize=7.5, color=GREY, va="top")
fig.text(0.015, 0.75, "alarm log", fontsize=9, weight="bold", color=TEXT, va="top")
log_text = fig.text(0.015, 0.725, "", fontsize=8, family="monospace", color=TEXT, va="top", linespacing=1.5)
fig.text(0.015, 0.06, "levels: detectors trained with 110 dB SPL at full scale;\n"
                      "on an uncalibrated mic the scores are hints", fontsize=7, color=GREY, va="bottom")

ax_ban = fig.add_axes([0.3, 0.84, 0.68, 0.12])
ax_ban.set_axis_off()
ban_box = ax_ban.add_patch(Rectangle((0, 0), 1, 1, transform=ax_ban.transAxes, color="#dddddd"))
ban_text = ax_ban.text(0.02, 0.5, "starting", fontsize=26, weight="bold", color="white", va="center",
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
ax_level.set(ylim=(-100, 0))
ax_level.set_ylabel("dBFS", fontsize=8)
ax_level.set_xlabel("seconds since start", fontsize=8)
ax_level.grid(color="#e5e5e5", lw=0.5)
ax_level.tick_params(labelsize=7)

state: dict = {}


def reset() -> None:
    spec[:] = np.nan
    state.update(t=[], level=[], score={k: [] for k in names}, drone={k: [] for k in names}, cols_shown=0,
                 t_now=0.0, wall0=time.perf_counter(), compute=0.0, overflows=0, on={k: False for k in names},
                 source="", status="starting", dirty=True)
    state.setdefault("log", [])


def log(msg: str) -> None:
    state["log"].append(msg)
    del state["log"][:-LOG_ROWS]
    log_text.set_text("\n".join(state["log"]))


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
    any_on = any(state["on"].values())
    running = state["status"] == "listening"
    ban_box.set_color(RED if any_on else GREEN if running else "#999999")
    ban_text.set_text("DRONE" if any_on else state["status"])
    det_text.set_text("\n".join(
        f"{k:<10} {state['score'][k][-1]:>8.3g}" + ("" if thr[k] is None else f" / {thr[k]:.3g}")
        + ("  DRONE" if state["on"][k] else "        ") for k in names) if len(t) else "")
    lag = 1000 * ((time.perf_counter() - state["wall0"]) - t_now) if len(t) else 0.0
    info.set_text(f"{state['source']}\n" + cfg.describe().replace(" | 64", "\n64", 1) + f"\nt = {t_now:.2f} s,  scoring {state['compute']:.1f} ms/window,"
                  f"  lag {lag:.0f} ms" + (f",  {state['overflows']} input overflows" if state["overflows"] else ""))
    fig.canvas.draw_idle()
    state["dirty"] = False


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
            rec_note.set_text(payload)
            state["dirty"] = True
        elif kind in ("end", "error"):
            state.update(status=payload if kind == "end" else "error", dirty=True)
            if kind == "error":
                log("ERROR " + payload)
                print(payload, file=sys.stderr)
            b_run.label.set_text("Start")
            ctl["record"] = False
            b_rec.label.set_text("Record")
    if state["dirty"]:
        redraw()
    if args.snapshot and state["t_now"] >= float(args.snapshot[0]) and not state.get("snapped"):
        state["snapped"] = True
        fig.savefig(args.snapshot[1], dpi=90)
        print(f"snapshot -> {args.snapshot[1]}")
        stop_worker()
        plt.close(fig)


def start_worker() -> None:
    reset()
    state["stop"] = threading.Event()
    state["thread"] = threading.Thread(target=worker, args=(state["stop"],), daemon=True)
    state["thread"].start()
    b_run.label.set_text("Stop")


def stop_worker() -> None:
    if state.get("stop") is not None:
        state["stop"].set()
        ctl["record"] = False
        state["thread"].join(timeout=3)


def on_run(_) -> None:
    if state.get("thread") is not None and state["thread"].is_alive():
        stop_worker()
    else:
        start_worker()


def on_rec(_) -> None:
    if state["status"] != "listening":
        return
    ctl["record"] = not ctl["record"]
    b_rec.label.set_text("Stop rec" if ctl["record"] else "Record")
    fig.canvas.draw_idle()


def on_clear(_) -> None:
    state["log"] = []
    log_text.set_text("")
    fig.canvas.draw_idle()


b_run.on_clicked(on_run)
b_rec.on_clicked(on_rec)
b_clear.on_clicked(on_clear)
fig.canvas.mpl_connect("close_event", lambda _: stop_worker())
timer = fig.canvas.new_timer(interval=100)
timer.add_callback(poll)
start_worker()
timer.start()
plt.show()
