"""A small window with Start section and End section buttons for a running scripts/live_gui.py.

    uv run python scripts/live_gui.py --device 2 --sections calls      # one terminal; press Start there
    uv run python scripts/live_control_gui.py                          # another terminal

It does what another script would do: every button press goes through dad.live_control, over
127.0.0.1, so what works here works from a script. The buttons always send their call, even when
the detector will refuse it, so the refusals can be tested too; the reply is in the log.

The banner shows what the detector reports, asked four times a second:
    no detector        nothing answers on this port
    idle               the live window is open but Start has not been pressed
    wrong mode         detecting, but its sections are not "on start/end calls", so calls are refused
    ready              detecting, no section open: nothing is being saved
    section N saving   a section is open; the time is how long, in audio seconds

"Start repeating sections" starts and ends sections over and over (dad.live_control.run_sections):
the section length is saved, the gap is not, both in the detector's audio seconds. While it runs
every other control is greyed out; "Stop repeating sections" ends a section that is open. A refused
call stops the repeating, with the reason in the log.

The log lists every call and its reply, with the time in the detector's audio clock. `--port` (or
the box in the window) must match the live window's --control-port.
"""
from __future__ import annotations

import argparse
import queue
import threading
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, TextBox

from dad import live_control as lc

POLL_S = 0.25
LOG_ROWS = 12
TEXT, GREY, RED, GREEN, AMBER = "#333333", "#777777", "#c0392b", "#1e8449", "#b9770e"
START_COLORS, END_COLORS = ("#cfe8d5", "#b5dcbf"), ("#f4c7c3", "#eeaaa4")   # (face, hover)
OFF_FACE, OFF_TEXT = "#e8e8e8", "#aaaaaa"                                   # a control that is locked

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--port", type=int, default=lc.DEFAULT_PORT)
args = ap.parse_args()

requests: queue.Queue = queue.Queue()
results: queue.Queue = queue.Queue()
state: dict = dict(port=args.port, status=None, error=None, connected=None, started_t=None, started_k=None, log=[], cycle=None, cycle_note="")
closing = threading.Event()
cycle_stop = threading.Event()


def worker() -> None:
    """All network calls happen here, one at a time, so the window never waits on them.
    A button's call goes first; with none waiting, it asks for the status."""
    while not closing.is_set():
        try:
            cmd = requests.get(timeout=POLL_S)
        except queue.Empty:
            cmd = "status"
        try:
            results.put((cmd, lc.send(cmd, state["port"], timeout=5.0), None))
        except lc.LiveControlError as e:
            results.put((cmd, None, str(e)))


fig = plt.figure(figsize=(7.5, 7))
fig.canvas.manager.set_window_title("Live section control")
fig.text(0.04, 0.965, "Live section control", fontsize=13, weight="bold", color=TEXT, va="top")
ax_ban = fig.add_axes([0.04, 0.76, 0.92, 0.15])
ax_ban.set_axis_off()
ban_box = ax_ban.add_patch(Rectangle((0, 0), 1, 1, transform=ax_ban.transAxes, color="#999999"))
ban_text = ax_ban.text(0.03, 0.64, "connecting", fontsize=22, weight="bold", color="white", va="center",
                       transform=ax_ban.transAxes)
ban_note = ax_ban.text(0.03, 0.22, "", fontsize=9, color="white", va="center", transform=ax_ban.transAxes)
b_start = Button(fig.add_axes([0.04, 0.655, 0.32, 0.085]), "Start section", color=START_COLORS[0],
                 hovercolor=START_COLORS[1])
b_end = Button(fig.add_axes([0.38, 0.655, 0.32, 0.085]), "End section", color=END_COLORS[0], hovercolor=END_COLORS[1])
t_port = TextBox(fig.add_axes([0.82, 0.68, 0.14, 0.045]), "port  ", initial=str(args.port))
fig.text(0.04, 0.635, "repeat sections", fontsize=9, weight="bold", color=TEXT, va="top")
t_length = TextBox(fig.add_axes([0.2, 0.55, 0.08, 0.045]), "section length (s)  ", initial="20")
t_gap = TextBox(fig.add_axes([0.38, 0.55, 0.08, 0.045]), "gap (s)  ", initial="0")
b_cycle = Button(fig.add_axes([0.5, 0.545, 0.32, 0.055]), "Start repeating sections")
for box in (t_length, t_gap):
    box.label.set_fontsize(9)
fig.text(0.04, 0.52, "calls and replies", fontsize=9, weight="bold", color=TEXT, va="top")
log_text = fig.text(0.04, 0.485, "", fontsize=8, family="monospace", color=TEXT, va="top", linespacing=1.5)


def log(msg: str) -> None:
    state["log"].append(f"{time.strftime('%H:%M:%S')}  {msg}"[:92])
    del state["log"][:-LOG_ROWS]
    log_text.set_text("\n".join(state["log"]))


def banner(text: str, note: str, colour: str) -> None:
    ban_text.set_text(text)
    ban_note.set_text(note)
    ban_box.set_color(colour)


def show_status() -> None:
    s = state["status"]
    if s is None:
        banner("no detector", f"nothing answers on 127.0.0.1:{state['port']}: is scripts/live_gui.py open?", "#999999")
    elif not s["detecting"]:
        banner("idle", "the live window is open: choose 'on start/end calls' there and press Start", "#999999")
    elif s["sections"] != "calls":
        banner("wrong mode", f"sections are '{s['sections']}': calls will be refused. Stop, choose "
                             "'on start/end calls', Start", AMBER)
    elif s["open"] is None:
        banner("ready", f"detecting, nothing being saved  |  t = {s['t_s']:.2f} s{state['cycle_note']}", GREEN)
    else:
        known = state["started_k"] == s["open"] and state["started_t"] is not None
        length = f"{s['t_s'] - state['started_t']:.2f} s" if known else "started elsewhere"
        banner(f"section {s['open']} saving   {length}", f"t = {s['t_s']:.2f} s{state['cycle_note']}", RED)


def run_cycle(length_s: float, gap_s: float, port: int) -> None:
    """The repeating, in its own thread. Its calls come back through `results` like a button's."""
    try:
        n = lc.run_sections(length_s, gap_s, port=port, stop=cycle_stop, report=lambda c, r: results.put((c, r, None)))
        results.put(("cycle", n, None))
    except lc.LiveControlError as e:
        results.put(("cycle", None, str(e)))


def set_cycling(on: bool, note: str = "") -> None:
    """While repeating, the only control that works is the button that stops it."""
    for b, (face, hover) in ((b_start, START_COLORS), (b_end, END_COLORS)):
        b.active = not on                                     # matplotlib: an inactive widget ignores clicks and keys
        b.color, b.hovercolor = (OFF_FACE, OFF_FACE) if on else (face, hover)
        b.ax.set_facecolor(b.color)
        b.label.set_color(OFF_TEXT if on else "black")
    for box in (t_length, t_gap, t_port):
        box.active = not on
        box.label.set_color(OFF_TEXT if on else "black")
        box.text_disp.set_color(OFF_TEXT if on else "black")
    b_cycle.label.set_text("Stop repeating sections" if on else "Start repeating sections")
    state["cycle_note"] = note


def on_cycle(_) -> None:
    if state["cycle"] is not None:
        cycle_stop.set()
        b_cycle.label.set_text("Stopping...")
        fig.canvas.draw_idle()
        return
    t_length.stop_typing()
    t_gap.stop_typing()
    try:
        length_s, gap_s = float(t_length.text.replace(",", ".")), float(t_gap.text.replace(",", "."))
    except ValueError:
        length_s, gap_s = float("nan"), float("nan")
    if not (0 < length_s <= 24 * 3600 and 0 <= gap_s <= 24 * 3600):
        log(f"repeat: length above 0 and gap 0 or more, not {t_length.text!r} and {t_gap.text!r}")
        fig.canvas.draw_idle()
        return
    cycle_stop.clear()
    state["cycle"] = threading.Thread(target=run_cycle, args=(length_s, gap_s, state["port"]), daemon=True)
    state["cycle"].start()
    set_cycling(True, f"  |  repeating: {length_s:g} s saved, {gap_s:g} s gap")
    log(f"repeat: started, {length_s:g} s sections, {gap_s:g} s gap")
    show_status()
    fig.canvas.draw_idle()


def on_reply(cmd: str, reply: dict | None, error: str | None) -> None:
    if cmd == "cycle":
        state["cycle"] = None
        set_cycling(False)
        log(f"repeat: STOPPED, {error}" if error else f"repeat: stopped after {reply} sections")
    elif cmd == "status":
        connected = error is None
        if connected != state["connected"]:
            log(f"detector {'found' if connected else 'not found'} on port {state['port']}")
        state.update(connected=connected, status=reply)
    elif error is not None:
        log(f"{cmd}: REFUSED  {error}")
    elif cmd == "start":
        state.update(started_k=reply["section"], started_t=reply["t_s"])
        log(lc.reply_message(cmd, reply))
    else:
        known = state["started_k"] == reply["section"] and state["started_t"] is not None
        length = f", {reply['t_s'] - state['started_t']:.2f} s long" if known else ""
        log(lc.reply_message(cmd, reply) + length)


def poll() -> None:
    changed = False
    while True:
        try:
            on_reply(*results.get_nowait())
        except queue.Empty:
            break
        changed = True
    if changed:
        show_status()
        fig.canvas.draw_idle()


def on_port(text: str) -> None:
    try:
        port = int(text)
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        log(f"port must be a number from 1 to 65535, not {text!r}")
        t_port.set_val(str(state["port"]))
        return
    if port != state["port"]:
        state.update(port=port, connected=None)


def on_close(_) -> None:
    closing.set()
    cycle_stop.set()


b_start.on_clicked(lambda _: requests.put("start"))
b_end.on_clicked(lambda _: requests.put("end"))
b_cycle.on_clicked(on_cycle)
t_port.on_submit(on_port)
fig.canvas.mpl_connect("close_event", on_close)
threading.Thread(target=worker, daemon=True).start()
timer = fig.canvas.new_timer(interval=100)
timer.add_callback(poll)
timer.start()
plt.show()
if state["cycle"] is not None:                                # closed while repeating: let it end the open section
    state["cycle"].join(timeout=6.0)
