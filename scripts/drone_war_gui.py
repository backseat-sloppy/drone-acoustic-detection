"""A window for the drone-in-war test (scripts/drone_war_grid.py): type the drone speeds and the
war levels as min, max and count (count values from min to max, both included, evenly spaced;
count 1 needs min = max), press Run, and watch one grid fill in: rows = drone speeds, columns =
war noise levels, CNN and comb in every panel, with a table of totals per war level.
Tick 'no drone' for an extra row with war only (false alarms) and 'no war' for an extra
column with the drone only; counts may then be 0. War level 0 dB = the war track as rendered. The war is
war-shaped noise (scripts/war_shaped_noise.py, seed 1000), as loud as the first 20 s of the war
track at the same dB.

    uv run python scripts/drone_war_gui.py

Runs in the background, so the window stays live; Stop ends a run after the current clip.
'Page 2: SNR' swaps the x axis from time to each window's SNR (drone over noise, 50-8000 Hz).
Click a panel to open it in its own window: drag with the toolbar's zoom tool, or scroll to zoom
the x axis around the mouse.
Results (results.csv, hops.csv, grid_speed_x_war.png and grid_speed_x_war_snr.png) go to
artifacts/drone_war/gui/.
Drones are rendered by Pure Data once per parameter set and cached.
Synthetic scaffolding (CLAUDE.md section 5): never detection performance.
"""
from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import drone_war_grid as dwg  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.widgets import Button, CheckButtons, TextBox  # noqa: E402

OUT = dwg.ROOT / "artifacts/drone_war/gui"
D = dwg.defaults()
FIELDS = [  # key, label, start, allowed min, allowed max, type
    ("speed_min", "drone speed min (km/h)", D.speed_min, 1, 1000, float),
    ("speed_max", "drone speed max (km/h)", D.speed_max, 1, 1000, float),
    ("speeds", "drone speed count", D.speeds, 0, 20, int),
    ("war_min_db", "war level min (dB)", D.war_min_db, -300, 30, float),
    ("war_max_db", "war level max (dB)", D.war_max_db, -300, 30, float),
    ("levels", "war level count", D.levels, 0, 20, int),
]
TEXT = "#333333"
TITLE = f"war-shaped noise, seed {dwg.WSN_SEED}"

fig = plt.figure(figsize=(17, 10))
fig.canvas.manager.set_window_title("Drone in war: virtual test")
fig.text(0.015, 0.975, "Drone in war: virtual test", fontsize=13, weight="bold", color=TEXT, va="top")
boxes = {}
for i, (key, label, v0, lo, hi, kind) in enumerate(FIELDS):
    y = 0.90 - i * 0.05 - (0.05 if i >= 3 else 0)        # room for the drone group's checkbox
    fig.text(0.015, y + 0.008, label, fontsize=9, color=TEXT)
    boxes[key] = TextBox(fig.add_axes([0.125, y, 0.05, 0.033]), "", initial=f"{v0:g}")
    fig.text(0.18, y + 0.008, f"{lo} to {hi}", fontsize=8, color="#777777")
TICKS = {}
for key, label, y in (("no_drone", "no drone (a row with war only)", 0.735),
                      ("no_war", "no war (a column with the drone only)", 0.535)):
    ax = fig.add_axes([0.012, y, 0.2, 0.035])
    ax.axis("off")
    TICKS[key] = CheckButtons(ax, [label], [False])
fig.text(0.015, 0.515, "count: min to max, evenly spaced; count 1 needs min = max.\n"
         "war: war-shaped noise, as loud as the war track's first 20 s.\n"
         "war 0 dB = the war track as rendered.\n"
         "click a panel to open it big; scroll there to zoom.", fontsize=8, color="#777777", va="top")
b_run = Button(fig.add_axes([0.015, 0.40, 0.08, 0.04]), "Run")
b_stop = Button(fig.add_axes([0.10, 0.40, 0.08, 0.04]), "Stop")
b_page = Button(fig.add_axes([0.185, 0.40, 0.08, 0.04]), "Page 2: SNR")
PAGES = {"time": "Page 2: SNR", "snr": "Page 1: time"}       # page shown -> label of the button to the other
b_flush = Button(fig.add_axes([0.015, 0.06, 0.08, 0.04]), "Flush plots")
ax_confirm = fig.add_axes([0.10, 0.06, 0.08, 0.04])
b_confirm = Button(ax_confirm, "Confirm", color="#f4c7c3", hovercolor="#ee9a93")
ax_confirm.set_visible(False)                                 # only shown after Flush plots
status = fig.text(0.015, 0.38, "Type the values and press Run.", fontsize=8, color=TEXT, va="top", wrap=True)
table = fig.text(0.015, 0.34, "", fontsize=7, family="monospace", color=TEXT, va="top")

state = dict(thread=None, stop=threading.Event(), levels={}, rows=[], meta=None, grid=[], page="time")
events: queue.Queue = queue.Queue()


def params():
    """The defaults with the values from the boxes; ValueError names a bad box or a bad min/max/count."""
    a = dwg.defaults()
    for key, label, _, lo, hi, kind in FIELDS:
        try:
            v = kind(boxes[key].text.strip().replace(",", "."))
        except ValueError:
            raise ValueError(f"{label}: type a {'whole ' if kind is int else ''}number") from None
        if not lo <= v <= hi:
            raise ValueError(f"{label}: {lo} to {hi}")
        setattr(a, key, v)
    for key, box in TICKS.items():
        setattr(a, key, box.get_status()[0])
    dwg.speeds_and_gains(a)                                  # min above max, or count and min/max disagree
    a.wars = 1                                               # one war clip: the track's first 20 s
    a.out = OUT
    return a


def worker(a):
    try:
        r = dwg.run(a, progress=lambda m: events.put(("status", m)),
                    on_level=lambda g, res, rows: events.put(("level", (g, res, list(rows)))),
                    stop=state["stop"].is_set)
        for x, name in (("time", "grid_speed_x_war.png"), ("snr", "grid_speed_x_war_snr.png")):
            dwg.plot_speed_war(r["res"], r["speeds"], r["gains"], r["offsets"][0], r["thr"], OUT / name,
                               title=TITLE, x=x)
        events.put(("status", f"done: {len(r['rows']) // len(dwg.DETECTORS)} mixes, "
                              f"saved in {OUT.relative_to(dwg.ROOT).as_posix()}"))
    except InterruptedError:
        events.put(("status", "stopped"))
    except Exception as e:                                   # shown in the window, not lost in a thread
        events.put(("status", f"error: {e}"))
    events.put(("done", None))


def clear():
    for x in state["grid"]:
        x.remove()
    fig.legends.clear()
    state["grid"] = []


def draw():
    """The whole grid so far (rows = speeds, columns = the war levels done) and the table."""
    clear()
    speeds, gains, offset, thr = state["meta"]
    state["grid"] = dwg.draw_speed_war(fig, state["levels"], speeds, gains, offset, thr, rect=(0.27, 0.0, 1.0, 0.95),
                                       title=TITLE, x=state["page"])
    s = dwg.summarise(state["rows"])
    pct = lambda v: "    -" if v is None else f"{100 * v:5.1f}"  # noqa: E731
    n = sum(v != dwg.NO_DRONE for v in speeds)
    lines = [f"{n} speeds" + (" + no drone" if dwg.NO_DRONE in speeds else "") + ", war-shaped noise", "",
             f"{'':8s}{'CNN':>17s}{'comb':>19s}",
             f"{'war':8s}" + f"{'recall':>7s}{'false+':>7s}{'>=3s':>5s} " * 2]
    for g in [g for g in gains if g in state["levels"]]:
        row = f"{'no war' if g == dwg.OFF else dwg.war_tag(g):8s}"
        for k in dwg.DETECTORS:
            v = s[(g, k)]
            row += f"{pct(v['recall']):>7s}{pct(v['false_positive_rate']):>7s}{v['alarm_3s']:5d} "
        lines.append(row)
    lines += ["", "recall, false+: % of 1 s windows", ">=3s: speeds alarmed 3 s before impact"]
    table.set_text("\n".join(lines))


def on_page(_):
    state["page"] = "snr" if state["page"] == "time" else "time"
    b_page.label.set_text(PAGES[state["page"]])
    if state["meta"] and state["levels"]:
        draw()
    fig.canvas.draw_idle()


def scroll_zoom(event):
    """Scroll up zooms in, down zooms out, on the x axis around the mouse."""
    ax = event.inaxes
    if ax is None or event.xdata is None:
        return
    f = 0.8 if event.button == "up" else 1.25
    lo, hi = ax.get_xlim()
    ax.set_xlim(event.xdata - (event.xdata - lo) * f, event.xdata + (hi - event.xdata) * f)
    ax.figure.canvas.draw_idle()


def on_click(event):
    """A click on a grid panel opens that panel alone in its own window, with the zoom toolbar."""
    ax = event.inaxes
    tool = getattr(fig.canvas, "toolbar", None)
    if event.button != 1 or not hasattr(ax, "cell") or (tool and tool.mode) or not state["meta"]:
        return
    v, g = ax.cell
    if g not in state["levels"]:
        return
    speeds, gains, offset, thr = state["meta"]
    big = plt.figure(figsize=(11, 5.5))
    name = f"{dwg.speed_label(v)}, " + ("no war" if g == dwg.OFF else f"war {g:+g} dB")
    big.canvas.manager.set_window_title(name)
    drawn = dwg.draw_speed_war(big, {g: state["levels"][g]}, [v], [g], offset, thr,
                               rect=(0.0, 0.06, 1.0, 1 - 0.9 / big.get_figheight()), title=f"{TITLE}  |  {name}",
                               x=state["page"])
    for t in drawn[-2:]:                                     # the grid's "war noise level →" headings: one panel needs none
        t.remove()
    big.canvas.mpl_connect("scroll_event", scroll_zoom)
    big.show()


def poll():
    changed = False
    while not events.empty():
        kind, x = events.get()
        if kind == "status":
            status.set_text(x)
        elif kind == "level":
            g, res, rows = x
            state["levels"][g] = res
            state["rows"] = rows
            draw()
        elif kind == "done":
            state["thread"] = None
        changed = True
    if changed:
        fig.canvas.draw_idle()


def on_run(_):
    if state["thread"]:
        return
    ax_confirm.set_visible(False)                            # Run cancels a pending flush
    try:
        a = params()
    except ValueError as e:
        status.set_text(str(e))
        fig.canvas.draw_idle()
        return
    speeds, gains = dwg.speeds_and_gains(a)
    offset = dwg.war_offsets(a, len(dwg._sources()["war"]) / dwg.es.FS)[0]
    clear()
    table.set_text("")
    state.update(levels={}, rows=[], meta=(speeds, gains, offset, dwg._detectors()[0]))
    state["stop"].clear()
    status.set_text("starting (the first run loads the models and the war track)")
    fig.canvas.draw_idle()
    state["thread"] = threading.Thread(target=worker, args=(a,), daemon=True)
    state["thread"].start()


def plot_files():
    return sorted((dwg.ROOT / "artifacts/drone_war").rglob("*.png"))


def on_flush(_):
    if state["thread"]:
        status.set_text("a run is going: Stop it before flushing plots")
    else:
        files = plot_files()
        mb = sum(p.stat().st_size for p in files) / 1e6
        status.set_text(f"Delete {len(files)} plot files ({mb:.1f} MB) in artifacts/drone_war and clear the window? "
                        "Press Confirm. Results (CSV) and cached drones are kept.")
        ax_confirm.set_visible(True)
    fig.canvas.draw_idle()


def on_confirm(_):
    ax_confirm.set_visible(False)
    if state["thread"]:                                      # a run started between Flush and Confirm
        status.set_text("a run is going: nothing deleted")
    else:
        files = plot_files()
        for p in files:
            p.unlink()
        clear()
        table.set_text("")
        state.update(levels={}, rows=[])
        status.set_text(f"deleted {len(files)} plot files")
    fig.canvas.draw_idle()


b_run.on_clicked(on_run)
b_flush.on_clicked(on_flush)
b_confirm.on_clicked(on_confirm)
b_stop.on_clicked(lambda _: state["stop"].set())
b_page.on_clicked(on_page)
fig.canvas.mpl_connect("button_press_event", on_click)
timer = fig.canvas.new_timer(interval=300)
timer.add_callback(poll)
timer.start()
plt.show()
