"""A window for the drone-in-war test (scripts/drone_war_grid.py): type how many drone speeds
(100-200 km/h), war clips and war levels (-20 to 0 dB) to run, press Run, and watch each war
level's grid appear (rows = drone speeds, columns =
war clips, CNN / LightGBM / comb in every panel) with its totals.

    uv run python scripts/drone_war_gui.py

Runs in the background, so the window stays live; Stop ends a run after the current clip.
Results (results.csv, hops.csv and a PNG per level) go to artifacts/drone_war/gui/.
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
from matplotlib.widgets import Button, TextBox  # noqa: E402

OUT = dwg.ROOT / "artifacts/drone_war/gui"
D = dwg.defaults()
FIELDS = [  # key, label, start; each a whole number of steps over a fixed range
    ("speeds", f"drone speeds, {D.speed_min:.0f}-{D.speed_max:.0f} km/h", D.speeds),
    ("wars", "war clips, spread over the 10 min track", D.wars),
    ("levels", f"war levels, {D.war_min_db:+.0f} to {D.war_max_db:+.0f} dB", D.levels),
]
TEXT = "#333333"

fig = plt.figure(figsize=(17, 10))
fig.canvas.manager.set_window_title("Drone in war: virtual test")
fig.text(0.015, 0.975, "Drone in war: virtual test", fontsize=13, weight="bold", color=TEXT, va="top")
boxes = {}
for i, (key, label, v0) in enumerate(FIELDS):
    y = 0.87 - i * 0.07
    fig.text(0.015, y + 0.045, label, fontsize=9, color=TEXT)
    boxes[key] = TextBox(fig.add_axes([0.015, y, 0.06, 0.035]), "", initial=str(v0))
b_run = Button(fig.add_axes([0.015, 0.65, 0.08, 0.04]), "Run")
b_stop = Button(fig.add_axes([0.10, 0.65, 0.08, 0.04]), "Stop")
b_prev = Button(fig.add_axes([0.015, 0.60, 0.08, 0.04]), "< level")
b_next = Button(fig.add_axes([0.10, 0.60, 0.08, 0.04]), "level >")
b_flush = Button(fig.add_axes([0.015, 0.06, 0.08, 0.04]), "Flush plots")
ax_confirm = fig.add_axes([0.10, 0.06, 0.08, 0.04])
b_confirm = Button(ax_confirm, "Confirm", color="#f4c7c3", hovercolor="#ee9a93")
ax_confirm.set_visible(False)                                 # only shown after Flush plots
status = fig.text(0.015, 0.57, "Type the counts and press Run.", fontsize=8, color=TEXT, va="top", wrap=True)
table = fig.text(0.015, 0.47, "", fontsize=8, family="monospace", color=TEXT, va="top")

state = dict(thread=None, stop=threading.Event(), levels={}, rows=[], shown=None, meta=None, grid=[])
events: queue.Queue = queue.Queue()


def params():
    """The defaults with the three counts from the boxes; ValueError names a bad box."""
    a = dwg.defaults()
    for key, label, _ in FIELDS:
        try:
            v = int(boxes[key].text.strip())
        except ValueError:
            raise ValueError(f"{label}: type a whole number") from None
        if not 1 <= v <= 20:
            raise ValueError(f"{label}: 1 to 20")
        setattr(a, key, v)
    a.out = OUT
    return a


def worker(a):
    try:
        r = dwg.run(a, progress=lambda m: events.put(("status", m)),
                    on_level=lambda g, res, rows: events.put(("level", (g, res, list(rows)))),
                    stop=state["stop"].is_set)
        for g in r["gains"]:
            dwg.plot_level(r["res"][g], r["speeds"], r["offsets"], r["thr"], g, OUT / f"grid_war{g:+.0f}dB.png")
        events.put(("status", f"done: {len(r['rows']) // 3} mixes. Results in {OUT}"))
    except InterruptedError:
        events.put(("status", "stopped"))
    except Exception as e:                                   # shown in the window, not lost in a thread
        events.put(("status", f"error: {e}"))
    events.put(("done", None))


def draw(g):
    for ax in state["grid"]:
        ax.remove()
    fig.legends.clear()
    speeds, offsets, thr = state["meta"]
    state["grid"] = dwg.draw_grid(fig, state["levels"][g], speeds, offsets, thr, g, rect=(0.27, 0.0, 1.0, 0.92))
    state["shown"] = g
    s = dwg.summarise([r for r in state["rows"] if r["war_gain_db"] == g])
    pct = lambda v: "   -" if v is None else f"{100 * v:5.1f}%"  # noqa: E731
    lines = [f"war {g:+.0f} dB, {s[(g, 'CNN')]['mixes']} mixes", "",
             f"{'':9s}{'recall':>8s}{'false+':>8s}{'>=3s':>6s}{'none':>6s}"]
    for k in dwg.DETECTORS:
        v = s[(g, k)]
        lines.append(f"{k:9s}{pct(v['recall']):>8s}{pct(v['false_positive_rate']):>8s}{v['alarm_3s']:6d}{v['no_alarm']:6d}")
    lines += ["", "recall, false+: per 1 s window", ">=3s: mixes alarmed 3 s before impact",
              "none: mixes with no alarm before impact"]
    table.set_text("\n".join(lines))


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
            draw(g)
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
    speeds = [float(v) for v in dwg.np.linspace(a.speed_min, a.speed_max, a.speeds)]
    src = dwg._sources()
    offsets = [float(o) for o in dwg.np.round(dwg.np.linspace(0, len(src["war"]) / dwg.es.FS - a.length, a.wars), 1)]
    state.update(levels={}, rows=[], shown=None, meta=(speeds, offsets, dwg._detectors()[0]))
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
        for ax in state["grid"]:
            ax.remove()
        fig.legends.clear()
        table.set_text("")
        state.update(grid=[], levels={}, rows=[], shown=None)
        status.set_text(f"deleted {len(files)} plot files")
    fig.canvas.draw_idle()


def step(d):
    done = sorted(state["levels"])
    if state["shown"] is None or not done:
        return
    i = min(max(done.index(state["shown"]) + d, 0), len(done) - 1)
    draw(done[i])
    fig.canvas.draw_idle()


b_run.on_clicked(on_run)
b_flush.on_clicked(on_flush)
b_confirm.on_clicked(on_confirm)
b_stop.on_clicked(lambda _: state["stop"].set())
b_prev.on_clicked(lambda _: step(-1))
b_next.on_clicked(lambda _: step(+1))
timer = fig.canvas.new_timer(interval=300)
timer.add_callback(poll)
timer.start()
plt.show()
