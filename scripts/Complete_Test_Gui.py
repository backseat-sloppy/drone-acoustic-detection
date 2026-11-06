"""Complete test: full control over large batches of drone-in-noise test rounds, no graphs on the front.

    uv run python scripts/Complete_Test_Gui.py

Two pages, a slide show (the arrows at the top, the page names, or Ctrl+Left / Ctrl+Right):

    Real drone test       real drone recordings (WAV files you add), each cut to its first 10 s
    Synthetic drone test  Pure Data drones, one per speed, 10 s from start to impact (as drone_war_gui)

On each page:

    Drones               the files or speeds, and the drone level(s) in dB SPL (synthetic: "as rendered"
                         keeps the physical level of the render)
    Background layers    (open with the arrow) war soundscape, war-shaped noise, wind, and cars,
                         helicopter, insects (listed, not set up yet). Tick a layer and give one or more
                         levels in dB SPL; every combination of the ticked layers' levels is a background
    Test rounds          soundscape length: the drone starts at every step (1 s) from 0 until it ends with
                         the soundscape (length - 10 + 1 starts); optional no-drone rounds per background
    Detectors            any of the networks and comb
    Run                  the live count of rounds, Start / Stop, progress and time left, the results window

Levels are real levels: full scale in every WAV here is 110 dB SPL. A background layer's level is its
long-term level over the whole source (all 600 s of the war track), so its calm and loud stretches
keep their difference. The engine (scripts/complete_test_engine.py) explains the scoring.

Results go to artifacts/complete_test/<time>_<kind>/: rounds.csv (one row per round and detector),
summary.csv (means per detector and per varied setting), scores.npz (every hop's score), config.json.
The results window shows the summary, a sortable and filterable table of all rounds, and for the
selected round its log-mel, every detector's confidence and the dB SPL of each layer over time, and
plays the mix, the drone alone or the background alone. "Open results..." reopens an earlier run.

Synthetic scaffolding (CLAUDE.md section 5): numbers from these mixes are not detection performance.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, str(Path(__file__).parent))
import complete_test_engine as E  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("TkAgg") if os.environ.get("MPLBACKEND", "").lower() != "agg" else None
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from dad.config import DEFAULT  # noqa: E402
from dad.features import LogMel, hz_to_mel, mel_to_hz  # noqa: E402

TITLE = "Complete Test"
PAGES = [("real", "Real drone test"), ("synthetic", "Synthetic drone test")]
DEFAULT_LEVELS = {"war soundscape": "60", "war-shaped noise": "50", "wind": "50", "cars": "", "helicopter": "",
                  "insects": ""}
COLORS = {"it6 CNN": "#2a78d6", "it9 CNN": "#e8710a", "it9 CRNN": "#9334e6", "it10 CNN": "#d93025",
          "e123 aug CRNN": "#00897b", "e123 mix CRNN": "#c2185b", "e123 augmix CRNN": "#795548",
          "it11r CRNN 600k real-thr": "#3949ab",
          "comb": "#1baf7a"}

# Entry 124's arms (report entries 124-126): it11r CRNN 600k + speed/EQ, + real-recording mixing, + both; seed 0.
# Registered here only, so the older GUIs' model lists stay as they are.
_E123 = E.dwg.ROOT / "artifacts/e123_arms"
E.dwg.MODELS.update({f"e123 {n} CRNN": (_E123 / f"e123_{n}.json", _E123 / f"e123_{n}_crnn.pt")
                     for n in ("aug", "mix", "augmix")})
# Entry 129: the it11r CRNN 600k checkpoint at a threshold set on real background (5% FA, it9's rule).
E.dwg.MODELS["it11r CRNN 600k real-thr"] = (E.dwg.ROOT / "artifacts/e129_realthr/it11r_crnn600k_s0_realthr.json",
                                            E.dwg.ROOT / "artifacts/it11r_crnn600k/it11r_crnn600k_s0_crnn.pt")
PART_COLORS = {"drone": "#d93025", "war soundscape": "#6b4e16", "war-shaped noise": "#7a7a7a", "wind": "#2a9d8f",
               "mix": "#111111"}
BG, PANEL, ACCENT, MUTED = "#f4f5f7", "#ffffff", "#2a78d6", "#6b7280"
LOGMEL = LogMel(DEFAULT)
BAND_HZ = E.np.asarray(mel_to_hz(np.linspace(hz_to_mel(DEFAULT.fmin), hz_to_mel(DEFAULT.fmax), DEFAULT.n_mels + 2))[1:-1])


def parse_levels(text: str, what: str) -> list[float]:
    """'40 50, 60' -> [40, 50, 60]; ValueError says what is wrong."""
    parts = text.replace(",", " ").replace(";", " ").split()
    out = []
    for p in parts:
        try:
            v = float(p)
        except ValueError:
            raise ValueError(f"{what}: '{p}' is not a number") from None
        if not -20 <= v <= 140:
            raise ValueError(f"{what}: {v:g} dB SPL is outside -20 to 140")
        if v not in out:
            out.append(v)
    return out


def detector_names() -> list[str]:
    return [*E.dwg.MODELS, "comb"]


# ------------------------------------------------------------------ one page

class Page(ttk.Frame):
    """All controls of one test kind. plan() turns them into an engine Plan (ValueError if unusable)."""

    def __init__(self, master, app, kind: str, title: str):
        super().__init__(master, style="Page.TFrame", padding=(18, 10))
        self.app, self.kind, self.title = app, kind, title
        self.layers = E.make_layers()
        self.files: list[str] = []
        self.thread = None
        self.test = None
        self.stop_flag = threading.Event()
        self.events: queue.Queue = queue.Queue()
        self._build()
        self.update_count()

    # ---------- layout
    def _section(self, row, text, **kw):
        f = ttk.LabelFrame(self, text=f"  {text}  ", style="Card.TLabelframe", padding=(12, 8))
        f.grid(row=row, column=0, sticky="nsew", pady=(0, 10), **kw)
        return f

    def _build(self):
        self.columnconfigure(0, weight=1)
        ttk.Label(self, text=self.title, style="H1.TLabel", font=("Segoe UI", 18, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(self, text=("Real drone recordings mixed into chosen background layers at real levels."
                              if self.kind == "real" else
                              "Pure Data drones flying at the microphone, mixed into chosen background layers."),
                  style="Sub.TLabel").grid(row=1, column=0, sticky="w", pady=(0, 10))
        self._build_drones(self._section(2, "Drones"))
        self._build_layers(self._section(3, "Background layers"))
        self._build_rounds(self._section(4, "Test rounds"))
        self._build_detectors(self._section(5, "Detectors"))
        self._build_run(self._section(6, "Run"))

    def _build_drones(self, f):
        f.columnconfigure(1, weight=1)
        self.v_drone_levels = tk.StringVar(value="60")
        if self.kind == "real":
            ttk.Label(f, text="Drone files (first 10 s of each):").grid(row=0, column=0, sticky="nw")
            box = ttk.Frame(f)
            box.grid(row=0, column=1, sticky="ew", padx=(8, 0))
            box.columnconfigure(0, weight=1)
            self.lb_files = tk.Listbox(box, height=5, activestyle="none", relief="flat", highlightthickness=1,
                                       highlightbackground="#d0d4da", font=("Consolas", 9))
            self.lb_files.grid(row=0, column=0, sticky="ew")
            btns = ttk.Frame(box)
            btns.grid(row=0, column=1, sticky="n", padx=(6, 0))
            ttk.Button(btns, text="Add files...", command=self.add_files).pack(fill="x")
            ttk.Button(btns, text="Remove", command=self.remove_files).pack(fill="x", pady=2)
            ttk.Button(btns, text="Clear", command=self.clear_files).pack(fill="x")
            r = 1
        else:
            self.v_speeds = tk.StringVar(value="100 150 200")
            self.v_seed, self.v_pscale, self.v_elev = tk.StringVar(value="1"), tk.StringVar(value="0.85"), tk.StringVar(value="40")
            ttk.Label(f, text="Drone speeds (km/h):").grid(row=0, column=0, sticky="w")
            ttk.Entry(f, textvariable=self.v_speeds).grid(row=0, column=1, sticky="ew", padx=(8, 0))
            sub = ttk.Frame(f)
            sub.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
            for i, (lbl, var) in enumerate((("drone sound seed", self.v_seed), ("pitch scale", self.v_pscale),
                                            ("elevation (deg)", self.v_elev))):
                ttk.Label(sub, text=lbl, style="Muted.TLabel").grid(row=0, column=2 * i, padx=(0 if i == 0 else 12, 4))
                ttk.Entry(sub, textvariable=var, width=6).grid(row=0, column=2 * i + 1)
            self.v_speeds.trace_add("write", lambda *_: self.update_count())
            r = 2
        ttk.Label(f, text="Drone level(s), dB SPL:").grid(row=r, column=0, sticky="w", pady=(8, 0))
        lv = ttk.Frame(f)
        lv.grid(row=r, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        self.e_drone_levels = ttk.Entry(lv, textvariable=self.v_drone_levels, width=22)
        self.e_drone_levels.pack(side="left")
        self.v_as_rendered = tk.BooleanVar(value=self.kind == "synthetic")
        if self.kind == "synthetic":
            ttk.Checkbutton(lv, text="as rendered (physical level)", variable=self.v_as_rendered,
                            command=self._as_rendered).pack(side="left", padx=(10, 0))
            self._as_rendered()
        ttk.Label(lv, text="RMS over the drone's sound; one or more", style="Muted.TLabel").pack(side="left", padx=(10, 0))
        self.v_drone_levels.trace_add("write", lambda *_: self.update_count())

    def _as_rendered(self):
        self.e_drone_levels.configure(state="disabled" if self.v_as_rendered.get() else "normal")
        self.update_count()

    def _build_layers(self, f):
        f.columnconfigure(0, weight=1)
        head = ttk.Frame(f)
        head.grid(row=0, column=0, sticky="ew")
        self.v_layers_open = tk.BooleanVar(value=True)
        self.b_layers = ttk.Button(head, text="▾ hide layers", style="Link.TButton", command=self._toggle_layers)
        self.b_layers.pack(side="left")
        self.l_layers_sum = ttk.Label(head, text="", style="Muted.TLabel")
        self.l_layers_sum.pack(side="left", padx=10)
        self.f_layers = ttk.Frame(f)
        self.f_layers.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        g = self.f_layers
        g.columnconfigure(3, weight=1)
        for c, h in enumerate(("", "layer", "levels (dB SPL)", "about")):
            ttk.Label(g, text=h, style="Head.TLabel").grid(row=0, column=c, sticky="w", padx=4)
        self.v_layer_on, self.v_layer_lv = {}, {}
        for r, (name, L) in enumerate(self.layers.items(), 1):
            on = tk.BooleanVar(value=name == "war soundscape" and L.available)
            lv = tk.StringVar(value=DEFAULT_LEVELS.get(name, ""))
            self.v_layer_on[name], self.v_layer_lv[name] = on, lv
            cb = ttk.Checkbutton(g, variable=on, command=self.update_count)
            cb.grid(row=r, column=0, padx=4, pady=2)
            ttk.Label(g, text=name, style="Bold.TLabel" if L.available else "Muted.TLabel").grid(row=r, column=1, sticky="w", padx=4)
            e = ttk.Entry(g, textvariable=lv, width=18)
            e.grid(row=r, column=2, sticky="w", padx=4)
            ttk.Label(g, text=L.note, style="Muted.TLabel" if not L.available else "TLabel").grid(row=r, column=3, sticky="w", padx=4)
            if not L.available:
                cb.state(["disabled"])
                e.state(["disabled"])
            lv.trace_add("write", lambda *_: self.update_count())
        r += 1
        st = ttk.Frame(g)
        st.grid(row=r, column=1, columnspan=3, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(st, text="soundscape start in the war track (s):").pack(side="left")
        self.v_sc_start = tk.StringVar(value="")
        ttk.Entry(st, textvariable=self.v_sc_start, width=8).pack(side="left", padx=6)
        ttk.Label(st, text="blank = drawn once per batch (seed 1000); every round uses the same stretch",
                  style="Muted.TLabel").pack(side="left")

    def _toggle_layers(self):
        open_ = not self.v_layers_open.get()
        self.v_layers_open.set(open_)
        (self.f_layers.grid if open_ else self.f_layers.grid_remove)()
        self.b_layers.configure(text="▾ hide layers" if open_ else "▸ show layers")

    def _build_rounds(self, f):
        self.v_len, self.v_step, self.v_nodrone = tk.StringVar(value="20"), tk.StringVar(value="1"), tk.BooleanVar(value=True)
        ttk.Label(f, text="Soundscape length (s):").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(f, from_=10, to=600, increment=1, textvariable=self.v_len, width=8,
                    command=self.update_count).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(f, text="drone length 10 s, starts every").grid(row=0, column=2, sticky="w")
        ttk.Entry(f, textvariable=self.v_step, width=5).grid(row=0, column=3, sticky="w", padx=4)
        ttk.Label(f, text="s until it ends with the soundscape").grid(row=0, column=4, sticky="w")
        ttk.Checkbutton(f, text="add one no-drone round per background (false alarms)", variable=self.v_nodrone,
                        command=self.update_count).grid(row=1, column=0, columnspan=5, sticky="w", pady=(6, 0))
        for v in (self.v_len, self.v_step):
            v.trace_add("write", lambda *_: self.update_count())

    def _build_detectors(self, f):
        self.v_det = {}
        for i, d in enumerate(detector_names()):
            v = tk.BooleanVar(value=d in ("it6 CNN", "comb"))
            self.v_det[d] = v
            tk.Checkbutton(f, text="DSP (comb)" if d == "comb" else d, variable=v, command=self.update_count,
                           fg=COLORS.get(d, "#333"), bg=PANEL, activebackground=PANEL, selectcolor=PANEL,
                           font=("Segoe UI", 10, "bold")).grid(row=i // 6, column=i % 6, padx=(0, 14), sticky="w")

    def _build_run(self, f):
        f.columnconfigure(0, weight=1)
        self.l_count = ttk.Label(f, text="", style="Count.TLabel")
        self.l_count.grid(row=0, column=0, columnspan=4, sticky="w")
        self.l_count_detail = ttk.Label(f, text="", style="Muted.TLabel")
        self.l_count_detail.grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 8))
        self.pb = ttk.Progressbar(f, mode="determinate")
        self.pb.grid(row=2, column=0, sticky="ew", padx=(0, 10))
        self.b_start = ttk.Button(f, text="▶  Start test", style="Accent.TButton", command=self.start)
        self.b_start.grid(row=2, column=1, padx=2)
        self.b_stop = ttk.Button(f, text="■  Stop", command=self.stop, state="disabled")
        self.b_stop.grid(row=2, column=2, padx=2)
        self.b_results = ttk.Button(f, text="Results window", command=self.open_results, state="disabled")
        self.b_results.grid(row=2, column=3, padx=2)
        self.l_status = ttk.Label(f, text="Set up the test, then Start.", style="Muted.TLabel")
        self.l_status.grid(row=3, column=0, columnspan=4, sticky="w", pady=(6, 0))

    # ---------- drones
    def add_files(self):
        paths = filedialog.askopenfilenames(parent=self, title="Drone recordings",
                                            filetypes=[("Audio", "*.wav *.flac *.ogg *.mp3"), ("All", "*.*")])
        for p in paths:
            if p not in self.files:
                self.files.append(p)
                self.lb_files.insert("end", Path(p).name)
        self.update_count()

    def remove_files(self):
        for i in reversed(self.lb_files.curselection()):
            self.lb_files.delete(i)
            del self.files[i]
        self.update_count()

    def clear_files(self):
        self.lb_files.delete(0, "end")
        self.files.clear()
        self.update_count()

    # ---------- plan
    def plan(self) -> E.Plan:
        try:
            length = float(self.v_len.get())
            step = float(self.v_step.get())
        except ValueError:
            raise ValueError("soundscape length and start step must be numbers") from None
        if self.kind == "real":
            drones = list(self.files)
        else:
            try:
                drones = [float(v) for v in self.v_speeds.get().replace(",", " ").split()]
            except ValueError:
                raise ValueError("drone speeds must be numbers in km/h") from None
            if any(not 0 < v <= 1000 for v in drones):
                raise ValueError("drone speeds must be above 0 and at most 1000 km/h")
        levels = None if (self.kind == "synthetic" and self.v_as_rendered.get()) else \
            parse_levels(self.v_drone_levels.get(), "drone level")
        if levels == []:
            raise ValueError("give at least one drone level in dB SPL")
        layers = {}
        for name, on in self.v_layer_on.items():
            if on.get() and self.layers[name].available:
                lv = parse_levels(self.v_layer_lv[name].get(), name)
                if not lv:
                    raise ValueError(f"{name}: give at least one level in dB SPL, or untick it")
                layers[name] = lv
        start = self.v_sc_start.get().strip()
        synth = {}
        if self.kind == "synthetic":
            try:
                synth = dict(drone_seed=int(self.v_seed.get()), pscale=float(self.v_pscale.get()),
                             elev_deg=float(self.v_elev.get()))
            except ValueError:
                raise ValueError("drone sound seed, pitch scale and elevation must be numbers") from None
        p = E.Plan(self.kind, length, drones, levels, layers, [d for d, v in self.v_det.items() if v.get()],
                   step_s=step, no_drone_rounds=self.v_nodrone.get(),
                   soundscape_start_s=float(start) if start else None, **({"synth": synth} if synth else {}))
        return p

    def update_count(self, *_):
        if not hasattr(self, "l_count"):
            return
        try:
            p = self.plan()
        except ValueError as e:
            self.l_count.configure(text="—", style="Count.TLabel")
            self.l_count_detail.configure(text=str(e), style="Warn.TLabel")
            return
        c, probs = p.count(), p.check()
        self.l_count.configure(text=f"{c['total']:,} test rounds".replace(",", " "))
        parts = [f"{c['starts']} starts", f"{c['drones']} drone{'s' * (c['drones'] != 1)}",
                 f"{c['drone_levels']} drone level{'s' * (c['drone_levels'] != 1)}" if p.drone_levels is not None
                 else "drone as rendered", f"{c['backgrounds']} background{'s' * (c['backgrounds'] != 1)}"]
        detail = " × ".join(parts) + f" = {c['drone_rounds']}"
        if p.no_drone_rounds:
            detail += f"  + {c['backgrounds']} no-drone"
        detail += f"   ·   {c['detector_runs']} detector runs"
        if probs:
            detail = "  ·  ".join(probs)
        self.l_count_detail.configure(text=detail, style="Warn.TLabel" if probs else "Muted.TLabel")
        bits = [f"{n} {', '.join(f'{v:g}' for v in lv)} dB" for n, lv in p.layers.items()]
        self.l_layers_sum.configure(text="  ·  ".join(bits) if bits else "no background layer ticked")

    # ---------- run
    def start(self):
        try:
            p = self.plan()
        except ValueError as e:
            messagebox.showerror(TITLE, str(e), parent=self)
            return
        probs = p.check()
        if probs:
            messagebox.showerror(TITLE, "\n".join(probs), parent=self)
            return
        self.test = E.Test(p)
        self.stop_flag.clear()
        self.pb.configure(maximum=len(self.test.rounds), value=0)
        self.b_start.state(["disabled"])
        self.b_stop.state(["!disabled"])
        self.b_results.state(["disabled"])
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()
        self.after(150, self._poll)

    def _work(self):
        try:
            self.test.run(progress=lambda s: self.events.put(("status", s)),
                          on_round=lambda i, r: self.events.put(("round", i)), stop=self.stop_flag.is_set)
            self.events.put(("done", f"done: {len(self.test.results)} rounds. Results in {self.test.out}"))
        except Exception as e:                               # shown, not lost in the thread
            self.events.put(("done", f"error: {type(e).__name__}: {e}"))

    def _poll(self):
        done = False
        while not self.events.empty():
            kind, x = self.events.get()
            if kind == "status":
                self.l_status.configure(text=x)
            elif kind == "round":
                self.pb.configure(value=x)
            else:
                self.l_status.configure(text=x)
                done = True
        if done:
            self.b_start.state(["!disabled"])
            self.b_stop.state(["disabled"])
            if self.test and self.test.results:
                self.b_results.state(["!disabled"])
                self.open_results()
        else:
            self.after(150, self._poll)

    def stop(self):
        self.stop_flag.set()
        self.l_status.configure(text="stopping after this round...")

    def open_results(self):
        if self.test and self.test.results:
            Results(self.app, self.test)


# ------------------------------------------------------------------ results window

class Results(tk.Toplevel):
    """Summary, every round in a sortable table, and the selected round's log-mel, confidences and
    dB SPL per layer over time, with sound."""

    def __init__(self, app, test: E.Test):
        super().__init__(app)
        self.app, self.test = app, test
        self.title(f"{TITLE} results: {test.out.name}")
        self.geometry("1500x940")
        self.configure(bg=BG)
        self.by_round = {r["round"]: r for r in test.rounds}
        self.scores = self._scores()
        self._wav = None
        self._build()

    def _scores(self) -> dict:
        p = self.test.out / "scores.npz"
        if p.exists():
            d = np.load(p)
            return {k: d[k] for k in d.files}
        return {}

    def _build(self):
        top = ttk.Frame(self, padding=(12, 8))
        top.pack(fill="x")
        c = self.test.plan.count()
        ttk.Label(top, text=f"{self.test.plan.kind} drone test · {len(self.test.results)} of {c['total']} rounds · "
                            f"soundscape {self.test.plan.length_s:g} s from {self.test.start_s:g} s in the war track",
                  style="H2.TLabel").pack(side="left")
        ttk.Button(top, text="Open folder", command=lambda: os.startfile(self.test.out)).pack(side="right")
        ttk.Button(top, text="Open rounds.csv", command=lambda: os.startfile(self.test.out / "rounds.csv")).pack(side="right", padx=6)

        pw = ttk.PanedWindow(self, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        left = ttk.Frame(pw)
        right = ttk.Frame(pw)
        pw.add(left, weight=2)
        pw.add(right, weight=3)

        # summary
        ttk.Label(left, text="Summary (mean over rounds)", style="Bold.TLabel").pack(anchor="w")
        cols = ("detector", "setting", "value", "rounds", "recall", "FPR", "detected", "with FA", "delay s",
                "conf drone", "conf none")
        self.t_sum = ttk.Treeview(left, columns=cols, show="headings", height=9)
        for col, w in zip(cols, (80, 110, 120, 50, 55, 55, 62, 58, 55, 70, 65)):
            self.t_sum.heading(col, text=col)
            self.t_sum.column(col, width=w, anchor="w" if col in ("detector", "setting", "value") else "e")
        self.t_sum.pack(fill="x", pady=(2, 10))
        for s in getattr(self.test, "summary", []):
            self.t_sum.insert("", "end", values=(s["detector"], s["setting"], s["value"], s["rounds"],
                                                 *(fmt_pct(s[k]) for k in ("recall", "false_positive_rate",
                                                                           "share_detected", "share_with_false_alarm")),
                                                 fmt(s["delay_s"]), fmt_pct(s["mean_conf_drone"]),
                                                 fmt_pct(s["mean_conf_no_drone"])))

        # rounds table with filter
        bar = ttk.Frame(left)
        bar.pack(fill="x")
        ttk.Label(bar, text="Rounds", style="Bold.TLabel").pack(side="left")
        ttk.Label(bar, text="detector").pack(side="left", padx=(16, 4))
        self.v_fdet = tk.StringVar(value="all")
        cb = ttk.Combobox(bar, textvariable=self.v_fdet, values=["all", *self.test.plan.detectors], width=12, state="readonly")
        cb.pack(side="left")
        ttk.Label(bar, text="search").pack(side="left", padx=(16, 4))
        self.v_search = tk.StringVar()
        ttk.Entry(bar, textvariable=self.v_search, width=18).pack(side="left")
        self.l_shown = ttk.Label(bar, text="", style="Muted.TLabel")
        self.l_shown.pack(side="right")
        self.cols = ["round", "drone", "level", "start", *self.test.plan.layers, "detector", "recall", "FPR",
                     "delay s", "false alarm", "conf drone", "conf none"]
        box = ttk.Frame(left)
        box.pack(fill="both", expand=True, pady=(2, 0))
        self.t_rounds = ttk.Treeview(box, columns=self.cols, show="headings", selectmode="browse")
        for col in self.cols:
            self.t_rounds.heading(col, text=col, command=lambda c=col: self._sort(c))
            self.t_rounds.column(col, width=54 if col not in ("drone", "detector") else 110,
                                 anchor="w" if col in ("drone", "detector") else "e")
        sb = ttk.Scrollbar(box, orient="vertical", command=self.t_rounds.yview)
        self.t_rounds.configure(yscrollcommand=sb.set)
        self.t_rounds.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.t_rounds.bind("<<TreeviewSelect>>", self._select)
        self.sort_col, self.sort_rev = "round", False
        self.v_fdet.trace_add("write", lambda *_: self._fill())
        self.v_search.trace_add("write", lambda *_: self._fill())
        self._fill()

        # round view
        ctl = ttk.Frame(right)
        ctl.pack(fill="x")
        self.l_round = ttk.Label(ctl, text="Select a round in the table.", style="Bold.TLabel")
        self.l_round.pack(side="left")
        for txt, what in (("Play background", "background"), ("Play drone", "drone"), ("Play mix", "mix")):
            ttk.Button(ctl, text=txt, command=lambda w=what: self._play(w)).pack(side="right", padx=2)
        ttk.Button(ctl, text="Stop sound", command=stop_sound).pack(side="right", padx=(2, 8))
        self.fig = Figure(figsize=(9, 8), dpi=100, facecolor="white")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        NavigationToolbar2Tk(self.canvas, right, pack_toolbar=True)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.cur = None

    def _row_values(self, rnd: int, det: str, res: dict) -> tuple:
        r = self.by_round[rnd]
        lv = "as rendered" if r["drone"] is not None and r["drone_level_db"] is None else fmt(r["drone_level_db"])
        return (rnd, E.drone_label(r["drone"]) if r["drone"] is not None else "none", lv, fmt(r["start_s"]),
                *(f"{r['background'][n]:g}" for n in self.test.plan.layers), det, fmt_pct(res["recall"]),
                fmt_pct(res["false_positive_rate"]), fmt(res["delay_s"]), "yes" if res["any_false_alarm"] else "",
                fmt_pct(res["mean_conf_drone"]), fmt_pct(res["mean_conf_no_drone"]))

    def _fill(self):
        self.t_rounds.delete(*self.t_rounds.get_children())
        want, q = self.v_fdet.get(), self.v_search.get().lower().strip()
        rows = []
        for rnd, dets in self.test.results.items():
            for det, res in dets.items():
                if want != "all" and det != want:
                    continue
                v = self._row_values(rnd, det, res)
                if q and q not in " ".join(map(str, v)).lower():
                    continue
                rows.append(v)
        i = self.cols.index(self.sort_col)
        rows.sort(key=lambda v: sort_key(v[i]), reverse=self.sort_rev)
        for v in rows:
            self.t_rounds.insert("", "end", values=v, tags=("fa",) if v[self.cols.index("false alarm")] else ())
        self.t_rounds.tag_configure("fa", foreground="#c0392b")
        self.l_shown.configure(text=f"{len(rows)} rows · click a header to sort")

    def _sort(self, col):
        self.sort_rev = not self.sort_rev if col == self.sort_col else False
        self.sort_col = col
        self._fill()

    def _select(self, _=None):
        sel = self.t_rounds.selection()
        if sel:
            self.show_round(int(self.t_rounds.item(sel[0], "values")[0]))

    def show_round(self, rnd: int):
        r = self.by_round[rnd]
        m = self.test.rebuild(r)
        self.cur = (rnd, m)
        n = len(m["mix"])
        starts = np.arange(0, n - DEFAULT.samples_per_window + 1, DEFAULT.samples_per_hop)
        t_end = (starts + DEFAULT.samples_per_window) / E.SR
        self.fig.clear()
        ax1, ax2, ax3 = self.fig.subplots(3, 1, sharex=True, gridspec_kw=dict(height_ratios=[1.1, 1, 1]))
        lm = 10 / np.log(10) * LOGMEL.frames(m["mix"].astype(np.float32))
        top = float(lm.max())
        ax1.imshow(lm, origin="lower", aspect="auto", cmap="magma", vmin=top - 80, vmax=top, interpolation="nearest",
                   extent=(0, n / E.SR, 0, lm.shape[0]))
        ticks = [0, 16, 32, 48, 63]
        ax1.set_yticks([t + 0.5 for t in ticks], [f"{BAND_HZ[t]:.0f}" for t in ticks])
        ax1.set_ylabel("log-mel band (Hz)")
        title = (f"round {rnd}: " + (E.drone_label(r["drone"]) if r["drone"] is not None else "no drone")
                 + (f" at {r['drone_level_db']:g} dB SPL" if r["drone_level_db"] is not None else
                    " as rendered" if r["drone"] is not None else "")
                 + (f", starts {r['start_s']:g} s" if r["start_s"] is not None else "")
                 + "".join(f" · {k} {v:g} dB" for k, v in r["background"].items()))
        ax1.set_title(title, fontsize=9, loc="left")
        for k in m["truth"].drones:
            for ax in (ax2, ax3):
                ax.axvspan(k["start_s"], k["end_s"], color="#d93025", alpha=0.06, lw=0)
        for det in self.test.plan.detectors:
            s = self.scores.get(f"{rnd}|{det}")
            if s is None or len(s) != len(t_end):
                continue
            thr = self.test.thr[det]
            y = E.dwg.squash(s, thr) if det == "comb" else s
            ax2.plot(t_end, y, color=COLORS.get(det, "#333"), lw=1.3, label=det if det != "comb" else "DSP (comb)")
            ax2.axhline(0.5 if det == "comb" else thr, color=COLORS.get(det, "#333"), lw=0.8, ls="--")
        ax2.set(ylim=(0, 1.02), ylabel="confidence")
        ax2.legend(loc="upper left", fontsize=8, frameon=False, ncol=5)
        ax2.set_title("each detector's confidence (dashed = threshold; red band = drone sound; "
                      "decision at the end of its 1 s window)", fontsize=8, loc="left", color=MUTED)
        parts = dict(m["parts"], mix=m["mix"])
        lv = E.hop_levels(parts, t_end)
        for name, y in lv.items():
            ax3.plot(t_end, y, color=PART_COLORS.get(name, "#555"), lw=2.0 if name == "mix" else 1.3,
                     ls="-" if name != "mix" else ":", label=name)
        fin = np.concatenate([y[np.isfinite(y)] for y in lv.values()]) if lv else np.array([0.0])
        ax3.set(ylabel="dB SPL (1 s windows)", ylim=(max(float(fin.min()) - 5, 0), float(fin.max()) + 5),
                xlabel="time (s)", xlim=(0, n / E.SR))
        ax3.legend(loc="upper left", fontsize=8, frameon=False, ncol=6)
        for ax in (ax2, ax3):
            ax.grid(color="#e8e8e8", lw=0.5)
        res = self.test.results[rnd]
        self.l_round.configure(text=f"Round {rnd}:  " + "   ".join(
            f"{d}: recall {fmt_pct(x['recall'])}, FPR {fmt_pct(x['false_positive_rate'])}" for d, x in res.items()))
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _play(self, what):
        if not self.cur:
            return
        _, m = self.cur
        drone = m["parts"].get("drone", np.zeros_like(m["mix"]))
        x = {"mix": m["mix"], "drone": drone, "background": m["mix"] - drone}[what]
        g = 0.9 / max(float(np.max(np.abs(m["mix"]))), 1e-12)   # the mix's gain for all three: levels stay comparable
        self._wav = play(np.clip(x * g, -1, 1), self._wav)


# ------------------------------------------------------------------ helpers

def fmt(v) -> str:
    return "" if v is None else f"{v:.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def fmt_pct(v) -> str:
    return "" if v is None else f"{100 * v:.0f}%"


def sort_key(v):
    s = str(v).rstrip("%")
    try:
        return (0, float(s))
    except ValueError:
        return (1, s)


def play(x: np.ndarray, old: str | None) -> str | None:
    """Plays x (16 kHz) from a temp WAV; returns its path so the next call can delete it."""
    if sys.platform != "win32":
        return old
    import winsound
    winsound.PlaySound(None, 0)
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    f.close()
    sf.write(f.name, x.astype(np.float32), E.SR, subtype="PCM_16")
    if old:
        Path(old).unlink(missing_ok=True)
    winsound.PlaySound(f.name, winsound.SND_FILENAME | winsound.SND_ASYNC)
    return f.name


def stop_sound():
    if sys.platform == "win32":
        import winsound
        winsound.PlaySound(None, 0)


def load_test(folder: Path) -> E.Test:
    """An earlier run from its folder: plan and start from config.json, results from rounds.csv."""
    import csv
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    p = cfg["plan"]
    plan = E.Plan(**{k: p[k] for k in ("kind", "length_s", "drones", "drone_levels", "layers", "detectors", "step_s",
                                       "no_drone_rounds", "soundscape_start_s", "synth")})
    t = E.Test(plan, folder)
    t.start_s = cfg["soundscape_start_s"]
    t.prepare(progress=lambda s: None)
    num = lambda v: None if v in ("", "None") else (v == "True") if v in ("True", "False") else float(v)  # noqa: E731
    with open(folder / "rounds.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            t.results.setdefault(int(row["round"]), {})[row["detector"]] = {
                k: num(row[k]) for k in ("tp", "fn", "fp", "tn", "left_out", "recall", "false_positive_rate",
                                         "precision", "first_alarm_s", "delay_s", "lead_s", "any_alarm",
                                         "any_false_alarm", "mean_conf_drone", "mean_conf_no_drone")}
    E.write_summary(t)
    return t


# ------------------------------------------------------------------ the app

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(TITLE)
        self.geometry("1180x940")
        self.minsize(980, 760)
        self.configure(bg=BG)
        style(self)
        bar = tk.Frame(self, bg="#1f2937", height=52)
        bar.pack(fill="x")
        tk.Label(bar, text="  ◉  Complete Test", bg="#1f2937", fg="white", font=("Segoe UI", 14, "bold")).pack(side="left", pady=10)
        tk.Button(bar, text="Open results...", command=self.open_results, bg="#374151", fg="white", relief="flat",
                  activebackground="#4b5563", activeforeground="white", padx=12).pack(side="right", padx=12, pady=10)
        nav = tk.Frame(bar, bg="#1f2937")
        nav.pack(side="right", padx=20)
        tk.Button(nav, text="◀", command=lambda: self.go(-1), bg="#1f2937", fg="white", relief="flat",
                  font=("Segoe UI", 12), activebackground="#374151").pack(side="left")
        self.tabs = []
        for i, (_, title) in enumerate(PAGES):
            b = tk.Button(nav, text=title, command=lambda i=i: self.show(i), relief="flat", padx=14,
                          font=("Segoe UI", 10, "bold"), activebackground="#374151")
            b.pack(side="left", padx=2)
            self.tabs.append(b)
        tk.Button(nav, text="▶", command=lambda: self.go(1), bg="#1f2937", fg="white", relief="flat",
                  font=("Segoe UI", 12), activebackground="#374151").pack(side="left")
        self.dots = tk.Label(bar, text="", bg="#1f2937", fg="#9ca3af", font=("Segoe UI", 10))
        self.dots.pack(side="right")

        holder = ttk.Frame(self, style="Page.TFrame")
        holder.pack(fill="both", expand=True)
        canvas = tk.Canvas(holder, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(holder, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(canvas, style="Page.TFrame")
        win = canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units")
                        if str(e.widget).startswith(str(self)) and not isinstance(e.widget, (tk.Listbox,)) else None)
        self.pages = [Page(self.inner, self, kind, title) for kind, title in PAGES]
        self.cur = 0
        self.show(0)
        self.bind("<Control-Left>", lambda e: self.go(-1))
        self.bind("<Control-Right>", lambda e: self.go(1))

    def show(self, i):
        for p in self.pages:
            p.pack_forget()
        self.cur = i % len(self.pages)
        self.pages[self.cur].pack(fill="both", expand=True)
        for j, b in enumerate(self.tabs):
            b.configure(bg=ACCENT if j == self.cur else "#374151", fg="white")
        self.dots.configure(text="  ".join("●" if j == self.cur else "○" for j in range(len(self.pages))))

    def go(self, step):
        self.show(self.cur + step)

    def open_results(self):
        d = filedialog.askdirectory(parent=self, title="A run folder in artifacts/complete_test",
                                    initialdir=str(E.OUT_ROOT) if E.OUT_ROOT.exists() else None)
        if not d:
            return
        try:
            Results(self, load_test(Path(d)))
        except Exception as e:                               # noqa: BLE001  (shown, not raised)
            messagebox.showerror(TITLE, f"cannot open {d}:\n{type(e).__name__}: {e}", parent=self)


def style(root):
    s = ttk.Style(root)
    s.theme_use("clam")
    root.option_add("*Font", ("Segoe UI", 10))
    s.configure(".", background=BG, font=("Segoe UI", 10))
    s.configure("Page.TFrame", background=BG)
    s.configure("Card.TLabelframe", background=PANEL, bordercolor="#dfe3e8", relief="solid")
    s.configure("Card.TLabelframe.Label", background=BG, foreground="#111827", font=("Segoe UI", 11, "bold"))
    for name in ("TLabel", "TCheckbutton", "TFrame"):
        s.configure(name, background=PANEL)
    s.configure("Page.TFrame", background=BG)
    s.configure("H1.TLabel", background=BG, font=("Segoe UI", 18, "bold"), foreground="#111827")
    s.configure("H2.TLabel", font=("Segoe UI", 12, "bold"), foreground="#111827", background=BG)
    s.configure("Sub.TLabel", background=BG, foreground=MUTED)
    s.configure("Muted.TLabel", foreground=MUTED, background=PANEL)
    s.configure("Warn.TLabel", foreground="#b45309", background=PANEL)
    s.configure("Bold.TLabel", font=("Segoe UI", 10, "bold"))
    s.configure("Head.TLabel", foreground=MUTED, font=("Segoe UI", 9))
    s.configure("Count.TLabel", font=("Segoe UI", 20, "bold"), foreground=ACCENT)
    s.configure("Accent.TButton", background=ACCENT, foreground="white", font=("Segoe UI", 10, "bold"), padding=(14, 6))
    s.map("Accent.TButton", background=[("disabled", "#9db8e0"), ("active", "#1d5fb4")])
    s.configure("Link.TButton", background=PANEL, foreground=ACCENT, borderwidth=0, padding=0)
    s.map("Link.TButton", background=[("active", PANEL)])
    s.configure("Treeview", rowheight=20, font=("Segoe UI", 9))
    s.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))


if __name__ == "__main__":
    App().mainloop()
