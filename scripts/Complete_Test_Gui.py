"""Complete test: full control over large batches of drone-in-noise test rounds, no graphs on the front.

    uv run python scripts/Complete_Test_Gui.py

Three pages, a slide show (the arrows at the top, the page names, or Ctrl+Left / Ctrl+Right):

    Real drone test       real drone recordings (WAV files you add), each cut to its first 10 s
                          (a file of 5-10 s is used whole)
    Synthetic drone test  Pure Data drones, one per speed, 10 s from start to impact (as drone_war_gui)
    Modified drone test   real drone recordings treated as a steady drone heard at 1 m and flown
                          straight at the microphone: the file's last 10 s (or all of a 5-10 s file),
                          its last sample the impact. Per speed (km/h) the drone is v/3.6 x the time
                          left before impact away; the level falls with that distance (spherical
                          spreading and air absorption, optional Doppler). Each file at each speed.

On each page:

    Drones               the files or speeds, and the drone level(s) in dB SPL (synthetic: "as rendered"
                         keeps the physical level of the render)
    Background layers    (open with the arrow) war soundscape, war-shaped noise, wind, and cars,
                         helicopter, insects (listed, not set up yet). Tick a layer and give its levels in
                         dB SPL as min, max and # (as drone_war_gui: # values from min to max, both
                         included, evenly spaced; # 1 needs min = max); every combination of the ticked
                         layers' levels is a background
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
import time
import tkinter as tk
from dataclasses import asdict
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
PAGES = [("real", "Real drone test"), ("synthetic", "Synthetic drone test"), ("modified", "Modified drone test")]
SUBTITLES = {"real": "Real drone recordings mixed into chosen background layers at real levels.",
             "synthetic": "Pure Data drones flying at the microphone, mixed into chosen background layers.",
             "modified": "Real drone recordings flown at the microphone: the end of each file is the impact, and the "
                         "level follows the distance at each moment for the chosen speed."}
DEFAULT_LEVELS = {"war soundscape": ("60", "60", "1"), "war-shaped noise": ("50", "50", "1"), "wind": ("50", "50", "1"),
                  "cars": ("", "", ""), "helicopter": ("", "", ""), "insects": ("", "", "")}
COLORS = {"it6 CNN": "#2a78d6", "it9 CNN": "#e8710a", "it9 CRNN": "#9334e6", "it10 CNN": "#d93025",
          "comb": "#1baf7a"}
PART_COLORS = {"drone": "#d93025", "war soundscape": "#6b4e16", "war-shaped noise": "#7a7a7a", "wind": "#2a9d8f",
               "mix": "#111111"}
BG, PANEL, ACCENT, MUTED = "#f4f5f7", "#ffffff", "#2a78d6", "#6b7280"
LOGMEL = LogMel(DEFAULT)
BAND_HZ = E.np.asarray(mel_to_hz(np.linspace(hz_to_mel(DEFAULT.fmin), hz_to_mel(DEFAULT.fmax), DEFAULT.n_mels + 2))[1:-1])


def layer_steps(name: str, lo: str, hi: str, n: str) -> list[float]:
    """A layer's levels from its min, max and # boxes, by drone_war_gui's rule (drone_war_grid.steps):
    # values from min to max, both included, evenly spaced; # 1 needs min = max. ValueError says what."""
    try:
        lo_, hi_ = float(lo), float(hi)
    except ValueError:
        raise ValueError(f"{name}: min and max must be numbers of dB SPL") from None
    try:
        n_ = int(n)
    except ValueError:
        raise ValueError(f"{name}: # must be a whole number") from None
    if not 1 <= n_ <= 50:
        raise ValueError(f"{name}: # must be 1 to 50")
    for v in (lo_, hi_):
        if not -20 <= v <= 140:
            raise ValueError(f"{name}: {v:g} dB SPL is outside -20 to 140")
    try:
        return E.dwg.steps(lo_, hi_, n_, f"{name} level")
    except ValueError as e:                                  # its words, with the box's name
        raise ValueError(str(e).replace("the count", "#").replace("count 1", "# 1")) from None


def detector_names() -> list[str]:
    return [*E.dwg.MODELS, "comb"]


def approach_preview(speeds: list[float], closest_m: float, length_s: float = E.DRONE_S) -> str:
    """Where a modified drone starts for each speed, and how much quieter than at impact (spreading
    only; air absorption takes a little more off the treble), for a 10 s file."""
    if not speeds or closest_m <= 0:
        return ""
    bits = []
    for v in speeds[:6]:
        r0 = float(np.hypot(closest_m, v / 3.6 * length_s))
        bits.append(f"{v:g} km/h: {r0:.0f} m, {20 * np.log10(r0 / closest_m):.0f} dB below impact")
    return "start (10 s before impact)  ·  " + "  ·  ".join(bits) + ("  ·  …" if len(speeds) > 6 else "")


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
        ttk.Label(self, text=SUBTITLES[self.kind], style="Sub.TLabel").grid(row=1, column=0, sticky="w", pady=(0, 10))
        self._build_drones(self._section(2, "Drones"))
        self._build_layers(self._section(3, "Background layers"))
        self._build_rounds(self._section(4, "Test rounds"))
        self._build_detectors(self._section(5, "Detectors"))
        self._build_run(self._section(6, "Run"))

    def _build_drones(self, f):
        f.columnconfigure(1, weight=1)
        self.v_drone_levels = tk.StringVar(value="60")
        if self.kind in ("real", "modified"):
            ttk.Label(f, text="Drone files (first 10 s of each; 5-10 s files whole):" if self.kind == "real" else
                      "Drone files (last 10 s of each = up to impact;\n5-10 s files whole):").grid(row=0, column=0, sticky="nw")
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
            if self.kind == "modified":
                self.v_speeds = tk.StringVar(value="50 100 150")
                self.v_closest = tk.StringVar(value=f"{E.APPROACH['closest_m']:g}")
                self.v_doppler = tk.BooleanVar(value=E.APPROACH["doppler"])
                ttk.Label(f, text="Speeds (km/h):").grid(row=1, column=0, sticky="w", pady=(8, 0))
                ttk.Entry(f, textvariable=self.v_speeds).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
                sub = ttk.Frame(f)
                sub.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
                ttk.Label(sub, text="distance at impact (m)", style="Muted.TLabel").pack(side="left", padx=(0, 4))
                ttk.Entry(sub, textvariable=self.v_closest, width=6).pack(side="left")
                ttk.Checkbutton(sub, text="Doppler (pitch rises while it approaches)", variable=self.v_doppler,
                                command=self.update_count).pack(side="left", padx=(12, 0))
                self.l_approach = ttk.Label(f, text="", style="Muted.TLabel")
                self.l_approach.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(2, 0))
                for v in (self.v_speeds, self.v_closest):
                    v.trace_add("write", lambda *_: self.update_count())
                r = 4
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
        ttk.Label(f, text="Drone level at 1 m, dB SPL:" if self.kind == "modified" else "Drone level, dB SPL:").grid(
            row=r, column=0, sticky="w", pady=(8, 0))
        lv = ttk.Frame(f)
        lv.grid(row=r, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        d0 = "80" if self.kind == "modified" else "60"
        self.v_drone_levels = (tk.StringVar(value=d0), tk.StringVar(value=d0), tk.StringVar(value="1"))
        self.e_drone_levels = []
        for lbl, v, w in zip(("min", "max", "#"), self.v_drone_levels, (7, 7, 4)):
            ttk.Label(lv, text=lbl, style="Muted.TLabel").pack(side="left", padx=(0 if lbl == "min" else 8, 3))
            e = ttk.Entry(lv, textvariable=v, width=w)
            e.pack(side="left")
            self.e_drone_levels.append(e)
            v.trace_add("write", lambda *_: self.update_count())
        self.l_drone_preview = ttk.Label(lv, text="", style="Muted.TLabel", width=26)
        self.l_drone_preview.pack(side="left", padx=(10, 0))
        self.v_as_rendered = tk.BooleanVar(value=self.kind == "synthetic")
        if self.kind == "synthetic":
            ttk.Checkbutton(lv, text="as rendered (physical level)", variable=self.v_as_rendered,
                            command=self._as_rendered).pack(side="left", padx=(10, 0))
            self._as_rendered()
        ttk.Label(f, text=("the file's RMS is set to this as if heard at 1 m, then lowered by the distance; "
                           if self.kind == "modified" else "RMS over the drone's sound; ")
                  + "# values from min to max, both included, evenly spaced; # 1 needs min = max",
                  style="Muted.TLabel").grid(row=r + 1, column=1, sticky="w", padx=(8, 0))

    def _as_rendered(self):
        for e in self.e_drone_levels:
            e.configure(state="disabled" if self.v_as_rendered.get() else "normal")
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
        g.columnconfigure(6, weight=1)
        for c, h in enumerate(("", "layer", "min (dB SPL)", "max (dB SPL)", "#", "", "about")):
            ttk.Label(g, text=h, style="Head.TLabel").grid(row=0, column=c, sticky="w", padx=4)
        self.v_layer_on, self.v_layer_lv = {}, {}
        for r, (name, L) in enumerate(self.layers.items(), 1):
            on = tk.BooleanVar(value=name == "war soundscape" and L.available)
            lo, hi, n = DEFAULT_LEVELS.get(name, ("", "", ""))
            vs = (tk.StringVar(value=lo), tk.StringVar(value=hi), tk.StringVar(value=n))
            self.v_layer_on[name], self.v_layer_lv[name] = on, vs
            cb = ttk.Checkbutton(g, variable=on, command=self.update_count)
            cb.grid(row=r, column=0, padx=4, pady=2)
            ttk.Label(g, text=name, style="Bold.TLabel" if L.available else "Muted.TLabel").grid(row=r, column=1, sticky="w", padx=4)
            entries = [ttk.Entry(g, textvariable=v, width=7 if i < 2 else 4) for i, v in enumerate(vs)]
            for i, e in enumerate(entries):
                e.grid(row=r, column=2 + i, sticky="w", padx=4)
            preview = ttk.Label(g, text="", style="Muted.TLabel", width=26)
            preview.grid(row=r, column=5, sticky="w", padx=4)
            ttk.Label(g, text=L.note, style="Muted.TLabel" if not L.available else "TLabel").grid(row=r, column=6, sticky="w", padx=4)
            if not L.available:
                cb.state(["disabled"])
                for e in entries:
                    e.state(["disabled"])
            for v in vs:
                v.trace_add("write", lambda *_: self.update_count())
            self.layer_preview = getattr(self, "layer_preview", {})
            self.layer_preview[name] = preview
        r += 1
        ttk.Label(g, text="# values from min to max, both included, evenly spaced; # 1 needs min = max "
                          "(as in drone_war_gui)", style="Muted.TLabel").grid(row=r, column=1, columnspan=6, sticky="w",
                                                                             padx=4, pady=(4, 0))
        r += 1
        st = ttk.Frame(g)
        st.grid(row=r, column=1, columnspan=6, sticky="w", padx=4, pady=(6, 0))
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
        ttk.Label(f, text="drone length up to 10 s, starts every").grid(row=0, column=2, sticky="w")
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
                           font=("Segoe UI", 10, "bold")).grid(row=0, column=i, padx=(0, 14), sticky="w")

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
        ttk.Button(f, text="Save plan...", command=self.save_plan).grid(row=2, column=4, padx=2)
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
        speeds, approach = None, None
        if self.kind in ("real", "modified"):
            drones = list(self.files)
            if self.kind == "modified":
                try:
                    speeds = [float(v) for v in self.v_speeds.get().replace(",", " ").split()]
                    approach = dict(closest_m=float(self.v_closest.get()), doppler=bool(self.v_doppler.get()))
                except ValueError:
                    raise ValueError("speeds and distance at impact must be numbers") from None
        else:
            try:
                drones = [float(v) for v in self.v_speeds.get().replace(",", " ").split()]
            except ValueError:
                raise ValueError("drone speeds must be numbers in km/h") from None
            if any(not 0 < v <= 1000 for v in drones):
                raise ValueError("drone speeds must be above 0 and at most 1000 km/h")
        levels = None if (self.kind == "synthetic" and self.v_as_rendered.get()) else \
            layer_steps("drone", *(v.get() for v in self.v_drone_levels))
        layers = {}
        for name, on in self.v_layer_on.items():
            if on.get() and self.layers[name].available:
                layers[name] = layer_steps(name, *(v.get() for v in self.v_layer_lv[name]))
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
                   soundscape_start_s=float(start) if start else None, **({"synth": synth} if synth else {}),
                   **({"speeds": speeds, "approach": approach} if self.kind == "modified" else {}))
        return p

    def update_count(self, *_):
        if not hasattr(self, "l_count"):
            return
        if self.kind == "synthetic" and self.v_as_rendered.get():   # the drone's levels, or what is wrong
            self.l_drone_preview.configure(text="physical level of each render", style="Muted.TLabel")
        else:
            try:
                vals = layer_steps("drone", *(v.get() for v in self.v_drone_levels))
                txt = ", ".join(f"{v:g}" for v in vals)
                self.l_drone_preview.configure(text=(txt if len(txt) <= 24 else txt[:22] + "…") + " dB",
                                               style="Muted.TLabel")
            except ValueError as e:
                self.l_drone_preview.configure(text=str(e).split(": ", 1)[-1][:26], style="Warn.TLabel")
        for name, lbl in self.layer_preview.items():         # each layer's values, or what is wrong
            if not (self.v_layer_on[name].get() and self.layers[name].available):
                lbl.configure(text="", style="Muted.TLabel")
                continue
            try:
                vals = layer_steps(name, *(v.get() for v in self.v_layer_lv[name]))
                txt = ", ".join(f"{v:g}" for v in vals)
                lbl.configure(text=(txt if len(txt) <= 24 else txt[:22] + "…") + " dB", style="Muted.TLabel")
            except ValueError as e:
                lbl.configure(text=str(e).split(": ", 1)[-1][:26], style="Warn.TLabel")
        try:
            p = self.plan()
        except ValueError as e:
            self.l_count.configure(text="—", style="Count.TLabel")
            self.l_count_detail.configure(text=str(e), style="Warn.TLabel")
            return
        if self.kind == "modified":
            self.l_approach.configure(text=approach_preview(p.speeds, p.approach["closest_m"]))
        c, probs = p.count(), p.check()
        self.l_count.configure(text=f"{c['total']:,} test rounds".replace(",", " "))
        n_f, n_v = len(p.drones), len(p.speeds or [])
        drones_txt = (f"{n_f} file{'s' * (n_f != 1)} × {n_v} speed{'s' * (n_v != 1)}" if self.kind == "modified"
                      else f"{c['drones']} drone{'s' * (c['drones'] != 1)}")
        parts = [f"{c['starts']} starts", drones_txt,
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
    def checked_plan(self) -> E.Plan | None:
        """The plan, or None after saying what is wrong with it."""
        try:
            p = self.plan()
        except ValueError as e:
            messagebox.showerror(TITLE, str(e), parent=self)
            return None
        probs = p.check()
        if probs:
            messagebox.showerror(TITLE, "\n".join(probs), parent=self)
            return None
        return p

    def save_plan(self):
        """The plan as JSON, for scripts/complete_test_run.py (the same test without the GUI, on the server)."""
        p = self.checked_plan()
        if p is None:
            return
        f = filedialog.asksaveasfilename(parent=self, title="Plan for complete_test_run.py", defaultextension=".json",
                                         initialfile=f"plan_{p.kind}.json", filetypes=[("JSON", "*.json")],
                                         initialdir=str(E.OUT_ROOT) if E.OUT_ROOT.exists() else None)
        if f:
            Path(f).write_text(json.dumps(dict(plan=asdict(p)), indent=1), encoding="utf-8")
            self.l_status.configure(text=f"plan saved: {f}")

    def start(self):
        p = self.checked_plan()
        if p is None:
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
        ttk.Button(top, text="Conclusions", style="Accent.TButton",
                   command=lambda: Conclusions(self, self.test)).pack(side="right", padx=6)
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
        ttk.Button(ctl, text="Stop sound", command=self._stop_play).pack(side="right", padx=(2, 8))
        self.fig = Figure(figsize=(9, 8), dpi=100, facecolor="white")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        NavigationToolbar2Tk(self.canvas, right, pack_toolbar=True)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.cur = None
        self.axes, self._head, self._head_job = [], [], None   # the playhead: one line per graph

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
        self._stop_play()
        self.fig.clear()
        ax1, ax2, ax3 = self.fig.subplots(3, 1, sharex=True, gridspec_kw=dict(height_ratios=[1.1, 1, 1]))
        self.axes = [ax1, ax2, ax3]
        lm = 10 / np.log(10) * LOGMEL.frames(m["mix"].astype(np.float32))
        top = float(lm.max())
        ax1.imshow(lm, origin="lower", aspect="auto", cmap="magma", vmin=top - 80, vmax=top, interpolation="nearest",
                   extent=(0, n / E.SR, 0, lm.shape[0]))
        ticks = [0, 16, 32, 48, 63]
        ax1.set_yticks([t + 0.5 for t in ticks], [f"{BAND_HZ[t]:.0f}" for t in ticks])
        ax1.set_ylabel("log-mel band (Hz)")
        title = (f"round {rnd}: " + (E.drone_label(r["drone"]) if r["drone"] is not None else "no drone")
                 + (f" at {r['drone_level_db']:g} dB SPL" + (" at 1 m" if self.test.plan.kind == "modified" else "")
                    if r["drone_level_db"] is not None else
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
        self._stop_play()
        self._wav = play(np.clip(x * g, -1, 1), self._wav)
        self._start_head(len(x) / E.SR, time.perf_counter())   # the clock starts with the sound

    def _start_head(self, seconds: float, t0: float):
        """A vertical line across the log-mel and the graphs that follows the sound as it plays. Only
        the line is redrawn (blitting over a saved background), so it moves smoothly."""
        self._head = [ax.axvline(0, color="#00e5ff" if i == 0 else "#111111", lw=1.6, alpha=0.9, zorder=10,
                                 animated=True) for i, ax in enumerate(self.axes)]
        self.canvas.draw()                                   # the figure without the (animated) line
        self._head_bg = self.canvas.copy_from_bbox(self.fig.bbox)
        self._head_cid = self.canvas.mpl_connect("draw_event", self._new_head_bg)   # resized: save it again
        self._head_t0, self._head_len = t0, seconds
        self._tick_head()

    def _new_head_bg(self, _):
        self._head_bg = self.canvas.copy_from_bbox(self.fig.bbox)

    def _tick_head(self):
        t = time.perf_counter() - self._head_t0
        if t > self._head_len or not self._head:
            self._remove_head()
            return
        self.canvas.restore_region(self._head_bg)
        for ln in self._head:
            ln.set_xdata([t, t])
            ln.axes.draw_artist(ln)
        self.canvas.blit(self.fig.bbox)
        self._head_job = self.after(30, self._tick_head)

    def _remove_head(self):
        if self._head_job:
            self.after_cancel(self._head_job)
            self._head_job = None
        if self._head:
            self.canvas.mpl_disconnect(self._head_cid)
            for ln in self._head:
                ln.remove()
            self._head = []
            self.canvas.draw_idle()

    def _stop_play(self):
        stop_sound()
        self._remove_head()


# ------------------------------------------------------------------ conclusions

CURVE_ABOVE = 4                                              # more values than this: a curve, not bars


def parameter_values(test: E.Test) -> dict[str, list]:
    """Every setting that takes more than one value in the test, with its values in order."""
    modified = test.plan.kind == "modified"
    names = {"drone": "drone", **({"file": "drone file", "speed_kmh": "speed (km/h)"} if modified else {}),
             "drone_level_db": "drone level at 1 m (dB SPL)" if modified else "drone level (dB SPL)",
             "start_s": "drone start (s)", **{n: f"{n} (dB SPL)" for n in test.plan.layers}}
    out = {}
    for key, label in names.items():
        vals = {setting(r, key) for r in test.rounds if setting(r, key) is not None}
        if len(vals) > 1:
            out[key] = (label, sorted(vals, key=lambda v: (isinstance(v, str), v)))
    return out


def setting(r: dict, key: str):
    if key in r["background"]:
        return r["background"][key]
    if r["drone"] is None:                                   # a no-drone round has no drone setting
        return None
    return E.drone_setting(r["drone"], key) if key in ("drone", "file", "speed_kmh") else r[key]


def confidences(test: E.Test, det: str, key: str, value) -> dict:
    """Mean confidence of `det` over the rounds where `key` = value: true = over every drone window,
    false = over every no-drone window (no-drone rounds included), each window counted once."""
    sums = new_sums()
    for rnd, res in test.results.items():
        r = test.by_round[rnd]                               # set by Conclusions
        if det in res and setting(r, key) == value:
            add_conf(sums, res[det])
    return final_conf(sums)


def new_sums() -> dict:
    return dict(true=0.0, nt=0, false=0.0, nf=0, rounds=0)


def add_conf(sums: dict, x: dict):
    """One round's detector row into window-weighted sums of true and false confidence."""
    sums["rounds"] += 1
    n_t, n_f = int(x["tp"] + x["fn"]), int(x["fp"] + x["tn"])
    if n_t and x["mean_conf_drone"] is not None:
        sums["true"] += x["mean_conf_drone"] * n_t
        sums["nt"] += n_t
    if n_f and x["mean_conf_no_drone"] is not None:
        sums["false"] += x["mean_conf_no_drone"] * n_f
        sums["nf"] += n_f


def final_conf(sums: dict) -> dict:
    return dict(true=sums["true"] / sums["nt"] if sums["nt"] else None,
                false=sums["false"] / sums["nf"] if sums["nf"] else None, rounds=sums["rounds"])


def level_ratio_points(test: E.Test, det: str, layer: str) -> list[tuple[float, dict]]:
    """Nicolai's x-axis: the drone's level over the layer's level, both in dB SPL (drone 30 dB and war
    60 dB -> 0.5; the same level -> 1). Drone rounds are grouped by that ratio (3 decimals) and give
    true and false confidence as in confidences(); a layer at 0 dB SPL or below has no ratio. A drone
    "as rendered" uses its measured level (rebuilt for runs made before it was stored), and so does a
    modified drone, whose set level is the level at 1 m, not what the microphone hears."""
    groups = {}
    measured = test.plan.kind == "modified"
    for rnd, res in test.results.items():
        r = test.by_round[rnd]
        lv = r["background"].get(layer)
        if r["drone"] is None or det not in res or lv is None or lv <= 0:
            continue
        d_db = None if measured else r["drone_level_db"]
        if d_db is None:
            d_db = res[det].get("drone_db")
            if d_db is None:
                d_db = E.snr(test.rebuild(r))["drone_db"]
                for x in res.values():
                    x["drone_db"] = d_db
        if d_db is None:
            continue
        add_conf(groups.setdefault(round(d_db / lv, 3), new_sums()), res[det])
    return [(x, final_conf(s)) for x, s in sorted(groups.items())]


def ratio_layer(test: E.Test, key: str) -> str | None:
    """The layer a setting's graph is drawn against on the dB-ratio axis: the layer itself, or for the
    drone level the first ticked layer; None for other settings or without layers."""
    if key in test.plan.layers:
        return key
    if key == "drone_level_db" and test.plan.layers:
        return next(iter(test.plan.layers))
    return None


class Conclusions(tk.Toplevel):
    """One graph per varied setting: a detector's mean confidence on drone windows (true) and on
    no-drone windows (false) for each value. Bars up to CURVE_ABOVE values, a curve above that for
    numbers. The dashed line is the detector's threshold (comb: 0.5, as drawn)."""

    def __init__(self, master, test: E.Test):
        super().__init__(master)
        self.test = test
        test.by_round = {r["round"]: r for r in test.rounds}
        self.params = parameter_values(test)
        self.title(f"{TITLE} conclusions: {test.out.name}")
        self.geometry("1300x900")
        bar = ttk.Frame(self, padding=(12, 8))
        bar.pack(fill="x")
        ttk.Label(bar, text="Conclusions", style="H2.TLabel").pack(side="left")
        ttk.Label(bar, text="detector").pack(side="left", padx=(20, 4))
        self.v_det = tk.StringVar(value=test.plan.detectors[0])
        ttk.Combobox(bar, textvariable=self.v_det, values=test.plan.detectors, state="readonly", width=12).pack(side="left")
        ttk.Button(bar, text="Save PNG", command=self.save).pack(side="right")
        self.l_text = ttk.Label(self, text="", style="Muted.TLabel", padding=(12, 0), justify="left")
        self.l_text.pack(fill="x")
        self.fig = Figure(figsize=(12, 8), dpi=100, facecolor="white")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        NavigationToolbar2Tk(self.canvas, self, pack_toolbar=True)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.v_det.trace_add("write", lambda *_: self.draw())
        self.draw()

    def draw(self):
        det = self.v_det.get()
        thr = 0.5 if det == "comb" else self.test.thr[det]
        self.fig.clear()
        if not self.params:
            self.fig.text(0.5, 0.5, "No setting takes more than one value in this test: nothing to compare.",
                          ha="center", va="center", color=MUTED)
            self.canvas.draw_idle()
            self.l_text.configure(text="")
            return
        n = len(self.params)
        cols = 1 if n == 1 else 2
        rows = (n + cols - 1) // cols
        axes = self.fig.subplots(rows, cols, squeeze=False).ravel()
        lines = []
        for ax, (key, (label, vals)) in zip(axes, self.params.items()):
            layer = ratio_layer(self.test, key)
            if layer:                                        # x = drone dB / layer dB, per round
                pts = level_ratio_points(self.test, det, layer)
                vals, c = [p[0] for p in pts], [p[1] for p in pts]
                title = f"{label} -> drone dB / {layer} dB"
                xlabel = f"drone level / {layer} level, both in dB SPL (1 = same level, 0.5 = drone at half the dB)"
            else:
                c = [confidences(self.test, det, key, v) for v in vals]
                title, xlabel = label, label
            t = [np.nan if x["true"] is None else 100 * x["true"] for x in c]
            f = [np.nan if x["false"] is None else 100 * x["false"] for x in c]
            numeric = all(isinstance(v, (int, float)) for v in vals)
            if not vals:
                ax.text(0.5, 0.5, f"no ratio: {layer} is at 0 dB SPL or below in every round", ha="center",
                        va="center", color=MUTED, transform=ax.transAxes)
            elif len(vals) > CURVE_ABOVE and numeric:
                ax.plot(vals, t, "-o", color="#2a78d6", lw=2, ms=4, label="true (drone windows)")
                ax.plot(vals, f, "-o", color="#d93025", lw=2, ms=4, label="false (no-drone windows)")
            else:
                x = np.arange(len(vals))
                bt = ax.bar(x - 0.2, t, 0.4, color="#2a78d6", label="true (drone windows)")
                bf = ax.bar(x + 0.2, f, 0.4, color="#d93025", label="false (no-drone windows)")
                for b in (*bt, *bf):
                    h = b.get_height()
                    if np.isfinite(h):
                        ax.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", fontsize=8)
                ticks = [(f"{v:.2f}" if layer else fmt(v)) if not isinstance(v, str)
                         else (v if len(v) <= 14 else v[:13] + "~") for v in vals]
                ax.set_xticks(x, ticks, rotation=20 if not numeric else 0, fontsize=8)
            if layer and vals and 1.0 >= min(vals) and len(vals) > CURVE_ABOVE:
                ax.axvline(1.0, color="#999", lw=0.8, ls=":")    # drone and layer at the same level
            ax.set_xlabel(xlabel, fontsize=8 if layer else 10)
            ax.axhline(100 * thr, color="#555", ls="--", lw=1, label=f"threshold {100 * thr:.0f}%")
            ax.set(ylim=(0, 105), ylabel="mean confidence (%)")
            ax.set_title(title, fontsize=10, loc="left", weight="bold")
            ax.grid(axis="y", color="#e8e8e8", lw=0.5)
            lines.append(f"{title}: " + ";  ".join(
                f"at {(f'{v:.2f}' if layer else fmt(v)) if not isinstance(v, str) else v} true {pct(x['true'])}, "
                f"false {pct(x['false'])}" for v, x in zip(vals, c)))
        for ax in axes[n:]:
            ax.set_visible(False)
        self.fig.suptitle(f"{det if det != 'comb' else 'DSP (comb)'}: mean confidence on drone windows (true) "
                          f"and no-drone windows (false), per setting", fontsize=11, x=0.01, ha="left")
        handles, labels = axes[0].get_legend_handles_labels()   # one legend for all graphs, out of the bars' way
        self.fig.legend(handles, labels, loc="upper right", ncol=3, fontsize=8, frameon=False)
        self.fig.tight_layout(rect=(0, 0, 1, 0.94))
        self.canvas.draw_idle()
        self.l_text.configure(text="\n".join(lines))

    def save(self):
        p = self.test.out / f"conclusions_{self.v_det.get().replace(' ', '_')}.png"
        self.fig.savefig(p, dpi=120)
        messagebox.showinfo(TITLE, f"Saved {p}", parent=self)


def pct(v) -> str:
    return "-" if v is None else f"{100 * v:.0f}%"


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
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    p = cfg["plan"]
    plan = E.Plan(**{k: p[k] for k in ("kind", "length_s", "drones", "drone_levels", "layers", "detectors", "step_s",
                                       "no_drone_rounds", "soundscape_start_s", "synth")},
                  **{k: p[k] for k in ("speeds", "approach") if k in p})        # modified runs only
    t = E.Test(plan, folder)
    t.start_s = cfg["soundscape_start_s"]
    t.prepare(progress=lambda s: None)
    t.results = E.read_results(folder)
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
