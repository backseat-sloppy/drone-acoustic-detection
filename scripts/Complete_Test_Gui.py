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
    Background layers    (open with the arrow) war soundscape, war-shaped noise, wind, wind and cars from
                         Pure Data, and helicopter, insects (listed, not set up yet). Tick a layer and give its levels in
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
selected round its log-mel (arrows above it switch between the view for us and what each network is
fed), every detector's confidence and the dB SPL of each layer over time, and plays the mix, the
drone alone or the background alone. "Open results..." reopens an earlier run.

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
from contextlib import nullcontext
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
from matplotlib.lines import Line2D  # noqa: E402

from dad.config import DEFAULT  # noqa: E402
from dad.features import LogMel, hz_to_mel, mel_to_hz  # noqa: E402
from dad.realrun import featurise  # noqa: E402

TITLE = "Complete Test"
PAGES = [("real", "Real drone test"), ("synthetic", "Synthetic drone test"), ("modified", "Modified drone test")]
SUBTITLES = {"real": "Real drone recordings mixed into chosen background layers at real levels.",
             "synthetic": "Pure Data drones flying at the microphone, mixed into chosen background layers.",
             "modified": "Real drone recordings flown at the microphone: the end of each file is the impact, and the "
                         "level follows the distance at each moment for the chosen speed."}
DEFAULT_LEVELS = {"war soundscape": ("60", "60", "1"), "war-shaped noise": ("50", "50", "1"), "wind": ("50", "50", "1"),
                  "wind (Pd)": ("50", "50", "1"), "cars": ("50", "50", "1"), "helicopter": ("", "", ""), "insects": ("", "", "")}
COLORS = {"it6 CNN": "#2a78d6", "it9 CNN": "#e8710a", "it9 CRNN": "#9334e6", "it10 CNN": "#d93025",
          "It9B": "#0b8a8a", "it16 CNN 600k s0": "#b8860b", "it16 CNN 600k s1": "#d4a72c",
          "it16 CNN 600k s2": "#8a6d0b", "it16 CRNN 600k s0": "#c2185b", "it16 CRNN 600k s1": "#e5609a",
          "it16 CRNN 600k s2": "#880e4f", "it19 CRNN colour s1": "#ff6f00", "it17 CNN 2s 600k s0": "#1b5e20", "it17 CNN 2s 600k s1": "#43a047",
          "it17 CNN 2s 600k s2": "#81c784", "it17 CRNN 2s 600k s0": "#311b92", "it17 CRNN 2s 600k s1": "#5e35b1",
          "it17 CRNN 2s 600k s2": "#9575cd", "comb": "#1baf7a"}
PD_CONTROLS = {"wind (Pd)": ("wind", (("speed_ms", "speed (m/s, assumed)", "quiet: 1  ·  storm: 25"),
                                      ("reso", "resonance (0-127)", "open field: 10  ·  city: 90"))),
               "cars": ("cars", (("speed_kmh", "speed (km/h)", "town: 50  ·  motorway: 130"),
                                 ("dist_m", "closest distance (m)", "near: 10  ·  far: 40")))}
WIND_LEVELS = "quiet: 25  ·  storm: 50"
LEVEL_HINTS = {"war soundscape": "Test 1: 90",
               "war-shaped noise": "Test 1: 90",
               "wind": WIND_LEVELS, "wind (Pd)": WIND_LEVELS,
               "cars": "near (10 m): 57  ·  far (40 m): 45"}
PD_LABELS = {"wind_speed_ms": "wind (Pd) speed (m/s, assumed)", "wind_reso": "wind (Pd) resonance (0-127)",
             "car_speed_kmh": "car speed (km/h)", "car_dist_m": "car closest distance (m)"}
PART_COLORS = {"drone": "#d93025", "war soundscape": "#6b4e16", "war-shaped noise": "#7a7a7a", "wind": "#2a9d8f",
               "wind (Pd)": "#2a9d8f", "cars": "#c2570c", "mix": "#111111"}
BG, PANEL, ACCENT, MUTED = "#f4f5f7", "#ffffff", "#2a78d6", "#6b7280"
LOGMEL = LogMel(DEFAULT)
BAND_HZ = E.np.asarray(mel_to_hz(np.linspace(hz_to_mel(DEFAULT.fmin), hz_to_mel(DEFAULT.fmax), DEFAULT.n_mels + 2))[1:-1])
# The log-mel of a full-scale 1 kHz sine, dB: subtracted so the colour bar reads dBFS with that sine at 0.
FS_DB = float(10 / np.log(10) * LOGMEL.frames(np.sin(2 * np.pi * 1000 * np.arange(4 * DEFAULT.n_fft) / E.SR)).max())


LAYER_GUIDELINES = (
    "About column: wind speed in m/s is an assumed scale, not measured (10 m/s is the sound's default), so do not report "
    "results per m/s. Resonance for city and open field is a guess. The wind levels rest on one study, the storm "
    "level extrapolated. Car levels come from the EU pass-by limits (68-72 dB(A) at 7.5 m) and are an upper limit.",
    "Confidence follows the dB difference between drone and background, not the absolute level. Keep levels between "
    "about 60 and 100 dB SPL: below that the frontend's floor changes the scores, above that a real microphone would "
    "clip and this test does not.",
    "wind (Pd) and cars: their settings change the character of the sound, not its loudness. The level you set is "
    "still the average over the whole render, whatever the speed or distance. Each new combination is rendered once, "
    "at the start of the test (about a minute per wind combination, seconds for cars), then kept.",
    "cars: at 50 km/h and 10 m a pass-by peaks about 11 dB above the level you set (1 s windows), and the typical "
    "level is about 11 dB below it.",
    "cars: the engine recordings are from FSD50K dev, which training draws from. Low confidence on cars may partly be "
    "memory of the engines: say so when reporting results.")
MODEL_LIM = (-3, 3)                                         # colour scale of the network's input, z-scores
SPEC_NOTES = {"ui": "the whole round on one scale: 80 dB down from its loudest point",
              "model": "each 125 ms from its own 1 s window: that window's level removed, scaled by the training statistics"}


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


def speed_steps(lo: str, hi: str, n: str) -> list[float]:
    """The modified page's speeds (km/h) from its min, max and # boxes, by the same rule as layer_steps."""
    try:
        lo_, hi_ = float(lo), float(hi)
    except ValueError:
        raise ValueError("speed: min and max must be numbers of km/h") from None
    try:
        n_ = int(n)
    except ValueError:
        raise ValueError("speed: # must be a whole number") from None
    if not 1 <= n_ <= 50:
        raise ValueError("speed: # must be 1 to 50")
    for v in (lo_, hi_):
        if not 0 < v <= 1000:
            raise ValueError(f"speed: {v:g} km/h is outside 0 to 1000")
    try:
        return E.dwg.steps(lo_, hi_, n_, "speed")
    except ValueError as e:
        raise ValueError(str(e).replace("the count", "#").replace("count 1", "# 1")) from None


def pd_steps(name: str, lo: str, hi: str, n: str) -> list[float]:
    """A Pure Data setting's values from its min, max and # boxes, by the same rule as layer_steps;
    Plan.check holds the setting's range."""
    try:
        lo_, hi_ = float(lo), float(hi)
    except ValueError:
        raise ValueError(f"{name}: min and max must be numbers") from None
    try:
        n_ = int(n)
    except ValueError:
        raise ValueError(f"{name}: # must be a whole number") from None
    if not 1 <= n_ <= 50:
        raise ValueError(f"{name}: # must be 1 to 50")
    try:
        return E.dwg.steps(lo_, hi_, n_, name)
    except ValueError as e:
        raise ValueError(str(e).replace("the count", "#").replace("count 1", "# 1")) from None


def detector_names() -> list[str]:
    return [*E.dwg.MODELS, "comb"]


def model_view(x: np.ndarray, sc, batch: int = 256) -> np.ndarray:
    """What network scorer `sc` is fed for the mix x, as one image over time (bands, frames): z-scores,
    exactly NetScorer.score's input. The network sees its own window (1 s or 2 s); a hop is 12.5 frames,
    so the first window is shown whole and each later one adds its newest 12 or 13 frames, normalised by
    that window."""
    n, hop = sc.cfg.samples_per_window, sc.cfg.samples_per_hop
    starts = np.arange(0, len(x) - n + 1, hop)
    per_hop = hop / sc.cfg.hop_length
    parts = []
    for i in range(0, len(starts), batch):
        a = np.stack([x[s:s + n] for s in starts[i:i + batch]]).astype(np.float32)
        for k, z in enumerate((featurise(a, sc.variant, sc.fe) - sc.mu) / sc.sigma, i):
            keep = z.shape[1] if k == 0 else int(round((k + 1) * per_hop) - round(k * per_hop))
            parts.append(z[:, -keep:])
    return np.concatenate(parts, axis=1)


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
                self.v_speeds = (tk.StringVar(value="50"), tk.StringVar(value="150"), tk.StringVar(value="3"))
                self.v_closest = tk.StringVar(value=f"{E.APPROACH['closest_m']:g}")
                self.v_doppler = tk.BooleanVar(value=E.APPROACH["doppler"])
                ttk.Label(f, text="Speeds (km/h):").grid(row=1, column=0, sticky="w", pady=(8, 0))
                sp = ttk.Frame(f)
                sp.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
                for lbl, v, w in zip(("min", "max", "#"), self.v_speeds, (7, 7, 4)):
                    ttk.Label(sp, text=lbl, style="Muted.TLabel").pack(side="left", padx=(0 if lbl == "min" else 8, 3))
                    ttk.Entry(sp, textvariable=v, width=w).pack(side="left")
                self.l_speed_preview = ttk.Label(sp, text="", style="Muted.TLabel", width=26)
                self.l_speed_preview.pack(side="left", padx=(10, 0))
                sub = ttk.Frame(f)
                sub.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
                ttk.Label(sub, text="distance at impact (m)", style="Muted.TLabel").pack(side="left", padx=(0, 4))
                ttk.Entry(sub, textvariable=self.v_closest, width=6).pack(side="left")
                ttk.Checkbutton(sub, text="Doppler (pitch rises while it approaches)", variable=self.v_doppler,
                                command=self.update_count).pack(side="left", padx=(12, 0))
                self.l_approach = ttk.Label(f, text="", style="Muted.TLabel")
                self.l_approach.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(2, 0))
                for v in (*self.v_speeds, self.v_closest):
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
        g.columnconfigure(7, weight=1)
        for c, h in enumerate(("", "layer", "min (dB SPL)", "max (dB SPL)", "#", "also without", "", "about")):
            ttk.Label(g, text=h, style="Head.TLabel").grid(row=0, column=c, sticky="w", padx=4)
        self.v_layer_on, self.v_layer_lv, self.v_layer_off, self.v_pd, self.pd_preview = {}, {}, {}, {}, {}
        r = 0
        for name, L in self.layers.items():
            r += 1
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
            off = tk.BooleanVar(value=False)
            self.v_layer_off[name] = off
            cb_off = ttk.Checkbutton(g, variable=off, command=self.update_count)
            cb_off.grid(row=r, column=5, padx=4)
            preview = ttk.Label(g, text="", style="Muted.TLabel", width=22)
            preview.grid(row=r, column=6, sticky="w", padx=4)
            ttk.Label(g, text=LEVEL_HINTS.get(name, L.note) if L.available else L.note,   # what to enter; if unavailable, why
                      style="Muted.TLabel" if not L.available else "TLabel").grid(row=r, column=7, sticky="w", padx=4)
            if not L.available:
                cb.state(["disabled"])
                cb_off.state(["disabled"])
                for e in entries:
                    e.state(["disabled"])
            for v in vs:
                v.trace_add("write", lambda *_: self.update_count())
            self.layer_preview = getattr(self, "layer_preview", {})
            self.layer_preview[name] = preview
            if name in PD_CONTROLS and L.available:
                r = self._pd_rows(g, r, name)
        r += 1
        ttk.Label(g, text="# values from min to max, both included, evenly spaced; # 1 needs min = max "
                          "(as in drone_war_gui)", style="Muted.TLabel").grid(row=r, column=1, columnspan=7, sticky="w",
                                                                             padx=4, pady=(4, 0))
        r += 1
        st = ttk.Frame(g)
        st.grid(row=r, column=1, columnspan=7, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(st, text="soundscape start in the war track (s):").pack(side="left")
        self.v_sc_start = tk.StringVar(value="")
        ttk.Entry(st, textvariable=self.v_sc_start, width=8).pack(side="left", padx=6)
        ttk.Label(st, text="blank = drawn once per batch (seed 1000); every round uses the same stretch",
                  style="Muted.TLabel").pack(side="left")
        r += 1
        guide = ttk.Frame(g)
        guide.grid(row=r, column=1, columnspan=7, sticky="w", padx=4, pady=(8, 0))
        ttk.Label(guide, text="Guidelines", style="Bold.TLabel").pack(anchor="w")
        for text in LAYER_GUIDELINES:
            ttk.Label(guide, text=f"•  {text}", style="Muted.TLabel", wraplength=980, justify="left").pack(anchor="w")

    def _pd_rows(self, g, r: int, name: str) -> int:
        """A row per setting of a Pure Data layer, under the layer, with the same min, max and # boxes
        as a level. Returns the last row used."""
        sec, settings = PD_CONTROLS[name]
        for key, label, hint in settings:
            r += 1
            start = f"{E.PD_SETTINGS[sec][key]:g}"
            vs = (tk.StringVar(value=start), tk.StringVar(value=start), tk.StringVar(value="1"))
            self.v_pd.setdefault(sec, {})[key] = vs
            ttk.Label(g, text=f"    {label}", style="Muted.TLabel").grid(row=r, column=1, sticky="w", padx=4)
            for i, v in enumerate(vs):
                ttk.Entry(g, textvariable=v, width=7 if i < 2 else 4).grid(row=r, column=2 + i, sticky="w", padx=4)
                v.trace_add("write", lambda *_: self.update_count())
            preview = ttk.Label(g, text="", style="Muted.TLabel", width=22)
            preview.grid(row=r, column=6, sticky="w", padx=4)
            self.pd_preview[name, sec, key] = preview
            ttk.Label(g, text=hint).grid(row=r, column=7, sticky="w", padx=4)
        return r

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
            v = tk.BooleanVar(value=d in ("it9 CRNN", "comb"))
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
                speeds = speed_steps(*(v.get() for v in self.v_speeds))
                try:
                    approach = dict(closest_m=float(self.v_closest.get()), doppler=bool(self.v_doppler.get()))
                except ValueError:
                    raise ValueError("distance at impact must be a number") from None
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
        pd_settings = {}
        for name, (sec, _) in PD_CONTROLS.items():
            if name in layers:                               # an unticked layer's boxes do not count
                pd_settings[sec] = {k: pd_steps(f"{name} {k}", *(v.get() for v in vs)) for k, vs in self.v_pd[sec].items()}
        p = E.Plan(self.kind, length, drones, levels, layers, [d for d, v in self.v_det.items() if v.get()],
                   step_s=step, no_drone_rounds=self.v_nodrone.get(), pd_settings={**E.PD_SETTINGS, **pd_settings},
                   also_without=[n for n in layers if self.v_layer_off[n].get()],
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
        if self.kind == "modified":                          # the speeds, or what is wrong
            try:
                txt = ", ".join(f"{v:g}" for v in speed_steps(*(v.get() for v in self.v_speeds)))
                self.l_speed_preview.configure(text=(txt if len(txt) <= 22 else txt[:20] + "…") + " km/h", style="Muted.TLabel")
            except ValueError as e:
                self.l_speed_preview.configure(text=str(e).split(": ", 1)[-1][:26], style="Warn.TLabel")
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
        for (name, sec, key), lbl in self.pd_preview.items():   # the same for the Pure Data settings
            if not self.v_layer_on[name].get():
                lbl.configure(text="", style="Muted.TLabel")
                continue
            try:
                txt = ", ".join(f"{v:g}" for v in pd_steps(key, *(v.get() for v in self.v_pd[sec][key])))
                lbl.configure(text=txt if len(txt) <= 24 else txt[:22] + "…", style="Muted.TLabel")
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
        bits = [f"{n} {', '.join(f'{v:g}' for v in lv)} dB" + (" + off" if n in p.also_without else "") for n, lv in p.layers.items()]
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
        self.cols = ["round", "drone", "level", "start", *self.test.plan.layers, *self.test.plan.pd_swept(), "detector", "recall", "FPR",
                     "delay s", "false alarm", "conf drone", "conf none"]
        box = ttk.Frame(left)
        box.pack(fill="both", expand=True, pady=(2, 0))
        self.t_rounds = ttk.Treeview(box, columns=self.cols, show="headings", selectmode="browse")
        for col in self.cols:
            self.t_rounds.heading(col, text=col, command=lambda c=col: self._sort(c))
            self.t_rounds.column(col, width=110 if col in ("drone", "detector") else 90 if col in PD_LABELS else 54,
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
        nav = ttk.Frame(right)                               # the log-mel's slide show: for us, or per network
        nav.pack(fill="x", pady=(4, 0))
        self.spec_slides = ["ui", *(d for d in self.test.plan.detectors if d != "comb"
                                    and self.test.scorers[d].cfg.kind == "mfe")]
        self.spec_i = 0
        state = "normal" if len(self.spec_slides) > 1 else "disabled"
        ttk.Button(nav, text="◀", width=3, state=state, command=lambda: self._spec_go(-1)).pack(side="left")
        self.l_spec = ttk.Label(nav, text="Log-mel for us", style="Bold.TLabel", width=22, anchor="center")
        self.l_spec.pack(side="left", padx=4)
        ttk.Button(nav, text="▶", width=3, state=state, command=lambda: self._spec_go(1)).pack(side="left")
        self.l_spec_note = ttk.Label(nav, text="", style="Muted.TLabel")
        self.l_spec_note.pack(side="left", padx=10)
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
                *(f"{r['background'][n]:g}" if n in r["background"] else "off" for n in self.test.plan.layers),
                *(f"{r['pd'][k]:g}" if k in r["pd"] else "" for k in self.test.plan.pd_swept()), det, fmt_pct(res["recall"]),
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
        grid = self.fig.subplots(3, 2, sharex="col", gridspec_kw=dict(height_ratios=[1.1, 1, 1], width_ratios=[60, 1]))
        ax1, ax2, ax3 = grid[:, 0]                           # the bar has its own column: the time axes stay aligned
        grid[1, 1].remove()
        grid[2, 1].remove()
        self.axes = [ax1, ax2, ax3]
        self._views = {}                                     # this round's images, made when first shown
        self._im = ax1.imshow(np.zeros((2, 2)), origin="lower", aspect="auto", cmap="magma", interpolation="nearest")
        self._bar = self.fig.colorbar(self._im, cax=grid[0, 1])
        self._bar.ax.tick_params(labelsize=7)
        self._paint_spec()
        ticks = [0, 16, 32, 48, 63]
        ax1.set_yticks([t + 0.5 for t in ticks], [f"{BAND_HZ[t]:.0f}" for t in ticks])
        ax1.set_ylabel("log-mel band (Hz)")
        title = (f"round {rnd}: " + (E.drone_label(r["drone"]) if r["drone"] is not None else "no drone")
                 + (f" at {r['drone_level_db']:g} dB SPL" + (" at 1 m" if self.test.plan.kind == "modified" else "")
                    if r["drone_level_db"] is not None else
                    " as rendered" if r["drone"] is not None else "")
                 + (f", starts {r['start_s']:g} s" if r["start_s"] is not None else "")
                 + "".join(f" · {k} {v:g} dB" for k, v in r["background"].items())
                 + "".join(f" · {PD_LABELS[k]} {r['pd'][k]:g}" for k in self.test.plan.pd_swept() if k in r["pd"])
                 + "".join(f" · {n} off" for n in r["off"]))
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
        ax2.axhline(0.5, color="#888", ls="--", lw=0.9)          # a fixed 50 % mark, not a threshold (Mads)
        ax2.set(ylim=(0, 1.02), ylabel="confidence")
        ax2.legend(loc="upper left", fontsize=8, frameon=False, ncol=5)
        ax2.set_title("each detector's confidence (dashed = 50 %; red band = drone sound; "
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

    def _spec_view(self, key: str) -> dict:
        """The image of slide `key` for the shown round, with its colour scale."""
        if key not in self._views:
            mix = self.cur[1]["mix"]
            if key == "ui":                                  # the round's own 80 dB: the numbers move with the level
                z = 10 / np.log(10) * LOGMEL.frames(mix.astype(np.float32)) - FS_DB
                top = float(z.max())
                v = dict(z=z, lim=(top - 80, top), ticks=np.arange(np.floor(top / 10) * 10, top - 80, -10), label="dBFS",
                         end=len(mix) / E.SR)
            else:                                            # a fixed scale: comparable between rounds
                n, hop = self.test.scorers[key].cfg.samples_per_window, DEFAULT.samples_per_hop
                v = dict(z=model_view(mix, self.test.scorers[key]), lim=MODEL_LIM, ticks=np.arange(MODEL_LIM[0], MODEL_LIM[1] + 1),
                         label="model input (z-score)", end=(n + (len(mix) - n) // hop * hop) / E.SR)
            self._views[key] = v
        return self._views[key]

    def _paint_spec(self):
        key = self.spec_slides[self.spec_i]
        v = self._spec_view(key)
        self._im.set_data(v["z"])
        self._im.set_extent((0, v["end"], 0, v["z"].shape[0]))
        self._im.set_clim(*v["lim"])
        self._bar.set_ticks(v["ticks"])
        self._bar.set_label(v["label"], fontsize=8)
        self.l_spec.configure(text="Log-mel for us" if key == "ui" else f"What {key} is fed")
        self.l_spec_note.configure(text=SPEC_NOTES["ui" if key == "ui" else "model"])

    def _spec_go(self, step: int):
        self.spec_i = (self.spec_i + step) % len(self.spec_slides)
        if self.cur:
            self._paint_spec()
            self.fig.tight_layout()
            self.canvas.draw_idle()
        else:
            self.l_spec.configure(text="Log-mel for us" if self.spec_i == 0 else f"What {self.spec_slides[self.spec_i]} is fed")
            self.l_spec_note.configure(text=SPEC_NOTES["ui" if self.spec_i == 0 else "model"])

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
SUMMARY_LAYERS = (("War soundscape summary", ("war soundscape",)), ("Wind summary", ("wind", "wind (Pd)")),
                  ("Car summary", ("cars",)))
TOP_IN, HEAD_IN, ROW_IN, NOTE_IN = 0.7, 0.55, 3.1, 0.4       # heights on the Conclusions page, inches


def parameter_values(test: E.Test) -> dict[str, list]:
    """Every setting that takes more than one value in the test, with its values in order."""
    modified = test.plan.kind == "modified"
    names = {**({"speed_kmh": "speed (km/h), all drones", "file": "drone file, all speeds"} if modified  # not per file x speed
                else {"drone": "drone"}),
             "drone_level_db": "drone level at 1 m (dB SPL)" if modified else "drone level (dB SPL)",
             "start_s": "drone start (s)", **{n: f"{n} (dB SPL)" for n in test.plan.layers}}
    out = {}
    for key, label in names.items():
        vals = values_of(test, key)
        if len(vals) > 1:
            out[key] = (label, vals)
    return out


def values_of(test: E.Test, key: str) -> list:
    """The values `key` takes over the rounds, in order."""
    vals = {setting(r, key) for r in test.rounds if setting(r, key) is not None}
    return sorted(vals, key=lambda v: (not isinstance(v, str), v))   # a layer's "off" comes first


def setting(r: dict, key: str):
    if key in E.PD_ROUND_KEYS:                               # no-drone rounds have these too; None when the layer is off
        return r["pd"].get(key)
    if key in r["off"]:
        return "off"
    if key in r["background"]:
        return r["background"][key]
    if r["drone"] is None:                                   # a no-drone round has no drone setting
        return None
    if key == "speed_kmh" and not isinstance(r["drone"], str):   # a synthetic drone is its speed
        return float(r["drone"])
    return E.drone_setting(r["drone"], key) if key in ("drone", "file", "speed_kmh") else r[key]


def summary_sections(test: E.Test) -> list[dict]:
    """What the Conclusions page shows, in order: Overall (drone dB / layer dB, one graph per layer); the
    four summaries, always there, each graph on the setting's own values even when it has only one;
    then every other setting that varies, on its own values too.
    -> dict(title, charts=[(key, label, values, layer of a ratio graph or None)], note=why no graph)."""
    layers, varied, used = test.plan.layers, parameter_values(test), set()

    def section(title, keys, note):
        charts = []
        for key, label in keys:
            used.add(key)
            charts.append((key, label, values_of(test, key), None))
        return dict(title=title, charts=charts, note="" if charts else note)

    speed = [("speed_kmh", "speed (km/h), all drones")] if test.plan.kind != "real" else []
    out = [dict(title="Overall", charts=[(n, "", [], n) for n in layers], note="No background layer in this test, so no ratio."),
           section("Speed summary", speed, "A real drone test has no set speed: the recordings are played as they are.")]
    if test.plan.kind == "synthetic":                        # the same thing as "drone" there
        used.add("drone")
    for title, names in SUMMARY_LAYERS:
        keys = [(n, f"{n} level (dB SPL)") for n in names if n in layers]
        keys += [(k, PD_LABELS[k]) for k in test.plan.pd_combos()[0] if E.PD_ROUND_KEYS[k][0] in names]   # also at one value
        out.append(section(title, keys, f"Not in this test: tick {' or '.join(names)} under Background layers."))
    other = [(k, lab, v, None) for k, (lab, v) in varied.items() if k not in used]
    if other:
        out.append(dict(title="Other settings", charts=other, note=""))
    return out


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


def confidence_over_time(test: E.Test, det: str, scores: dict) -> dict[float, dict]:
    """Modified drone tests: per speed, `det`'s confidence at each moment of the drone file, over every
    drone round at that speed (all files, levels, starts and backgrounds). Time is time in the drone
    file with the impact at 10 s (E.DRONE_S); a 5-10 s file is aligned on its impact, so x = 7 is 3 s
    before impact for every file. A point is the decision at the end of its window (1 s, or 2 s for
    the 2 s models: the window covers the seconds before it); hops are binned to 0.125 s. comb is squashed as drawn elsewhere.
    -> speed -> dict(x, mean, p25, p75, n) with numpy arrays."""
    if test.plan.kind != "modified" or test.drones is None:
        return {}
    hop, win = DEFAULT.samples_per_hop / E.SR, DEFAULT.samples_per_window / E.SR
    thr = test.thr[det]
    bins: dict[float, dict[int, list]] = {}
    for rnd in test.results:
        r = test.by_round[rnd]
        s = scores.get(f"{rnd}|{det}")
        if r["drone"] is None or s is None:
            continue
        speed = E.split_key(r["drone"])[1]
        impact = r["start_s"] + test.drones.span[r["drone"]][2]
        t_end = np.arange(len(s)) * hop + win
        x = E.DRONE_S - (impact - t_end)
        y = E.dwg.squash(s, thr) if det == "comb" else s
        b = bins.setdefault(speed, {})
        for k, v in zip(np.round(x / hop).astype(int), y):
            if np.isnan(v):                                  # a 2 s model's hops before its first window
                continue
            b.setdefault(int(k), []).append(float(v))
    out = {}
    for speed, b in sorted(bins.items()):
        ks = sorted(k for k in b if -2 / hop <= k <= (E.DRONE_S + 1) / hop)
        vals = [np.asarray(b[k]) for k in ks]
        out[speed] = dict(x=np.array(ks) * hop, mean=np.array([v.mean() for v in vals]),
                          p25=np.array([np.percentile(v, 25) for v in vals]),
                          p75=np.array([np.percentile(v, 75) for v in vals]), n=np.array([len(v) for v in vals]))
    return out


class Conclusions(tk.Toplevel):
    """A scrollable page: Overall (the drone-to-layer dB ratio), summaries of speed, war soundscape, wind
    and cars, then any other setting that varies (summary_sections). Each graph is a detector's mean confidence on drone windows (true) and on
    no-drone windows (false) for each value of the setting, every round at that value pooled. Bars up to
    CURVE_ABOVE values, a curve above that for numbers. No threshold lines; a fixed dashed 50 % mark on
    every graph instead (Mads, 2026-11-06). Modified drone tests also get, in the speed summary, one
    "mean confidence over time" graph per speed: all rounds at that speed together, the mean confidence
    at each moment of the drone file (impact at 10 s), see confidence_over_time()."""

    def __init__(self, master, test: E.Test):
        super().__init__(master)
        self.test = test
        test.by_round = {r["round"]: r for r in test.rounds}
        self.title(f"{TITLE} conclusions: {test.out.name}")
        self.geometry("1300x900")
        bar = ttk.Frame(self, padding=(12, 8))
        bar.pack(fill="x")
        ttk.Label(bar, text="Conclusions", style="H2.TLabel").pack(side="left")
        ttk.Label(bar, text="detector").pack(side="left", padx=(20, 4))
        self.v_det = tk.StringVar(value=test.plan.detectors[0])
        ttk.Combobox(bar, textvariable=self.v_det, values=test.plan.detectors, state="readonly", width=12).pack(side="left")
        self.scores = {}
        if test.plan.kind == "modified":                     # per-hop scores for confidence over time
            p = test.out / "scores.npz"
            with np.load(p) if p.exists() else nullcontext({}) as d:
                self.scores = {k: d[k] for k in getattr(d, "files", [])}
        ttk.Button(bar, text="Save PNG", command=self.save).pack(side="right")
        holder = ttk.Frame(self)
        self.scroll = tk.Canvas(holder, bg="white", highlightthickness=0)
        sb = ttk.Scrollbar(holder, orient="vertical", command=self.scroll.yview)
        self.scroll.configure(yscrollcommand=sb.set)
        inner = tk.Frame(self.scroll, bg="white")
        win = self.scroll.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: self.scroll.configure(scrollregion=self.scroll.bbox("all")))
        self.scroll.bind("<Configure>", lambda e: self.scroll.itemconfigure(win, width=e.width))
        self.fig = Figure(figsize=(12, 8), dpi=100, facecolor="white")
        self.canvas = FigureCanvasTkAgg(self.fig, master=inner)
        self.canvas.get_tk_widget().pack(fill="x")
        self.l_text = tk.Label(inner, text="", bg="white", fg=MUTED, justify="left", anchor="w", padx=12, pady=8, wraplength=1240)
        self.l_text.pack(fill="x")
        NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False).pack(side="bottom", fill="x")
        holder.pack(fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.scroll.pack(side="left", fill="both", expand=True)
        self.bind("<MouseWheel>", self._wheel)               # replaces matplotlib's, and stops the App's scrolling too
        self.v_det.trace_add("write", lambda *_: self.draw())
        self.draw()

    def _wheel(self, e):
        self.scroll.yview_scroll(int(-e.delta / 120), "units")
        return "break"

    def draw(self):
        """Lays the page out in inches from the top: a heading per section, then its graphs two to a
        row (a lone graph takes the whole width). Modified runs add one over-time graph per speed."""
        det = self.v_det.get()
        thr = 0.5 if det == "comb" else self.test.thr[det]
        curves = confidence_over_time(self.test, det, self.scores)
        secs = summary_sections(self.test)
        for s in secs:
            s["curves"] = curves if s["title"] == "Speed summary" else {}
            s["cells"] = len(s["charts"]) + len(s["curves"])
            s["rows"] = (s["cells"] + 1) // 2
        height = TOP_IN + sum(HEAD_IN + (s["rows"] * ROW_IN if s["cells"] else NOTE_IN) for s in secs)
        self.fig.clear()
        self.fig.set_size_inches(self.fig.get_figwidth(), height, forward=False)
        self.canvas.get_tk_widget().configure(height=int(height * self.fig.dpi))
        name = det if det != "comb" else "DSP (comb)"
        self.fig.text(0.012, 1 - 0.28 / height, f"{name}: mean confidence on drone windows (true) and no-drone windows (false)",
                      fontsize=12, weight="bold", va="center")
        self.fig.text(0.012, 1 - 0.54 / height, "Each bar or point pools every round at that value, over all the other settings.",
                      fontsize=9, color=MUTED, va="center")
        lines, y = [], TOP_IN
        for s in secs:
            top = 1 - y / height
            self.fig.add_artist(Line2D([0.012, 0.988], [top, top], color="#c9ced6", lw=1, transform=self.fig.transFigure))
            self.fig.text(0.012, 1 - (y + 0.3) / height, s["title"], fontsize=13, weight="bold", va="center")
            y += HEAD_IN
            if not s["cells"]:
                self.fig.text(0.012, 1 - (y + 0.14) / height, s["note"], fontsize=10, color=MUTED, va="center")
                y += NOTE_IN
                continue
            axes = [self._axes(i, s["cells"], y, height) for i in range(s["cells"])]
            for ax, chart in zip(axes, s["charts"]):
                lines.append(self._chart(ax, det, *chart))
            if s["charts"]:
                handles, labels = axes[0].get_legend_handles_labels()
                self.fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.988, 1 - (y - HEAD_IN + 0.3) / height),
                                ncol=3, fontsize=8, frameon=False)
            if s["curves"]:
                lines += self._over_time(axes[len(s["charts"]):], s["curves"], det, thr)
            y += s["rows"] * ROW_IN
        self.canvas.draw_idle()
        self.l_text.configure(text="\n".join(lines))
        self.scroll.yview_moveto(0)

    def _axes(self, i: int, cells: int, y: float, height: float):
        """Cell i of a section whose first row starts y inches down the page."""
        h = ROW_IN - 1.05                                    # the rest holds its title and x label
        bottom = 1 - (y + (i // 2) * ROW_IN + 0.4 + h) / height
        if cells == 1:
            return self.fig.add_axes((0.06, bottom, 0.92, h / height))
        return self.fig.add_axes((0.06 if i % 2 == 0 else 0.56, bottom, 0.42, h / height))

    def _chart(self, ax, det: str, key: str, label: str, vals: list, layer: str | None) -> str:
        """One setting's graph, or with `layer` its drone-to-layer ratio graph; returns its line of numbers."""
        if layer:                                            # x = drone dB / layer dB, per round
            pts = level_ratio_points(self.test, det, layer)
            vals, c = [p[0] for p in pts], [p[1] for p in pts]
            title = f"drone dB / {layer} dB"
            xlabel = f"drone level / {layer} level, both in dB SPL (1 = same level, 0.5 = drone at half the dB)"
        else:
            c = [confidences(self.test, det, key, v) for v in vals]
            title, xlabel = label, label
        t = [np.nan if x["true"] is None else 100 * x["true"] for x in c]
        f = [np.nan if x["false"] is None else 100 * x["false"] for x in c]
        nums = [(i, v) for i, v in enumerate(vals) if v != "off"]   # a layer's "off" is not on the number line
        off = vals.index("off") if "off" in vals else None
        numeric = all(isinstance(v, (int, float)) for _, v in nums)
        if not vals:
            ax.text(0.5, 0.5, f"no ratio: {layer} is at 0 dB SPL or below in every round", ha="center",
                    va="center", color=MUTED, transform=ax.transAxes)
        elif len(nums) > CURVE_ABOVE and numeric:
            xs = [v for _, v in nums]
            ax.plot(xs, [t[i] for i, _ in nums], "-o", color="#2a78d6", lw=2, ms=4, label="true (drone windows)")
            ax.plot(xs, [f[i] for i, _ in nums], "-o", color="#d93025", lw=2, ms=4, label="false (no-drone windows)")
            if off is not None:                              # dotted: the same two without this layer
                for y, col in ((t[off], "#2a78d6"), (f[off], "#d93025")):
                    if np.isfinite(y):
                        ax.axhline(y, color=col, ls=":", lw=1.2)
                        ax.text(xs[-1], y + 1.5, "without this layer", color=col, fontsize=8, ha="right")
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
            span = max(len(vals), CURVE_ABOVE)               # few bars keep their width, centred
            ax.set_xlim((len(vals) - 1 - span) / 2, (len(vals) - 1 + span) / 2)
        if layer and vals and 1.0 >= min(vals) and len(vals) > CURVE_ABOVE:
            ax.axvline(1.0, color="#999", lw=0.8, ls=":")    # drone and layer at the same level
        ax.set_xlabel(xlabel, fontsize=8 if layer else 10)
        ax.axhline(50, color="#888", ls="--", lw=1, label="50%")
        ax.set(ylim=(0, 105), ylabel="mean confidence (%)")
        ax.set_title(title, fontsize=10, loc="left", weight="bold")
        ax.grid(axis="y", color="#e8e8e8", lw=0.5)
        ax.set_axisbelow(True)
        if len(vals) == 1:                                   # every such graph averages the same rounds
            ax.set_title("one value, compares nothing", loc="right", fontsize=8, color=MUTED)
        return f"{title}: " + ";  ".join(
            f"at {(f'{v:.2f}' if layer else fmt(v)) if not isinstance(v, str) else v} true {pct(x['true'])}, "
            f"false {pct(x['false'])}" for v, x in zip(vals, c))

    def _over_time(self, axes, curves: dict, det: str, thr: float) -> list[str]:
        """One graph per speed: "all the rounds at this speed together: the mean confidence during the
        drone clip". Mean (line), middle half of the rounds (band), time in the drone file with the
        impact at 10 s; the same axes on every speed, so speeds compare. Returns the text lines."""
        color = COLORS.get(det, "#333")
        lines = []
        for k, (ax, (speed, c)) in enumerate(zip(axes, curves.items())):
            ax.axvspan(0, E.DRONE_S, color="#d93025", alpha=0.05, lw=0)
            ax.axvline(E.DRONE_S, color="#d93025", lw=1.2)
            ax.text(E.DRONE_S, 102, " impact", color="#d93025", fontsize=8, va="bottom")
            ax.fill_between(c["x"], 100 * c["p25"], 100 * c["p75"], color=color, alpha=0.18, lw=0,
                            label="middle half of the rounds")
            ax.plot(c["x"], 100 * c["mean"], color=color, lw=2, label="mean confidence")
            start_m = float(np.hypot(self.test.plan.approach.get("closest_m", 1.0), speed / 3.6 * E.DRONE_S))
            ax.set_title(f"confidence over time · {speed:g} km/h, all {int(c['n'].max())} rounds "
                         f"(starts {start_m:.0f} m away)", fontsize=10, loc="left", weight="bold")
            ax.axhline(50, color="#888", ls="--", lw=1, label="50%")
            ax.set(ylim=(0, 108), xlim=(-2, E.DRONE_S + 1), ylabel="mean confidence (%)")
            ax.set_xticks(range(-2, int(E.DRONE_S) + 2))
            ax.set_xlabel("time in the drone file (s); impact at 10 s (a shorter file aligned on its impact)", fontsize=8)
            ax.grid(color="#e8e8e8", lw=0.5)
            if k == 0:
                ax.legend(loc="upper left", fontsize=7, frameon=False)
            at = []
            for before in (1, 2, 3, 5, 8):
                i = np.flatnonzero(np.isclose(c["x"], E.DRONE_S - before))
                if i.size:
                    at.append(f"{before} s before impact: {100 * c['mean'][i[0]]:.0f}%")
            lines.append(f"{speed:g} km/h, mean confidence over time ({int(c['n'].max())} rounds): " + ",  ".join(at))
        return lines

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
                  **{k: p[k] for k in ("speeds", "approach", "pd_settings", "also_without") if k in p})   # not in every run's config
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
