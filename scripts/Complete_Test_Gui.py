"""Complete test: full control over large batches of drone-in-noise test rounds, no graphs on the front.

    uv run python scripts/Complete_Test_Gui.py

Three pages, a slide show (the arrows at the top, the page names, or Ctrl+Left / Ctrl+Right):

    Real drone test       real drone recordings (WAV files you add), each cut to its first 10 s
                          (a file of 5-10 s is used whole)
    Modified drone test   real drone recordings treated as a steady drone heard at 1 m and flown
                          straight at the microphone: the file's last 10 s (or all of a 5-10 s file),
                          its last sample the impact. Per speed (km/h) the drone is v/3.6 x the time
                          left before impact away; the level falls with that distance (spherical
                          spreading and air absorption, optional Doppler). Each file at each speed.
    Live demo test        the modified drone test in real time, out loud (scripts/live_test_engine.py):
                          one soundscape that loops from Play to the end. Source: virtual (scored on
                          the mix) or microphone (played out of the speaker, scored on what the
                          microphone hears). Drones: automatic (the plan's rounds one after the
                          other) or manual (Send drone flies in the next drone of the plan). A live
                          view (spectrogram, every detector's confidence, level) runs while it
                          plays; the results are a run folder like the other pages'.

The Synthetic drone test page (Pure Data drones) was replaced by the Live demo test on 2026-11-12; its runs
still open with "Open results...", and the engine still runs synthetic plans.

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
long-term level over the whole source (all 38.5 s of WarSoundScape.wav, looped), so its calm and loud stretches
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
import threading
import time
import tkinter as tk
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, str(Path(__file__).parent))
import complete_test_engine as E  # noqa: E402
import live_test_engine as L  # noqa: E402

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
PAGES = [("real", "Real drone test"), ("modified", "Modified drone test"), ("live", "Live demo test")]   # the synthetic
# page was replaced by the live page on 2026-11-12 (Nicolai); synthetic runs still open with "Open results..."
SUBTITLES = {"real": "Real drone recordings mixed into chosen background layers at real levels.",
             "live": "The modified drone test in real time, out loud: one soundscape that loops until the test is over, "
                     "drones flown in by the plan or by you, scored on the mix or on what a microphone hears.",
             "synthetic": "Pure Data drones flying at the microphone, mixed into chosen background layers.",
             "modified": "Real drone recordings flown at the microphone: the end of each file is the impact, and the "
                         "level follows the distance at each moment for the chosen speed."}
DEFAULT_LEVELS = {"war soundscape": ("60", "60", "1"), "war-shaped noise": ("50", "50", "1"), "wind": ("50", "50", "1"),
                  "wind (Pd)": ("50", "50", "1"), "cars": ("5", "5", "1"), "helicopter": ("", "", ""), "insects": ("", "", "")}
COLORS = {"it6 CNN": "#2a78d6", "it9 CNN": "#e8710a", "it9 CRNN": "#9334e6", "it10 CNN": "#d93025",
          "It9B": "#0b8a8a", "it16 CNN 600k s0": "#b8860b", "it16 CNN 600k s1": "#d4a72c",
          "it16 CNN 600k s2": "#8a6d0b", "it16 CRNN 600k s0": "#c2185b", "it16 CRNN 600k s1": "#e5609a",
          "it16 CRNN 600k s2": "#880e4f", "it19 CRNN colour s1": "#ff6f00", "it17 CNN 2s 600k s0": "#1b5e20", "it17 CNN 2s 600k s1": "#43a047",
          "it17 CNN 2s 600k s2": "#81c784", "it17 CRNN 2s 600k s0": "#311b92", "it17 CRNN 2s 600k s1": "#5e35b1",
          "it17 CRNN 2s 600k s2": "#9575cd", "it20 CNN bp s0": "#006064", "it20 CNN bp s1": "#0097a7",
          "it20 CNN bp s2": "#4dd0e1", "it20 CRNN bp s0": "#3e2723", "it20 CRNN bp s1": "#6d4c41",
          "it20 CRNN bp s2": "#a1887f", "it21 CRNN uav16 s0": "#263238", "it21 CRNN uav16 s1": "#607d8b",
          "it21 CRNN uav16 s2": "#b0bec5", "it22 CNN pcen s0": "#1a237e", "it22 CNN pcen s1": "#3949ab",
          "it22 CNN pcen s2": "#7986cb", "it22 CRNN pcen s0": "#b71c1c", "it22 CRNN pcen s1": "#e53935",
          "it22 CRNN pcen s2": "#ef9a9a", "comb": "#1baf7a"}   # it20, it21, it22: the same colours as detect_gui
PD_CONTROLS = {"wind (Pd)": ("wind", (("speed_ms", "speed (m/s, assumed)", "quiet: 1  ·  storm: 25"),
                                      ("reso", "resonance (0-127)", "open field: 10  ·  city: 90"))),
               "cars": ("cars", (("speed_kmh", "speed (km/h)", "town: 50  ·  motorway: 130"),
                                 ("dist_m", "closest distance (m)", "distant: 50  ·  near: 10")))}
DET_PER_ROW = 5                                              # detector ticks per line on the setup page (they wrap)
PD_GUI_START = {("cars", "dist_m"): 50.0}                    # distant cars; the engine's own default stays 10 m
WIND_LEVELS = "quiet: 25  ·  storm: 50"
LEVEL_HINTS = {"war soundscape": "Test 1: 90",
               "war-shaped noise": "Test 1: 90",
               "wind": WIND_LEVELS, "wind (Pd)": WIND_LEVELS,
               "cars": "distant cars: 1-10"}
FINAL_TAG = "  -  THE FINAL EVALUATION (sealed car)"
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
    "level extrapolated. Cars are distant cars: about 1-10 dB SPL (10_11_TBITest used 1-20 at 50 m). The EU pass-by "
    "limits (68-72 dB(A) at 7.5 m) would be a near car, far louder than intended here.",
    "Confidence follows the dB difference between drone and background, not the absolute level. Keep levels between "
    "about 60 and 100 dB SPL: below that the frontend's floor changes the scores, above that a real microphone would "
    "clip and this test does not.",
    "wind (Pd) and cars: their settings change the character of the sound, not its loudness. The level you set is "
    "still the average over the whole render, whatever the speed or distance. Each new combination is rendered once, "
    "at the start of the test (about a minute per wind combination, seconds for cars), then kept.",
    "cars: one car (car_idle) passes once, starting with the round; it is not looped, so after its pass-by (about "
    "21.5 s) the round has no car. The level you set is the average over the pass-by. At 50 km/h and 10 m a pass-by "
    "peaks about 11 dB above that level (1 s windows).",
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


FA_MAX = 200.0                                               # the false-alarm box's ceiling (2 x the highest no-drone peak)


def fa_pct(s: str) -> float:
    """The false-alarm box, in %: up to 100 the share of no-drone windows that must lie below the threshold
    (100 = the highest no-drone peak); above 100 the threshold is that peak times the % (101 = just above it)."""
    try:
        p = float(str(s).strip().rstrip("%"))
    except ValueError:
        raise ValueError("false alarms: a number in %, for example 95") from None
    if not 50 <= p <= FA_MAX:
        raise ValueError(f"false alarms: from 50 to {FA_MAX:g} %")
    return p


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
    """The detectors offered on the setup page: the recurrent networks (CRNN, It9B) and comb. The DS-CNNs
    are left out at Nicolai's request (2026-11-10); they stay in dwg.MODELS, so older results that used
    them still open. A model whose run file or checkpoint is missing on this machine is not offered (run
    files are git-ignored, so a pulled checkpoint can arrive without the threshold it needs)."""
    return [m for m, (run, ckpt) in E.dwg.MODELS.items()
            if not str(ckpt).endswith("_dscnn.pt") and Path(run).exists() and Path(ckpt).exists()] + ["comb"]


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
    START_LABEL = "soundscape start in the war track (s):"
    START_NOTE = "blank = drawn once per batch (seed 1000); every round uses the same stretch"

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
                self.v_pitches = (tk.StringVar(value="1"), tk.StringVar(value="1"), tk.StringVar(value="1"))
                self.v_pitch_random = tk.BooleanVar(value=False)
                ttk.Label(f, text="Pitch (x, 1 = as recorded):").grid(row=2, column=0, sticky="w", pady=(8, 0))
                pi = ttk.Frame(f)
                pi.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
                for lbl, v, w in zip(("min", "max", "#"), self.v_pitches, (7, 7, 4)):
                    ttk.Label(pi, text=lbl, style="Muted.TLabel").pack(side="left", padx=(0 if lbl == "min" else 8, 3))
                    ttk.Entry(pi, textvariable=v, width=w).pack(side="left")
                    v.trace_add("write", lambda *_: self.update_count())
                self.l_pitch_preview = ttk.Label(pi, text="", style="Muted.TLabel", width=26)
                self.l_pitch_preview.pack(side="left", padx=(10, 0))
                ttk.Label(pi, text=f"allowed {E.PITCH_LIMITS[0]:g}-{E.PITCH_LIMITS[1]:g}  ·  suggested 0.8-1.25  ·  "
                                   "light: 0.8  ·  as recorded: 1  ·  heavy: 1.25", style="Muted.TLabel").pack(side="left")
                rnd = ttk.Frame(f)
                rnd.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(2, 0))
                ttk.Checkbutton(rnd, text="add randomness", variable=self.v_pitch_random, command=self.update_count).pack(side="left")
                ttk.Checkbutton(rnd, text="Doppler (pitch rises while it approaches)", variable=self.v_doppler,
                                command=self.update_count).pack(side="left", padx=(16, 0))
                ttk.Label(rnd, text=f"randomness: the pitch wanders about {100 * E.PITCH_WOBBLE:g} % around its value throughout",
                          style="Muted.TLabel").pack(side="left", padx=(16, 0))
                ttk.Label(f, text="Pitch is the speed of the four motors. In hover, pitch = √(drone weight ÷ weight when recorded): "
                                  "x0.8 = 36 % lighter, x1.1 = 21 % heavier (a load), x1.25 = 1.56 times as heavy, x2 = 4 times. "
                                  "A propeller-physics rule, not measured on our drones; flying fast also raises the motor speed.",
                          style="Muted.TLabel", wraplength=760, justify="left").grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(2, 0))
                sub = ttk.Frame(f)
                sub.grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
                ttk.Label(sub, text="distance at impact (m)", style="Muted.TLabel").pack(side="left", padx=(0, 4))
                ttk.Entry(sub, textvariable=self.v_closest, width=6).pack(side="left")
                self.l_approach = ttk.Label(f, text="", style="Muted.TLabel")
                self.l_approach.grid(row=6, column=1, sticky="w", padx=(8, 0), pady=(2, 0))
                for v in (*self.v_speeds, self.v_closest):
                    v.trace_add("write", lambda *_: self.update_count())
                r = 7
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
        ttk.Label(st, text=self.START_LABEL).pack(side="left")
        self.v_sc_start = tk.StringVar(value="")
        ttk.Entry(st, textvariable=self.v_sc_start, width=8).pack(side="left", padx=6)
        ttk.Label(st, text=self.START_NOTE, style="Muted.TLabel").pack(side="left")
        r += 1
        final = ttk.Frame(g)
        final.grid(row=r, column=1, columnspan=7, sticky="w", padx=4, pady=(8, 0))
        self.v_final = tk.BooleanVar(value=False)
        self.b_final = tk.Checkbutton(final, text="The Final Evaluation: OFF", variable=self.v_final, indicatoron=False, width=26,
                                      font=("Segoe UI", 10, "bold"), selectcolor="#f4c7c3", command=self._flip_final)
        self.b_final.pack(side="left")
        ttk.Label(final, text="on: the car is the sealed car, which no model has trained on; off: the development car (idle)",
                  style="Muted.TLabel").pack(side="left", padx=8)
        r += 1
        guide = ttk.Frame(g)
        guide.grid(row=r, column=1, columnspan=7, sticky="w", padx=4, pady=(8, 0))
        ttk.Label(guide, text="Guidelines", style="Bold.TLabel").pack(anchor="w")
        for text in LAYER_GUIDELINES:
            ttk.Label(guide, text=f"•  {text}", style="Muted.TLabel", wraplength=980, justify="left").pack(anchor="w")

    def _flip_final(self):
        self.b_final.configure(text=f"The Final Evaluation: {'ON' if self.v_final.get() else 'OFF'}")
        self.update_count()

    def _pd_rows(self, g, r: int, name: str) -> int:
        """A row per setting of a Pure Data layer, under the layer, with the same min, max and # boxes
        as a level. Returns the last row used."""
        sec, settings = PD_CONTROLS[name]
        for key, label, hint in settings:
            r += 1
            start = f"{PD_GUI_START.get((sec, key), E.PD_SETTINGS[sec][key]):g}"
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
        self._build_buffer(f, row=3)
        fa = ttk.Frame(f)
        fa.grid(row=2, column=0, columnspan=5, sticky="w", pady=(4, 0))
        ttk.Label(fa, text="false alarms:").pack(side="left")
        self.v_fa = tk.StringVar(value="95")
        ttk.Entry(fa, textvariable=self.v_fa, width=5).pack(side="left", padx=4)
        ttk.Label(fa, text="%  of the no-drone windows lie below the threshold (100 = the highest no-drone peak, above 100 = "
                           "that peak x the %); Conclusions gives the threshold and the time before impact at it",
                  style="Muted.TLabel").pack(side="left")
        for v in (self.v_len, self.v_step, self.v_fa):
            v.trace_add("write", lambda *_: self.update_count())

    def _build_buffer(self, f, row: int):
        """The buffer after impact: seconds the drone phase lasts past the drone's end (E.with_buffer)."""
        bf = ttk.Frame(f)
        bf.grid(row=row, column=0, columnspan=5, sticky="w", pady=(4, 0))
        ttk.Label(bf, text="buffer after impact (s):").pack(side="left")
        self.v_buffer = tk.StringVar(value="1")
        ttk.Entry(bf, textvariable=self.v_buffer, width=5).pack(side="left", padx=4)
        ttk.Label(bf, text="the \"there is a drone\" phase lasts this much longer than the drone (past the impact; past "
                           "the end of a real file): the detectors' cooling counts with the drone, not as a false alarm",
                  style="Muted.TLabel").pack(side="left")
        self.v_buffer.trace_add("write", lambda *_: self.update_count())

    def _build_detectors(self, f):
        self.v_det = {}
        for i, d in enumerate(detector_names()):
            v = tk.BooleanVar(value=d in ("it9 CRNN", "comb"))
            self.v_det[d] = v
            tk.Checkbutton(f, text="DSP (comb)" if d == "comb" else d, variable=v, command=self.update_count,
                           fg=COLORS.get(d, "#333"), bg=PANEL, activebackground=PANEL, selectcolor=PANEL,
                           font=("Segoe UI", 10, "bold")).grid(row=i // DET_PER_ROW, column=i % DET_PER_ROW,
                                                               padx=(0, 14), pady=1, sticky="w")

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
        try:
            buffer_s = float(self.v_buffer.get().replace(",", "."))
        except ValueError:
            raise ValueError("buffer after impact must be a number of seconds") from None
        speeds, approach, pitches = None, None, None
        if self.kind in ("real", "modified"):
            drones = list(self.files)
            if self.kind == "modified":
                speeds = speed_steps(*(v.get() for v in self.v_speeds))
                pitches = pd_steps("pitch", *(v.get() for v in self.v_pitches))
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
                   also_without=[n for n in layers if self.v_layer_off[n].get()], final_evaluation=self.v_final.get(),
                   car_once=True,                            # one car, one pass-by from the round's start (entry 199)
                   fa_percentile=fa_pct(self.v_fa.get()), buffer_after_s=buffer_s,
                   soundscape_start_s=float(start) if start else None, **({"synth": synth} if synth else {}),
                   **({"speeds": speeds, "approach": approach, "pitches": pitches, "pitch_random": self.v_pitch_random.get()}
                      if self.kind == "modified" else {}))
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
            try:
                txt = ", ".join(f"x{v:g}" for v in pd_steps("pitch", *(v.get() for v in self.v_pitches)))
                txt = (txt if len(txt) <= 22 else txt[:20] + "…") + (", random" if self.v_pitch_random.get() else "")
                self.l_pitch_preview.configure(text=txt, style="Muted.TLabel")
            except ValueError as e:
                self.l_pitch_preview.configure(text=str(e).split(": ", 1)[-1][:26], style="Warn.TLabel")
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
        self.l_count.configure(text=f"{c['total']:,} test round{'s' * (c['total'] != 1)}".replace(",", " "))
        n_f, n_v, n_p = len(p.drones), len(p.speeds or []), len(p.pitches or [1.0])
        drones_txt = (f"{n_f} file{'s' * (n_f != 1)} × {n_v} speed{'s' * (n_v != 1)} × {n_p} pitch{'es' * (n_p != 1)}"
                      if self.kind == "modified"
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
        bits = [f"{n} {', '.join(f'{v:g}' for v in lv)} dB" + (" + off" if n in p.also_without else "")
                + (" (sealed)" if n == "cars" and p.final_evaluation else "")
                for n, lv in p.layers.items()]
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


# ------------------------------------------------------------------ the live page

LIVE_SPAN_S = 30.0                                           # seconds shown in the live view
LIVE_MODES = {("virtual", "automatic"): "The plan's rounds play one after the other in real time, out loud, over a war "
                                        "soundscape that loops. Each round is scored on the mix it played.",
              ("virtual", "manual"): "The soundscape loops; each press of Send drone flies in the next drone of the plan. "
                                     "Each round is scored on the mix it played.",
              ("microphone", "automatic"): "The plan's rounds play one after the other out of the speaker; the detectors "
                                           "score what the microphone hears.",
              ("microphone", "manual"): "The soundscape loops out of the speaker; each press of Send drone flies in the "
                                        "next drone of the plan. The detectors score what the microphone hears."}
LIVE_UNTIL_STOP = {"automatic": "The test ends when the plan's last round has played, or when you press Stop; then the "
                                "Results window opens once, with every finished round.",
                   "manual": "There is no number of rounds: a round ends at the drone's impact + the buffer, the next "
                             "starts at once, and the test runs until you press Stop; then the Results window opens."}
DRONE_SPAN = "#f9d5d1"


def audio_devices(kind: str) -> list[tuple[str, int | None]]:
    """(label, index) of every input or output device, the system default first."""
    try:
        import sounddevice as sd
        devs, apis = sd.query_devices(), sd.query_hostapis()
    except Exception:                                        # noqa: BLE001  (no audio system: the default only)
        return [("system default", None)]
    key = "max_output_channels" if kind == "output" else "max_input_channels"
    return [("system default", None)] + [(f"{i}: {d['name']} ({apis[d['hostapi']]['name']}, {d[key]} ch)", i)
                                         for i, d in enumerate(devs) if d[key] > 0]


class LivePage(Page):
    """The Live page: the modified drone test's settings, played out loud in real time on one continuous
    soundscape (scripts/live_test_engine.py). Source: virtual (scored on the mix) or microphone (scored on
    what the microphone hears); trigger: automatic (the plan's rounds one after the other) or manual (Send
    drone flies in the next drone round). Everything is set before Play and locked while it runs; then only
    Send drone and Stop act. The results are a run folder like the other pages', for Results and Conclusions."""

    def __init__(self, master, app, kind: str, title: str):
        self.run = None
        super().__init__(master, app, "modified", title)    # the modified page's settings and plan

    # ---------- layout
    START_LABEL = "where the war soundscape starts (s):"
    START_NOTE = "blank = drawn once (seed 1000); from there it loops without a break until the test is over"

    def _build(self):
        """Top to bottom in the order of use: how it runs, what it plays, Play, and the live view."""
        self.columnconfigure(0, weight=1)
        ttk.Label(self, text=self.title, style="H1.TLabel", font=("Segoe UI", 18, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(self, text=SUBTITLES["live"], style="Sub.TLabel").grid(row=1, column=0, sticky="w", pady=(0, 10))
        self.f_modes = self._section(2, "1  ·  Source and drones")
        self.setup = [self._section(3, "2  ·  Drones"), self._section(4, "3  ·  Background layers"),
                      self._section(5, "4  ·  Test rounds"), self._section(6, "5  ·  Detectors")]
        self.f_play = self._section(7, "6  ·  Play")
        self.f_view = self._section(8, "Live view")
        self._build_modes(self.f_modes)
        self._build_drones(self.setup[0])
        self._build_layers(self.setup[1])
        self._build_rounds(self.setup[2])
        self._build_detectors(self.setup[3])
        self._build_play(self.f_play)
        self._build_view(self.f_view)
        self._mode_changed()

    def _build_modes(self, f):
        f.columnconfigure(1, weight=1)
        self.v_source, self.v_trigger = tk.StringVar(value="virtual"), tk.StringVar(value="automatic")
        for row, (label, var, opts) in enumerate((("source", self.v_source, L.SOURCES), ("drones", self.v_trigger, L.TRIGGERS))):
            ttk.Label(f, text=label).grid(row=row, column=0, sticky="w", pady=2)
            box = ttk.Frame(f)
            box.grid(row=row, column=1, sticky="w", padx=8, pady=2)
            for o in opts:
                tk.Radiobutton(box, text=o.capitalize(), value=o, variable=var, indicatoron=False, width=12,
                               font=("Segoe UI", 10, "bold"), selectcolor="#cfe0f7", bg=PANEL, relief="ridge",
                               command=self._mode_changed).pack(side="left")
        self.l_mode = ttk.Label(f, text="", wraplength=950, justify="left")
        self.l_mode.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 10))
        self.outs, self.ins = audio_devices("output"), audio_devices("input")
        self.v_out, self.v_in = tk.StringVar(value=self.outs[0][0]), tk.StringVar(value=self.ins[0][0])
        self.v_play_gain, self.v_in_ch, self.v_mic_gain = tk.StringVar(value="0"), tk.StringVar(value="0"), tk.StringVar(value="0")
        dev = ttk.Frame(f)
        dev.grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Label(dev, text="speaker").grid(row=0, column=0, sticky="w")
        ttk.Combobox(dev, textvariable=self.v_out, values=[a for a, _ in self.outs], state="readonly", width=60).grid(
            row=0, column=1, sticky="w", padx=8)
        ttk.Label(dev, text="playback gain (dB)").grid(row=0, column=2, sticky="e", padx=(12, 4))
        ttk.Entry(dev, textvariable=self.v_play_gain, width=6).grid(row=0, column=3, sticky="w")
        self.mic_widgets = [ttk.Label(dev, text="microphone"),
                            ttk.Combobox(dev, textvariable=self.v_in, values=[a for a, _ in self.ins], state="readonly", width=60),
                            ttk.Label(dev, text="mic gain (dB)"), ttk.Entry(dev, textvariable=self.v_mic_gain, width=6),
                            ttk.Label(dev, text="channel"), ttk.Entry(dev, textvariable=self.v_in_ch, width=6)]
        places = [("w", 0), ("w", (8, 0)), ("e", (12, 4)), ("w", 0), ("e", (12, 4)), ("w", 0)]   # (sticky, padx) per column
        for col, (w, (sticky, padx)) in enumerate(zip(self.mic_widgets, places)):
            w.grid(row=1, column=col, sticky=sticky, padx=padx, pady=(6, 0))
        ttk.Label(f, text="Full scale is 110 dB SPL in every file. Playback gain changes only what goes to the speaker "
                          "(not the virtual scores); mic gain scales the recording before the detectors hear it. A speaker "
                          "and a microphone are not calibrated, so with the microphone the detectors do not hear the plan's "
                          "dB SPL. The microphone delay is measured each round; use a microphone without echo "
                          "cancellation (a laptop's built-in array removes the speaker's sound).",
                  style="Muted.TLabel", wraplength=950, justify="left").grid(row=4, column=0, columnspan=2, sticky="w",
                                                                            pady=(8, 0))

    def _build_rounds(self, f):
        """The modified page's round settings, in live terms: a round is a stretch of the one soundscape."""
        self.v_len, self.v_step, self.v_nodrone = tk.StringVar(value="20"), tk.StringVar(value="1"), tk.BooleanVar(value=True)
        ttk.Label(f, text="Round length (s):").grid(row=0, column=0, sticky="w")
        self.e_len = ttk.Spinbox(f, from_=10, to=600, increment=1, textvariable=self.v_len, width=8, command=self.update_count)
        self.e_len.grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(f, text="the drone (up to 10 s, ending in its impact) starts at 0 s, then every").grid(row=0, column=2, sticky="w")
        self.e_step = ttk.Entry(f, textvariable=self.v_step, width=5)
        self.e_step.grid(row=0, column=3, sticky="w", padx=4)
        ttk.Label(f, text="s, as long as it fits in the round: one round per start").grid(row=0, column=4, sticky="w")
        self.l_step_note = ttk.Label(f, text="", style="Muted.TLabel")
        self.l_step_note.grid(row=1, column=0, columnspan=5, sticky="w")
        self.cb_nodrone = ttk.Checkbutton(f, text="add one no-drone round per background (false alarms; they play after "
                                                  "the drone rounds)", variable=self.v_nodrone, command=self.update_count)
        self.cb_nodrone.grid(row=2, column=0, columnspan=5, sticky="w", pady=(6, 0))
        fa = ttk.Frame(f)
        fa.grid(row=3, column=0, columnspan=5, sticky="w", pady=(4, 0))
        ttk.Label(fa, text="false alarms:").pack(side="left")
        self.v_fa = tk.StringVar(value="95")
        ttk.Entry(fa, textvariable=self.v_fa, width=5).pack(side="left", padx=4)
        ttk.Label(fa, text="%  of the no-drone windows lie below the threshold; Conclusions gives the threshold and the "
                           "time before impact at it", style="Muted.TLabel").pack(side="left")
        self._build_buffer(f, row=4)
        for v in (self.v_len, self.v_step, self.v_fa):
            v.trace_add("write", lambda *_: self.update_count())

    def _build_detectors(self, f):
        """The detectors, and the alarm threshold the tester may set for all of them."""
        super()._build_detectors(f)
        th = ttk.Frame(f)
        th.grid(row=99, column=0, columnspan=DET_PER_ROW, sticky="w", pady=(8, 0))
        ttk.Label(th, text="alarm threshold (confidence, 0-1):").pack(side="left")
        self.v_alarm = tk.StringVar(value="")
        ttk.Entry(th, textvariable=self.v_alarm, width=6).pack(side="left", padx=4)
        ttk.Label(th, text="blank = each detector's own threshold (frozen on validation; comb's own is 0.5 on its "
                           "confidence). Set, every detector alarms at this confidence: the DRONE banner, the dashed "
                           "lines and the alarm numbers in the results (recall, false positives, first alarm, lead). "
                           "The confidences themselves do not change.", style="Muted.TLabel", wraplength=820,
                  justify="left").pack(side="left")
        self.v_alarm.trace_add("write", lambda *_: self.update_count())

    def _build_play(self, f):
        f.columnconfigure(0, weight=1)
        self.l_count = ttk.Label(f, text="", style="Count.TLabel")
        self.l_count.grid(row=0, column=0, sticky="w")
        self.l_count_detail = ttk.Label(f, text="", style="Muted.TLabel")
        self.l_count_detail.grid(row=1, column=0, sticky="w")
        self.l_duration = ttk.Label(f, text="", style="Muted.TLabel")
        self.l_duration.grid(row=2, column=0, sticky="w", pady=(0, 8))
        bar = ttk.Frame(f)
        bar.grid(row=3, column=0, sticky="ew")
        bar.columnconfigure(5, weight=1)
        self.b_start = ttk.Button(bar, text="▶  Play", style="Accent.TButton", command=self.start)
        self.b_start.grid(row=0, column=0, padx=(0, 4))
        self.b_stop = ttk.Button(bar, text="■  Stop", command=self.stop, state="disabled")
        self.b_stop.grid(row=0, column=1, padx=4)
        self.b_send = tk.Button(bar, text="✈  Send drone", command=self.send, fg="white", disabledforeground="#f5f5f5",
                                activebackground="#b3261e", activeforeground="white", relief="flat",
                                font=("Segoe UI", 11, "bold"), padx=16, pady=3)
        self.b_send.grid(row=0, column=2, padx=4)               # manual only: _mode_changed hides it otherwise
        self._send_ready(False)
        self.b_results = ttk.Button(bar, text="Results window", command=self.open_results, state="disabled")
        self.b_results.grid(row=0, column=3, padx=4)
        ttk.Button(bar, text="Save plan...", command=self.save_plan).grid(row=0, column=4, padx=4)
        self.pb = ttk.Progressbar(bar, mode="determinate")
        self.pb.grid(row=0, column=5, sticky="ew", padx=(10, 0))
        self.l_status = ttk.Label(f, text="Set up 1-5 above, then Play. While it plays, only Stop (and Send drone) act; "
                                          "the Results window opens once, when the plan is done or you press Stop.",
                                  style="Muted.TLabel", wraplength=1000, justify="left")
        self.l_status.grid(row=4, column=0, sticky="w", pady=(6, 0))

    def _send_ready(self, ready: bool):
        """Send drone: red when a press would fly a drone in now, grey otherwise."""
        if str(self.b_send.cget("state")) != ("normal" if ready else "disabled") or not hasattr(self, "_send_set"):
            self.b_send.configure(state="normal" if ready else "disabled", bg="#d93025" if ready else "#c4c7cc")
            self._send_set = True

    def _build_view(self, f):
        f.columnconfigure(0, weight=1)
        self.l_round = ttk.Label(f, text="", style="Bold.TLabel")
        self.l_round.grid(row=0, column=0, sticky="w")
        self.fig = Figure(figsize=(11, 5.4), dpi=90)
        self.fig.subplots_adjust(left=0.07, right=0.985, top=0.98, bottom=0.08)
        gs = self.fig.add_gridspec(4, 1, height_ratios=[0.55, 1.6, 1.3, 0.55], hspace=0.12)
        self.ax_ban = self.fig.add_subplot(gs[0])
        self.ax_spec = self.fig.add_subplot(gs[1])
        self.ax_score = self.fig.add_subplot(gs[2], sharex=self.ax_spec)
        self.ax_level = self.fig.add_subplot(gs[3], sharex=self.ax_spec)
        self.ax_ban.set_axis_off()
        from matplotlib.patches import Rectangle
        self.ban_box = self.ax_ban.add_patch(Rectangle((0, 0), 1, 1, transform=self.ax_ban.transAxes, color="#bbbbbb"))
        self.ban_text = self.ax_ban.text(0.015, 0.5, "press Play", fontsize=22, weight="bold", color="white", va="center",
                                         transform=self.ax_ban.transAxes)
        self.det_text = self.ax_ban.text(0.985, 0.5, "", fontsize=9, family="monospace", color="white", va="center",
                                         ha="right", transform=self.ax_ban.transAxes, linespacing=1.3)
        self.dist_text = self.ax_ban.text(0.43, 0.5, "", fontsize=20, weight="bold", color="white", va="center",
                                          ha="center", transform=self.ax_ban.transAxes, linespacing=1.1)
        self.n_cols = int(round(LIVE_SPAN_S * DEFAULT.sample_rate / DEFAULT.hop_length))
        self.spec = np.full((DEFAULT.n_mels, self.n_cols), np.nan, np.float32)
        cmap = matplotlib.colormaps["magma"].with_extremes(bad="#000004")
        self.img = self.ax_spec.imshow(self.spec, origin="lower", aspect="auto", cmap=cmap, interpolation="nearest",
                                       extent=(-LIVE_SPAN_S, 0, 0, DEFAULT.n_mels), vmin=-100, vmax=-20)
        ticks = [0, 16, 32, 48, 63]
        self.ax_spec.set_yticks([t + 0.5 for t in ticks], [f"{BAND_HZ[t]:.0f}" for t in ticks])
        self.ax_spec.set_ylabel("Hz", fontsize=8)
        self.ax_spec.tick_params(labelbottom=False, labelsize=7)
        self.ax_score.set(ylim=(0, 1.02))
        self.ax_score.set_ylabel("confidence", fontsize=8)
        self.ax_score.grid(color="#e5e5e5", lw=0.5)
        self.ax_score.tick_params(labelbottom=False, labelsize=7)
        (self.level_line,) = self.ax_level.plot([], [], color="#777777", lw=1.0)
        self.ax_level.set(ylim=(-100, 0), xlim=(0, LIVE_SPAN_S))
        self.ax_level.set_ylabel("dBFS", fontsize=8)
        self.ax_level.set_xlabel("seconds since Play  (red: a drone is flying; orange: the buffer after impact, still "
                                 "counted as drone; with the microphone, moved by its delay)", fontsize=8)
        self.ax_level.grid(color="#e5e5e5", lw=0.5)
        self.ax_level.tick_params(labelsize=7)
        self.canvas = FigureCanvasTkAgg(self.fig, master=f)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="ew")
        ttk.Label(f, text="Finished rounds (newest on top)", style="Bold.TLabel").grid(row=2, column=0, sticky="w", pady=(8, 2))
        box = ttk.Frame(f)
        box.grid(row=3, column=0, sticky="ew")
        box.columnconfigure(0, weight=1)
        cols = ("round", "drone", "detector", "false", "lead")
        self.t_log = ttk.Treeview(box, columns=cols, show="headings", height=7, style="Log.Treeview")
        for c, (head, w, anchor) in zip(cols, (("Round", 60, "center"), ("Drone", 330, "w"), ("Detector", 190, "w"),
                                              ("False alarms", 110, "center"), ("Time before impact", 260, "w"))):
            self.t_log.heading(c, text=head)
            self.t_log.column(c, width=w, anchor=anchor, stretch=c == "drone")
        sb = ttk.Scrollbar(box, orient="vertical", command=self.t_log.yview)
        self.t_log.configure(yscrollcommand=sb.set)
        self.t_log.grid(row=0, column=0, sticky="ew")
        sb.grid(row=0, column=1, sticky="ns")
        ttk.Style(self).configure("Log.Treeview", rowheight=24, font=("Segoe UI", 10))
        ttk.Style(self).configure("Log.Treeview.Heading", font=("Segoe UI", 10, "bold"))
        self.t_log.tag_configure("alarm", foreground="#b3261e")
        self.t_log.tag_configure("odd", background="#f4f5f7")
        ttk.Label(f, text="False alarms: separate alarms in the round's no-drone part (one alarm can last several hops), at "
                          "the alarm threshold in force. Time before impact: the first alarm from the drone, counted back "
                          "from the impact.", style="Muted.TLabel", wraplength=1000, justify="left").grid(
            row=4, column=0, sticky="w", pady=(2, 0))
        self.lines, self.thr_lines, self.view_extra = {}, [], []
        self._reset_view([])
        self.canvas.draw_idle()

    def _reset_view(self, detectors: list[str]):
        self.spec[:] = np.nan
        self.vt, self.vlevel, self.vconf = [], [], {d: [] for d in detectors}
        self.spans, self.bounds, self.log = [], [], []
        self.lag_s, self.thr_view, self.cols_shown, self.t_now = 0.0, {}, 0, 0.0
        for ln in list(self.lines.values()) + self.thr_lines:
            ln.remove()
        self.lines = {d: self.ax_score.plot([], [], color=COLORS.get(d, "#333333"), lw=1.3, label=d)[0] for d in detectors}
        self.thr_lines = []
        if self.ax_score.get_legend():
            self.ax_score.get_legend().remove()
        if detectors:
            self.ax_score.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=len(detectors))
        self.t_log.delete(*self.t_log.get_children())
        self.l_round.configure(text="")
        self.dirty = True

    # ---------- modes and plan
    def _mode_changed(self):
        mic = self.v_source.get() == "microphone"
        manual = self.v_trigger.get() == "manual"
        for w in self.mic_widgets:
            w.state(["!disabled"] if mic else ["disabled"])
        for w in (self.e_len, self.e_step, self.cb_nodrone):  # automatic only
            w.state(["disabled"] if manual else ["!disabled"])
        self.l_step_note.configure(text="manual: none of these apply. A round runs from the end of the last one (or Play) "
                                        "to the drone's impact + the buffer after impact, so the wait before you send a "
                                        "drone is its no-drone part; the test runs until Stop." if manual else "")
        (self.b_send.grid if manual else self.b_send.grid_remove)()
        (self.pb.grid_remove if manual else self.pb.grid)()
        self.l_mode.configure(text=LIVE_MODES[self.v_source.get(), self.v_trigger.get()] + " "
                              + LIVE_UNTIL_STOP[self.v_trigger.get()])
        self.update_count()

    def plan(self) -> E.Plan:
        p = super().plan()
        alarm = self.v_alarm.get().strip().replace(",", ".") if hasattr(self, "v_alarm") else ""
        try:
            alarm = float(alarm) if alarm else None
        except ValueError:
            raise ValueError("alarm threshold: a confidence from 0 to 1, or blank") from None
        if alarm is not None and not 0 < alarm < 1:
            raise ValueError("alarm threshold: above 0 and below 1, or blank")
        p.live = dict(source=self.v_source.get(), trigger=self.v_trigger.get(), threshold=alarm)
        if p.live["trigger"] == "manual":                    # the plan only lists the drone settings: one each
            p.step_s, p.no_drone_rounds = p.length_s, False
        return p

    def update_count(self, *_):
        super().update_count()
        if not hasattr(self, "l_duration") or not hasattr(self, "v_source"):
            return
        try:
            p = self.plan()
        except ValueError:
            self.l_duration.configure(text="")
            return
        if p.check():                                        # the line above says what is wrong
            self.l_duration.configure(text="")
            return
        c = p.count()
        if p.live["trigger"] == "automatic":
            s = c["total"] * p.length_s
            self.l_duration.configure(text=f"plays for about {s / 60:.1f} min: {c['total']} rounds × {p.length_s:g} s, "
                                           f"after {L.LEAD_IN_S:g} s of background")
        else:                                                # manual: no number of rounds, the test runs until Stop
            n = c["drone_rounds"]
            self.l_count.configure(text="runs until you press Stop")
            self.l_count_detail.configure(text=f"{n} drone setting{'s' * (n != 1)} ({len(p.drones)} file"
                                               f"{'s' * (len(p.drones) != 1)} × {len(p.speeds or [])} speed(s) × "
                                               f"{len(p.pitches or [1])} pitch(es) × {c['drone_levels']} level(s) × "
                                               f"{c['backgrounds']} background(s)), flown in turn by Send drone and over "
                                               f"again", style="Muted.TLabel")
            self.l_duration.configure(text=f"each round: the wait before you send the drone, then the drone up to its "
                                           f"impact, then {p.buffer_after_s:g} s of buffer; Send drone lights up again "
                                           "when the round is over")

    # ---------- run
    def _devices(self):
        out = dict(self.outs)[self.v_out.get()]
        inp = dict(self.ins)[self.v_in.get()]
        try:
            play, mic, ch = float(self.v_play_gain.get()), float(self.v_mic_gain.get()), int(self.v_in_ch.get())
        except ValueError:
            raise ValueError("playback gain and mic gain must be numbers of dB, the channel a whole number") from None
        if not (-60 <= play <= 40 and -60 <= mic <= 60):
            raise ValueError("playback gain from -60 to 40 dB, mic gain from -60 to 60 dB")
        return out, inp, ch, play, mic

    def _lock(self, on: bool):
        """While playing, every setting is locked; unlocking leaves what was disabled before (unavailable layers,
        the microphone boxes in virtual mode) disabled."""
        if on:
            self._was_off = set()
        for f in [self.f_modes, *self.setup]:
            self._set_state(f, on)

    def _set_state(self, w, locked: bool):
        for c in w.winfo_children():
            try:
                if isinstance(c, ttk.Widget):
                    if locked:
                        if c.instate(["disabled"]):
                            self._was_off.add(str(c))
                        c.state(["disabled"])
                    elif str(c) not in self._was_off:
                        c.state(["!disabled"])
                elif isinstance(c, (tk.Checkbutton, tk.Radiobutton, tk.Button, tk.Entry, tk.Listbox, tk.Spinbox)):
                    if locked:
                        if str(c.cget("state")) == "disabled":
                            self._was_off.add(str(c))
                        c.configure(state="disabled")
                    elif str(c) not in self._was_off:
                        c.configure(state="normal")
            except tk.TclError:
                pass
            self._set_state(c, locked)

    def start(self):
        p = self.checked_plan()
        if p is None:
            return
        try:
            out, inp, ch, play, mic = self._devices()
        except ValueError as e:
            messagebox.showerror(TITLE, str(e), parent=self)
            return
        self.events = queue.Queue()
        self.run = L.LiveRun(p, self.events, out_dev=out, in_dev=inp, in_channel=ch, play_gain_db=play, mic_gain_db=mic)
        self.test = self.run.test
        self._reset_view(p.detectors)
        self.pb.configure(maximum=len(self.run.order), value=0)
        self.b_start.state(["disabled"])
        self.b_stop.state(["!disabled"])
        self.b_results.state(["disabled"])
        self._lock(True)
        self.l_status.configure(text="preparing: drones, layers and detectors...")
        self.error = None
        self.run.start()
        self.after(50, self._show_play)
        self.after(100, self._poll)

    def _show_play(self):
        """Scroll the page so Play and the live view are in sight."""
        self.app.update_idletasks()
        h = self.app.inner.winfo_height()
        if h > 0:
            self.app.page_canvas.yview_moveto(max(0.0, (self.winfo_y() + self.f_play.winfo_y() - 10) / h))

    def _after_run(self):
        """When a run has ended: its Results window, or why there is none."""
        if self.test and self.test.results:
            self.b_results.state(["!disabled"])
            self.open_results()
        elif self.error:
            messagebox.showerror(TITLE, f"The test stopped with an error, before any round was scored:\n\n{self.error}",
                                 parent=self)
        else:
            messagebox.showinfo(TITLE, "No round was finished, so there are no results: a round is scored when it has "
                                       "played to its end (Stop drops the round that is playing).", parent=self)

    def stop(self):
        if self.run:
            self.run.stop()
            self.l_status.configure(text="stopping: the round that is playing is dropped; finished rounds are kept...")

    def send(self):
        if self.run and self.run.send_drone():
            self._send_ready(False)

    def _poll(self):
        done = False
        while not self.events.empty():
            kind, x = self.events.get()
            if kind == "dec":
                self._on_dec(x)
            elif kind == "round_start":
                self.spans.append(x)
                lvl = "" if x["level"] is None else f" at {x['level']:g} dB SPL at 1 m"
                bg = "  ·  ".join(f"{k} {v:g} dB" for k, v in x["background"].items()) or "no background"
                of = "" if x["total"] is None else f" of {x['total']}"
                self.l_round.configure(text=f"round {x['index']}{of}: {x['label']}{lvl}  ·  {bg}")
            elif kind == "round_done":
                self.pb.configure(value=x["scored"])
                self._log_round(x)
            elif kind == "lag":
                self.lag_s = x / 1000
                self.dirty = True
            elif kind == "warn":
                self.l_status.configure(text=f"note: {x}")
            elif kind == "status":
                self.l_status.configure(text=x)
            else:                                            # done, error
                self.l_status.configure(text=x if kind == "done" else f"error: {x}")
                if kind == "error":
                    self.error = x
                done = True
        if self.dirty:
            self._redraw()
        running = self.run is not None and not done
        self._send_ready(running and self.run.can_send())
        if done:
            self._lock(False)
            self.b_start.state(["!disabled"])
            self.b_stop.state(["disabled"])
            self.dirty = True
            self._redraw()
            self.after(200, self._after_run)
        else:
            self.after(100, self._poll)

    def _log_round(self, x: dict):
        """One finished round into the table: per detector its false alarms and its time before impact."""
        for i, (det, s) in enumerate(reversed(list(x["sums"].items()))):   # inserted on top: keep the plan's order
            n = s["false_alarms"]
            if not x["drone"]:
                lead = "no drone in this round"
            elif s["lead_s"] is None:
                lead = "no alarm before impact"
            else:
                lead = f"{s['lead_s']:.1f} s"
            first = i == len(x["sums"]) - 1
            self.t_log.insert("", 0, values=(x["round"] if first else "", x["label"] if first else "",
                                             "DSP (comb)" if det == "comb" else det, n, lead),
                              tags=(("alarm",) if n else ()) + (("odd",) if x["round"] % 2 else ()))   # rounds banded

    def _on_dec(self, d: dict):
        if d["cols"] is not None:
            c = min(d["cols"].shape[1], self.n_cols)
            self.spec[:, :-c] = self.spec[:, c:]
            self.spec[:, -c:] = d["cols"][:, -c:]
        self.cols_shown = d["t"]
        self.vt.append(d["t"])
        self.vlevel.append(d["level_dbfs"])
        for k, v in d["conf"].items():
            self.vconf.setdefault(k, []).append(v)
        keep = int(LIVE_SPAN_S / (DEFAULT.samples_per_hop / E.SR)) + 4
        del self.vt[:-keep], self.vlevel[:-keep]
        for k in self.vconf:
            del self.vconf[k][:-keep]
        self.t_now = d["t"]
        self.dirty = True

    def _thresholds(self) -> dict:
        thr = getattr(self.run.test, "thr", None) if self.run else None
        alarm = self.run.plan.live.get("threshold") if self.run else None
        return {} if not thr else {k: alarm if alarm is not None else 0.5 if k == "comb" else float(v) for k, v in thr.items()}

    def _redraw(self):
        self.dirty = False
        t_now = self.t_now
        x0, x1 = max(t_now, LIVE_SPAN_S) - LIVE_SPAN_S, max(t_now, LIVE_SPAN_S)
        self.img.set_extent((t_now - LIVE_SPAN_S, t_now, 0, DEFAULT.n_mels))
        if np.isfinite(self.spec).any():
            lo, top = (float(v) for v in np.nanpercentile(self.spec, [2, 99.5]))
            self.img.set_clim(max(lo, top - 80), top)
        self.img.set_data(self.spec)
        thr = self._thresholds()
        if thr and not self.thr_lines:
            self.thr_lines = [self.ax_score.axhline(v, color=COLORS.get(k, "#333333"), lw=0.8, ls="--")
                              for k, v in thr.items() if k in self.lines]
        t = np.asarray(self.vt)
        for k, ln in self.lines.items():
            y = np.asarray(self.vconf.get(k, []), dtype=float)
            ln.set_data(t[:len(y)], np.clip(y, 0, 1.02))
        self.level_line.set_data(t, self.vlevel)
        for a in self.view_extra:
            a.remove()
        self.view_extra = []
        lag = self.lag_s if self.run and self.run.mic else 0.0
        for s in self.spans:
            if s["drone"] is not None:
                a, b = s["drone"][0] + lag, s["drone"][1] + lag
                buf = self.run.plan.buffer_after_s if self.run else 0.0
                if b + buf >= x0 and a <= x1:
                    self.view_extra += [ax.axvspan(a, b, color=DRONE_SPAN, alpha=0.6, lw=0, zorder=0)
                                        for ax in (self.ax_score, self.ax_level)]
                    if buf > 0:                              # the buffer after impact: still the drone phase
                        self.view_extra += [ax.axvspan(b, b + buf, facecolor="#f4a261", alpha=0.3, lw=0, zorder=0,
                                                       hatch="//", edgecolor="#e07b22")
                                            for ax in (self.ax_score, self.ax_level)]
            if x0 <= s["t0"] + lag <= x1:
                self.view_extra += [ax.axvline(s["t0"] + lag, color="#555555", lw=0.8, ls=":")
                                    for ax in (self.ax_score, self.ax_level)]
        self.ax_spec.set_xlim(x0, x1)
        running = self.run is not None and self.run.running
        last = {k: v[-1] for k, v in self.vconf.items() if v and np.isfinite(v[-1])}
        on = [k for k, v in last.items() if k in thr and v >= thr[k]]
        self.ban_box.set_color("#c0392b" if running and on else "#1e8449" if running else "#999999")
        self.ban_text.set_text("DRONE" if running and on else "listening" if running else
                               "press Play" if self.run is None else
                               "stopped" if self.run.stop_flag.is_set() else "finished")
        self.det_text.set_text("\n".join(f"{k[:22]:<22} {v:5.2f}" + (f" / {thr[k]:.2f}" if k in thr else "")
                                         + ("  DRONE" if k in on else "       ") for k, v in last.items()))
        self.dist_text.set_text(self._distance(t_now, lag) if running else "")
        self.canvas.draw_idle()

    def _distance(self, t: float, lag: float) -> str:
        """How far the flying drone is from the microphone at t (the end of the window the detectors just heard),
        by the approach of the modified test: hypot(distance at impact, v x time left), with the time left before
        impact; "" with no drone flying."""
        for s in reversed(self.spans):
            if s.get("impact") is None or s.get("speed_kmh") is None or s["drone"] is None:
                continue
            impact = s["impact"] + lag
            if s["drone"][0] + lag <= t < impact:
                left = impact - t
                r = float(np.hypot(s["closest_m"], s["speed_kmh"] / 3.6 * left))
                return f"drone {r:,.0f} m away\n{left:.1f} s to impact".replace(",", " ")
            if impact <= t < impact + 1.5:
                return "IMPACT"
        return ""

    def open_results(self):
        if self.test and self.test.results:
            w = Results(self.app, self.test)
            w.lift()                                         # in front of the main window, which had the focus
            w.focus_force()


# ------------------------------------------------------------------ results window

class Results(tk.Toplevel):
    """Summary, every round in a sortable table, and the selected round's log-mel, confidences and
    dB SPL per layer over time, with sound."""

    def __init__(self, app, test: E.Test):
        super().__init__(app)
        self.app, self.test = app, test
        self.title(f"{TITLE} results: {test.out.name}" + (FINAL_TAG if test.plan.final_evaluation else ""))
        self.geometry("1500x940")
        self.configure(bg=BG)
        self.by_round = {r["round"]: r for r in test.rounds}
        self.scores = self._scores()
        self._sound = None
        self.protocol("WM_DELETE_WINDOW", lambda: (self._stop_play(), self.destroy()))   # closing stops the sound
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
        lv = self.test.plan.live
        buf = f" · buffer after impact {self.test.plan.buffer_after_s:g} s"
        head = (f"live demo test ({lv['source']}, manual) · {len(self.test.results)} rounds, each to impact + buffer, "
                "until Stop · one continuous soundscape" if lv and lv["trigger"] == "manual" else
                f"live demo test ({lv['source']}, {lv['trigger']}) · {len(self.test.results)} of {c['total']} rounds · "
                f"{self.test.plan.length_s:g} s each, one continuous soundscape" if lv else
                f"{self.test.plan.kind} drone test · {len(self.test.results)} of {c['total']} rounds · "
                f"soundscape {self.test.plan.length_s:g} s from {self.test.start_s:g} s in the war track")
        if lv and lv.get("threshold") is not None:
            buf += f" · alarm threshold {lv['threshold']:g} (set by the tester)"
        ttk.Label(top, text=head + buf, style="H2.TLabel").pack(side="left")
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
        ctl = ttk.Frame(right)                               # the sound buttons on their own line: a long round text
        ctl.pack(fill="x")                                   # (several detectors) used to push them out of sight
        for txt, what in (("Play mix", "mix"), ("Play drone", "drone"), ("Play background", "background")):
            ttk.Button(ctl, text=txt, command=lambda w=what: self._play(w)).pack(side="left", padx=2)
        ttk.Button(ctl, text="Stop sound", command=self._stop_play).pack(side="left", padx=(8, 2))
        self.l_round = ttk.Label(right, text="Select a round in the table.", style="Bold.TLabel", justify="left")
        self.l_round.pack(fill="x", pady=(4, 0))
        self.l_round.bind("<Configure>", lambda e: self.l_round.configure(wraplength=max(e.width - 10, 200)))
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
        heard = getattr(self.test, "heard", None)            # a live microphone run: the mix is what the microphone heard
        h = heard(r) if heard else None
        if h is not None:
            m = {**m, "mix": h}
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
        buf = self.test.plan.buffer_after_s
        for k in m["truth"].drones:
            for ax in (ax2, ax3):
                ax.axvspan(k["start_s"], k["end_s"], color="#d93025", alpha=0.06, lw=0)
                if buf > 0:                                  # the buffer: still the drone phase (E.with_buffer)
                    ax.axvspan(k["end_s"], k["end_s"] + buf, facecolor="#f4a261", alpha=0.25, lw=0,
                               hatch="//", edgecolor="#e07b22")
                    ax.axvline(k["end_s"], color="#d93025", lw=0.9, ls=":")
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
        ax2.set_title("each detector's confidence (dashed = 50 %; red band = drone sound"
                      + (f"; orange = buffer after impact, {buf:g} s, still counted as drone" if buf > 0 else "")
                      + "; decision at the end of its 1 s window)", fontsize=8, loc="left", color=MUTED)
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
        self._start_head(len(x) / E.SR)                      # drawn first, so drawing does not delay the sound
        self._sound = Sound(np.clip(x * g, -1, 1))           # the line waits for the sound's own clock (Sound.elapsed)

    def _start_head(self, seconds: float):
        """A vertical line across the log-mel and the graphs that follows the sound as it plays. Only
        the line is redrawn (blitting over a saved background), so it moves smoothly."""
        self._head = [ax.axvline(0, color="#00e5ff" if i == 0 else "#111111", lw=1.6, alpha=0.9, zorder=10,
                                 animated=True) for i, ax in enumerate(self.axes)]
        self.canvas.draw()                                   # the figure without the (animated) line
        self._head_bg = self.canvas.copy_from_bbox(self.fig.bbox)
        self._head_cid = self.canvas.mpl_connect("draw_event", self._new_head_bg)   # resized: save it again
        self._head_len = seconds
        self._head_job = self.after(10, self._tick_head)

    def _new_head_bg(self, _):
        self._head_bg = self.canvas.copy_from_bbox(self.fig.bbox)

    def _tick_head(self):
        if not self._head:
            return
        snd = self._sound
        t = snd.elapsed() if snd else None
        if t is None:                                        # the sound has not reached the speaker yet: no line
            if snd and snd.active():
                self._head_job = self.after(10, self._tick_head)
            else:
                self._remove_head()
            return
        if t > self._head_len or (not snd.active() and t > self._head_len - 0.5):   # done, even if the clock stops
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
        if self._sound:
            self._sound.stop()
            self._sound = None
        self._remove_head()


# ------------------------------------------------------------------ conclusions

CURVE_ABOVE = 4                                              # more values than this: a curve, not bars
COMPARE = "comparison"                                       # the Conclusions choice that draws every detector together
SPARE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SUMMARY_LAYERS = (("War soundscape summary", ("war soundscape",)), ("Wind summary", ("wind", "wind (Pd)")),
                  ("Car summary", ("cars",)))
TOP_IN, HEAD_IN, ROW_IN, NOTE_IN = 0.7, 0.55, 3.1, 0.4       # heights on the Conclusions page, inches


def parameter_values(test: E.Test) -> dict[str, list]:
    """Every setting that takes more than one value in the test, with its values in order."""
    modified = test.plan.kind == "modified"
    names = {**({"speed_kmh": "speed (km/h), all drones", "file": "drone file, all speeds",   # not per file x speed
                 "pitch": "drone pitch (x), all drones"} if modified
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
    return E.drone_setting(r["drone"], key) if key in ("drone", "file", "speed_kmh", "pitch") else r[key]


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


def fa_threshold(test: E.Test, det: str, scores: dict, pct: float) -> tuple[float | None, int]:
    """Up to 100: the confidence that `pct` % of `det`'s windows in the no-drone rounds lie below (all
    backgrounds together; each 1 s window one value, comb squashed as drawn elsewhere); 100 is the highest
    one. Above 100: that highest one times pct / 100, so 101 is just above every no-drone window (a
    threshold over 1 can never be reached). -> (threshold, windows)."""
    vals = [scores[f"{rnd}|{det}"] for rnd in test.results
            if test.by_round[rnd]["drone"] is None and f"{rnd}|{det}" in scores]
    if not vals:
        return None, 0
    v = np.concatenate(vals)
    v = v[~np.isnan(v)]
    if det == "comb":
        v = E.dwg.squash(v, test.thr[det])
    if not v.size:
        return None, 0
    return (float(np.percentile(v, pct)) if pct <= 100 else float(v.max()) * pct / 100), int(v.size)


def round_curves(test: E.Test, det: str, scores: dict) -> dict[int, tuple]:
    """Modified drone tests: per drone round, `det`'s confidence in each window at least half drone and no
    later than impact, keyed by its time before impact in hops. -> rnd -> (hops before impact, confidence,
    index into the score grid); comb squashed as drawn elsewhere, NaN hops left out."""
    hop, base = DEFAULT.samples_per_hop / E.SR, DEFAULT.window_seconds
    win = base if det == "comb" else E.dwg.model_cfg(det).window_seconds
    out = {}
    for rnd in test.results:
        r = test.by_round[rnd]
        s = scores.get(f"{rnd}|{det}")
        if r["drone"] is None or s is None:
            continue
        s0, _, imp = test.drones.span[r["drone"]]
        if imp is None:
            continue
        y = E.dwg.squash(s, test.thr[det]) if det == "comb" else s
        t_end = np.arange(len(y)) * hop + base               # on_grid: every detector on the 1 s window's time axis
        impact = r["start_s"] + imp
        idx = np.flatnonzero((t_end - win / 2 >= r["start_s"] + s0) & (t_end <= impact + 1e-9) & np.isfinite(y))
        out[rnd] = (np.round((impact - t_end[idx]) / hop).astype(int), y[idx], idx)
    return out


def curve_lead(curves: dict, rounds: list, thr: float) -> dict:
    """The time before impact of a group of rounds (Mads, 2026-11-11; replaces the per-round rules of entries
    213 and 220): their mean confidence at each moment before impact, as in the "confidence over time"
    graphs, and the first moment it is at or above thr. A moment counts only when at least half the
    rounds have a window there. -> dict(lead s or None, rounds, k = that moment in hops before impact)."""
    sums: dict[int, list] = {}
    rs = [r for r in rounds if r in curves]
    for r in rs:
        for k, v in zip(curves[r][0], curves[r][1]):
            a = sums.setdefault(int(k), [0.0, 0])
            a[0] += float(v)
            a[1] += 1
    hop = DEFAULT.samples_per_hop / E.SR
    for k in sorted(sums, reverse=True):                     # earliest moment first
        tot, n = sums[k]
        if n * 2 >= len(rs) and tot / n >= thr:
            return dict(lead=k * hop, rounds=len(rs), k=k)
    return dict(lead=None, rounds=len(rs), k=None)


SNR_BIN_DB = 2.0                                             # width of an SNR bin
SNR_MIN_WINDOWS = 10                                         # a bin with fewer windows is not drawn


def window_snrs(test: E.Test, scores: dict) -> dict[str, np.ndarray]:
    """round|window -> the SNR (dB) of every score-grid window of a drone round (E.window_snr): read from
    window_snr.npz, or, for a run made before it was stored, measured once by rebuilding each drone round
    from its parts and saved there."""
    p = test.out / "window_snr.npz"
    if p.exists():
        with np.load(p) as d:
            return {k: d[k] for k in d.files}
    out = {}
    for rnd in test.results:
        r = test.by_round[rnd]
        n = max((len(v) for k, v in scores.items() if k.split("|")[0] == str(rnd)), default=0)
        if r["drone"] is None or not n:
            continue
        for w, x in E.round_window_snrs(test.rebuild(r), n).items():
            out[f"{rnd}|{w}"] = x
    if out:
        np.savez_compressed(p, **out)
    return out


def det_window(det: str) -> float:
    return DEFAULT.window_seconds if det == "comb" else E.dwg.model_cfg(det).window_seconds


def snr_pairs(test: E.Test, det: str, scores: dict, snrs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Every drone window of every drone round: (its SNR in dB, `det`'s confidence there), with the SNR over
    the detector's own window (1 s or 2 s); comb squashed as drawn elsewhere."""
    w = f"{det_window(det):g}"
    xs, ys = [], []
    for rnd in test.results:
        s, q = scores.get(f"{rnd}|{det}"), snrs.get(f"{rnd}|{w}")
        if s is None or q is None:
            continue
        y = E.dwg.squash(s, test.thr[det]) if det == "comb" else s
        n = min(len(y), len(q))
        ok = np.isfinite(q[:n]) & np.isfinite(y[:n])
        xs.append(q[:n][ok])
        ys.append(y[:n][ok])
    return (np.concatenate(xs), np.concatenate(ys)) if xs else (np.array([]), np.array([]))


def snr_bins(x: np.ndarray, y: np.ndarray, thr: float) -> dict:
    """Windows grouped by SNR in SNR_BIN_DB bins (centres): mean confidence, its middle half, the share at or
    above thr, and the window count; bins under SNR_MIN_WINDOWS left out."""
    k = np.floor(x / SNR_BIN_DB).astype(int)
    rows = []
    for b in np.unique(k):
        v = y[k == b]
        if v.size >= SNR_MIN_WINDOWS:
            rows.append(((b + 0.5) * SNR_BIN_DB, v.mean(), np.percentile(v, 25), np.percentile(v, 75), np.mean(v >= thr), v.size))
    a = np.array(rows).reshape(-1, 6)
    return dict(x=a[:, 0], mean=a[:, 1], p25=a[:, 2], p75=a[:, 3], share=a[:, 4], n=a[:, 5].astype(int))


def crossing(x: np.ndarray, y: np.ndarray, level: float) -> float | None:
    """The first x where y reaches level, interpolated between bins; None if it never does (or starts above)."""
    above = np.flatnonzero(y >= level)
    if not above.size or above[0] == 0:
        return None
    i = above[0]
    return float(x[i - 1] + (level - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))


def snr_over_time(test: E.Test, snrs: dict) -> dict[float, dict]:
    """Modified drone tests: per speed, the SNR (1 s windows) at each moment of the drone file, impact at
    10 s, over every drone round at that speed: median and middle half. -> speed -> dict(x, med, p25, p75, n)."""
    if test.plan.kind != "modified" or test.drones is None:
        return {}
    hop = DEFAULT.samples_per_hop / E.SR
    bins: dict[float, dict[int, list]] = {}
    for rnd in test.results:
        r = test.by_round[rnd]
        q = snrs.get(f"{rnd}|1")
        if r["drone"] is None or q is None or test.drones.span[r["drone"]][2] is None:
            continue
        impact = r["start_s"] + test.drones.span[r["drone"]][2]
        x = E.DRONE_S - (impact - E.grid_t_end(len(q)))
        b = bins.setdefault(E.split_key(r["drone"])[1], {})
        for k, v in zip(np.round(x / hop).astype(int), q):
            if np.isfinite(v):
                b.setdefault(int(k), []).append(float(v))
    out = {}
    for speed, b in sorted(bins.items()):
        ks = sorted(k for k in b if -2 / hop <= k <= (E.DRONE_S + 1) / hop)
        vals = [np.asarray(b[k]) for k in ks]
        out[speed] = dict(x=np.array(ks) * hop, med=np.array([np.median(v) for v in vals]),
                          p25=np.array([np.percentile(v, 25) for v in vals]),
                          p75=np.array([np.percentile(v, 75) for v in vals]), n=np.array([len(v) for v in vals]))
    return out


def snr_at_alarm(test: E.Test, det: str, scores: dict, snrs: dict, thr: float) -> list[float]:
    """Modified drone tests: the moment all drone rounds' mean confidence first reaches thr (curve_lead),
    and every round's window SNR at that moment, over the detector's own window."""
    curves = round_curves(test, det, scores)
    k = curve_lead(curves, list(curves), thr)["k"]
    if k is None:
        return []
    out = []
    for rnd, (ks, _, idx) in curves.items():
        q = snrs.get(f"{rnd}|{det_window(det):g}")
        j = np.flatnonzero(ks == k)
        if q is not None and j.size and idx[j[0]] < len(q) and np.isfinite(q[idx[j[0]]]):
            out.append(float(q[idx[j[0]]]))
    return out


class Conclusions(tk.Toplevel):
    """A scrollable page: Overall (the drone-to-layer dB ratio), summaries of speed, war soundscape, wind
    and cars, then any other setting that varies (summary_sections). Each graph is a detector's mean confidence on drone windows (true) and on
    no-drone windows (false) for each value of the setting, every round at that value pooled. Bars up to
    CURVE_ABOVE values, a curve above that for numbers. No threshold lines; a fixed dashed 50 % mark on
    every graph instead (Mads, 2026-11-06). Modified drone tests also get, in the speed summary, one
    "mean confidence over time" graph per speed: all rounds at that speed together, the mean confidence
    at each moment of the drone file (impact at 10 s), see confidence_over_time(). With more than one
    detector, the choice "comparison" draws them all together on two questions: confidence before impact
    and confidence during distractors (see _draw_compare)."""

    def __init__(self, master, test: E.Test):
        super().__init__(master)
        self.test = test
        test.by_round = {r["round"]: r for r in test.rounds}
        self.title(f"{TITLE} conclusions: {test.out.name}" + (FINAL_TAG if test.plan.final_evaluation else ""))
        self.geometry("1300x900")
        bar = ttk.Frame(self, padding=(12, 8))
        bar.pack(fill="x")
        ttk.Label(bar, text="Conclusions", style="H2.TLabel").pack(side="left")
        ttk.Label(bar, text="detector").pack(side="left", padx=(20, 4))
        self.v_det = tk.StringVar(value=test.plan.detectors[0])
        choices = [*test.plan.detectors, *([COMPARE] if len(test.plan.detectors) > 1 else [])]
        ttk.Combobox(bar, textvariable=self.v_det, values=choices, state="readonly", width=14).pack(side="left")
        self.scores = {}                                     # per-hop scores: over time, false alarms, window SNR
        p = test.out / "scores.npz"
        with np.load(p) if p.exists() else nullcontext({}) as d:
            self.scores = {k: d[k] for k in getattr(d, "files", [])}
        self._snr = None                                     # window_snrs(), on first use
        ttk.Button(bar, text="Save PNG", command=self.save).pack(side="right")
        ttk.Label(bar, text="false alarms").pack(side="left", padx=(20, 4))   # the threshold's percentile, from the plan
        self.v_fa = tk.StringVar(value=f"{getattr(test.plan, 'fa_percentile', 95.0):g}")
        e_fa = ttk.Entry(bar, textvariable=self.v_fa, width=5)
        e_fa.pack(side="left")
        ttk.Label(bar, text="%  (Enter to apply)", style="Muted.TLabel").pack(side="left", padx=(2, 0))
        e_fa.bind("<Return>", lambda e: self.draw())
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
        if det == COMPARE:
            return self._draw_compare()
        thr = 0.5 if det == "comb" else self.test.thr[det]
        curves = confidence_over_time(self.test, det, self.scores)
        secs = [self._fa_section([det]), self._snr_section([det]), *summary_sections(self.test)]
        for s in secs:
            if "fa" in s or "snr" in s:                      # cells set by _fa_section / _snr_section
                s["rows"] = (s["cells"] + 1) // 2
                continue
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
        self.fig.text(0.012, 1 - 0.54 / height, ("THE FINAL EVALUATION, sealed car.  " if self.test.plan.final_evaluation else "")
                      + "Each bar or point pools every round at that value, over all the other settings.",
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
            if "fa" in s or "snr" in s:
                lines += self._draw_fa(axes, s["fa"]) if "fa" in s else self._draw_snr(axes, s["snr"])
                y += s["rows"] * ROW_IN
                continue
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

    def _color(self, det: str) -> str:
        """The comparison page's colour: the validated categorical order by the detector's place in the plan
        (COLORS has near-identical purples for it9 CRNN and the it17 CRNNs, which cannot be told apart)."""
        return SPARE_COLORS[self.test.plan.detectors.index(det) % len(SPARE_COLORS)]

    def _fa_section(self, dets: list) -> dict:
        """The top section: from the false-alarm % (front page, or the box here), each detector's threshold =
        that percentile of its no-drone windows, then the time before impact at that threshold for all
        rounds and for the rounds with each background layer on, and per level of each layer that varies."""
        t, title = self.test, "False alarms: the threshold, and the time before impact at it"
        try:
            pct = fa_pct(self.v_fa.get())
        except ValueError as e:
            return dict(title=title, cells=0, note=str(e), fa=None)
        if t.plan.kind != "modified":
            return dict(title=title, cells=0, note="Time before impact needs a modified drone test: only it has an impact.", fa=None)
        if not any(t.by_round[r]["drone"] is None for r in t.results):
            return dict(title=title, cells=0, fa=None,
                        note="No no-drone rounds in this run: tick \"add one no-drone round per background\" to get a threshold.")
        if not self.scores:
            return dict(title=title, cells=0, note="scores.npz is missing from the run folder.", fa=None)
        per = {}
        for d in dets:
            thr, nwin = fa_threshold(t, d, self.scores, pct)
            per[d] = dict(thr=thr, nwin=nwin, curves=round_curves(t, d, self.scores) if thr is not None else {})
        drone_rounds = [rnd for rnd in t.results if t.by_round[rnd]["drone"] is not None]
        groups = [("all rounds", drone_rounds)]
        for n in t.plan.layers:
            on = [rnd for rnd in drone_rounds if setting(t.by_round[rnd], n) not in (None, "off")]
            if on and len(on) < len(drone_rounds):           # a layer on in every round is just "all rounds"
                on_set = set(on)
                groups.append((f"{n} on", on))
                groups.append((f"{n} off", [rnd for rnd in drone_rounds if rnd not in on_set]))
        params = []                                          # every setting that varies: speed, file, levels, Pd settings...
        for key, (label, vals) in parameter_values(t).items():
            if key in t.plan.layers and len([v for v in vals if v != "off"]) < 2:
                continue                                     # only on/off: already a bar on the left
            params.append((key, label))
        params += [(k, PD_LABELS[k]) for k in t.plan.pd_combos()[0] if len(values_of(t, k)) > 1]
        fa = dict(pct=pct, per=per, groups=groups, params=params, dets=dets)
        return dict(title=title, cells=1 + len(params), note="", fa=fa)

    def _snr_section(self, dets: list) -> dict:
        """Window SNR: the drone's power against the background's in the same seconds, from the separate
        parts. Two views: continuous (confidence and detection against each window's SNR, all windows
        pooled) and, for modified tests, the SNR over the time before impact with the SNR at first alarm."""
        t, title = self.test, "SNR: the drone against the simultaneous background, window by window"
        if not self.scores:
            return dict(title=title, cells=0, note="scores.npz is missing from the run folder.", snr=None)
        if self._snr is None:
            self._snr = window_snrs(t, self.scores)
        if not self._snr:
            return dict(title=title, cells=0, snr=None, note="No drone round with a background: no SNR to measure.")
        try:
            pct = fa_pct(self.v_fa.get())
        except ValueError:
            pct = None
        has_nd = any(t.by_round[r]["drone"] is None for r in t.results)
        per = {}
        for d in dets:
            thr, how = None, ""
            if pct is not None and has_nd:
                thr = fa_threshold(t, d, self.scores, pct)[0]
                how = f"{pct:g} % false-alarm threshold"
            if thr is None:
                thr, how = (0.5 if d == "comb" else t.thr[d]), "its own threshold"
            x, y = snr_pairs(t, d, self.scores, self._snr)
            per[d] = dict(thr=thr, how=how, bins=snr_bins(x, y, thr) if x.size else None, n=int(x.size),
                          first=snr_at_alarm(t, d, self.scores, self._snr, thr) if t.plan.kind == "modified" else [])
        over = snr_over_time(t, self._snr)
        return dict(title=title, cells=2 + (1 if over else 0), note="", snr=dict(per=per, over=over, dets=dets))

    def _draw_snr(self, axes, sn: dict) -> list[str]:
        dets = sn["dets"]
        col = (lambda d: self._color(d)) if len(dets) > 1 else (lambda d: COLORS.get(d, "#2a78d6"))
        ax_c, ax_s = axes[0], axes[1]
        lines = ["SNR = 10 log10(drone power / background power) over the same window (1 s, or the detector's 2 s), "
                 "from the round's separate drone and background parts; only windows whose middle is inside the drone's sound."]
        for d in dets:
            p = sn["per"][d]
            b = p["bins"]
            if b is None:
                lines.append(f"SNR · {d}: no drone windows with a background")
                continue
            if len(dets) == 1:
                ax_c.fill_between(b["x"], 100 * b["p25"], 100 * b["p75"], color=col(d), alpha=0.18, lw=0, label="middle half")
            ax_c.plot(b["x"], 100 * b["mean"], "-o", color=col(d), lw=2, ms=3, label=d)
            ax_s.plot(b["x"], 100 * b["share"], "-o", color=col(d), lw=2, ms=3,
                      label=f"{d}: threshold {100 * p['thr']:.0f} % ({p['how']})")
            c50, s50 = crossing(b["x"], b["mean"], 0.5), crossing(b["x"], b["share"], 0.5)
            txt = (f"SNR · {d} ({p['n']} drone windows): mean confidence reaches 50 % at "
                   + ("-" if c50 is None else f"{c50:+.1f} dB") + ";  half the windows at or above the threshold "
                   f"({100 * p['thr']:.0f} %, {p['how']}) at " + ("-" if s50 is None else f"{s50:+.1f} dB"))
            if p["first"]:
                f = np.array(p["first"])
                txt += (f";  SNR when the mean confidence first reaches the threshold ({f.size} rounds): median {np.median(f):+.1f} dB, "
                        f"middle half {np.percentile(f, 25):+.1f} to {np.percentile(f, 75):+.1f} dB")
            lines.append(txt)
        if len(dets) == 1 and sn["per"][dets[0]]["bins"] is not None:   # how many windows lie behind each point
            b = sn["per"][dets[0]]["bins"]
            tw = ax_s.twinx()
            tw.bar(b["x"], b["n"], SNR_BIN_DB * 0.9, color="#bbb", alpha=0.35, zorder=0)
            tw.set_ylabel("windows per bin", fontsize=8, color="#888")
            tw.tick_params(labelsize=7, colors="#888")
            ax_s.set_zorder(tw.get_zorder() + 1)
            ax_s.patch.set_visible(False)
        for ax in (ax_c, ax_s):
            ax.axhline(50, color="#888", ls="--", lw=1)
            ax.axvline(0, color="#999", lw=0.8, ls=":")
            ax.set_xlabel(f"SNR in the window (dB; {SNR_BIN_DB:g} dB bins, at least {SNR_MIN_WINDOWS} windows; 0 = drone as loud as the background)", fontsize=8)
            ax.set_ylim(0, 105)
            ax.grid(color="#e8e8e8", lw=0.5)
        ax_c.set_ylabel("mean confidence (%)")
        ax_c.set_title("continuous: confidence against the window's SNR", fontsize=10, loc="left", weight="bold")
        ax_c.legend(loc="upper left", fontsize=8, frameon=False)
        ax_s.set_ylabel("drone windows at or above threshold (%)")
        ax_s.set_title("continuous: detected windows against the window's SNR", fontsize=10, loc="left", weight="bold")
        ax_s.legend(loc="upper left", fontsize=7, frameon=False)
        if sn["over"]:
            ax = axes[2]
            for k, (speed, c) in enumerate(sn["over"].items()):
                cc = SPARE_COLORS[k % len(SPARE_COLORS)]
                ax.fill_between(c["x"], c["p25"], c["p75"], color=cc, alpha=0.12, lw=0)
                ax.plot(c["x"], c["med"], color=cc, lw=2, label=f"{speed:g} km/h")
                at = []
                for before in (1, 3, 5, 7):
                    i = np.flatnonzero(np.isclose(c["x"], E.DRONE_S - before))
                    if i.size:
                        at.append(f"{before} s before {c['med'][i[0]]:+.1f} dB")
                lines.append(f"SNR before impact · {speed:g} km/h ({int(c['n'].max())} rounds), median: " + ",  ".join(at))
            ax.axvspan(E.DRONE_S - 7, E.DRONE_S - 3, color="#888", alpha=0.1, lw=0)
            ax.axvline(E.DRONE_S, color="#d93025", lw=1.2)
            ax.axhline(0, color="#999", lw=0.8, ls=":")
            ax.set(xlim=(-2, E.DRONE_S + 1), ylabel="SNR (dB), 1 s windows")
            ax.set_xticks(range(-2, int(E.DRONE_S) + 2))
            ax.set_xlabel("time in the drone file (s); impact at 10 s; shaded 3-7 s before impact; band = middle half", fontsize=8)
            ax.set_title("SNR over the time before impact (median over rounds, per speed)", fontsize=10, loc="left", weight="bold")
            ax.grid(color="#e8e8e8", lw=0.5)
            ax.legend(loc="upper left", fontsize=8, frameon=False)
        return lines

    def _draw_fa(self, axes, fa: dict) -> list[str]:
        t, dets, pct = self.test, fa["dets"], fa["pct"]
        col = (lambda d: self._color(d)) if len(dets) > 1 else (lambda d: COLORS.get(d, "#2a78d6"))
        lines = []
        ax = axes[0]
        names = [g for g, _ in fa["groups"]]
        w = 0.8 / max(len(dets), 1)
        for j, d in enumerate(dets):
            p = fa["per"][d]
            if p["thr"] is None:
                lines.append(f"False alarms {pct:g} % · {d}: no no-drone windows, no threshold")
                continue
            how = (f"{pct:g} % of its {p['nwin']} no-drone windows lie below it" if pct < 100 else
                   f"the highest of its {p['nwin']} no-drone windows" if pct == 100 else
                   f"{pct:g} % of the highest of its {p['nwin']} no-drone windows: above every one"
                   + ("; over 100 % confidence, so never reached" if p["thr"] > 1 else ""))
            lines.append(f"False alarms {pct:g} % · {d}: threshold {100 * p['thr']:.1f} % confidence ({how})")
            meds = []
            for g, rnds in fa["groups"]:
                s = curve_lead(p["curves"], rnds, p["thr"])
                meds.append(0.0 if s["rounds"] and s["lead"] is None else s["lead"])
                if s["rounds"]:
                    lines.append(f"    {g}: the mean confidence of its {s['rounds']} rounds first reaches the threshold "
                                 + ("never before impact (0 s)" if s["lead"] is None else f"{s['lead']:.2f} s before impact"))
            xs = np.arange(len(names)) + (j - (len(dets) - 1) / 2) * w
            ax.bar(xs, [0 if m is None else m for m in meds], w * 0.9, color=col(d),
                   label=f"{d}: threshold {100 * p['thr']:.0f} %")
            for x, m in zip(xs, meds):
                if m is not None:
                    ax.text(x, m + 0.15, f"{m:.1f}", ha="center", fontsize=7)
        ax.set_xticks(range(len(names)), names, fontsize=8)
        ax.set(ylim=(0, E.DRONE_S + 1), ylabel="seconds before impact")
        ax.set_title(f"when the mean confidence first reaches the {pct:g} % false-alarm threshold (s before impact)",
                     fontsize=10, loc="left", weight="bold")
        ax.axhline(3, color="#888", ls="--", lw=1)
        ax.grid(axis="y", color="#e8e8e8", lw=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        for ax, (key, label) in zip(axes[1:], fa["params"]):
            vals = values_of(t, key)
            nums = [v for v in vals if v != "off"]
            numeric = all(isinstance(v, (int, float)) for v in nums)
            xs = nums if numeric else list(range(len(nums)))  # a drone file is a name: one place per file
            for d in dets:
                p = fa["per"][d]
                if p["thr"] is None:
                    continue
                by = {}
                for rnd in p["curves"]:
                    by.setdefault(setting(t.by_round[rnd], key), []).append(rnd)

                def lead(v):
                    s = curve_lead(p["curves"], by.get(v, []), p["thr"])
                    return None if not s["rounds"] else 0.0 if s["lead"] is None else s["lead"]
                mean = [lead(v) for v in nums]
                ax.plot(xs, [np.nan if m is None else m for m in mean], "-o", color=col(d), lw=2, ms=4, label=d)
                off = lead("off") if "off" in by else None
                if off is not None:
                    ax.axhline(off, color=col(d), ls=":", lw=1.2)
                lines.append(f"    {label} · {d}: s before impact " + ("" if off is None else f"off {off:.2f};  ")
                             + ";  ".join(f"{fmt(v)} {m:.2f}" for v, m in zip(nums, mean) if m is not None))
            if not numeric:
                ax.set_xticks(xs, [v if len(v) <= 14 else v[:13] + "~" for v in map(str, nums)], rotation=20, fontsize=8)
            ax.set(ylim=(0, E.DRONE_S + 1), ylabel="seconds before impact")
            ax.set_xlabel(label + ("; dotted = layer off" if "off" in vals else ""))
            ax.set_title(f"{label}: time before impact at the threshold", fontsize=10, loc="left", weight="bold")
            ax.axhline(3, color="#888", ls="--", lw=1)
            ax.grid(color="#e8e8e8", lw=0.5)
            if len(dets) > 1 and ax is axes[1]:
                ax.legend(loc="lower left", fontsize=7, frameon=False)
        return lines

    def _draw_compare(self):
        """The "comparison" page: every detector together, on two questions only.
        1. Confidence before impact (modified drone tests): each detector's mean confidence over time, all
           speeds pooled (each speed's rounds weighted by their count), with the 3-7 s before impact shaded;
           and the test 2 figure (mean over those 4 s, every 0.125 s bin counted equally) per speed.
        2. Confidence during distractors: confidence on no-drone windows (lower is better), against each
           background layer's level, and by which distractors are on (every layer but the war soundscape).
        One pass over the rounds fills every sum."""
        t, dets = self.test, list(self.test.plan.detectors)
        layers = [n for n in t.plan.layers if len(values_of(t, n)) > 1]
        dist = [n for n in t.plan.layers if n != "war soundscape"]
        sums, groups = {}, {}
        for rnd, res in t.results.items():
            r = t.by_round[rnd]
            on = [n for n in dist if setting(r, n) not in (None, "off")]
            g = " + ".join(on) if on else "no distractor"
            for d in dets:
                if d not in res:
                    continue
                for n in layers:
                    add_conf(sums.setdefault((d, n, setting(r, n)), new_sums()), res[d])
                add_conf(groups.setdefault((d, g), new_sums()), res[d])
        gnames = sorted({g for _, g in groups}, key=lambda g: (g != "no distractor", g.count("+"), g))
        curves = {d: confidence_over_time(t, d, self.scores) for d in dets}
        has_time = any(curves.values())

        secs = [self._fa_section(dets), self._snr_section(dets),
                dict(title="Confidence before impact", cells=2 if has_time else 0,
                     note="Only a modified drone test has an impact; this test has no confidence over time."),
                dict(title="Confidence during distractors: no-drone windows (lower is better)", cells=len(layers) + 1, note="")]
        for s in secs:
            s["rows"] = (s["cells"] + 1) // 2
        height = TOP_IN + sum(HEAD_IN + (s["rows"] * ROW_IN if s["cells"] else NOTE_IN) for s in secs)
        self.fig.clear()
        self.fig.set_size_inches(self.fig.get_figwidth(), height, forward=False)
        self.canvas.get_tk_widget().configure(height=int(height * self.fig.dpi))
        self.fig.text(0.012, 1 - 0.28 / height, "Comparison: " + ", ".join(d if d != "comb" else "DSP (comb)" for d in dets),
                      fontsize=12, weight="bold", va="center")
        self.fig.text(0.012, 1 - 0.54 / height, ("THE FINAL EVALUATION, sealed car.  " if t.plan.final_evaluation else "")
                      + "Each point pools every round at that value, over all the other settings.", fontsize=9, color=MUTED, va="center")
        lines, y, axes_of = [], TOP_IN, []
        for s in secs:
            top = 1 - y / height
            self.fig.add_artist(Line2D([0.012, 0.988], [top, top], color="#c9ced6", lw=1, transform=self.fig.transFigure))
            self.fig.text(0.012, 1 - (y + 0.3) / height, s["title"], fontsize=13, weight="bold", va="center")
            y += HEAD_IN
            if not s["cells"]:
                self.fig.text(0.012, 1 - (y + 0.14) / height, s["note"], fontsize=10, color=MUTED, va="center")
                y += NOTE_IN
                axes_of.append([])
                continue
            axes_of.append([self._axes(i, s["cells"], y, height) for i in range(s["cells"])])
            y += s["rows"] * ROW_IN
        if axes_of[0]:
            lines += self._draw_fa(axes_of[0], secs[0]["fa"])
        if axes_of[1]:
            lines += self._draw_snr(axes_of[1], secs[1]["snr"])
        if has_time:
            lines += self._compare_impact(*axes_of[2], curves)
        lines += self._compare_distractors(axes_of[3], layers, sums, groups, gnames)
        self.canvas.draw_idle()
        self.l_text.configure(text="\n".join(lines))
        self.scroll.yview_moveto(0)

    def _compare_impact(self, ax_t, ax_s, curves: dict) -> list[str]:
        lo, hi = E.DRONE_S - 7, E.DRONE_S - 3
        in_win = (lambda x: (x >= lo - 1e-9) & (x <= hi + 1e-9))
        ax_t.axvspan(lo, hi, color="#888", alpha=0.12, lw=0)
        ax_t.text((lo + hi) / 2, 102, "3-7 s before impact", ha="center", va="bottom", fontsize=8, color=MUTED)
        ax_t.axvline(E.DRONE_S, color="#d93025", lw=1.2)
        ax_t.text(E.DRONE_S, 102, " impact", color="#d93025", fontsize=8, va="bottom")
        lines = []
        for d, per in curves.items():
            if not per:
                continue
            pool = {}
            for c in per.values():
                for x, m, n in zip(c["x"], c["mean"], c["n"]):
                    a = pool.setdefault(round(float(x), 4), [0.0, 0])
                    a[0] += m * n
                    a[1] += n
            xs = np.array(sorted(pool))
            ys = np.array([pool[x][0] / pool[x][1] for x in xs])
            fig2 = 100 * ys[in_win(xs)].mean()
            ax_t.plot(xs, 100 * ys, color=self._color(d), lw=2, label=f"{d}: {fig2:.1f}%")
            speeds = sorted(per)
            f_sp = [100 * per[s]["mean"][in_win(per[s]["x"])].mean() for s in speeds]
            ax_s.plot(speeds, f_sp, "-o", color=self._color(d), lw=2, ms=4, label=d)
            lines.append(f"Before impact · {d}: mean confidence 3-7 s before impact {fig2:.1f}% (all speeds);  per speed "
                         + ",  ".join(f"{s:g} km/h {v:.1f}%" for s, v in zip(speeds, f_sp)))
        ax_t.axhline(50, color="#888", ls="--", lw=1)
        ax_t.set(ylim=(0, 108), xlim=(-2, E.DRONE_S + 1), ylabel="mean confidence (%)")
        ax_t.set_xticks(range(-2, int(E.DRONE_S) + 2))
        ax_t.set_xlabel("time in the drone file (s); impact at 10 s", fontsize=8)
        ax_t.set_title("mean confidence over time, all speeds", fontsize=10, loc="left", weight="bold")
        ax_t.grid(color="#e8e8e8", lw=0.5)
        ax_t.legend(loc="upper left", fontsize=8, frameon=False, title="mean 3-7 s before impact", title_fontsize=8)
        ax_s.axhline(50, color="#888", ls="--", lw=1)
        ax_s.set(ylim=(0, 105), ylabel="mean confidence 3-7 s before impact (%)")
        ax_s.set_xlabel("speed (km/h)")
        ax_s.set_title("test 2 figure per speed", fontsize=10, loc="left", weight="bold")
        ax_s.grid(color="#e8e8e8", lw=0.5)
        ax_s.legend(loc="lower left", fontsize=8, frameon=False)
        return lines

    def _compare_distractors(self, axes, layers: list, sums: dict, groups: dict, gnames: list) -> list[str]:
        dets = list(self.test.plan.detectors)
        false = (lambda s: None if s is None or not s["nf"] else 100 * s["false"] / s["nf"])
        lines = []
        for ax, n in zip(axes, layers):
            vals = values_of(self.test, n)
            nums = [v for v in vals if v != "off"]
            for d in dets:
                col = self._color(d)
                ys = [false(sums.get((d, n, v))) for v in nums]
                ax.plot(nums, [np.nan if v is None else v for v in ys], "-o", color=col, lw=2, ms=4, label=d)
                off = false(sums.get((d, n, "off"))) if "off" in vals else None
                if off is not None:                          # dotted: the same detector with this layer off
                    ax.axhline(off, color=col, ls=":", lw=1.2)
                lines.append(f"Distractors · {n} · {d}: no-drone confidence " + ("" if off is None else f"off {off:.0f}%;  ")
                             + ";  ".join(f"{fmt(v)} dB {y:.0f}%" for v, y in zip(nums, ys) if y is not None))
            ax.set(ylim=(0, 105), ylabel="no-drone confidence (%)")
            ax.set_xlabel(f"{n} level (dB SPL)" + ("; dotted = layer off" if "off" in vals else ""))
            ax.set_title(f"{n}", fontsize=10, loc="left", weight="bold")
            ax.axhline(50, color="#888", ls="--", lw=1)
            ax.grid(color="#e8e8e8", lw=0.5)
        ax = axes[len(layers)]
        w = 0.8 / max(len(dets), 1)
        for j, d in enumerate(dets):
            ys = [false(groups.get((d, g))) for g in gnames]
            xs = np.arange(len(gnames)) + (j - (len(dets) - 1) / 2) * w
            ax.bar(xs, [0 if v is None else v for v in ys], w * 0.9, color=self._color(d), label=d)
            for x, v in zip(xs, ys):
                if v is not None:
                    ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=7)
            lines.append(f"Distractors · by what is on · {d}: " + ";  ".join(
                f"{g} {v:.0f}%" for g, v in zip(gnames, ys) if v is not None))
        ax.set_xticks(range(len(gnames)), gnames, fontsize=8)
        ax.set(ylim=(0, 105), ylabel="no-drone confidence (%)")
        ax.set_title("by which distractors are on (war soundscape as set)", fontsize=10, loc="left", weight="bold")
        ax.axhline(50, color="#888", ls="--", lw=1)
        ax.grid(axis="y", color="#e8e8e8", lw=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
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


class Sound:
    """Plays x (16 kHz, -1..1) through sounddevice and says how far the listener has got, read from the
    audio device's own clock: elapsed() is None until the first sample reaches the speaker, then the
    seconds since. The playhead follows that, so it neither runs ahead of a slow device start-up (winsound,
    used before, gave no start time) nor drifts."""

    def __init__(self, x: np.ndarray):
        import sounddevice as sd
        self._stop = sd.CallbackStop
        self.x, self.pos, self.dac0 = np.ascontiguousarray(x, dtype=np.float32), 0, None
        self.stream = sd.OutputStream(samplerate=E.SR, channels=1, dtype="float32", callback=self._fill)
        self.stream.start()

    def _fill(self, out, frames, ti, status):
        if self.dac0 is None:                                # when this first buffer will be heard, on the stream's clock
            t = ti.outputBufferDacTime                       # some Windows host APIs report 0 here
            self.dac0 = t if t and t >= ti.currentTime > 0 else self.stream.time + self.stream.latency
        chunk = self.x[self.pos:self.pos + frames]
        out[:len(chunk), 0] = chunk
        out[len(chunk):] = 0
        self.pos += len(chunk)
        if len(chunk) < frames:
            raise self._stop

    def elapsed(self) -> float | None:
        if self.dac0 is None:
            return None
        t = self.stream.time - self.dac0
        return t if t >= 0 else None

    def active(self) -> bool:
        return bool(self.stream.active)

    def stop(self):
        self.stream.abort()
        self.stream.close()


def load_test(folder: Path) -> E.Test:
    """An earlier run from its folder: plan and start from config.json, results from rounds.csv."""
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    p = cfg["plan"]
    plan = E.Plan(**{k: p[k] for k in ("kind", "length_s", "drones", "drone_levels", "layers", "detectors", "step_s",
                                       "no_drone_rounds", "soundscape_start_s", "synth")},
                  **{k: p[k] for k in ("speeds", "approach", "pitches", "pitch_random", "pd_settings", "also_without", "final_evaluation", "car_engines", "car_once", "fa_percentile", "live") if k in p},   # not in every run's config
                  war_file=p.get("war_file", str(E.WAR_TRACK)),   # a run from before WarSoundScape.wav used the 600 s track
                  buffer_after_s=p.get("buffer_after_s", 0.0))    # a run from before the buffer counted from the impact on
    t = L.LiveTest(plan, folder) if plan.live else E.Test(plan, folder)
    if plan.live:
        t.load_live()                                        # each round's place in the one continuous soundscape
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
        canvas = self.page_canvas = tk.Canvas(holder, bg=BG, highlightthickness=0)
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
        self.pages = [(LivePage if kind == "live" else Page)(self.inner, self, kind, title) for kind, title in PAGES]
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
