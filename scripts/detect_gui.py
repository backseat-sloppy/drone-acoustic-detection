"""Find drones in your own audio files: open one or more files and a CNN and comb listen to them.

    uv run python scripts/detect_gui.py

"Open audio files..." picks any number of WAV, FLAC, OGG or MP3 files. Each file is read as
dad.wavio reads it (channel 0, resampled to 16 kHz) and every detector scores it hop by hop as it
would run live: a 1 s window every 0.125 s, a decision stamped at the window's end, drone when the
score reaches the model's frozen threshold. Nothing else is set: no sliders.

Detectors: the tick boxes at the bottom left pick the networks and comb ("DSP (comb)", last) that
run, any one or more, for the next batch opened. Each ticked model gets its own column in the list (with more than one, a
compact "38%*" column, * = alarmed) and its own colour in the graphs, with its own dashed
threshold. Each uses the threshold its own run froze at 5% false alarms on validation (It9B: 10%):
  it6 CNN   artifacts/models/it6_final_none_simdist_dscnn.pt (model 6, as scripts/drone_war_gui.py)
  it9 CNN   artifacts/models/final_dscnn_none_simdist_dscnn.pt (report entry 096)
  it9 CRNN  artifacts/models/final_crnn_none_simdist_crnn.pt (report entry 096)
  it10 CNN  artifacts/it10_clean/clean_s0_dscnn.pt (seed 0, trained on synthetic drones only)
  It9B      artifacts/models/It9B_none_simdist_crnn.pt (it9 CRNN with drone windows weighted 2, entry 160)
  it16 CNN/CRNN 600k s0-s2  artifacts/models/it16_600k_s*_none_simdist_{dscnn,crnn}.pt (600,000 training windows, entry 168)
  it19 CRNN colour s1  artifacts/models/it19_colour_s1_none_simdist_colour_crnn.pt (600k plus pink and brown noise as no drone, entry 171)
  it17 CNN/CRNN 2s 600k s0-s2  artifacts/models/it16_win2s600k_s*_none_simdist_{dscnn,crnn}.pt (the same with a 2 s window, entry 170)
  it20 CNN/CRNN bp s0-s2  artifacts/models/it20_bp_s*_none_simdist_{dscnn,crnn}.pt (600k with an 80 Hz - 6 kHz bandpass, entry 192)
  it21 CRNN uav16 s0-s2  artifacts/models/it21_uav16_s*_none_simdist_colour_crnn.pt (it19's recipe with UAV16 in the pool, iteration 21)
  it24 CNN 2s war s0-s2  artifacts/models/it24_2s_mad_s*_none_simdist_colour_mad_dscnn.pt (2 s colour recipe with MAD war audio, iteration 24)
  it24 CRNN 2s war s0-s2  artifacts/models/it24_2s_mad_s*_none_simdist_colour_mad_crnn.pt (the same with the CRNN)
comb, when ticked, keeps model 6's frozen threshold whichever models are picked.

War noise (optional): tick "add war noise" before opening files, pick its source and type one or
more SNRs in dB, separated by spaces, commas or semicolons (each -20 to +30, at most 6, default
"0 10 30"; lower = more noise; decimals with a point). The seed is 1000 for the whole batch, so
every file, at every SNR, gets the same noise from the same start. Sources:

    war-shaped noise  steady noise with the war's long-term spectrum (scripts/war_shaped_noise.py)
    war track         a window of the 600 s war soundscape (data/processed/war/war_track.wav,
                      scripts/war_track.py): real FSD50K war sounds coming and going, its calm,
                      normal and intense phases included. One start per batch, drawn from the seed
                      so that the longest file fits, shown in the title; a window that runs past
                      the end wraps to 0.

The noise is scaled so the file's broadband RMS is SNR dB above the noise's RMS over the same
stretch (docs/war_shaped_noise.md). Every SNR is its own row in the list; the detectors score the mix, Play
plays the mix, and the mix is saved as a 16 kHz float WAV in artifacts/detect_gui/mixes/. Below the
spectrogram the file gets one confidence graph per SNR, the selected one in bold; click a graph to
select that SNR (and hear it with Play). The setting applies to the next batch opened.

"dummy": one extra row at the top of the list, "DUMMY war track": a stretch of the war track
alone, no file and no drone, as long as the first file (10 s if none can be read), at the track's
own level, from the batch's war-track start (the same stretch the files get). Every alarm on it is
a false alarm; compare it with the files to see what a drone adds. It does not need "add war
noise", and does not depend on the SNR.

"Slide show" (top right) shows one SNR graph at a time over the whole right side, in the same
window; "< SNR" / "SNR >" or the Left / Right keys step through the file's SNRs (Play plays the
slide shown), Previous / Next and the list still change file, and Escape or the button goes back.

The list on the left gives each file's verdict per detector: DRONE if any window alarmed, with the
mean confidence in % (the area under its confidence curve over the scored time; comb as drawn,
s/(s+threshold)). The graph titles keep the share of windows that alarmed, the first alarm and the
longest unbroken alarm; results.csv has both mean_confidence and flagged_share. Click a file in the
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
from dad.features import LogMel, hz_to_mel, mel_to_hz  # noqa: E402
from dad.wavio import Clip, read_wav, stream  # noqa: E402

OUT = dwg.ROOT / "artifacts/detect_gui/results.csv"
MIXES = OUT.parent / "mixes"
SNR_MIN, SNR_MAX, SNR_DEFAULT = -20.0, 30.0, "0 10 30"      # dB, file RMS over noise RMS
SNR_MAX_COUNT = 6                                            # graphs under the spectrogram
NOISE_LOOP_S = 60.0                                          # longer files repeat it (the noise is periodic)
WAR_TRACK = dwg.es.WAR / "war_track.wav"                     # 600 s, 44.1 kHz (scripts/war_track.py)
SOURCES = {"war-shaped noise": "wsn", "war track (600 s)": "track"}   # radio label -> source
AUDIO_TYPES = [("Audio files", "*.wav *.flac *.ogg *.mp3"), ("All files", "*.*")]
LOGMEL = LogMel(DEFAULT)                                      # what the networks see (before level normalisation)
BAND_HZ = mel_to_hz(np.linspace(hz_to_mel(DEFAULT.fmin), hz_to_mel(DEFAULT.fmax), DEFAULT.n_mels + 2))[1:-1]
HOP_S = DEFAULT.samples_per_hop / DEFAULT.sample_rate
TEXT, GREY, RED = "#333333", "#777777", "#c0392b"
NAMES = {"CNN": "CNN", "comb": "DSP (comb)"}                 # detector -> name in the stats window
LIST_ROWS = 16                                               # files shown in the list at once
MODELS = dwg.MODELS
MODEL_COLORS = dict(zip(MODELS, ["#2a78d6", "#e8710a", "#9334e6", "#d93025", "#0b8a8a", "#b8860b", "#d4a72c",
                                 "#8a6d0b", "#c2185b", "#e5609a", "#880e4f", "#ff6f00", "#1b5e20", "#43a047", "#81c784",
                                 "#311b92", "#5e35b1", "#9575cd", "#006064", "#0097a7", "#4dd0e1", "#3e2723",
                                 "#6d4c41", "#a1887f",
                                 "#263238", "#607d8b", "#b0bec5",                       # it21 CRNN uav16 s0-s2
                                 "#1a237e", "#3949ab", "#7986cb", "#b71c1c", "#e53935", "#ef9a9a",   # it22 CNN, CRNN pcen s0-s2
                                 "#00695c", "#f9a825",                                # it23 CRNN relabel 600k, colour s1
                                 "#827717", "#afb42b", "#c0ca33",                       # it24 CNN 2s war s0-s2
                                 "#004d40", "#00897b", "#80cbc4"]))                     # it24 CRNN 2s war s0-s2
assert len(MODEL_COLORS) == len(MODELS), "add a colour for every model in drone_war_grid.MODELS"
COMB_COLOR = dwg.COLORS["comb"]

fig = plt.figure(figsize=(15, 9))
fig.canvas.manager.set_window_title("Find drones in audio files")
fig.text(0.015, 0.975, "Find drones in audio files", fontsize=13, weight="bold", color=TEXT, va="top")
ax_noise = fig.add_axes([0.015, 0.905, 0.15, 0.04], frameon=False)
c_noise = CheckButtons(ax_noise, ["add war noise"], [False])
for lbl in c_noise.labels:
    lbl.set_fontsize(8)
t_snr = TextBox(fig.add_axes([0.2, 0.91, 0.065, 0.03]), "SNR dB ", initial=SNR_DEFAULT)
t_snr.label.set_fontsize(8)
fig.text(0.135, 0.9, f"one or more, {SNR_MIN:+g} to {SNR_MAX:+g}, lower = more noise", fontsize=6.5, color=GREY,
         va="top")
r_source = RadioButtons(fig.add_axes([0.015, 0.835, 0.2, 0.055], frameon=False), list(SOURCES), active=0)
for lbl in r_source.labels:
    lbl.set_fontsize(8)
c_dummy = CheckButtons(fig.add_axes([0.17, 0.85, 0.1, 0.03], frameon=False), ["dummy"], [False])
c_dummy.labels[0].set_fontsize(8)
fig.text(0.172, 0.85, "+1 row: war track, no drone", fontsize=6.5, color=GREY, va="top")
b_open = Button(fig.add_axes([0.015, 0.785, 0.25, 0.045]), "Open audio files...")
b_prev = Button(fig.add_axes([0.015, 0.737, 0.12, 0.04]), "< Previous")
b_next = Button(fig.add_axes([0.145, 0.737, 0.12, 0.04]), "Next >")
b_play = Button(fig.add_axes([0.015, 0.692, 0.12, 0.04]), "Play")
b_hush = Button(fig.add_axes([0.145, 0.692, 0.12, 0.04]), "Stop sound")
status = fig.text(0.015, 0.677, "Open one or more audio files.", fontsize=8, color=TEXT, va="top", wrap=True)
ax_list = fig.add_axes([0.015, 0.15, 0.26, 0.49])
ax_model = fig.add_axes([0.015, 0.005, 0.25, 0.135], frameon=False)
ax_model.set_title("detectors, for the next batch opened (one or more)", fontsize=8, color=TEXT, loc="left")
DETECTOR_BOXES = [*MODELS, NAMES["comb"]]                    # the networks, then comb last
c_models = CheckButtons(ax_model, DETECTOR_BOXES, [i == 0 or b == NAMES["comb"] for i, b in enumerate(DETECTOR_BOXES)])
for lbl, b in zip(c_models.labels, DETECTOR_BOXES):
    lbl.set_fontsize(8)
    lbl.set_color(COMB_COLOR if b == NAMES["comb"] else MODEL_COLORS[b])
ax_list.axis("off")
ax_spec = fig.add_axes([0.33, 0.62, 0.65, 0.30])
CONF_BOX = (0.33, 0.07, 0.65, 0.50)                          # left, bottom, width, height of the graphs

SLIDE_BOX = (0.33, 0.08, 0.65, 0.73)                         # one graph over the whole right side
b_slides = Button(fig.add_axes([0.855, 0.945, 0.125, 0.035]), "Slide show")
ax_sprev = fig.add_axes([0.69, 0.945, 0.075, 0.035])
ax_snext = fig.add_axes([0.77, 0.945, 0.075, 0.035])
b_sprev, b_snext = Button(ax_sprev, "< SNR"), Button(ax_snext, "SNR >")
ax_sprev.set_visible(False)
ax_snext.set_visible(False)
slide_head = fig.text(0.33, 0.905, "", fontsize=11, weight="bold", color=TEXT, va="bottom")

state = dict(thread=None, files=[], cur=0, top=0, models=[next(iter(MODELS))], comb=True, loaded={}, conf_axes=[],
             slides=False)
events: queue.Queue = queue.Queue()


def detectors():
    """(thresholds, scorers) of the batch's ticked models, each under its own name, and comb if
    ticked; a network is loaded on first use (a few seconds) and kept."""
    thr, scorers = {}, {}
    for m in state["models"]:
        if m not in state["loaded"]:
            state["loaded"][m] = dwg._detectors(m)
        t, s = state["loaded"][m]
        thr[m], scorers[m] = t["CNN"], s["CNN"]
    if state["comb"]:                                        # model 6's frozen comb threshold, as before
        if "comb" not in state["loaded"]:
            state["loaded"]["comb"] = (dwg.comb_threshold(), dwg.load_scorer("comb", DEFAULT, "cpu"))
        thr["comb"], scorers["comb"] = state["loaded"]["comb"]
    state["thr"] = thr
    return thr, scorers


def loading() -> bool:
    return any(m not in state["loaded"] for m in state["models"])


def keys() -> list[str]:
    """The batch's detectors in display order: the ticked models, then comb if ticked."""
    return [*state["models"], *(["comb"] if state["comb"] else [])]


def comb_note(thr) -> str:
    """How comb is drawn, for the notes under the graphs; empty without comb."""
    return f", comb shown as s/(s+{thr['comb']:.2f})" if "comb" in thr else ""


def color(k) -> str:
    return COMB_COLOR if k == "comb" else MODEL_COLORS[k]


def shown_as(k):
    """Detector k as shown: the model's name, or "DSP (comb)"."""
    return NAMES["comb"] if k == "comb" else k


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


def war_track() -> np.ndarray:
    """The 600 s war soundscape at the frontend's rate, read and resampled once."""
    if "war_track" not in state:
        x, sr = sf.read(WAR_TRACK, dtype="float64")
        state["war_track"] = resample_poly(x, DEFAULT.sample_rate, sr)
    return state["war_track"]


def track_start(seconds: float, seed: int = dwg.WSN_SEED) -> float:
    """A start in the war track, drawn from seed, where a window of `seconds` fits (0 if none does).
    One start per batch, for the longest file, so every file and the dummy share it."""
    x, n = war_track(), int(seconds * DEFAULT.sample_rate)
    return int(np.random.default_rng(seed).integers(0, max(len(x) - n, 0) + 1)) / DEFAULT.sample_rate


def track_window(n: int, start_s: float) -> tuple[np.ndarray, float]:
    """n samples of the war track from start_s, and start_s; past the end it wraps to 0."""
    i = int(round(start_s * DEFAULT.sample_rate))
    return np.take(war_track(), np.arange(i, i + n), mode="wrap"), start_s


def add_noise(audio: np.ndarray, snr_db: float, seed: int, source: str = "wsn",
              start_s: float = 0.0) -> tuple[np.ndarray, float | None, np.ndarray]:
    """audio plus war noise from source ("wsn" or "track", from start_s), scaled so audio's
    broadband RMS is snr_db above the noise's; the track window's start in s (None for war-shaped
    noise); and the scaled noise on its own (the layer that was added)."""
    noise, offset = (war_noise(len(audio), seed), None) if source == "wsn" else track_window(len(audio), start_s)
    rms = lambda v: np.sqrt(np.mean(v ** 2))                  # noqa: E731
    if not np.any(noise):
        raise ValueError("the war track is silent in this window")
    layer = (noise * rms(audio) / rms(noise) / 10 ** (snr_db / 20)).astype(audio.dtype)
    return audio + layer, offset, layer


def score(clip, thr, scorers) -> tuple[dict, dict]:
    """Every detector over clip: decisions and a summary per detector. "mean" is the area under the
    confidence curve over the scored time, as a share: the mean confidence over all decisions (comb
    as drawn, s/(s+threshold))."""
    dec, summary = {}, {}
    for k, sc in scorers.items():
        d = stream(clip, sc, sc.cfg, thr[k])          # each model at its own window (1 s or 2 s)
        first = d.t_end[d.drone]
        p = peaks(d.drone)
        dec[k] = d
        conf = dwg.squash(d.score, thr[k]) if k == "comb" else d.score
        summary[k] = dict(drone=bool(d.drone.any()), flagged=float(d.drone.mean()), mean=float(np.mean(conf)),
                          first_alarm_s=float(first[0]) if len(first) else None,
                          longest_s=max((b - a + 1 for a, b in p), default=0) * HOP_S,
                          peaks=p, above_s=int(d.drone.sum()) * HOP_S)
    return dec, summary


def detect(path: str, noise: tuple[float, int, str, float] | None = None, group: int = 0) -> dict:
    """Every detector over one file, with war noise added first when
    noise = (SNR dB, seed, source, war-track start in s).
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
        snr, seed, source, start = noise
        if not np.any(clip.audio):
            return {**f, "error": "silent: no level to set the noise against"}
        try:
            clip.audio, offset, layer = add_noise(clip.audio, snr, seed, source, start)
        except ValueError as e:
            return {**f, "error": str(e)}
        f.update(audio=clip.audio, snr_db=snr, noise_seed=seed, noise_source=source, noise_offset_s=offset)
        MIXES.mkdir(parents=True, exist_ok=True)
        tag = "wsn" if source == "wsn" else f"wartrack{offset:.1f}s"
        f["mix_path"] = str(MIXES / f"{Path(path).stem}_{tag}_snr{snr:+g}dB_seed{seed}.wav")
        sf.write(f["mix_path"], clip.audio, DEFAULT.sample_rate, subtype="FLOAT")
    thr, scorers = detectors()
    f["dec"], f["summary"] = score(clip, thr, scorers)
    return f


def dummy_row(seconds: float, start_s: float) -> dict:
    """The dummy: the war track alone from start_s, no file and no drone, at the track's own level
    (as rendered, full scale 110 dB SPL like the detectors' input), scored like a file."""
    audio, offset = track_window(int(seconds * DEFAULT.sample_rate), start_s)
    f = dict(path=str(WAR_TRACK), name="DUMMY war track", group="dummy", dummy=True, audio=audio.astype(np.float32),
             seconds=len(audio) / DEFAULT.sample_rate, noise_seed=dwg.WSN_SEED, noise_source="track",
             noise_offset_s=offset)
    thr, scorers = detectors()
    f["dec"], f["summary"] = score(Clip(str(WAR_TRACK), f["audio"], DEFAULT.sample_rate, 1, "FLOAT"), thr, scorers)
    return f


def durations(paths) -> list[float]:
    """The length of every file that can be read and holds a window (detect() reports the others)."""
    out = []
    for p in paths:
        try:
            s = sf.info(p).duration
        except Exception:                                    # noqa: BLE001
            continue
        if s >= DEFAULT.window_seconds:
            out.append(s)
    return out


def worker(paths, snrs=None, source="wsn", dummy=False):
    """With dummy, first one row of the war track alone; then every file, at every SNR in snrs
    (None: no noise) of noise from source. One seed (1000) and one war-track start for the whole
    batch, so every file and the dummy get the same stretch of soundscape. One row per file and SNR."""
    try:
        lengths = durations(paths) or [10.0]
        start = track_start(max(lengths)) if (dummy or (snrs and source == "track")) else 0.0
        if dummy:
            events.put(("status", "listening to the dummy (war track alone)"
                                  + (" (this also loads the detectors)" if loading() else "")))
            events.put(("file", dummy_row(lengths[0], start)))
        for i, p in enumerate(paths):
            for snr in snrs or [None]:
                events.put(("status", f"listening to file {i + 1} of {len(paths)}: {Path(p).name}"
                                      + ("" if snr is None else f" at SNR {snr:+g} dB")
                                      + (" (the first file also loads the detectors)" if loading() else "")))
                events.put(("file", detect(p, None if snr is None else (snr, dwg.WSN_SEED, source, start), group=i)))
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
    ks = keys()
    compact = len(ks) > 2                                    # several models: "38%*" columns, * = alarmed
    head = "".join(f"{k.replace(' ', ''):>8s}" if compact else f"{k:>11s}" for k in ks)
    ax_list.text(0, 1, f"{'':3s}{'file':16s}{head}", family="monospace", fontsize=8, weight="bold", color=TEXT,
                 va="top")
    ax_list.cell_rows = {}
    for row, i in enumerate(range(state["top"], min(state["top"] + LIST_ROWS, len(files)))):
        f = files[i]
        tag = f" {f['snr_db']:+g}" if "snr_db" in f else ""      # the SNR stays visible, the name is cut
        name = f["name"] if len(f["name"]) + len(tag) <= 15 else f["name"][:14 - len(tag)] + "~"
        name += tag
        if "error" in f:
            cols = f"  {f['error']}"
        elif compact:
            cols = "".join(f"{100 * f['summary'][k]['mean']:6.0f}%{'*' if f['summary'][k]['drone'] else ' '}"
                           for k in ks)
        else:
            cols = "".join(f"{('DRONE' if f['summary'][k]['drone'] else 'none'):>6s}{100 * f['summary'][k]['mean']:4.0f}%"
                           for k in ks)
        y = 1 - (row + 1.3) / (LIST_ROWS + 3)
        hit = is_hit(f)
        ax_list.text(0, y, f"{'>' if i == state['cur'] else ' '}{i + 1:<2d}{name:16s}{cols}", family="monospace",
                     fontsize=8, color=RED if hit else TEXT, weight="bold" if i == state["cur"] else "normal",
                     va="top", picker=True).file_index = i
    more = len(files) - state["top"] - LIST_ROWS
    note = ("click a file to show it, a red one opens its stats; % = mean confidence, * = alarmed" if compact
            else "click a file to show it, a red one also opens its stats; % = mean confidence")
    if state["top"] or more > 0:
        note = f"files {state['top'] + 1}-{min(state['top'] + LIST_ROWS, len(files))} of {len(files)}; scroll here. " + note
    ax_list.text(0, 1 - (LIST_ROWS + 2) / (LIST_ROWS + 3), note, fontsize=7, color=GREY, va="top", wrap=True)


def describe(f) -> str:
    """A row's file and noise, for the spectrogram title and the slide head."""
    if f.get("dummy"):
        return (f"DUMMY: {noise_name(f)} alone, no drone, at the track's own level  ({f['seconds']:.1f} s, "
                f"seed {f['noise_seed']})")
    return (f"{f['name']}" + (f"  ({f['seconds']:.1f} s)" if "seconds" in f else "")
            + (f"  + {noise_name(f)}, SNR {f['snr_db']:+g} dB (seed {f['noise_seed']})" if "snr_db" in f else ""))


def noise_name(f) -> str:
    if f.get("noise_source") == "track":
        return f"war track from {f['noise_offset_s']:.1f} s"
    return "war-shaped noise"


def siblings(i: int) -> list[int]:
    """Rows of the same file as row i (one per SNR), in list order."""
    g = state["files"][i]["group"]
    return [j for j, f in enumerate(state["files"]) if f["group"] == g]


def conf_axes(n: int, box=CONF_BOX) -> list:
    """n confidence graphs stacked in box, sharing the spectrogram's time axis."""
    for ax in state["conf_axes"]:
        fig.delaxes(ax)
    state["conf_axes"] = []
    if not n:
        return []
    left, bottom, width, height = box
    gap = 0.035 if n > 1 else 0.0
    h = (height - gap * (n - 1)) / n
    state["conf_axes"] = [fig.add_axes([left, bottom + (n - 1 - r) * (h + gap), width, h], sharex=ax_spec)
                          for r in range(n)]
    return state["conf_axes"]


def plot_score(ax, d, y, k):
    """Confidence over time with alarms shaded. A short file has few decisions (a 1 s file has
    one, at 1.00 s), and a line through one point draws nothing, so few decisions get dots and
    each alarmed one shades the window it heard."""
    few = len(d.t_end) < 20
    ax.plot(d.t_end, y, color=color(k), lw=1.2, label=shown_as(k), marker="o" if few else None, ms=5,
            clip_on=not few)                                 # the last dot sits on the right edge
    if few:                                                  # overlapping windows merged, shaded once
        spans = []
        for t in d.t_end[d.drone]:
            a = max(t - d.t_end[0], 0)                  # the first decision ends one window in (1 or 2 s)
            if spans and a <= spans[-1][1]:
                spans[-1][1] = t
            else:
                spans.append([a, t])
        for a, b in spans:
            ax.axvspan(a, b, color=color(k), alpha=0.10, lw=0)
    else:
        ax.fill_between(d.t_end, 0, 1, where=d.drone, color=color(k), alpha=0.10, step="pre", lw=0)


def draw_conf(ax, f, thr, selected, legend, size=9, sep="   |   "):
    """One row's confidence over time: both detectors, thresholds dashed, alarms shaded."""
    head = "DUMMY, war track alone:  " if f.get("dummy") else f"SNR {f['snr_db']:+g} dB:  " if "snr_db" in f else ""
    if "error" in f:
        ax.text(0.5, 0.5, f["error"], ha="center", va="center", color=GREY, transform=ax.transAxes)
        ax.set_title(head, fontsize=size, color=TEXT, loc="left", weight="bold" if selected else "normal")
        return
    lines = []
    for k, d in f["dec"].items():
        y = dwg.squash(d.score, thr[k]) if k == "comb" else d.score
        plot_score(ax, d, y, k)
        ax.axhline(0.5 if k == "comb" else thr[k], color=color(k), lw=0.8, ls="--")
        s = f["summary"][k]
        if len(f["dec"]) > 2:                                # several models: short, so the title fits
            lines.append(f"{shown_as(k)}: " + (f"DRONE {100 * s['flagged']:.0f}%" if s["drone"] else "none"))
        else:
            lines.append(f"{shown_as(k)}: " + (f"DRONE  first alarm {s['first_alarm_s']:.2f} s, longest alarm "
                                               f"{s['longest_s']:.2f} s, {100 * s['flagged']:.0f}% of windows"
                                               if s["drone"] else "no drone heard"))
    ax.set(ylim=(0, 1), xlim=(0, f["seconds"]), ylabel="confidence")
    ax.set_title(head + sep.join(lines), fontsize=size, color=TEXT, loc="left",
                 weight="bold" if selected else "normal")
    if legend:
        ax.legend(loc="upper right", fontsize=size - 1, frameon=False)
    ax.grid(color="#e5e5e5", lw=0.5)
    if selected and "snr_db" in f and not state["slides"]:
        for side in ax.spines.values():
            side.set(linewidth=1.6, color=TEXT)


def draw_logmel(ax, audio):
    """The log-mel the networks get (dad.features.LogMel, the frozen frontend: 64 mel bands 50-8000 Hz,
    25 ms frames, 10 ms stride), in dB, 80 dB range; before the level normalisation each model does.
    One row per band, labelled with the band's centre frequency."""
    lm = 10 / np.log(10) * LOGMEL.frames(audio)              # natural log -> dB
    top = float(lm.max()) if lm.size else 0.0
    ax.imshow(lm, origin="lower", aspect="auto", cmap="magma", vmin=top - 80, vmax=top, interpolation="nearest",
              extent=(0, len(audio) / DEFAULT.sample_rate, 0, lm.shape[0]))
    ticks = [0, 8, 16, 24, 32, 40, 48, 56, 63]
    ax.set_yticks([t + 0.5 for t in ticks], [f"{BAND_HZ[t]:.0f}" for t in ticks])
    ax.set_ylabel("Hz (log-mel band centre)")


def draw_slide():
    """Slide show: the current row's confidence graph alone, filling the right side, with the
    file and its place among the file's SNRs above it."""
    files = state["files"]
    f, rows = files[state["cur"]], siblings(state["cur"])
    ax = conf_axes(1, SLIDE_BOX)[0]
    ax.row_index = state["cur"]
    thr = detectors()[0] if "error" not in f else {}
    draw_conf(ax, f, thr, selected=True, legend=True, size=12, sep="\n")
    ax.set_xlabel("time (s), each decision stamped at the end of its 1 s window", fontsize=11)
    ax.tick_params(labelsize=10)
    slide_head.set_text(f"slide {rows.index(state['cur']) + 1} of {len(rows)}:  {describe(f)}")
    if thr:
        ax.text(0.005, 0.015, f"dashed = threshold, shaded = alarm{comb_note(thr)}; "
                              "< SNR / SNR > or Left / Right keys step; Play plays this slide; Esc goes back",
                fontsize=8, color=GREY, transform=ax.transAxes)


def step_slide(step: int):
    """The previous or next SNR of the shown file, wrapping around."""
    if not state["files"]:
        return
    rows = siblings(state["cur"])
    show(rows[(rows.index(state["cur"]) + step) % len(rows)])


def toggle_slides(_=None):
    state["slides"] = not state["slides"]
    for ax in (ax_sprev, ax_snext):
        ax.set_visible(state["slides"])
    b_slides.label.set_text("Back to all graphs" if state["slides"] else "Slide show")
    redraw()


def draw_file():
    """Spectrogram of the current row, and one confidence graph per row of the same file
    (one per SNR; just one without noise). In the slide show, draw_slide instead."""
    ax_spec.clear()
    slide_head.set_text("")
    files = state["files"]
    ax_spec.set_visible(not (state["slides"] and files))
    if not files:
        conf_axes(0)
        return
    if state["slides"]:
        draw_slide()
        return
    f = files[state["cur"]]
    if "audio" in f:
        draw_logmel(ax_spec, f["audio"])
    ax_spec.set_title(f"{state['cur'] + 1}/{len(files)}  {describe(f)}", fontsize=10, color=TEXT, loc="left")
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
        note = f"dashed = threshold, shaded = alarm{comb_note(thr)}"
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
        w.writerow(["file", "seconds", "detector", "threshold", "verdict", "mean_confidence", "flagged_share",
                    "first_alarm_s",
                    "longest_alarm_s", "peaks_above_threshold", "time_above_threshold_s", "noise_snr_db",
                    "noise_seed", "noise_source", "noise_offset_s", "mix_file", "dummy", "error"])
        thr = state.get("thr", {})
        for f in state["files"]:
            for k in keys():
                s = f.get("summary", {}).get(k, {})
                w.writerow([f["path"], round(f["seconds"], 3) if "seconds" in f else "", shown_as(k), thr.get(k, ""),
                            verdict(f, k), s.get("mean", ""), s.get("flagged", ""), s.get("first_alarm_s", ""),
                            s.get("longest_s", ""),
                            len(s["peaks"]) if s else "", s.get("above_s", ""), f.get("snr_db", ""),
                            f.get("noise_seed", ""), f.get("noise_source", ""),
                            "" if f.get("noise_offset_s") is None else f["noise_offset_s"],
                            f.get("mix_path", ""), "yes" if f.get("dummy") else "", f.get("error", "")])


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
    ticked = [b for b, on in zip(DETECTOR_BOXES, c_models.get_status()) if on]
    models, comb = [b for b in ticked if b in MODELS], NAMES["comb"] in ticked
    if not ticked:
        status.set_text("tick at least one detector at the bottom left")
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
    state.update(files=[], cur=0, top=0, models=models, comb=comb)
    redraw()
    state["thread"] = threading.Thread(target=worker, args=(paths, snr, SOURCES[r_source.value_selected],
                                                            c_dummy.get_status()[0]), daemon=True)
    state["thread"].start()


def show(i):
    if not state["files"]:
        return
    state["cur"] = i % len(state["files"])
    if not state["top"] <= state["cur"] < state["top"] + LIST_ROWS:   # keep the current file in the list
        state["top"] = max(0, state["cur"] - LIST_ROWS + 1) if state["cur"] >= state["top"] else state["cur"]
    redraw()


def is_hit(f) -> bool:
    return "error" not in f and any(s["drone"] for s in f["summary"].values())


def open_stats(i):
    """A window with one file's peaks above threshold and time above threshold, per detector:
    confidence over time with each peak shaded and numbered, and a table of the peaks."""
    f = state["files"][i]
    thr = detectors()[0]
    win, axs = plt.subplots(len(f["dec"]), 1, figsize=(14, min(3.6 * len(f["dec"]), 10.5)), sharex=True,
                            squeeze=False)
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
        ax.plot(d.t_end, y, color=color(k), lw=1.2, marker="o" if few else None, ms=5, clip_on=not few)
        ax.axhline(line, color=color(k), lw=0.8, ls="--")
        rows = []
        for n, (a, b) in enumerate(s["peaks"], 1):
            t0, t1 = d.t_end[a], min(d.t_end[b] + HOP_S, f["seconds"])
            j = a + int(np.argmax(y[a:b + 1]))
            ax.axvspan(t0, t1, color=color(k), alpha=0.12, lw=0)
            ax.annotate(str(n), (d.t_end[j], y[j]), textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=7, color=TEXT)
            rows.append(f"{n:3d} {t0:8.2f} {t1:8.2f} {(b - a + 1) * HOP_S:7.2f} {y[j]:6.2f}")
        ax.set(ylim=(0, 1.08), xlim=(0, f["seconds"]), ylabel="confidence")
        ax.set_title(shown_as(k), fontsize=10, weight="bold", color=color(k), loc="left")
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
                         f"max = the highest confidence in the peak{comb_note(thr)}.",
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


def on_key(event):
    """Slide show keys: Left / Right step through the file's SNRs, Escape goes back."""
    if not state["slides"]:
        return
    if event.key in ("left", "right"):
        step_slide(-1 if event.key == "left" else 1)
    elif event.key == "escape":
        toggle_slides()


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
b_slides.on_clicked(toggle_slides)
b_sprev.on_clicked(lambda _: step_slide(-1) if state["slides"] else None)
b_snext.on_clicked(lambda _: step_slide(+1) if state["slides"] else None)
fig.canvas.mpl_connect("key_press_event", on_key)
timer = fig.canvas.new_timer(interval=300)
timer.add_callback(poll)
timer.start()
plt.show()
