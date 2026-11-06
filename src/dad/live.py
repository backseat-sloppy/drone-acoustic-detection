"""Run one or more detectors on a live microphone, hop by hop.

    uv run dad live --list-devices
    uv run dad live --device "Microphone Array" --detector "it6 CNN" --detector comb

The audio path is the one `dad.wavio` runs over a file, cut into blocks as they
arrive: channel 0 (or `--channel`), resampled to the frontend's 16 kHz, and every
hop the detectors score the last window. A decision is stamped with the time its
window ends, counted in audio samples since capture started, so a recording made
with `--record` and replayed through `dad wav` gives the same decisions (exactly,
when the device runs at 16 kHz; to within the resampler's edge when it does not).

Pieces, from the inside out:

    WindowBuffer  holds the last window of input audio and yields a window at
                  every hop boundary, however the blocks happen to be cut
    LivePipeline  WindowBuffer + detectors: blocks in, Decisions out
    mic_blocks    blocks from a sounddevice input stream
    wav_blocks    blocks from a file, paced like a microphone, for testing
    run           the loop: source -> pipeline -> display, CSV and recording

Nothing here needs a sound card except `mic_blocks`, so the tests feed files.

Levels: the networks were trained on audio whose full scale is 110 dB SPL and
their scores depend on level. An uncalibrated microphone gives hints, not
measurements; `--gain-db` shifts the input level before the detectors.
"""
from __future__ import annotations

import math
import queue
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from scipy.signal import resample_poly

from .config import DEFAULT, FrontendConfig

ROOT = Path(__file__).resolve().parents[2]
# Name -> (run file holding the threshold frozen on validation, checkpoint). The same
# models as scripts/drone_war_grid.py MODELS, so the live tool and the GUIs agree.
PRESETS = {
    "it6 CNN": ("artifacts/runs/it6_final.json", "artifacts/models/it6_final_none_simdist_dscnn.pt"),
    "it9 CNN": ("artifacts/runs/final_dscnn.json", "artifacts/models/final_dscnn_none_simdist_dscnn.pt"),
    "it9 CRNN": ("artifacts/runs/final_crnn.json", "artifacts/models/final_crnn_none_simdist_crnn.pt"),
    "it10 CNN": ("artifacts/it10_clean/clean_s0.json", "artifacts/it10_clean/clean_s0_dscnn.pt"),
    # Iteration 11.2 (report entries 143-146): seed-0 CRNNs under entry 143's rule, one rig held out.
    # N = trained on d3set (Nightjar held out), D = trained on Nightjar (d3set held out).
    "it11.2 N CRNN": ("artifacts/runs/it11_2_s0_nightjar.json",
                      "artifacts/models/it11_2_s0_nightjar_nightjar_simdist_crnn.pt"),
    "it11.2 N steady CRNN": ("artifacts/runs/it11_2_s0_nightjar.json",
                             "artifacts/models/it11_2_s0_nightjar_nightjar_simdist_steady_crnn.pt"),
    "it11.2 D CRNN": ("artifacts/runs/it11_2_s0_d3set.json",
                      "artifacts/models/it11_2_s0_d3set_d3set_simdist_crnn.pt"),
    "it11.2 D steady CRNN": ("artifacts/runs/it11_2_s0_d3set.json",
                             "artifacts/models/it11_2_s0_d3set_d3set_simdist_steady_crnn.pt"),
}
COMB_RUN = "artifacts/runs/it6_final.json"     # comb keeps model 6's frozen threshold, as in detect_gui
RESAMPLE_MARGIN_S = 0.05                       # extra input kept so the resampler's start-up falls outside the window


# ------------------------------------------------------------------ detectors

@dataclass
class Detector:
    name: str
    scorer: object                     # anything with score(windows (N, n)) -> (N,)
    threshold: float | None            # None: scores only, no True/False


def comb_threshold() -> float:
    import json
    rows = json.loads((ROOT / COMB_RUN).read_text(encoding="utf-8"))["results"]
    return next(float(r["deploy_threshold"]) for r in rows
                if "comb" in r["scorer"] and r.get("integrate", 1) == 1 and r["arm"] == "hold out none")


def load_detector(spec: str, cfg: FrontendConfig = DEFAULT, device: str = "cpu") -> Detector:
    """`SPEC` or `SPEC@THRESHOLD`. SPEC is a preset name (see PRESETS), `comb`, or a
    .pt / .pkl checkpoint. Presets and comb bring their frozen threshold; a bare
    checkpoint without `@THRESHOLD` gives scores only."""
    from .wavio import load_scorer, threshold_from_run
    spec, _, thr = spec.partition("@")
    threshold = float(thr) if thr else None
    if spec in PRESETS:
        run, ckpt = (str(ROOT / p) for p in PRESETS[spec])
        if threshold is None:
            threshold = threshold_from_run(run, ckpt, cfg.window_seconds)
        return Detector(spec, load_scorer(ckpt, cfg, device), threshold)
    if spec == "comb":
        return Detector("comb", load_scorer("comb", cfg, device), comb_threshold() if threshold is None else threshold)
    return Detector(Path(spec).stem, load_scorer(spec, cfg, device), threshold)


# ------------------------------------------------------------------- pipeline

class WindowBuffer:
    """The last window of input, and a window out at every hop boundary.

    Boundary k falls when (n + k * hop) frontend samples have arrived, as in
    `wavio.stream`: the first window comes one window in. Blocks are split at the
    boundaries, so the windows do not depend on how the device cuts its blocks.
    """

    def __init__(self, sr_in: int, cfg: FrontendConfig = DEFAULT) -> None:
        self.cfg, self.sr_in = cfg, int(sr_in)
        g = math.gcd(cfg.sample_rate, self.sr_in)
        self.up, self.down = cfg.sample_rate // g, self.sr_in // g
        self.resampling = self.sr_in != cfg.sample_rate
        margin = int(round(RESAMPLE_MARGIN_S * self.sr_in)) if self.resampling else 0
        self.n_in = int(math.ceil(cfg.samples_per_window * self.sr_in / cfg.sample_rate)) + margin
        self.buf = np.zeros(self.n_in, np.float32)
        self.received = 0              # input samples so far
        self.k = 0                     # next hop boundary

    def boundary(self, k: int) -> int:
        """Input samples needed for hop boundary k."""
        n_out = self.cfg.samples_per_window + k * self.cfg.samples_per_hop
        return int(round(n_out * self.sr_in / self.cfg.sample_rate))

    def _push(self, x: np.ndarray) -> None:
        m = len(x)
        if m >= self.n_in:
            self.buf[:] = x[-self.n_in:]
        elif m:
            self.buf[:-m] = self.buf[m:]
            self.buf[-m:] = x
        self.received += m

    def window(self) -> np.ndarray:
        n = self.cfg.samples_per_window
        if not self.resampling:
            return self.buf[-n:].copy()
        return resample_poly(self.buf.astype(np.float64), self.up, self.down)[-n:].astype(np.float32)

    def feed(self, x: np.ndarray) -> list[tuple[float, np.ndarray]]:
        """(t_end seconds, window at the frontend rate) for every boundary in x."""
        x = np.asarray(x, np.float32).ravel()
        out, i = [], 0
        while i < len(x):
            take = min(len(x) - i, self.boundary(self.k) - self.received)
            self._push(x[i:i + take])
            i += take
            if self.received == self.boundary(self.k):
                t_end = (self.cfg.samples_per_window + self.k * self.cfg.samples_per_hop) / self.cfg.sample_rate
                out.append((t_end, self.window()))
                self.k += 1
        return out


@dataclass
class Decision:
    t_end: float                       # seconds of audio since capture started
    level_dbfs: float                  # RMS of the window the detectors saw
    scores: dict[str, float]
    drone: dict[str, bool | None]      # R1: score >= threshold, None without a threshold
    compute_ms: float                  # scoring time of the batch this decision was in, per window
    window: np.ndarray | None = field(default=None, repr=False)   # the audio the detectors heard (gain applied)


class LivePipeline:
    def __init__(self, detectors: list[Detector], sr_in: int, cfg: FrontendConfig = DEFAULT,
                 gain_db: float = 0.0) -> None:
        self.detectors, self.cfg = detectors, cfg
        self.gain = float(10 ** (gain_db / 20))
        self.windows = WindowBuffer(sr_in, cfg)

    def feed(self, block: np.ndarray) -> list[Decision]:
        """Mono input samples in, a Decision for every hop boundary they cross.
        Windows that arrive together (a late or long block) are scored as one batch."""
        got = self.windows.feed(block)
        if not got:
            return []
        t_end = [t for t, _ in got]
        audio = np.stack([w for _, w in got]) * self.gain
        t0 = time.perf_counter()
        scores = {d.name: np.asarray(d.scorer.score(audio), dtype=np.float64) for d in self.detectors}
        ms = (time.perf_counter() - t0) * 1000 / len(got)
        level = 10 * np.log10(np.mean(audio.astype(np.float64) ** 2, axis=1) + 1e-20)
        thr = {d.name: d.threshold for d in self.detectors}
        return [Decision(t, float(level[i]), {k: float(s[i]) for k, s in scores.items()},
                         {k: None if thr[k] is None else bool(s[i] >= thr[k]) for k, s in scores.items()}, ms,
                         audio[i])
                for i, t in enumerate(t_end)]


# -------------------------------------------------------------------- sources

@dataclass
class Block:
    audio: np.ndarray                  # (frames, channels) float32 at the source rate
    status: str = ""                   # non-empty: the device reported a problem (an overflow)


def mic_blocks(device=None, samplerate: int | None = None, channels: int | None = None,
               blocksize: int = 0, stop: Callable[[], bool] = lambda: False) -> tuple[int, int, Iterator[Block]]:
    """(rate, channels, blocks) from a sounddevice input stream. The rate is the frontend's
    16 kHz when the device takes it, else the device's own rate (resampled in software).
    The callback only copies; scoring happens in the caller's thread."""
    import sounddevice as sd
    info = sd.query_devices(device, "input")
    channels = channels or int(info["max_input_channels"])
    if samplerate is None:
        try:
            sd.check_input_settings(device, channels=channels, samplerate=DEFAULT.sample_rate, dtype="float32")
            samplerate = DEFAULT.sample_rate
        except Exception:
            samplerate = int(info["default_samplerate"])
    q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(Block(indata.copy(), str(status) if status else ""))

    def blocks() -> Iterator[Block]:
        with sd.InputStream(device=device, samplerate=samplerate, channels=channels, dtype="float32",
                            blocksize=blocksize, callback=callback):
            while not stop():
                try:
                    yield q.get(timeout=0.5)
                except queue.Empty:
                    continue
                # Behind: hand over everything waiting at once, so it is scored as one batch.
                more = []
                while not q.empty():
                    more.append(q.get_nowait())
                for b in more:
                    yield b

    return int(samplerate), channels, blocks()


def wav_blocks(path, blocksize: int = 1024, realtime: bool = True,
               stop: Callable[[], bool] = lambda: False) -> tuple[int, int, Iterator[Block]]:
    """(rate, channels, blocks) from a file, as a microphone would deliver it: at its
    own rate and channel count, paced to real time unless `realtime` is False."""
    import soundfile as sf
    f = sf.SoundFile(str(path))

    def blocks() -> Iterator[Block]:
        t0 = time.perf_counter()
        sent = 0
        with f:
            while not stop():
                x = f.read(blocksize, dtype="float32", always_2d=True)
                if not len(x):
                    return
                sent += len(x)
                if realtime:
                    time.sleep(max(0.0, t0 + sent / f.samplerate - time.perf_counter()))
                yield Block(x)

    return int(f.samplerate), int(f.channels), blocks()


# ----------------------------------------------------------------------- loop

@dataclass
class Session:
    decisions: int = 0
    overflows: int = 0
    alarms: dict[str, int] = field(default_factory=dict)        # alarm onsets per detector
    first_alarm: dict[str, float] = field(default_factory=dict)
    max_compute_ms: float = 0.0
    audio_s: float = 0.0
    wall_s: float = 0.0


def _bar(v: float, thr: float | None, width: int = 10) -> str:
    """A small gauge; for comb (unbounded score) it is drawn as s/(s+threshold), as in detect_gui."""
    if thr is not None and thr > 1:
        v = v / (v + thr) if v > 0 else 0.0
    n = int(round(min(max(v, 0.0), 1.0) * width))
    return "#" * n + "." * (width - n)


def run(pipe: LivePipeline, sr_in: int, blocks: Iterator[Block], channel: int = 0,
        out_csv=None, record=None, duration: float | None = None, quiet: bool = False,
        print_fn: Callable[[str], None] = print) -> Session:
    """Feed blocks through the pipeline until the source ends, `duration` seconds of audio
    have passed, or Ctrl+C. Prints a live status line and a line at every alarm onset and
    end; writes every decision to `out_csv` and the raw input to `record` when given."""
    import csv
    s = Session(alarms={d.name: 0 for d in pipe.detectors})
    alarmed = {d.name: False for d in pipe.detectors}
    writer = rec = fh = None
    if out_csv:
        fh = open(out_csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(fh)
        writer.writerow(["t_end_s", "wall_clock", "level_dbfs", "detector", "score", "threshold", "drone",
                         "compute_ms"])
    t0 = time.perf_counter()
    tty = sys.stdout.isatty() and not quiet
    try:
        for b in blocks:
            if b.status:
                s.overflows += 1
            x = b.audio
            if record is not None:
                if rec is None:
                    import soundfile as sf
                    rec = sf.SoundFile(str(record), "w", samplerate=sr_in, channels=x.shape[1], subtype="FLOAT")
                rec.write(x)
            s.audio_s += len(x) / sr_in
            for d in pipe.feed(x[:, channel]):
                s.decisions += 1
                s.max_compute_ms = max(s.max_compute_ms, d.compute_ms)
                lag_ms = 1000 * ((time.perf_counter() - t0) - d.t_end)
                for det in pipe.detectors:
                    k, on = det.name, d.drone[det.name]
                    if writer:
                        writer.writerow([f"{d.t_end:.4f}", time.strftime("%H:%M:%S"), f"{d.level_dbfs:.1f}", k,
                                         f"{d.scores[k]:.6g}", "" if det.threshold is None else f"{det.threshold:.6g}",
                                         "" if on is None else int(on), f"{d.compute_ms:.2f}"])
                    if on and not alarmed[k]:
                        s.alarms[k] += 1
                        s.first_alarm.setdefault(k, d.t_end)
                    if on is not None and on != alarmed[k] and not quiet:
                        msg = (f"DRONE  {k}: score {d.scores[k]:.3g} >= {det.threshold:.3g}" if on
                               else f"clear  {k}")
                        print_fn(("\r" if tty else "") + f"[{time.strftime('%H:%M:%S')}  t={d.t_end:7.2f} s]  "
                                 f"{msg}".ljust(100))
                    alarmed[k] = bool(on)
                if tty:
                    parts = [f"{det.name} {_bar(d.scores[det.name], det.threshold)} {d.scores[det.name]:.3g}"
                             + ("  DRONE" if d.drone[det.name] else "") for det in pipe.detectors]
                    line = (f"t={d.t_end:7.2f} s  {d.level_dbfs:6.1f} dBFS  " + "  |  ".join(parts)
                            + f"  |  {d.compute_ms:.1f} ms, lag {lag_ms:.0f} ms")
                    sys.stdout.write("\r" + line[:160].ljust(160))
                    sys.stdout.flush()
            if writer and s.decisions % 80 == 0:
                fh.flush()
            if duration is not None and s.audio_s >= duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if tty:
            sys.stdout.write("\n")
        close = getattr(blocks, "close", None)      # a generator: closes the input stream now, not at GC
        if close is not None:
            close()
        if rec is not None:
            rec.close()
        if fh is not None:
            fh.close()
        s.wall_s = time.perf_counter() - t0
    return s
