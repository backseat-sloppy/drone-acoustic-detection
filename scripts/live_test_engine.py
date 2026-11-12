"""The engine behind the Live page of scripts/Complete_Test_Gui.py: the modified drone test, played out
loud in real time on one continuous timeline.

A live test takes the same Plan as the modified drone test (files, speeds, pitch, levels, background
layers, detectors) and the same rounds, in the same order. What differs:

    the timeline   the background layers play without a break from Play to the end: the war soundscape
                   loops again and again, and round after round takes its stretch of the one stream, so
                   round k's background starts where round k-1's ended (LiveTest.offset_s keeps where).
                   A round's levels hold for its own length_s seconds. A layer that plays once (the car)
                   starts at each round's start, as offline.
    real time      the stream goes to the speakers as it is made, and every detector scores it hop by hop
                   for the live display (LiveScorer). The first LEAD_IN_S seconds are background only.
    the trigger    automatic: each round follows the one before, with its drone at the plan's start in
                   it. manual: between rounds only the background plays; send_drone() starts the next
                   drone round at once, with its drone at the round's start (Plan.step_s = length_s, so
                   one start: 0 s), and the next press is taken when that round has ended. After the
                   last drone round the no-drone rounds (if any) play by themselves.
    the end        a test ends when the plan's last round has played, or when Stop is pressed. On Stop a
                   round that has played to its end is still scored; the round that is playing is
                   dropped. Either way the GUI opens the Results window once, at the end.
    the source     virtual: each round is scored on the digital mix it played, so its numbers are the
                   offline engine's on the same mix. microphone: an input stream records (started
                   first) while an output stream plays, each at a rate its device takes; each round is scored on what the microphone heard over that round, found by
                   cross-correlating the recording with the mix (the delay through speaker, air and
                   microphone, up to MAX_LAG_S). The heard audio of every round is kept in heard/.

Results go where the offline test's go, in the same files (rounds.csv, scores.npz, window_snr.npz,
summary.csv, config.json), so Results and Conclusions open them; plus live.json (per round: where it
sat on the timeline, the soundscape offset, the microphone delay and how well it was found).

The SNRs in rounds.csv and window_snr.npz come from the digital parts (what was sent to the speakers),
also in microphone mode: the room and the microphone are not in them. Levels are the plan's dB SPL in
WAV units (full scale = 110 dB SPL); a speaker and a microphone are not calibrated, so in microphone mode
the levels the detectors hear are not the plan's. Simulated drones on non-held-out sources: a tool, not a
detection result (CLAUDE.md section 5).
"""
from __future__ import annotations

import contextlib
import csv
import json
import math
import queue
import threading
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import complete_test_engine as E
from dad.config import DEFAULT
from dad.features import LogMel

SR = E.SR
CHUNK = 1600                                                 # samples per produced block: 0.1 s at 16 kHz
AHEAD_S = 0.5                                                # audio made ahead of the speaker
LEAD_IN_S = 3.0                                              # background only, before the first round
TAIL_S = 0.5                                                 # background after the last round
MAX_LAG_S = 2.0                                             # microphone mode: the longest delay searched (device
                                                             # latencies, the gap between the two streams' starts, air)
MIN_CORR = 0.05                                              # below this the delay is not trusted: the last good one is kept
RING_S = 120.0                                               # how much of the stream is kept in memory
HOP = DEFAULT.samples_per_hop
LOGMEL = LogMel(DEFAULT)
SOURCES = ("virtual", "microphone")
TRIGGERS = ("automatic", "manual")


# ------------------------------------------------------------------ a run on one timeline

class LiveTest(E.Test):
    """A Test whose rounds sit on one continuous timeline: round r's background starts at offset_s[r]
    seconds into the layers instead of the batch's one start. heard(r) is what the microphone heard over
    round r (None in virtual mode)."""

    def __init__(self, plan: E.Plan, out_dir: Path | None = None, device: str = "cpu"):
        super().__init__(plan, out_dir, device)
        self.offset_s: dict[int, float] = {}

    def rebuild(self, r: dict) -> dict:
        return E.mix_round(r, self.plan, self.layers_for(r), self.drones, self.offset_s.get(r["round"], self.start_s))

    def heard(self, r: dict) -> np.ndarray | None:
        f = self.out / "heard" / f"round_{r['round']:04d}.wav"
        return sf.read(f, dtype="float64")[0] if f.exists() else None

    def load_live(self) -> dict:
        """An earlier live run's live.json: the offsets back into offset_s."""
        f = self.out / "live.json"
        info = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        self.offset_s = {int(k): v["offset_s"] for k, v in info.get("rounds", {}).items()}
        return info


def out_folder(plan: E.Plan) -> Path:
    lv = plan.live or {}
    return E.OUT_ROOT / f"{time.strftime('%Y%m%d_%H%M%S')}_live_{lv.get('source', 'virtual')}_{lv.get('trigger', 'automatic')}"


# ------------------------------------------------------------------ small parts

class Ring:
    """The last `seconds` of a 16 kHz signal, addressed by absolute sample index."""

    def __init__(self, seconds: float):
        self.n = int(seconds * SR)
        self.buf = np.zeros(self.n, np.float32)
        self.end = 0                                         # samples written so far

    def write(self, x: np.ndarray) -> None:
        m = len(x)
        x = np.asarray(x, np.float32)[-self.n:]              # a block longer than the ring: only its end is kept
        idx = (self.end + m - len(x) + np.arange(len(x))) % self.n
        self.buf[idx] = x
        self.end += m

    def read(self, a: int, b: int) -> np.ndarray:
        """Samples a to b; what is older than the ring or not written yet reads as 0."""
        out = np.zeros(max(b - a, 0), np.float32)
        lo, hi = max(a, self.end - self.n, 0), min(b, self.end)
        if hi > lo:
            out[lo - a:hi - a] = self.buf[np.arange(lo, hi) % self.n]
        return out


class Resampler:
    """A stream from rate a to rate b without seams: the input is cut into blocks of a multiple of the
    ratio's denominator, and each block is resampled with `pad` samples of its neighbours on both sides
    and cut back. Output sample j is input time j * a / b exactly; the output lags by one block."""

    def __init__(self, sr_in: int, sr_out: int, block_s: float = 0.1, pad_s: float = 0.02):
        fr = Fraction(int(sr_out), int(sr_in))
        self.up, self.down = fr.numerator, fr.denominator
        self.same = self.up == self.down
        self.c = self.down * max(1, round(block_s * sr_in / self.down))
        self.p = self.down * max(1, math.ceil(pad_s * sr_in / self.down))
        self.buf = np.zeros(0)
        self.prev = np.zeros(self.p)
        self.cur = None

    def push(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float64)
        if self.same:
            return x
        self.buf = np.concatenate([self.buf, x])
        outs = []
        while len(self.buf) >= self.c:
            nxt, self.buf = self.buf[:self.c], self.buf[self.c:]
            if self.cur is not None:
                ctx = np.concatenate([self.prev[-self.p:], self.cur, nxt[:self.p]])
                y = resample_poly(ctx, self.up, self.down)
                k0 = self.p * self.up // self.down
                outs.append(y[k0:k0 + self.c * self.up // self.down])
                self.prev = self.cur
            self.cur = nxt
        return np.concatenate(outs) if outs else np.zeros(0)


class OutFifo:
    """Frames for the speaker, filled by the engine and emptied by the audio callback."""

    def __init__(self, channels: int):
        self.ch, self.parts, self.n, self.lock = channels, [], 0, threading.Lock()
        self.played = 0                                      # real frames handed to the device
        self.underflows = 0
        self.closed = False                                  # nothing more will come: silence is not an underflow

    def put(self, x: np.ndarray) -> None:
        with self.lock:
            self.parts.append(np.asarray(x, np.float32))
            self.n += len(x)

    def take(self, out: np.ndarray) -> None:
        frames, i = len(out), 0
        with self.lock:
            while i < frames and self.parts:
                p = self.parts[0]
                k = min(frames - i, len(p))
                out[i:i + k] = p[:k, None]
                if k == len(p):
                    self.parts.pop(0)
                else:
                    self.parts[0] = p[k:]
                i += k
            self.n -= i
            self.played += i
        if i < frames:
            out[i:] = 0
            if self.played and not self.closed:              # not the silence before the first block or after the last
                self.underflows += 1


def estimate_lag(ref: np.ndarray, rec: np.ndarray, max_lag: int) -> tuple[int, float]:
    """The delay (samples, 0 to max_lag) at which `rec` best matches `ref`, and the normalised peak (0-1)."""
    n = len(ref) + len(rec)
    nfft = 1 << (n - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(rec, nfft) * np.conj(np.fft.rfft(ref, nfft)), nfft)[:max_lag + 1]
    j = int(np.argmax(c))
    seg = rec[j:j + len(ref)]
    norm = float(np.sqrt(np.sum(ref.astype(np.float64) ** 2) * np.sum(seg.astype(np.float64) ** 2)))
    return j, float(c[j] / norm) if norm > 0 else 0.0


class LiveScorer:
    """Every detector on a 16 kHz stream, hop by hop on the shared grid (the first 1 s window ends at
    1 s; a 2 s model has no score before 2 s), for the live display. The audio it gets is the timeline's
    (virtual) or the microphone's (microphone), from the stream's first sample."""

    def __init__(self, scorers: dict, thr: dict):
        self.sc, self.thr = scorers, thr
        self.ring = Ring(10.0)
        self.k = 0
        self.cols = 0                                        # spectrogram frames handed out so far

    def feed(self, x: np.ndarray) -> list[dict]:
        self.ring.write(x)
        ends = []
        while DEFAULT.samples_per_window + self.k * HOP <= self.ring.end:
            ends.append(DEFAULT.samples_per_window + self.k * HOP)
            self.k += 1
        if not ends:
            return []
        conf = {}
        for name, sc in self.sc.items():
            n = sc.cfg.samples_per_window
            ok = [e for e in ends if e >= n]
            s = np.full(len(ends), np.nan)
            if ok:
                y = np.asarray(sc.score(np.stack([self.ring.read(e - n, e) for e in ok])), np.float64)
                s[len(ends) - len(ok):] = E.dwg.squash(y, self.thr[name]) if name == "comb" else y
            conf[name] = s
        out = []
        for i, e in enumerate(ends):
            w = self.ring.read(e - DEFAULT.samples_per_window, e)
            target = e // DEFAULT.hop_length
            c = min(target - self.cols, DEFAULT.samples_per_window // DEFAULT.hop_length - 3)
            cols = None
            if c > 0:
                tail = w[-((c - 1) * DEFAULT.hop_length + DEFAULT.n_fft):]
                cols = (10 / np.log(10) * LOGMEL.frames(tail)[:, -c:]).astype(np.float32)
            self.cols = target
            lvl = float(10 * np.log10(np.mean(w.astype(np.float64) ** 2) + 1e-20))
            out.append(dict(t=e / SR, conf={k: float(v[i]) for k, v in conf.items()}, level_dbfs=lvl, cols=cols))
        return out


# ------------------------------------------------------------------ devices

def pick_rate(dev, channels: int, kind: str) -> int:
    """A sample rate the device takes: 16 kHz first (no resampling), then its own, 48 and 44.1 kHz."""
    import sounddevice as sd
    own = int(sd.query_devices(dev, kind)["default_samplerate"])
    check = sd.check_output_settings if kind == "output" else sd.check_input_settings
    tried = []
    for sr in dict.fromkeys([SR, own, 48000, 44100]):
        try:
            check(dev, channels=channels, samplerate=sr, dtype="float32")
            return sr
        except Exception as e:                               # noqa: BLE001  (the next rate)
            tried.append(f"{sr} Hz: {e}")
    raise RuntimeError(f"no sample rate works for the {kind} device:\n" + "\n".join(tried))


# ------------------------------------------------------------------ the run

class LiveRun:
    """One live test. start() returns at once; the work runs in threads and reports to `events` as
    (kind, payload): status, round_start, round_done, dec, lag, warn, done, error."""

    def __init__(self, plan: E.Plan, events: queue.Queue, out_dev=None, in_dev=None, in_channel: int = 0,
                 play_gain_db: float = 0.0, mic_gain_db: float = 0.0):
        self.plan, self.events = plan, events
        self.source = plan.live["source"]
        self.trigger = plan.live["trigger"]
        self.mic = self.source == "microphone"
        self.out_dev, self.in_dev, self.in_channel = out_dev, in_dev, int(in_channel)
        self.play_gain = 10 ** (play_gain_db / 20)
        self.mic_gain = 10 ** (mic_gain_db / 20)
        self.test = LiveTest(plan, out_folder(plan))
        self.order = list(self.test.rounds)                  # drone rounds, then the no-drone rounds
        self.end_pos = None                                  # timeline sample where the stream ends (plan done)
        self.L = int(round(plan.length_s * SR))
        self.stop_flag = threading.Event()
        self.request = threading.Event()                     # manual: a drone was asked for
        self.thread = None
        self.next_i = 0
        self.current = None                                  # the round being played: dict(r, T, m)
        self.placed: list[dict] = []                         # every round started: r, T, m, done
        self.lag = None                                      # microphone delay, samples
        self.info_rounds: dict[int, dict] = {}
        self.scored = 0
        self.lock = threading.Lock()

    # ---------- controls (GUI thread)
    def start(self):
        self.thread = threading.Thread(target=self._main, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag.set()

    def can_send(self) -> bool:
        """Manual mode: a press now would start a drone (lead-in over, no round playing, a drone round next)."""
        with self.lock:
            return (self.trigger == "manual" and self.running and self.current is None and not self.request.is_set()
                    and not self.stop_flag.is_set() and self.next_i < len(self.order)
                    and self.order[self.next_i]["drone"] is not None and self.pos >= LEAD_IN_S * SR)

    def send_drone(self) -> bool:
        if not self.can_send():
            return False
        self.request.set()
        return True

    # ---------- the timeline
    def _bg_round(self) -> dict:
        """The round whose background plays between rounds: the next one, or the last."""
        return self.order[min(self.next_i, len(self.order) - 1)]

    def _idle(self, t: int, n: int) -> np.ndarray:
        """Background only, timeline samples t..t+n: the looped layers at the next round's levels."""
        r = self._bg_round()
        layers = self.test.layers_for(r)
        out = np.zeros(n)
        for name, lv in r["background"].items():
            if not layers[name].once:
                out += layers[name].window(n, (self.s0 + t) / SR, lv)
        return out

    def _maybe_start_round(self, t: int) -> None:
        if t < LEAD_IN_S * SR or self.stop_flag.is_set():
            return
        if self.next_i >= len(self.order):                   # the plan is done: a short tail, then the end
            if self.end_pos is None:
                self.end_pos = t + int((TAIL_S + (MAX_LAG_S if self.mic else 0.0)) * SR)
            return
        r = self.order[self.next_i]
        if self.trigger == "manual" and r["drone"] is not None and not self.request.is_set():
            return
        offset = (self.s0 + t) / SR
        m = E.mix_round(r, self.plan, self.test.layers_for(r), self.test.drones, offset)
        with self.lock:
            self.test.offset_s[r["round"]] = offset
            self.current = dict(r=r, T=t, m=m, done=False)
            self.placed.append(self.current)
            self.next_i += 1
            self.request.clear()
        span = m["truth"].drones[0] if m["truth"].drones else None
        self.events.put(("round_start", dict(
            round=r["round"], index=self.next_i, total=len(self.order), t0=t / SR, t1=(t + self.L) / SR,
            drone=None if span is None else (t / SR + span["start_s"], t / SR + span["end_s"]),
            label=E.drone_label(r["drone"]) if r["drone"] is not None else "no drone",
            level=r["drone_level_db"], background=dict(r["background"]))))

    def _next_chunk(self, pos: int, n: int) -> np.ndarray:
        out = np.zeros(n)
        i = 0
        while i < n:
            t = pos + i
            cur = self.current
            if cur is not None and t >= cur["T"] + self.L:
                with self.lock:
                    self.current = None
                cur = None
            if cur is None:
                self._maybe_start_round(t)
                cur = self.current
            if cur is not None:
                k = min(n - i, cur["T"] + self.L - t)
                out[i:i + k] = cur["m"]["mix"][t - cur["T"]:t - cur["T"] + k]
            else:
                k = n - i                                    # idle to the end of the block: a press waits for the next one
                out[i:i + k] = self._idle(t, k)
            i += k
        return out

    # ---------- the threads
    def _main(self):
        try:
            self._run()
        except Exception as e:                               # noqa: BLE001  (shown in the window)
            import traceback
            traceback.print_exc()
            self.events.put(("error", f"{type(e).__name__}: {e}"))
        finally:
            self.running = False

    running = False
    pos = 0

    def _run(self):
        import sounddevice as sd
        say = lambda s: self.events.put(("status", s))     # noqa: E731
        t = self.test
        t.prepare(say)
        if self.stop_flag.is_set():
            self.events.put(("done", "stopped before anything played"))
            return
        self.s0 = int(round(t.start_s * SR))
        out_ch = int(sd.query_devices(self.out_dev, "output")["max_output_channels"])
        out_ch = min(out_ch, 2)
        in_ch = int(sd.query_devices(self.in_dev, "input")["max_input_channels"]) if self.mic else 0
        if self.mic and not 0 <= self.in_channel < in_ch:
            raise ValueError(f"input channel {self.in_channel}: the microphone has {in_ch} channel(s)")
        sr = pick_rate(self.out_dev, out_ch, "output")
        in_sr = pick_rate(self.in_dev, in_ch, "input") if self.mic else None
        self.sr = sr
        self.fifo = OutFifo(out_ch)
        self.res_out = Resampler(SR, sr)
        self.res_in = Resampler(in_sr, SR) if self.mic else None
        self.digital = Ring(RING_S)                          # the timeline as sent (16 kHz)
        self.heard = Ring(RING_S) if self.mic else None      # the microphone (16 kHz)
        self.live_sc = LiveScorer(t.scorers, t.thr)
        inq: queue.Queue = queue.Queue()
        overflows = [0]

        def cb_out(outdata, frames, ti, status):
            self.fifo.take(outdata)

        def cb_in(indata, frames, ti, status):
            if status.input_overflow:
                overflows[0] += 1
            inq.put(indata[:, self.in_channel].copy())

        # Two streams, not one duplex stream: Windows refuses a duplex stream whose input and output are on
        # different host APIs (MME, WASAPI, ...). The microphone starts first, so the speaker's sound always
        # arrives later in the recording; each round's delay is measured (estimate_lag), which also absorbs
        # the gap between the two starts and any drift between the two devices' clocks.
        out_stream = sd.OutputStream(device=self.out_dev, samplerate=sr, channels=out_ch, dtype="float32",
                                     latency="high", callback=cb_out)
        streams = [out_stream]
        name = sd.query_devices(self.out_dev, "output")["name"]
        if self.mic:
            streams.insert(0, sd.InputStream(device=self.in_dev, samplerate=in_sr, channels=in_ch, dtype="float32",
                                             latency="high", callback=cb_in))
            name = f"{sd.query_devices(self.in_dev, 'input')['name']} ({in_sr} Hz) -> {name}"
        self.plan.live.update(sample_rate=sr, input_sample_rate=in_sr, device=name)
        out_lat = float(out_stream.latency)

        fields = E.row_fields(self.plan) + ["timeline_s", "mic_lag_ms", "mic_corr"]
        scores, snrs = {}, {}
        jobs: queue.Queue = queue.Queue()
        fh = open(t.out / "rounds.csv", "w", newline="", encoding="utf-8")
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        scorer = threading.Thread(target=self._scorer, args=(jobs, w, fh, scores, snrs), daemon=True)
        scorer.start()

        produced = 0                                         # timeline samples made
        fed = 0                                              # virtual: timeline samples given to the live display
        self.running = True
        say(f"playing: {name}, {sr} Hz" + (f", input channel {self.in_channel}" if self.mic else ""))
        t_wall = time.time()
        stopped_at = None
        with contextlib.ExitStack() as opened:
            for s in streams:                                # entered in order: the microphone first
                opened.enter_context(s)
            while True:
                # 1. make audio until AHEAD_S is queued for the speaker
                while self.fifo.n < AHEAD_S * sr and not self.fifo.closed:   # after Stop only background
                    x = self._next_chunk(produced, CHUNK)
                    self.digital.write(x)
                    y = np.clip(self.res_out.push(x) * self.play_gain, -1.0, 1.0)
                    if len(y):
                        self.fifo.put(y)
                    produced += CHUNK
                    self.pos = produced
                    if self.end_pos is not None and produced >= self.end_pos + 2 * CHUNK:   # 2 blocks more: the
                        self.fifo.closed = True              # resampler hands out the last real one
                # 2. what has been heard: the live display, and rounds that are complete
                played16 = int(self.fifo.played * SR / sr)
                if self.mic:
                    blocks = []
                    while not inq.empty():
                        blocks.append(inq.get_nowait())
                    if blocks:
                        x16 = self.res_in.push(np.concatenate(blocks)) * self.mic_gain
                        if len(x16):
                            self.heard.write(x16)
                            self._display(x16)
                    have = self.heard.end
                else:
                    target = max(0, played16 - int(out_lat * SR))
                    if target > fed:
                        self._display(self.digital.read(fed, target))
                        fed = target
                    have = played16
                for p in list(self.placed):
                    if p["done"]:
                        continue
                    need = p["T"] + self.L + (int(MAX_LAG_S * SR) if self.mic else 0)
                    if have >= need:
                        p["done"] = True
                        jobs.put(self._round_job(p))
                if self.mic and overflows[0]:
                    self.events.put(("warn", f"{overflows[0]} input overflow(s)"))
                    overflows[0] = 0
                # 3. the end: the plan is done, or Stop. After Stop a round that had played to its end is still
                # scored (with the microphone that needs up to MAX_LAG_S more recording); the round playing is dropped.
                if self.end_pos is not None and have >= self.end_pos and all(p["done"] for p in self.placed):
                    break
                if self.stop_flag.is_set():
                    if stopped_at is None:
                        stopped_at = time.time()
                        with self.lock:
                            dropped = self.current
                            self.current = None
                        if dropped is not None and dropped in self.placed and not dropped["done"]:
                            self.placed.remove(dropped)
                    if all(p["done"] for p in self.placed) or time.time() - stopped_at > MAX_LAG_S + 1.0:
                        break
                time.sleep(0.02)
        jobs.put(None)
        say("scoring the last rounds...")
        scorer.join()
        fh.close()
        np.savez_compressed(t.out / "scores.npz", **scores)
        np.savez_compressed(t.out / "window_snr.npz", **snrs)
        E.write_summary(t)
        cfg = json.loads((t.out / "config.json").read_text(encoding="utf-8"))
        cfg["plan"]["live"] = self.plan.live                 # with the rate and the devices it ran on
        (t.out / "config.json").write_text(json.dumps(cfg, indent=1, default=str), encoding="utf-8")
        live = dict(source=self.source, trigger=self.trigger, sample_rate=sr, device=name, lead_in_s=LEAD_IN_S,
                    play_gain_db=20 * np.log10(self.play_gain), mic_gain_db=20 * np.log10(self.mic_gain),
                    underflows=self.fifo.underflows, stopped=self.stop_flag.is_set(),
                    wall_s=round(time.time() - t_wall, 1), rounds={str(k): v for k, v in self.info_rounds.items()})
        (t.out / "live.json").write_text(json.dumps(live, indent=1), encoding="utf-8")
        n = len(t.results)
        self.events.put(("done", f"{'stopped' if self.stop_flag.is_set() else 'plan done'}: {n} of {len(self.order)} "
                                 f"round{'s' * (len(self.order) != 1)} scored. Results in {t.out}" + (f"  ·  {self.fifo.underflows} speaker underflow(s)"
                                                          if self.fifo.underflows else "")))

    def _display(self, x16: np.ndarray) -> None:
        for d in self.live_sc.feed(x16):
            self.events.put(("dec", d))

    def _round_job(self, p: dict) -> dict:
        """What the scorer needs for a finished round: the digital mix, and in microphone mode what was heard."""
        r, T, m = p["r"], p["T"], p["m"]
        job = dict(r=r, T=T, m=m, heard=None, lag=None, corr=None)
        if self.mic:
            rec = self.heard.read(T, T + self.L + int(MAX_LAG_S * SR))
            j, c = estimate_lag(m["mix"].astype(np.float32), rec, int(MAX_LAG_S * SR))
            if c >= MIN_CORR or self.lag is None:
                if c < MIN_CORR:
                    self.events.put(("warn", f"round {r['round']}: the microphone barely matches the mix "
                                             f"(correlation {c:.2f}); delay {1000 * j / SR:.0f} ms taken anyway"))
                self.lag = j
            else:
                self.events.put(("warn", f"round {r['round']}: correlation {c:.2f}, kept the last delay "
                                         f"{1000 * self.lag / SR:.0f} ms"))
            job.update(heard=rec[self.lag:self.lag + self.L].astype(np.float64), lag=self.lag, corr=c)
            self.events.put(("lag", 1000 * self.lag / SR))
        return job

    def _scorer(self, jobs: queue.Queue, w, fh, scores: dict, snrs: dict) -> None:
        t = self.test
        while True:
            job = jobs.get()
            if job is None:
                return
            try:
                r, m = job["r"], job["m"]
                ms = m if job["heard"] is None else dict(parts=m["parts"], mix=job["heard"], truth=m["truth"])
                res = E.score_round(ms, t.thr, t.scorers)
                q = E.snr(m)                                 # the digital parts: what was sent
                for v in res.values():
                    v["row"].update(q)
                t.results[r["round"]] = {k: v["row"] for k, v in res.items()}
                for k, v in res.items():
                    scores[f"{r['round']}|{k}"] = E.on_grid(v["dec"].score, t.scorers[k].cfg)
                    w.writerow(dict(round=r["round"], kind=self.plan.kind,
                                    drone=E.drone_label(r["drone"]) if r["drone"] is not None else "none",
                                    drone_level_db=r["drone_level_db"], start_s=r["start_s"],
                                    **{f"{n}_db": r["background"].get(n, "off") for n in self.plan.layers}, **r["pd"],
                                    detector=k, threshold=t.thr[k], **v["row"], timeline_s=round(job["T"] / SR, 3),
                                    mic_lag_ms=None if job["lag"] is None else round(1000 * job["lag"] / SR, 1),
                                    mic_corr=None if job["corr"] is None else round(job["corr"], 3)))
                if r["drone"] is not None and res:
                    n_grid = max(len(scores[f"{r['round']}|{k}"]) for k in res)
                    for win, x in E.round_window_snrs(m, n_grid).items():
                        snrs[f"{r['round']}|{win}"] = x
                fh.flush()
                if job["heard"] is not None:
                    (t.out / "heard").mkdir(exist_ok=True)
                    sf.write(t.out / "heard" / f"round_{r['round']:04d}.wav", np.clip(job["heard"], -1, 1).astype(np.float32),
                             SR, subtype="FLOAT")
                self.info_rounds[r["round"]] = dict(
                    timeline_s=round(job["T"] / SR, 4), offset_s=t.offset_s[r["round"]],
                    mic_lag_ms=None if job["lag"] is None else round(1000 * job["lag"] / SR, 2),
                    mic_corr=None if job["corr"] is None else round(job["corr"], 4))
                self.scored += 1
                self.events.put(("round_done", dict(round=r["round"], scored=self.scored,
                                                    label=E.drone_label(r["drone"]) if r["drone"] is not None else "no drone",
                                                    conf={k: (v["row"]["mean_conf_drone"], v["row"]["mean_conf_no_drone"])
                                                          for k, v in res.items()})))
            except Exception as e:                           # noqa: BLE001  (one round lost, the run goes on)
                import traceback
                traceback.print_exc()
                self.events.put(("warn", f"round {job['r']['round']} not scored: {type(e).__name__}: {e}"))
