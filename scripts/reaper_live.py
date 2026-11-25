"""The Live demo test with REAPER as the player: the Live page of scripts/Complete_Test_Gui.py, source "reaper".

The GUI makes the plan and its rounds exactly as for the other sources (live_test_engine.LiveRun: the same drones,
levels, background layers, round lengths, truth and scoring). What differs is where the sound comes from:

    the sound    each round's parts (E.mix_round, 16 kHz) are spread over the 39 speakers of the array
                 (scripts/reaper_array.py's layout): the drone from one direction per round (VBAP; the
                 approach is a straight dive, so its direction stays put), each background layer as the
                 array project does (wind from every speaker, each from its own point in the loop; cars,
                 helicopter and insects as one copy each from fixed speakers); the war soundscape as its tracks
                 (war_stems.py), each a point source, the whole scene in one direction, new each round (VBAP;
                 r["war_sources"] in live.json). The rounds are written as 39-channel
                 WAVs into a REAPER project in the run folder (reaper/live.RPP), which REAPER opens and plays.
    the control  REAPER's web remote (Preferences > Control/OSC/web > Web browser interface, port 8080):
                 the GUI presses Play and Stop, sets the play position, sets the master volume (playback gain)
                 and reads where REAPER is, every 50 ms.
    automatic    the rounds lie back to back after LEAD_IN_S of background, as on the virtual timeline;
                 REAPER plays straight through.
    manual       REAPER loops a background-only region of IDLE_S (5 min); Send drone jumps to the next drone's
                 region (PRE_S of background, the drone, the buffer after impact, then TAIL_S more), and when
                 the round is over the GUI jumps back into the loop where it would have been had it played on. A round here is that region, not the wait before it, and the
                 background jumps with it (a cut). Stop between drones adds no no-drone round.
    the score    always on the microphone, as in microphone mode: each round's recording is found by REAPER's
                 reported position, then by cross-correlation with what the 39 speakers played, summed.

Results go where every live run's go (artifacts/complete_test/<time>_live_reaper_<trigger>/), with heard/ and
reaper/ (the project and its WAVs; 39 x 16 kHz x 16 bit, about 1.2 MB per second of rounds).

Stereo test (the page's "REAPER: stereo test" box, plan.live["reaper_channels"] = 2): the 39 channels are folded to
outputs 1-2 by each speaker's azimuth (left = 90 degrees), power kept, so a laptop or a pair of speakers plays it all.
The speaker view then shows the two channels on speakers 1 and 2.
Simulated drones on non-held-out sources: a tool, not a detection result (CLAUDE.md section 5).
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import complete_test_engine as E  # noqa: E402
import live_test_engine as L  # noqa: E402
import reaper_array as RA  # noqa: E402
import war_stems as WS  # noqa: E402

SR = L.SR
PORT = 8080
PRE_S = 2.0                                                  # manual: background before the drone in its region
TAIL_S = L.MAX_LAG_S + 1.0                                   # background after a round: heard while the GUI catches up
OPEN_WAIT_S = 600.0                                          # REAPER may first ask to save the project it had open
IDLE_S = 300.0                                               # manual: the background loop between drones (whole
                                                             # passes of every layer; a short one repeats its start)
MARGIN_S = 0.5                                               # a round's recording starts this much before REAPER's
                                                             # position says; the cross-correlation finds the rest
REAPER_APP = {"darwin": ["open", "-a", "REAPER"],
              "win32": [r"C:\Program Files\REAPER (x64)\reaper.exe"]}


class Reaper:
    """REAPER's web remote: one GET per call, commands joined by ';', answers as tab-separated lines."""

    def __init__(self, port: int = PORT):
        self.base = f"http://127.0.0.1:{port}/_/"

    def __call__(self, *cmds: str) -> list[list[str]]:
        try:
            with urllib.request.urlopen(self.base + ";".join(cmds), timeout=2) as f:
                return [ln.split("\t") for ln in f.read().decode("utf-8", "replace").splitlines() if ln]
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"REAPER does not answer on port {PORT}. Start REAPER and turn on its web remote: "
                               "Preferences > Control/OSC/web > Add > Web browser interface, port 8080") from e

    def transport(self) -> tuple[int, float]:
        """(play state: 0 stopped, 1 playing, 2 paused; position in s)."""
        row = next(r for r in self("TRANSPORT") if r[0] == "TRANSPORT")
        return int(row[1]), float(row[2])

    def regions(self) -> list[str]:
        return [r[1] for r in self("REGION") if r[0] == "REGION"]


def spatial(r: dict, m: dict, layers: dict, offset: float) -> np.ndarray:
    """Round r's parts (m, mono) over the 39 speakers (n x 39). r gets the drone's direction."""
    n = len(m["mix"])
    out = np.zeros((n, RA.N), np.float32)
    for name, x in m["parts"].items():
        if name == "drone":
            seed = int(hashlib.md5(f"{r.get('round')}|{r['drone']}".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            r["array_azimuth_deg"], r["array_climb_deg"] = round(float(rng.uniform(0, 360)), 1), round(float(rng.uniform(5, 40)), 1)
            out += x[:, None] * RA.vbap(RA.unit(r["array_azimuth_deg"], r["array_climb_deg"])[None])[0]
            continue
        spread = RA.LAYERS.get(name, (None, None, "all"))[2]
        if spread == "stems":                                # each war track a point source: x split into its parts
            full, rr = layers[name].signal()
            i = int(round(offset * SR)) % len(full)
            parts = {k: np.take(v, np.arange(i, i + n), mode="wrap") * (E.rms_for(r["background"][name]) / rr)
                     for k, v in WS.load(SR).items()}
            if not np.allclose(sum(parts.values()), x, atol=1e-6):
                raise ValueError(f"the war layer is not the sum of {WS.OUT}: plan.war_file must be its mix.wav")
            seed = int(hashlib.md5(f"{r.get('round')}|{r.get('drone')}|{offset:.4f}|war".encode()).hexdigest()[:8], 16)
            r["war_sources"] = WS.place(np.random.default_rng(seed), parts)   # one place each round
            for k, v in parts.items():
                out += (v[:, None] * RA.moving_gains(r["war_sources"][k], n, SR)).astype(np.float32)
            continue
        spk = list(range(RA.N)) if spread == "all" else [s - 1 for s in spread]
        if spread != "all":                                  # one copy, the same on each of its speakers (entry 254)
            out[:, spk] += (x / len(spk))[:, None].astype(np.float32)
            continue
        lay, lv = layers[name], r["background"][name]        # steady noise: each speaker its own point in the loop
        step = len(lay.signal()[0]) / SR / len(spk)
        for j, s in enumerate(spk):
            y = x if j == 0 or lay.once else lay.window(n, offset + j * step, lv)
            out[:, s] += y / np.sqrt(len(spk))              # incoherent copies: the sum at the centre is the level
    return out


def stereo_fold() -> np.ndarray:
    """39 x 2: each speaker's sound to left and right by its azimuth (90 = left), power kept: a test on two speakers."""
    s = np.sin(np.radians(np.array(RA.LAYOUT, float)[:, 0]))
    return np.stack([np.sqrt((1 + s) / 2), np.sqrt((1 - s) / 2)], axis=1)


class ReaperRun(L.LiveRun):
    """A LiveRun whose sound REAPER plays on the array; scored on the microphone."""

    def __init__(self, plan: E.Plan, events, in_dev=None, in_channel: int | str = 0, play_gain_db: float = 0.0,
                 mic_gain_db: float = 0.0):
        WS.load(SR)                                          # the war soundscape as stems: its layer is their sum
        plan.war_file = str((WS.OUT / "mix.wav").relative_to(RA.ROOT))
        super().__init__(plan, events, None, in_dev, in_channel, play_gain_db, mic_gain_db)
        self.mic = True
        self.rp = Reaper()
        self.clipped = 0
        self.stereo = plan.live.get("reaper_channels", RA.N) == 2   # a test on two speakers: the 39 folded to left/right

    # ---------- the project
    def _write(self, f: Path, y: np.ndarray) -> None:
        self.clipped += int(np.sum(np.abs(y) >= 1.0))
        sf.write(f, np.clip(y, -1, 1), SR, subtype="PCM_16")

    def _slot(self, r: dict, P: float, offset: float, d: Path, k: int, tail_s: float = 0.0) -> dict:
        """Round r at project time P: its mix (scoring), its 39 channels (REAPER, with tail_s more background)."""
        t = self.test
        m = E.mix_round(r, self.plan, t.layers_for(r), t.drones, offset)
        ext = dict(r, length_s=len(m["mix"]) / SR + tail_s)
        y = spatial(r, E.mix_round(ext, self.plan, t.layers_for(r), t.drones, offset) if tail_s else m, t.layers_for(r), offset)
        if self.stereo:
            y = (y @ stereo_fold()).astype(np.float32)
        f = d / f"round_{k:04d}.wav"
        self._write(f, y)
        n = len(m["mix"])
        return dict(r=r, m=m, n=n, P=P, offset=offset, file=f.name, len_s=len(y) / SR, ref=y[:n].sum(axis=1))

    def _build(self, say) -> Path:
        """The rounds' WAVs and live.RPP in <run>/reaper/; self.slots in play order."""
        t = self.test
        d = t.out / "reaper"
        d.mkdir(parents=True, exist_ok=True)
        tag = t.out.name
        items, marks, self.slots = [], [], []
        if not self.manual:                                  # lead-in, the rounds back to back, a tail
            first = self.order[0]
            lead = self._slot(dict(first, drone=None, drone_level_db=None, start_s=None, length_s=L.LEAD_IN_S), 0.0,
                              t.start_s + self.skip_s, d, 0)
            items.append(RA.item(0, lead["len_s"], f"round_0000.wav", "lead-in"))
            marks.append(RA.region(1000, 0, L.LEAD_IN_S, f"{tag} lead-in"))
            P = L.LEAD_IN_S
            for i, r in enumerate(self.order):
                say(f"writing the rounds for the 39 speakers: {i + 1} of {len(self.order)}")
                last = i == len(self.order) - 1
                s = self._slot(r, P, t.start_s + self.skip_s + P, d, r["round"], tail_s=TAIL_S if last else 0.0)
                t.offset_s[r["round"]] = s["offset"]
                items.append(RA.item(P, s["len_s"], s["file"], f"round {r['round']}"))
                marks.append(RA.region(r["round"], P, P + s["n"] / SR, f"round {r['round']}: " +
                                       (E.drone_label(r["drone"]) if r["drone"] is not None else "no drone")))
                del s["m"]                                   # 18 MB a round: made again when the round starts (_loop)
                self.slots.append(s)
                P += s["n"] / SR
            self.end_P = P + TAIL_S - 0.5
            extra = "  LOOP 0\n"
        else:                                                # a background loop, then one region per drone setting
            idle_r = dict(self.templates[0], drone=None, drone_level_db=None, start_s=None, length_s=IDLE_S + RA.FADE_S)
            idle = self._slot(idle_r, 0.0, t.start_s, d, 0)
            y = sf.read(d / idle["file"])[0]                 # the loop's end runs on into its start
            f, n = int(RA.FADE_S * SR), int(IDLE_S * SR)
            w = np.linspace(0, np.pi / 2, f)[:, None]
            y[:f] = y[:f] * np.sin(w) + y[n:n + f] * np.cos(w)
            self._write(d / idle["file"], y[:n])
            items.append(RA.item(0, IDLE_S, idle["file"], "background loop"))
            marks.append(RA.region(1000, 0, IDLE_S, f"{tag} background loop"))
            P = IDLE_S + 1.0
            for k, tpl in enumerate(self.templates):
                say(f"writing the drones for the 39 speakers: {k + 1} of {len(self.templates)}")
                imp = t.drones.span[tpl["drone"]][2]
                r = dict(tpl, round=None, start_s=PRE_S, length_s=PRE_S + imp + self.plan.buffer_after_s)
                s = self._slot(r, P, t.start_s + P, d, k + 1, tail_s=TAIL_S)
                items.append(RA.item(P, s["len_s"], s["file"], f"drone {k + 1}"))
                marks.append(RA.region(k + 1, P, P + s["n"] / SR, f"drone {k + 1}: {E.drone_label(tpl['drone'])}"))
                self.slots.append(s)
                P += s["len_s"] + 1.0
            extra = f"  LOOP 1\n  SELECTION 0 {IDLE_S:.6f}\n  SELECTION2 0 {IDLE_S:.6f}\n"
        if self.clipped:
            self.events.put(("warn", f"{self.clipped} samples clipped in the speaker files: lower a level"))
        f = d / "live.RPP"
        f.write_text(RA.project(RA.track("Live test (39 speakers)", items), marks, extra,
                                out0=0 if self.stereo else RA.OUT0))
        RA.install_jsfx(d)
        self.tag = tag
        return f

    def _open(self, f: Path, say) -> None:
        self.rp.transport()                                  # REAPER is up and its web remote answers
        self.rp("1016")                                      # stop whatever plays
        self.rp("40859")                                     # a new project tab: no question about saving the last one
        cmd = REAPER_APP.get(sys.platform, REAPER_APP["win32"])
        subprocess.Popen([*cmd, str(f)])
        say("opening the project in REAPER (if REAPER asks to save the project it had open, answer it)")
        t0 = time.time()
        while not any(self.tag in name for name in self.rp.regions()):
            if time.time() - t0 > OPEN_WAIT_S or self.stop_flag.is_set():   # Stop in the GUI ends the wait sooner
                raise RuntimeError(f"REAPER did not open {f} within {OPEN_WAIT_S / 60:g} min: is a dialog open in "
                                   "REAPER (save the old project?)")
            time.sleep(0.5)
        self.rp("1016", "SET/POS/0", f"SET/REPEAT/{int(self.manual)}", f"SET/TRACK/0/VOL/{self.play_gain:.6f}")

    # ---------- the run
    def _run(self):
        import queue

        import sounddevice as sd
        say = lambda s: self.events.put(("status", s))     # noqa: E731
        in_ch = int(sd.query_devices(self.in_dev, "input")["max_input_channels"])
        L.check_channels(self.in_channels, in_ch)            # before anything is built: a wrong channel fails at once
        in_ch = max(self.in_channels) + 1                    # not all: PortAudio's callback overflows its stack at 94
        self.rp.transport()                                  # and REAPER's web remote answers
        t = self.test
        t.prepare(say)
        if self.stop_flag.is_set():
            self.events.put(("done", "stopped before anything played"))
            return
        self.s0 = int(round(t.start_s * SR))
        f = self._build(say)
        self._open(f, say)
        in_sr = L.pick_rate(self.in_dev, in_ch, "input")
        name = f"{sd.query_devices(self.in_dev, 'input')['name']} ({in_sr} Hz) <- REAPER, " + \
               ("2 speakers (stereo test: the 39 folded to left/right)" if self.stereo else "39 speakers")
        self.plan.live.update(sample_rate=None, input_sample_rate=in_sr, device=name, reaper_project=str(f),
                              input_channels=self.in_channels)
        self.res_in = L.Resampler(in_sr, SR)
        self.heard = L.Ring(L.RING_S)
        self.live_sc = L.LiveScorer(t.scorers, t.thr)
        inq: queue.Queue = queue.Queue()

        def cb_in(indata, frames, ti, status):
            inq.put(L.mix_in(indata, self.in_channels))

        jobs, scorer = self._open_results()
        t_wall = time.time()
        with sd.InputStream(device=self.in_dev, samplerate=in_sr, channels=in_ch, dtype="float32", callback=cb_in, **L.STREAM):
            time.sleep(0.3)
            self.rp("1007")                                  # Play
            self.running = True
            say(f"playing in REAPER; listening on {name}")
            self._loop(inq, jobs)
            self.rp("1016")
            self.running = False
        self._close_results(jobs, scorer, dict(device=name, input_sample_rate=in_sr, reaper_project=str(f),
                                               clipped_samples=self.clipped, wall_s=round(time.time() - t_wall, 1)))

    def _listen(self, inq) -> None:
        blocks = []
        while not inq.empty():
            blocks.append(inq.get_nowait())
        if blocks:
            x16 = self.res_in.push(np.concatenate(blocks)) * self.mic_gain
            if len(x16):
                self.heard.write(x16)
                self._display(x16)
        self.pos = self.heard.end

    def _score(self, jobs, p: dict) -> None:
        """The finished round p to the scorer; its audio is dropped from self.placed, which keeps every round."""
        jobs.put(self._round_job(p))
        p.pop("m")
        p.pop("ref")

    def _levels(self, s: dict, pos: float) -> None:
        a = int((pos - s["P"]) * SR)
        if 0 < a <= s["n"]:
            energy = {k: float(np.sum(np.square(x[max(0, a - L.CHUNK):a]))) for k, x in s["m"]["parts"].items()}
            self._part_levels(self.heard.end, energy, L.CHUNK)

    def _loop(self, inq, jobs) -> None:
        pending: list[dict] = []                             # rounds played, waiting for enough recording
        was_playing = False
        stop_at = None
        idle_since = None                                    # REAPER not playing since (wall clock)
        trace = open(self.test.out / "reaper" / "transport.csv", "w", encoding="utf-8")
        trace.write("wall_s,mic_s,state,position_s\n")
        t_wall = time.time()
        while True:
            time.sleep(0.05)
            self._listen(inq)
            state, pos = self.rp.transport()
            mic = self.heard.end
            trace.write(f"{time.time() - t_wall:.3f},{mic / SR:.4f},{state},{pos:.4f}\n")
            mic_at = lambda P: int(mic - (pos - P) * SR)    # noqa: E731  the recording's sample for project time P
            cur = self.current
            if self.stop_flag.is_set():
                if stop_at is None:
                    self.rp("1016")
                    stop_at = time.time()
                    if cur is not None and not any(p is cur for p in pending):
                        self.placed = [p for p in self.placed if p is not cur]   # the round playing is dropped
                        if self.manual:
                            self.test.rounds = [r for r in self.test.rounds if r is not cur["r"]]
                    self.current = None
                for p in list(pending):                       # REAPER is silent now; the recording runs on
                    if mic >= p["T"] + p["n"] + int(L.MAX_LAG_S * SR):
                        pending.remove(p)
                        p["done"] = True
                        self._score(jobs, p)
                if not pending or time.time() - stop_at > L.MAX_LAG_S + 2:
                    trace.close()
                    return
                continue
            if state == 1:
                was_playing, idle_since = True, None
            elif was_playing:                                # stopped in REAPER itself for 1 s: as Stop
                idle_since = idle_since or time.time()
                if time.time() - idle_since > 1.0:
                    self.events.put(("warn", f"REAPER stopped playing at {pos:.1f} s: the test stops"))
                    self.stop_flag.set()
                continue
            if self.manual:
                if cur is None and self.request.is_set():    # Send drone: jump to the next drone's region
                    s = self.slots[self.k_tpl % len(self.slots)]
                    self.k_tpl += 1
                    self.resume = (pos % IDLE_S, time.time())   # where the loop was: it goes on from there after
                    self.rp(f"SET/POS/{s['P']:.6f}")
                    r = dict(s["r"], round=len(self.test.rounds) + 1)
                    p = dict(s, r=r, done=False, T=None)
                    with self.lock:
                        self.test.rounds.append(r)
                        self.test.offset_s[r["round"]] = s["offset"]
                        self.current = p
                        self.placed.append(p)
                        self.request.clear()
                    continue
                if cur is not None and cur["T"] is None and cur["P"] <= pos < cur["P"] + 1.0:
                    cur["T"] = mic_at(cur["P"]) - int(MARGIN_S * SR)   # REAPER is in the region: anchor it
                    self._announce(cur["r"], cur["T"])
                if cur is not None and cur["T"] is not None:
                    self._levels(cur, pos)
                    if pos >= cur["P"] + cur["n"] / SR:      # the round has played: back to the loop, as if
                        q, t0 = self.resume                  # it had played on meanwhile
                        self.rp(f"SET/POS/{(q + time.time() - t0) % IDLE_S:.6f}")
                        pending.append(cur)
                        self.current = None
            else:
                if self.next_i < len(self.slots) and pos >= self.slots[self.next_i]["P"]:
                    s = self.slots[self.next_i]
                    p = dict(s, done=False, T=mic_at(s["P"]) - int(MARGIN_S * SR), m=self.test.rebuild(s["r"]))
                    del s["ref"]
                    with self.lock:
                        self.current = p
                        self.placed.append(p)
                        self.next_i += 1
                    self._announce(p["r"], p["T"])
                if cur is not None:
                    self._levels(cur, pos)
                    if pos >= cur["P"] + cur["n"] / SR:
                        pending.append(cur)
                        if self.current is cur:              # not the next round, which starts in the same pass
                            self.current = None
                if self.next_i >= len(self.slots) and self.current is None and not pending and pos >= self.end_P:
                    trace.close()
                    return
            for p in list(pending):
                if mic >= p["T"] + p["n"] + int(L.MAX_LAG_S * SR):
                    pending.remove(p)
                    p["done"] = True
                    self._score(jobs, p)
