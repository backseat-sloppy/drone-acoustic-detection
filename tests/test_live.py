"""The live pipeline must make the same decisions as `dad wav` on the same audio."""
import csv

import numpy as np
import pytest
import soundfile as sf

from dad.config import DEFAULT, FrontendConfig
from dad.live import Detector, LivePipeline, WindowBuffer, run, wav_blocks
from dad.realrun import CombScorer
from dad.wavio import Clip, read_wav, stream


class Probe:
    """A scorer that reports a fingerprint of each window it gets."""
    def score(self, audio):
        return audio.astype(np.float64) @ np.linspace(-1, 1, audio.shape[1])


def chunks(x, rng):
    i = 0
    while i < len(x):
        n = int(rng.integers(1, 3000))
        yield x[i:i + n]
        i += n


@pytest.mark.parametrize("hop", [0.125, 0.1, 0.05])
def test_windows_match_offline_stream_at_16k(hop):
    cfg = FrontendConfig(hop_seconds=hop)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(int(4.3 * cfg.sample_rate)).astype(np.float32) * 0.1
    off = stream(Clip("x", x, cfg.sample_rate, 1, "FLOAT"), Probe(), cfg, threshold=0.0)
    pipe = LivePipeline([Detector("probe", Probe(), 0.0)], cfg.sample_rate, cfg)
    got = [d for b in chunks(x, rng) for d in pipe.feed(b)]
    assert len(got) == len(off.score)
    np.testing.assert_allclose([d.t_end for d in got], off.t_end)
    np.testing.assert_allclose([d.scores["probe"] for d in got], off.score, rtol=1e-6, atol=1e-6)
    assert [d.drone["probe"] for d in got] == list(off.drone)


def test_block_cut_does_not_matter():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(3 * DEFAULT.sample_rate).astype(np.float32)
    a = WindowBuffer(DEFAULT.sample_rate)
    b = WindowBuffer(DEFAULT.sample_rate)
    wa = a.feed(x)
    wb = [w for c in chunks(x, rng) for w in b.feed(c)]
    assert len(wa) == len(wb) == 1 + (3 * DEFAULT.sample_rate - DEFAULT.samples_per_window) // DEFAULT.samples_per_hop
    for (ta, xa), (tb, xb) in zip(wa, wb):
        assert ta == tb
        np.testing.assert_array_equal(xa, xb)


def test_resampled_input_matches_whole_file_resampling(tmp_path):
    """A 48 kHz device: each window is resampled on its own, the file all at once. They agree
    except in the last millisecond or so, where the live resampler cannot see the future."""
    sr = 48_000
    t = np.arange(3 * sr) / sr
    x = (0.3 * np.sin(2 * np.pi * 180 * t) + 0.2 * np.sin(2 * np.pi * 2400 * t)).astype(np.float32)
    p = tmp_path / "tone48k.wav"
    sf.write(p, x, sr, subtype="FLOAT")
    whole = read_wav(p).audio
    buf = WindowBuffer(sr)
    got = [w for c in chunks(x, np.random.default_rng(2)) for w in buf.feed(c)]
    n, h = DEFAULT.samples_per_window, DEFAULT.samples_per_hop
    assert len(got) == 1 + (len(whole) - n) // h
    for k, (t_end, w) in enumerate(got):
        ref = whole[k * h:k * h + n]
        assert t_end == pytest.approx((k * h + n) / DEFAULT.sample_rate)
        np.testing.assert_allclose(w[:-32], ref[:-32], atol=2e-3)


def test_run_from_wav_writes_csv_and_a_replayable_recording(tmp_path):
    rng = np.random.default_rng(3)
    x = (0.05 * rng.standard_normal(int(2.5 * DEFAULT.sample_rate))).astype(np.float32)
    src = tmp_path / "in.wav"
    sf.write(src, np.stack([x, -x], axis=1), DEFAULT.sample_rate, subtype="FLOAT")
    comb = CombScorer(DEFAULT)
    pipe = LivePipeline([Detector("comb", comb, 1.0)], DEFAULT.sample_rate)
    sr, ch, blocks = wav_blocks(src, blocksize=700, realtime=False)
    assert (sr, ch) == (DEFAULT.sample_rate, 2)
    s = run(pipe, sr, blocks, out_csv=tmp_path / "d.csv", record=tmp_path / "rec.wav", quiet=True)
    off = stream(read_wav(src), comb, DEFAULT, threshold=1.0)
    assert s.decisions == len(off.score) == 13
    rows = list(csv.DictReader(open(tmp_path / "d.csv", encoding="utf-8")))
    np.testing.assert_allclose([float(r["score"]) for r in rows], off.score, rtol=1e-5)
    rec, rsr = sf.read(tmp_path / "rec.wav", dtype="float32")
    assert rsr == sr and rec.shape == (len(x), 2)
    np.testing.assert_array_equal(rec[:, 0], x)


def test_duration_stops_early(tmp_path):
    x = np.zeros(5 * DEFAULT.sample_rate, np.float32)
    sf.write(tmp_path / "z.wav", x, DEFAULT.sample_rate, subtype="FLOAT")
    pipe = LivePipeline([Detector("probe", Probe(), None)], DEFAULT.sample_rate)
    sr, _, blocks = wav_blocks(tmp_path / "z.wav", blocksize=1600, realtime=False)
    s = run(pipe, sr, blocks, duration=2.0, quiet=True)
    assert s.audio_s == pytest.approx(2.0)
    assert s.decisions == 9                          # windows ending at 1.0, 1.125, ..., 2.0 s


# ------------------------------------------------------------------ sections

def _session(tmp_path, seconds, section_s, sr=DEFAULT.sample_rate, block=1234):
    from dad.live import SectionRecorder
    rng = np.random.default_rng(4)
    x = (0.1 * rng.standard_normal((int(seconds * sr), 2))).astype(np.float32)
    det = Detector("probe", Probe(), 0.0)
    pipe = LivePipeline([det], sr)
    rec = SectionRecorder(tmp_path, "run", sr, 2, [det], section_s=section_s)
    decs = []
    for i in range(0, len(x), block):
        b = x[i:i + block]
        rec.audio(b)
        for d in pipe.feed(b[:, 0]):
            rec.decision(d, clock="12:00:00")
            decs.append(d)
    return x, decs, rec.close()


def test_sections_split_audio_exactly_and_side_by_side_in_csv(tmp_path):
    x, decs, files = _session(tmp_path, 50.0, 20.0)
    wavs = sorted(p for p in files if p.suffix == ".wav")
    assert [p.name for p in wavs] == ["run_section_001.wav", "run_section_002.wav", "run_section_003.wav"]
    parts = [sf.read(p, dtype="float32")[0] for p in wavs]
    assert [len(p) for p in parts] == [20 * DEFAULT.sample_rate, 20 * DEFAULT.sample_rate, 10 * DEFAULT.sample_rate]
    np.testing.assert_array_equal(np.concatenate(parts), x)

    rows = list(csv.reader(open(tmp_path / "run.csv", encoding="utf-8")))
    head, body = rows[0], rows[1:]
    assert head[:5] == ["t_in_section_s", "section_001_clock", "section_001_level_dbfs",
                        "section_001_probe_score", "section_001_probe_drone"]
    assert "section_003_probe_score" in head and "section_004_probe_score" not in head
    assert len(body) == 160                          # 0.125, 0.25, ..., 20.0 s into a section
    assert [float(r[0]) for r in body[:2]] == [0.125, 0.25] and float(body[-1][0]) == 20.0
    col = {h: i for i, h in enumerate(head)}
    # section 1 starts with the first window at 1.0 s; section 3 stops at 10 s
    assert body[6][col["section_001_probe_score"]] == "" and body[7][col["section_001_probe_score"]] != ""
    assert body[79][col["section_003_probe_score"]] != "" and body[80][col["section_003_probe_score"]] == ""
    # every decision is in its section's column at its offset, once
    by_t = {round(d.t_end, 4): d.scores["probe"] for d in decs}
    n = 0
    for r in body:
        for k in (1, 2, 3):
            v = r[col[f"section_{k:03d}_probe_score"]]
            if v:
                t = round((k - 1) * 20 + float(r[0]), 4)
                assert float(v) == pytest.approx(by_t[t], rel=1e-5, abs=1e-6)
                n += 1
    assert n == len(decs)


def test_sections_off_gives_one_wav_and_one_section(tmp_path):
    x, decs, files = _session(tmp_path, 25.0, None)
    assert sorted(p.name for p in files) == ["run.csv", "run.wav"]
    np.testing.assert_array_equal(sf.read(tmp_path / "run.wav", dtype="float32")[0], x)
    rows = list(csv.reader(open(tmp_path / "run.csv", encoding="utf-8")))
    assert not any(h.startswith("section_002") for h in rows[0])
    assert len(rows) - 1 == len(decs)


def test_sections_at_a_device_rate_not_a_multiple_of_the_hop(tmp_path):
    x, decs, files = _session(tmp_path, 7.0, 2.5, sr=44_100, block=1000)
    parts = [sf.read(p, dtype="float32")[0] for p in sorted(files) if p.suffix == ".wav"]
    assert [len(p) for p in parts] == [110_250, 110_250, 88_200]
    np.testing.assert_array_equal(np.concatenate(parts), x)


def test_excel_csv_holds_the_same_numbers(tmp_path):
    """sep=; first, `;` between fields, decimal commas; parsed back it equals the plain CSV."""
    from dad.live import SectionRecorder
    rng = np.random.default_rng(5)
    x = (0.1 * rng.standard_normal(int(12 * DEFAULT.sample_rate))).astype(np.float32)
    det = Detector("probe", Probe(), 0.0)
    for name, excel in (("plain", False), ("excel", True)):
        pipe = LivePipeline([det], DEFAULT.sample_rate)
        rec = SectionRecorder(tmp_path, name, DEFAULT.sample_rate, 1, [det], section_s=5.0, audio=False, excel=excel)
        for i in range(0, len(x), 999):
            for d in pipe.feed(x[i:i + 999]):
                rec.decision(d, clock="12:00:00")
        rec.close()
    raw = open(tmp_path / "excel.csv", encoding="utf-8", newline="").read()
    assert raw.startswith("sep=;\r\n") and not raw.startswith("﻿")
    plain = list(csv.reader(open(tmp_path / "plain.csv", encoding="utf-8")))
    excel = list(csv.reader(open(tmp_path / "excel.csv", encoding="utf-8"), delimiter=";"))[1:]
    assert excel[0] == plain[0]                      # same header
    assert "." not in "".join(c for r in excel[1:] for c in r)
    assert any("," in c for r in excel[1:] for c in r)
    for a, b in zip(plain[1:], excel[1:], strict=True):
        assert [c.replace(".", ",") for c in a] == b


# ------------------------------------------------------- sections by calls

def test_called_sections_save_only_between_begin_and_end(tmp_path):
    from dad.live import CalledSectionRecorder
    sr = DEFAULT.sample_rate
    rng = np.random.default_rng(6)
    x = (0.1 * rng.standard_normal((12 * sr, 2))).astype(np.float32)
    det = Detector("probe", Probe(), 0.0)
    pipe = LivePipeline([det], sr)
    rec = CalledSectionRecorder(tmp_path, "run", sr, 2, [det])
    marks = {int(2.3 * sr): "begin", int(5.1 * sr): "end", int(7.0 * sr): "begin"}   # left open: close() ends it
    block, got = 800, []
    for i in range(0, len(x), block):
        for at, what in marks.items():
            if i <= at < i + block:
                got.append(rec.begin() if what == "begin" else rec.end())
        b = x[i:i + block]
        rec.audio(b)
        for d in pipe.feed(b[:, 0]):
            rec.decision(d, clock="12:00:00")
    with pytest.raises(RuntimeError):
        rec.begin()                                   # section 2 is open
    files = rec.close()
    assert got == [1, 1, 2]
    # a mark takes effect at the block boundary where it is made
    w1 = sf.read(tmp_path / "run_section_001.wav", dtype="float32")[0]
    w2 = sf.read(tmp_path / "run_section_002.wav", dtype="float32")[0]
    start1 = rec.marks[0]["start_s"] * sr
    np.testing.assert_array_equal(w1, x[int(round(start1)):int(round(rec.marks[0]["end_s"] * sr))])
    assert len(w2) == len(x) - int(round(rec.marks[1]["start_s"] * sr))
    assert abs(start1 - 2.3 * sr) <= block

    rows = list(csv.reader(open(tmp_path / "run.csv", encoding="utf-8")))
    head, body = rows[0], rows[1:]
    assert head[0] == "n_in_section"
    assert head[1:4] == ["section_001_t_in_section_s", "section_001_clock", "section_001_level_dbfs"]
    assert not any(h.startswith("section_003") for h in head)
    col = {h: i for i, h in enumerate(head)}
    t1 = [float(r[col["section_001_t_in_section_s"]]) for r in body if r[col["section_001_t_in_section_s"]]]
    t2 = [float(r[col["section_002_t_in_section_s"]]) for r in body if r[col["section_002_t_in_section_s"]]]
    span1 = rec.marks[0]["end_s"] - rec.marks[0]["start_s"]
    assert 0 < t1[0] <= 0.125 and t1[-1] <= span1 and np.allclose(np.diff(t1), 0.125)
    assert len(t1) == pytest.approx(span1 / 0.125, abs=1)
    assert 0 < t2[0] <= 0.125 and len(t2) == pytest.approx((12 - rec.marks[1]["start_s"]) / 0.125, abs=1)
    assert sorted(p.name for p in files) == ["run.csv", "run_section_001.wav", "run_section_002.wav"]


def test_end_without_a_section_is_refused(tmp_path):
    from dad.live import CalledSectionRecorder
    rec = CalledSectionRecorder(tmp_path, "run", DEFAULT.sample_rate, 1, [Detector("probe", Probe(), 0.0)])
    with pytest.raises(RuntimeError):
        rec.end()
    assert rec.close() == []                          # nothing was saved, nothing written


def test_control_round_trip_and_errors():
    from dad import live_control as lc
    seen = []

    def handler(req):
        seen.append(req["cmd"])
        if req["cmd"] == "end":
            return dict(ok=False, error="no section is open")
        return dict(ok=True, section=3)

    srv = lc.ControlServer(handler, port=0).start()
    try:
        assert lc.start_section(port=srv.port) == 3
        with pytest.raises(lc.LiveControlError, match="no section is open"):
            lc.end_section(port=srv.port)
        assert lc.status(port=srv.port)["ok"] is True
        assert seen == ["start", "end", "status"]
    finally:
        srv.close()
    with pytest.raises(lc.LiveControlError, match="no live detector"):
        lc.start_section(port=srv.port, timeout=1.0)   # nothing listens there now


class FakeDetector:
    """Answers like the live GUI; its audio clock moves 0.2 s per status call."""

    def __init__(self):
        self.t, self.open, self.n, self.calls = 0.0, None, 0, []

    def __call__(self, req):
        cmd = req["cmd"]
        if cmd == "status":
            self.t += 0.2
            return dict(ok=True, detecting=True, sections="calls", open=self.open, t_s=round(self.t, 3))
        self.calls.append((cmd, round(self.t, 3)))
        if cmd == "start":
            self.n += 1
            self.open = self.n
        else:
            self.open = None
        return dict(ok=True, section=self.n, t_s=round(self.t, 3))


def test_start_and_end_report_success():
    from dad import live_control as lc
    fake, said = FakeDetector(), []
    srv = lc.ControlServer(fake, port=0).start()
    report = lambda c, r: said.append(lc.reply_message(c, r))
    try:
        lc.start_section(port=srv.port, report=report)
        lc.end_section(port=srv.port, report=report)
        lc.start_section(port=srv.port, report=None)
    finally:
        srv.close()
    assert said == ["section 1 started at t = 0.00 s", "section 1 ended at t = 0.00 s"]


def test_run_sections_keeps_length_and_gap_on_the_audio_clock():
    from dad import live_control as lc
    fake = FakeDetector()
    srv = lc.ControlServer(fake, port=0).start()
    try:
        assert lc.run_sections(1.0, 0.6, count=2, port=srv.port, report=None) == 2
    finally:
        srv.close()
    assert fake.calls == [("start", 0.0), ("end", 1.0), ("start", 1.6), ("end", 2.6)]   # no gap after the last


def test_stopping_run_sections_ends_the_open_section():
    import threading

    from dad import live_control as lc
    fake, stop = FakeDetector(), threading.Event()
    srv = lc.ControlServer(fake, port=0).start()
    try:
        assert lc.run_sections(100.0, port=srv.port, stop=stop,
                               report=lambda c, r: stop.set() if c == "start" else None) == 1
    finally:
        srv.close()
    assert [c for c, _ in fake.calls] == ["start", "end"] and fake.open is None


def test_run_sections_stops_if_the_section_is_ended_elsewhere():
    from dad import live_control as lc
    fake = FakeDetector()
    srv = lc.ControlServer(fake, port=0).start()

    def end_it(cmd, reply):
        fake.open = None

    try:
        with pytest.raises(lc.LiveControlError, match="ended by another call"):
            lc.run_sections(100.0, port=srv.port, report=end_it)
    finally:
        srv.close()
