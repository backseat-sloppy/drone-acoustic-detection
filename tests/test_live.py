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
