"""Tests for the invariants that actually matter.

The important one is `test_streaming_matches_full_window`. The incremental
ring-buffer frontend is the whole latency trick, and it is only legitimate if
it produces bit-comparable features to the naive full-window path. If that
test fails, every latency number in the report is measuring the wrong thing.
"""

from __future__ import annotations

import numpy as np
import pytest

from dad.bench import latency_budget, range_penalty
from dad.config import FrontendConfig
from dad.features import LogMel, StreamingLogMel, mel_filterbank, summary_stats
from dad.gate import HarmonicCombGate
from dad.synth import make_dataset, session_split


def test_config_frame_math():
    cfg = FrontendConfig()
    assert cfg.sample_rate == 16_000
    assert cfg.frames_per_window == 100       # 1.0 s / 10 ms stride
    assert cfg.samples_per_window == 16_000
    assert cfg.samples_per_hop == 2_000       # 125 ms
    assert cfg.n_freq_bins == 201


def test_mel_filterbank_shape_and_support():
    cfg = FrontendConfig()
    fb = mel_filterbank(cfg)
    assert fb.shape == (cfg.n_mels, cfg.n_freq_bins)
    assert np.all(fb >= 0.0)
    # Every filter must have some support, otherwise a mel band is dead.
    assert np.all(fb.sum(axis=1) > 0.0)


def test_streaming_matches_full_window():
    """Incremental frontend must agree with the naive one."""
    cfg = FrontendConfig()
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(cfg.samples_per_window * 3).astype(np.float32)

    full = LogMel(cfg).frames(audio)

    stream = StreamingLogMel(cfg)
    chunk = cfg.samples_per_hop
    for start in range(0, len(audio) - chunk + 1, chunk):
        stream.push(audio[start:start + chunk])

    assert stream.ready
    win = stream.window()
    assert win.shape == (cfg.n_mels, cfg.frames_per_window)

    # The streaming window holds the most recent frames of the same stream.
    n = win.shape[1]
    np.testing.assert_allclose(win, full[:, -n:], rtol=1e-4, atol=1e-4)


def test_streaming_frame_count_is_hop_sized_in_steady_state():
    """Per-decision cost must scale with hop, not window. 125 ms hop at a
    10 ms stride is 12 or 13 new frames, never 100."""
    cfg = FrontendConfig()
    rng = np.random.default_rng(1)
    stream = StreamingLogMel(cfg)
    stream.push(rng.standard_normal(cfg.samples_per_window).astype(np.float32))

    counts = [stream.push(rng.standard_normal(cfg.samples_per_hop).astype(np.float32))
              for _ in range(10)]
    assert all(10 <= c <= 14 for c in counts), counts
    assert sum(counts) / len(counts) == pytest.approx(12.5, abs=1.0)


def test_window_raises_before_ready():
    cfg = FrontendConfig()
    stream = StreamingLogMel(cfg)
    stream.push(np.zeros(cfg.samples_per_hop, dtype=np.float32))
    assert not stream.ready
    with pytest.raises(RuntimeError):
        stream.window()


def test_gate_recovers_blade_pass_frequency():
    """A synthetic harmonic comb at a known f0 must be found by the gate."""
    cfg = FrontendConfig()
    gate = HarmonicCombGate(cfg)
    sr = cfg.sample_rate
    t = np.arange(cfg.samples_per_window) / sr
    f0 = 180.0
    x = sum(0.8 ** k * np.sin(2 * np.pi * (k + 1) * f0 * t) for k in range(6))
    x = (x / np.std(x)).astype(np.float32)

    res = gate(x)
    assert res.score > gate.threshold
    assert abs(res.f0_hz - f0) < 10.0            # gate FFT resolution is ~7.8 Hz
    assert res.rpm == pytest.approx(60.0 * res.f0_hz / 2.0)


def test_gate_prefers_harmonic_over_noise():
    cfg = FrontendConfig()
    gate = HarmonicCombGate(cfg)
    rng = np.random.default_rng(2)
    sr = cfg.sample_rate
    t = np.arange(cfg.samples_per_window) / sr

    tonal = sum(0.8 ** k * np.sin(2 * np.pi * (k + 1) * 200.0 * t) for k in range(6))
    tonal = (tonal / np.std(tonal)).astype(np.float32)
    noise = rng.standard_normal(cfg.samples_per_window).astype(np.float32)

    assert gate(tonal).score > gate(noise).score


def test_summary_stats_is_fixed_length():
    cfg = FrontendConfig()
    fe = LogMel(cfg)
    rng = np.random.default_rng(3)
    for seconds in (0.5, 1.0, 2.0):
        audio = rng.standard_normal(int(seconds * cfg.sample_rate)).astype(np.float32)
        v = summary_stats(fe.frames(audio))
        assert v.shape == (80,)                  # 20 MFCC x (mean, std, dmean, dstd)
        assert np.all(np.isfinite(v))


def test_session_split_has_no_leakage():
    clips = make_dataset(n_sessions=6, clips_per_session=10)
    train, test = session_split(clips, test_sessions=2, seed=0)
    assert train and test
    assert not ({c.session for c in train} & {c.session for c in test})


def test_latency_budget_compute_is_a_rounding_error():
    """The council's central claim, asserted as a test."""
    cfg = FrontendConfig(window_seconds=1.0, hop_seconds=0.125)
    budget = latency_budget(cfg, compute_ms=8.0)
    assert budget["cold_onset_ms"] == pytest.approx(1008.0)
    assert budget["steady_state_ms"] == pytest.approx(70.5)
    assert budget["compute_share_pct"] < 1.0


def test_n_of_m_smoothing_costs_latency():
    """3-of-5 voting is not free: it adds 4 hops of delay."""
    cfg = FrontendConfig(window_seconds=0.25, hop_seconds=0.125)
    plain = latency_budget(cfg, compute_ms=2.0)
    voted = latency_budget(cfg, compute_ms=2.0, n_of_m=(3, 5))
    assert voted["cold_onset_ms"] - plain["cold_onset_ms"] == pytest.approx(500.0)


def test_range_penalty_matches_the_corrected_arithmetic():
    """The council disagreed on this number and the chairman resolved it.

    1 s down to 125 ms is three doublings, 4.5 dB, which is 0.75 of a 6 dB
    range doubling. Range falls to 2**-0.75 = 0.595x, about a 40% loss.
    It does NOT halve.
    """
    r = range_penalty(0.125, reference_s=1.0)
    assert r["snr_delta_db"] == pytest.approx(-4.5)
    assert r["range_factor"] == pytest.approx(0.5946, abs=1e-3)
    assert r["range_change_pct"] == pytest.approx(-40.5, abs=0.5)

    # Doubling the window the other way buys range back, sub-linearly.
    assert range_penalty(2.0)["range_factor"] == pytest.approx(1.1892, abs=1e-3)
    assert range_penalty(1.0)["range_factor"] == pytest.approx(1.0)
