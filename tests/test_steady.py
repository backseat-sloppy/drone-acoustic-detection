"""Iteration 11's steady-noise generators and how RangeSampler uses them."""
from __future__ import annotations

import types

import numpy as np

from dad import rangerun as rg
from dad import realrun as rr
from dad import steady
from dad.config import DEFAULT

FS = 16_000


def _slope(x):
    """Fitted power slope in dB/octave between 125 Hz and 4 kHz."""
    f = np.fft.rfftfreq(len(x), 1 / FS)
    p = np.abs(np.fft.rfft(x)) ** 2
    sel = (f > 125) & (f < 4000)
    return np.polyfit(np.log2(f[sel]), 10 * np.log10(p[sel]), 1)[0]


def test_slope_noise_has_the_slope_asked_for():
    rng = np.random.default_rng(0)
    for s in (-12.0, -6.0, 0.0, 3.0):
        x = steady.slope_noise(4 * FS, s, rng)
        assert abs(np.sqrt(np.mean(x ** 2)) - 1.0) < 1e-3
        assert abs(_slope(x) - s) < 0.5, (s, _slope(x))


def test_phase_copy_keeps_magnitude_and_rms_but_not_the_waveform():
    rng = np.random.default_rng(1)
    t = np.arange(FS) / FS
    x = (np.sin(2 * np.pi * 200 * t) * (1 + np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    y = steady.phase_copy(x, rng)
    assert np.allclose(np.abs(np.fft.rfft(y)), np.abs(np.fft.rfft(x)), rtol=1e-3, atol=1e-2)
    assert abs(np.mean(y ** 2) / np.mean(x ** 2) - 1) < 1e-4
    assert np.corrcoef(x, y)[0, 1] < 0.5


def test_wobble_stays_within_three_db():
    rng = np.random.default_rng(2)
    for _ in range(20):
        g = steady.wobble(np.ones(FS, np.float32), rng)
        assert 20 * np.log10(g.max()) <= 3.0 + 1e-4 and 20 * np.log10(g.min()) >= -3.0 - 1e-4


def _stub(recipe):
    """A RangeSampler with one drone cell (a 1 kHz tone) and white rig backgrounds; no data needed."""
    s = object.__new__(rg.RangeSampler)
    s.recipe, s.cfg, s.variant = recipe, DEFAULT, rr.VARIANTS["buried"]
    tone = (0.1 * np.sin(2 * np.pi * 1000 * np.arange(FS) / FS)).astype(np.float32)
    s.cells = [rg.Cell("x", 0, 50, lambda rng: tone.copy(), 1.0)]
    s._cell_cdf = np.array([1.0])
    s._rig_background = lambda rng, src=None: (
        lambda b: (b, rr.inband_power(b)))((0.01 * rng.standard_normal(FS)).astype(np.float32))
    s.fsd = types.SimpleNamespace(_background=s._rig_background)
    s.rig_neg_share = 1.0
    s._rig_share, s._rig_names = np.array([1.0]), ["x"]
    return s, tone


def test_steady_negatives_and_maskers_come_at_the_recipe_shares_and_snr():
    rng = np.random.default_rng(3)
    s, tone = _stub(rg.Recipe("t", {(0, 50): 1.0}, near_bury=False, steady_neg=0.15, steady_mask=0.30))
    s.steady = lambda rng: (np.full(FS, 7.0, np.float32), 49.0)   # marker: a constant no draw produces
    neg = [s.negative(rng) for _ in range(4000)]
    assert abs(np.mean([np.all(x == 7.0) for x in neg]) - 0.15) < 0.02
    pos = [s.positive(rng) for _ in range(4000)]
    masked = [x for x in pos if not np.allclose(x, tone)]
    assert abs(len(masked) / len(pos) - 0.30) < 0.03
    # drone over masker: the added constant's power against the tone's in-band power
    snr = [10 * np.log10(rr.inband_power(tone) / np.mean((x - tone) ** 2)) for x in masked]
    assert -5.0 - 1e-6 <= min(snr) and max(snr) <= 15.0 + 1e-6


def test_default_recipes_draw_no_steady_noise():
    rng = np.random.default_rng(4)
    s, tone = _stub(rg.Recipe("t", {(0, 50): 1.0}, near_bury=False))
    s.steady = lambda rng: (_ for _ in ()).throw(AssertionError("steady noise drawn"))
    for _ in range(200):
        s.negative(rng)
        assert np.allclose(s.positive(rng), tone)
    assert rg.RECIPES["simdist"].steady_neg == rg.RECIPES["simdist"].steady_mask == 0.0


def test_real_steady_draw_has_rig_level_and_the_window_length():
    rng = np.random.default_rng(5)
    s, _ = _stub(rg.Recipe("t", {(0, 50): 1.0}, steady_neg=1.0))
    for _ in range(50):
        x, p = rg.RangeSampler.steady(s, rng)
        assert len(x) == FS and np.isfinite(x).all()
        assert abs(10 * np.log10(p / 1e-4)) < 3.5   # white rig background at 0.01 RMS, wobble up to +-3 dB
