"""The stem builder's gate: it lowers a floor under events and keeps them; it eats a steady tone
when the tone is its own profile, which is why steady sources are not gated (report entry 052)."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("build_stems", Path(__file__).parents[1] / "scripts" / "build_stems.py")
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)
FS = bs.FS


def test_gate_keeps_events_and_lowers_floor():
    rng = np.random.default_rng(0)
    t = np.arange(4 * FS) / FS
    events = np.sin(2 * np.pi * 800 * t) * ((t % 1) < 0.2)              # 0.2 s tone bursts each second
    x = events + 0.01 * rng.standard_normal(len(t))
    profile = np.percentile(np.abs(bs.spec(x)[2]), 20, axis=1)
    y, _ = bs.gate(x, profile)
    assert np.percentile(bs.frame_db(x), 10) - np.percentile(bs.frame_db(y), 10) > 10
    assert bs.events_kept_db(x, y) > -1


def test_gate_eats_a_steady_source_given_its_own_floor():
    t = np.arange(4 * FS) / FS
    x = np.sin(2 * np.pi * 300 * t)
    y, _ = bs.gate(x, np.percentile(np.abs(bs.spec(x)[2]), 20, axis=1))
    assert bs.harmonic_kept_db(x, y) < -10


def test_finish_hits_target_and_limits_peak():
    x = np.zeros(FS); x[::4410] = 1.0                                   # spiky: peak-limited
    y, _ = bs.finish(np.random.default_rng(1).standard_normal(FS) * 0.1, -26)
    assert abs(bs.active_rms_db(y) + 26) < 0.1
    z, _ = bs.finish(x, -26)
    assert abs(np.abs(z).max() - 10 ** (-1 / 20)) < 1e-6


def test_glimpse_proportion():
    """Disjoint bands at equal level: the drone is never masked; same band and much louder: always.
    (A masker 40 dB up in a far band still takes some cells through window leakage, as upward spread
    of masking does in hearing, so the first case is at equal level.)"""
    sc_spec = importlib.util.spec_from_file_location("scenes", Path(__file__).parents[1] / "scripts" / "scenes.py")
    sc = importlib.util.module_from_spec(sc_spec)
    sc_spec.loader.exec_module(sc)
    t = np.arange(2 * FS) / FS
    tone = np.sin(2 * np.pi * 3000 * t)
    assert sc.glimpse_proportion(tone, np.sin(2 * np.pi * 200 * t)) > 0.95
    assert sc.glimpse_proportion(tone, 10 * np.sin(2 * np.pi * 3000 * t + 1)) < 0.05


def test_gate_on_a_mostly_silent_clip_is_finite():
    """A clip that is digital silence over most of its length has an all-zero profile (report entry 060)."""
    x = np.zeros(4 * FS)
    x[FS:FS + 2000] = np.hanning(2000)
    y, _ = bs.gate(x, np.percentile(np.abs(bs.spec(x)[2]), 20, axis=1))
    assert np.isfinite(y).all() and bs.events_kept_db(x, y) > -1
