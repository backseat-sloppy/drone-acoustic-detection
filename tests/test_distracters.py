"""Distracters: distance steps, level with distance, motions, seeded (report entry 063)."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("distracters", Path(__file__).parents[1] / "scripts" / "distracters.py")
dt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dt)
FS = dt.FS


def stems():
    t = np.arange(4 * FS) / FS
    hum = np.sin(2 * np.pi * 120 * t)                          # low tone: air absorption negligible
    return {c: {"a.wav": hum * (np.hanning(len(t)) if dt.CATALOGUE[c]["motion"] == "once" else 1)}
            for c in dt.CATALOGUE}


def test_steps():
    near, far = dt.CATALOGUE["mower"]["dist"]
    assert dt.step_distance("mower", 1) is None
    assert np.isclose(dt.step_distance("mower", 2), far) and np.isclose(dt.step_distance("mower", 10), near)
    x, row = dt.render("mower", 1, 5.0, np.random.default_rng(0), stems())
    assert row is None and not x.any()


def test_level_follows_distance_and_seed():
    s = stems()
    _, near = dt.render("generator", 10, 5.0, np.random.default_rng(0), s, semitones=0)
    _, far = dt.render("generator", 2, 5.0, np.random.default_rng(0), s, semitones=0)
    want = 20 * np.log10(far["distance_m"] / near["distance_m"])
    assert abs(near["rms_at_mic_db"] - far["rms_at_mic_db"] - want) < 0.5
    a, ra = dt.render("car", 6, 20.0, np.random.default_rng(4), s)
    b, rb = dt.render("car", 6, 20.0, np.random.default_rng(4), s)
    assert np.array_equal(a, b) and ra == rb


def test_motions():
    s = stems()
    for cls in dt.CATALOGUE:
        x, row = dt.render(cls, 6, 20.0, np.random.default_rng(1), s)
        assert len(x) == 20 * FS and np.isfinite(x).all()
        assert 0 <= row["start_s"] < row["end_s"] <= 20.0
        if row["motion"] == "once":
            assert abs(row["end_s"] - row["start_s"] - 4.0) < 0.6          # length changes with pitch
        else:
            assert row["start_s"] == 0 and row["end_s"] == 20.0
    r = dt.wander(20 * FS, 0.1, np.random.default_rng(0))
    assert r.min() > 0 and 0.02 < np.median(r) < 0.5
