"""The environment simulator (iteration 7): the war window and its gain, wind at its dB SPL, distracters
by seed, and a row that rebuilds its environment."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("env_sim", SCRIPTS / "env_sim.py")
es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(es)
import distracters as dt  # noqa: E402

FS = es.FS


def fake_src():
    rng = np.random.default_rng(0)
    war = rng.standard_normal(60 * FS)
    events = [dict(cls="small_arms", start_s="10.0", end_s="12.0", distance_m="500", peak_at_mic_db="100.0"),
              dict(cls="tank", start_s="40.0", end_s="50.0", distance_m="300", peak_at_mic_db="70.0")]
    phases = [dict(start_s="0", end_s="30", phase="calm", factor="0.3"),
              dict(start_s="30", end_s="60", phase="intense", factor="2.5")]
    t = np.arange(3 * FS) / FS
    stems = {c: {"a.wav": np.sin(2 * np.pi * 150 * t)} for c in dt.CATALOGUE}
    return dict(war=war, war_events=events, phases=phases, wind=rng.standard_normal(3 * FS), distracters=stems)


def test_war_window_and_gain():
    src, n = fake_src(), 20 * FS
    mix, layers, info = es.build(10, 5.0, 1, {}, n, 0, src)
    assert np.array_equal(mix, src["war"][5 * FS:25 * FS])
    assert [e["cls"] for e in info["war_events"]] == ["small_arms"] and info["war_events"][0]["start_s"] == 5.0
    assert info["war_phases"] == ["calm"]
    low, _, info6 = es.build(6, 5.0, 1, {}, n, 0, src)
    assert np.allclose(low, mix * 10 ** (-20 / 20)) and info6["war_events"][0]["peak_at_mic_db"] == 80.0
    assert not es.build(1, 5.0, 1, {}, n, 0, src)[0].any()


def test_wind_level_and_distracters_by_seed():
    src, n = fake_src(), 20 * FS
    mix, _, _ = es.build(1, 0, 6, {}, n, 0, src)
    assert abs(20 * np.log10(es.active_rms(mix) / es.P_REF) - 60) < 0.2      # step 6 = 60 dB SPL
    a, _, ia = es.build(1, 0, 1, {"car": 6, "mower": 4}, n, 7, src)
    b, _, ib = es.build(1, 0, 1, {"car": 6, "mower": 4}, n, 7, src)
    assert np.array_equal(a, b) and ia == ib and [d["cls"] for d in ia["distracters"]] == ["car", "mower"]
    row = es.row_for(1, "x.wav", 20.0, 7, 1, 0, 1, {"car": 6, "mower": 4}, a, ia)
    assert row["D_car_step"] == 6 and row["D_birds_step"] == 1 and row["seed"] == 7
    assert es.parse_distracters("none") == {} and es.parse_distracters("car:6,mower:4") == {"car": 6, "mower": 4}
