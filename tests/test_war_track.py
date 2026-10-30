"""The war track: fixed by its seed, events inside their catalogue's ranges, never silent for long."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("war_track", Path(__file__).parents[1] / "scripts" / "war_track.py")
wt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wt)
FS = wt.FS


def fake_stems():
    rng = np.random.default_rng(1)
    names = {c.get("stems", k) for k, c in wt.CATALOGUE.items()}
    out = {cls: {f"{cls}_{i}.wav": rng.standard_normal(2 * FS) * np.hanning(2 * FS) for i in range(2)}
           for cls in names}
    out[wt.SALVO["cls"]][wt.SALVO["stem"]] = rng.standard_normal(FS) * np.hanning(FS)
    return out


def test_seeded_and_in_range():
    stems = fake_stems()
    a, ev, ph = wt.render(60.0, 3, stems)
    b, ev2, _ = wt.render(60.0, 3, stems)
    assert np.array_equal(a, b) and ev == ev2
    assert np.isfinite(a).all() and len(a) == 60 * FS
    assert ph[0]["start_s"] == 0 and ph[-1]["end_s"] == 60.0
    for e in ev:
        lo, hi = (wt.SALVO if e["cls"] == "mlrs_salvo" else wt.CATALOGUE[e["cls"]])["dist"]
        assert lo <= e["distance_m"] <= hi and 0 <= e["start_s"] < 60
    assert {"distant_front", "small_arms"} <= {e["cls"] for e in ev}


def test_distant_front_is_steady():
    t = wt.times(30.0, 600.0, None, np.random.default_rng(0))
    assert 250 < len(t) < 350 and np.diff(t).max() < 30
