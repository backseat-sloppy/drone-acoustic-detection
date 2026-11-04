import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_scenes as ms  # noqa: E402


def test_place_and_mix_hit_the_snr():
    rng = np.random.default_rng(0)
    n = 16000
    bed, lo, hi = ms.place(rng.standard_normal(4000), n, rng, fill=True)
    assert (lo, hi) == (0, n) and len(bed) == n
    ev, lo, hi = ms.place(np.ones(1000), n, rng, fill=False)
    assert hi - lo == 1000 and ev[lo:hi].all() and not ev[:lo].any() and not ev[hi:].any()
    drone, lo, hi = ms.place(np.sin(np.arange(6000) / 3.0) + 2, n, rng, fill=False)

    out = ms.mix([bed, ev], drone, -12.0).astype(np.float64)
    active = drone != 0
    scale = out[~active].std() / (bed + ev)[~active].std()
    d = out - scale * (bed + ev)
    snr = 10 * np.log10(np.mean(d[active] ** 2) / np.mean((scale * (bed + ev))[active] ** 2))
    assert abs(snr - (-12.0)) < 0.01
    assert np.abs(out).max() <= 10 ** (-0.1 / 20) + 1e-6


def test_mix_drone_alone_is_only_levelled():
    drone = np.concatenate([np.zeros(500), np.sin(np.arange(4000) / 5.0)])
    out = ms.mix([], drone, None).astype(np.float64)
    assert not out[:500].any()
    assert np.allclose(out / np.abs(out).max(), drone / np.abs(drone).max(), atol=1e-6)


def test_transients_are_outside_the_snr_reference():
    rng = np.random.default_rng(1)
    n = 16000
    bed = rng.standard_normal(n)
    drone = np.sin(np.arange(n) / 4.0)
    boom = np.zeros(n)
    boom[8000:8400] = 50 * rng.standard_normal(400)
    a = ms.mix([bed], drone, -6.0).astype(np.float64)
    b = ms.mix([bed], drone, -6.0, [boom]).astype(np.float64)
    # same drone-to-bed ratio with and without the boom: b is a + boom, then levelled
    ka, kb = np.std(a[:8000]), np.std(b[:8000])
    assert np.allclose(a[:8000] / ka, b[:8000] / kb, atol=1e-4)
