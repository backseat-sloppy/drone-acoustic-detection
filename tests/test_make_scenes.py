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


def test_chain_is_seeded_and_keeps_the_level_rule():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(16000)
    c = ms.chain_params(np.random.default_rng([7, 2]))
    assert c == ms.chain_params(np.random.default_rng([7, 2]))          # same seed, same chain
    y = ms.level(ms.apply_chain(x, c))
    assert len(y) == len(x) and np.isfinite(y).all()
    assert abs(20 * np.log10(np.sqrt(np.mean(y.astype(np.float64) ** 2))) + 26) < 0.1 or np.abs(y).max() > 0.98
    # the high-pass removes DC
    assert abs(ms.apply_chain(np.ones(16000), c)[8000:].mean()) < 0.05


def test_realism_detune_scales_with_pitch():
    p = ms.realism_params(400.0, np.random.default_rng(0))
    assert len(p["detune"]) == 4 and all(abs(d) <= 20 for d in p["detune"])   # 5 % of 400 Hz
    assert -25 <= p["noise_db"] <= -3 and 300 <= p["noise_fc"] <= 4000 and 0.5 <= p["mic_h"] <= 3


def test_drone_ids_and_the_glasgow_split_by_seed():
    import argparse
    import re
    import pytest
    assert ms.drone_ids("1-4,9") == {1, 2, 3, 4, 9} and ms.drone_ids("17") == {17}
    root = ms.ROOT / "data/extracted/Glasgow"
    if not root.exists():
        pytest.skip("Glasgow not extracted")
    a = argparse.Namespace(real_drones=root, p_real=1.0, drone_ids="1-16", val_drone_ids="17-20", val_seed=1800)
    num = lambda f: int(re.search(r"_d(\d+)_", f).group(1))  # noqa: E731
    assert all(num(ms.real_drone(s, a)) <= 16 for s in range(0, 1800, 37))       # training seeds: d1-d16
    assert all(17 <= num(ms.real_drone(s, a)) <= 20 for s in range(1800, 2160, 7))  # validation seeds: d17-d20
    assert ms.real_drone(5, argparse.Namespace(**{**vars(a), "p_real": 0.0})) is None


def test_steady_noise_slope_is_seeded():
    a, pa = ms.steady_noise(11, 16000)
    b, pb = ms.steady_noise(11, 16000)
    assert np.array_equal(a, b) and pa == pb and abs(np.sqrt(np.mean(a ** 2)) - 1) < 1e-9
    # a steep negative slope puts more power low than high
    for seed in range(40):
        x, p = ms.steady_noise(seed, 32000)
        if p["slope_db_oct"] < -9:
            P = np.abs(np.fft.rfft(x)) ** 2
            f = np.fft.rfftfreq(32000, 1 / 16000)
            assert P[(f > 100) & (f < 400)].mean() > 10 * P[(f > 3000) & (f < 6000)].mean()
            break
