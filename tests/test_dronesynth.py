import math
import numpy as np

from dad.dronesynth import drone


def test_drone_is_finite_unit_rms_and_never_silent_at_the_start():
    rng = np.random.default_rng(0)
    for _ in range(10):
        x, p = drone(rng, 16_000)
        assert x.shape == (16_000,) and np.isfinite(x).all()
        assert abs(x.std() - 1.0) < 1e-6
        assert 0.4 <= p["pscale"] <= 3.0 and p["motion"] in ("approach", "flyby")
        # The window is cut after the first emitted sample has arrived: no silent lead-in.
        assert np.abs(x[:800]).max() > 0.05


def test_real_drone_path_and_stem_split():
    """Iteration 25: a recorded source through the simulated path keeps its lines, has no silent
    lead-in, and the stem split never puts one clip in both train and val, nor any FSD50K-eval clip."""
    import pytest
    from dad.dronesynth import STEMS, _load_stems, real_drone

    fs, rng = 16_000, np.random.default_rng(1)
    t = np.arange(fs * 10) / fs
    src = (3000 * sum(np.sin(2 * np.pi * 200 * k * t) / k for k in (1, 2, 3))).astype(np.int16)
    for _ in range(10):
        x, p = real_drone(rng, (src, 1.0), fs)
        assert x.shape == (fs,) and np.isfinite(x).all() and abs(x.std() - 1.0) < 1e-6
        assert np.abs(x[:800]).max() > 0.05
        assert 0.8 <= p["rate"] <= 1.25 and 5 <= p["height_m"] <= 120
        # The strongest bin is still one of the three lines (the ground reflection can notch any one of
        # them; Doppler at up to 60 m/s moves them by up to 22 %).
        f = np.fft.rfftfreq(fs, 1 / fs)
        peak = f[np.argmax(np.abs(np.fft.rfft(x)) * (f > 50))] / (200 * p["rate"])
        assert any(0.78 * k < peak < 1.25 * k for k in (1, 2, 3))
    if not (STEMS / "manifest.csv").exists():
        pytest.skip("no stems on this machine")
    import csv, zlib
    rows = [r for r in csv.DictReader(open(STEMS / "manifest.csv", encoding="utf-8")) if r["group"] == "wind"]
    val = {r["id"] for r in rows if r["source"] == "fsd50k_dev" and zlib.crc32(r["id"].encode()) % 5 == 0}
    train = {r["id"] for r in rows if r["source"] == "fsd50k_dev"} - val
    assert len(_load_stems(("wind",), fs, "val")) == len(val) and len(_load_stems(("wind",), fs, "train")) == len(train)
    assert val and train and not val & train


def test_uav16_sources_split_by_batch():
    """Iteration 26: UAV16 train and val drones come from different acquisition batches, only the
    closest hovers are used, and the plain noise splits 80/20 without overlap."""
    import pytest
    from dad import dronesynth as ds
    if not (ds.UAV16 / "metadata" / "recordings.csv").exists():
        pytest.skip("UAV16 not linked at data/extracted/UAV16")
    rows = [r for r in ds._u16_rows(ds.UAV16) if r["has_uav"] == "true"]
    near = lambda r: math.hypot(float(r["horizontal_distance_m"]), float(r["height_m"])) <= 15.0
    held = {r["recording_id"] for r in rows if near(r) and r["acquisition_batch_id"] in ds.U16_HELD}
    train = {r["recording_id"] for r in rows if near(r) and r["acquisition_batch_id"] not in ds.U16_HELD}
    assert held and train and not held & train
    v = ds.uav16_drones("val")
    assert len(v) == len(held) and all(r <= 15.0 for _, r in v)
    tr, va = ds.uav16_noise("train")[0], ds.uav16_noise("val")[0]
    assert 3.5 < len(tr) / len(va) < 4.5


def test_quiet_scenes():
    """Iteration 27: near-empty scenes are drawn, silence stays exact zeros without a confuser, and a
    drone over silence is not scaled to nothing."""
    import pytest
    from dad import dronesynth as ds
    if not (ds.STEMS / "manifest.csv").exists():
        pytest.skip("no stems")
    s = ds.SynthSampler([], seed=0, p_quiet=1.0, p_quiet_confuser=0.0, bed_groups=("wind",), confuser_groups=("distracter",),
                       stem_split="train")
    rng = np.random.default_rng(0)
    beds = [s.scene(rng) for _ in range(200)]
    kinds = {i["bed"] for _, i in beds}
    assert kinds == {"quiet", "silence"}
    assert all(not x.any() for x, i in beds if i["bed"] == "silence")
    d = rng.standard_normal(s.n)
    assert np.allclose(s.mix(np.zeros(s.n), d, -20.0).std(), 1e-2, rtol=0.2)


def test_vehicle_pass_bys():
    """Iteration 28: with p_vehicle=1 every confuser is a road vehicle driven past the microphone, from
    the vehicle classes only, low and close; with p_vehicle=0 none are loaded."""
    import pytest
    from dad import dronesynth as ds
    if not (ds.STEMS / "manifest.csv").exists():
        pytest.skip("no stems")
    assert ds.SynthSampler([], seed=0, bed_groups=("wind",), confuser_groups=("distracter",),
                           stem_split="train").vehicles == []
    s = ds.SynthSampler([], seed=0, p_confuser=1.0, p_vehicle=1.0, bed_groups=("wind",),
                        confuser_groups=("distracter",), stem_split="train")
    assert s.vehicles
    rng = np.random.default_rng(0)
    scenes = [s.scene(rng) for _ in range(20)]
    assert all(i["confuser"] == "vehicle" for x, i in scenes) and all(np.isfinite(x).all() for x, _ in scenes)
    y, p = ds.real_drone(rng, s.vehicles[0], s.n, **ds.VEHICLE_PATH)
    assert p["motion"] == "flyby" and p["height_m"] <= 2.0 and p["closest_m"] <= 40.0 and len(y) == s.n


def test_steady_noises_slope():
    """Iteration 29: colour_noise follows its drawn slope (white flat, brown 6 dB/octave down, +3 up) when no
    low-pass is drawn; synth_wind sits below 800 Hz; both are unit RMS."""
    from dad import dronesynth as ds
    fs, n = 16_000, 16_000
    f = np.fft.rfftfreq(n, 1 / fs)
    band = lambda x, lo, hi: np.mean(np.abs(np.fft.rfft(x))[(f >= lo) & (f < hi)] ** 2)
    old = ds.STEADY_SLOPE_DB_OCT
    try:
        for slope in (0.0, -6.0, 3.0):
            ds.STEADY_SLOPE_DB_OCT = (slope, slope)
            got = []
            for seed in range(40):
                x = ds.colour_noise(np.random.default_rng(seed), n, fs)
                assert abs(x.std() - 1.0) < 1e-6
                r = np.random.default_rng(seed)
                r.uniform()                                  # the slope draw
                if r.random() >= 0.5:                        # no low-pass in this draw
                    got.append(10 * np.log10(band(x, 1000, 2000) / band(x, 500, 1000)))
            assert got and abs(np.mean(got) - slope) < 1.0
    finally:
        ds.STEADY_SLOPE_DB_OCT = old
    w = ds.synth_wind(np.random.default_rng(0), n, fs)
    assert abs(w.std() - 1.0) < 1e-6 and band(w, 50, 800) > 10 * band(w, 2000, 8000)


def test_apply_ir_is_circular_convolution():
    """Iteration 30.5: an impulse gives the IR folded to the window; noise gives a periodic signal's steady state."""
    from dad.dronesynth import apply_ir
    rng = np.random.default_rng(0)
    ir, x, d = rng.standard_normal(2500), rng.standard_normal(1000), np.zeros(1000)
    d[0] = 1.0
    assert np.allclose(apply_ir(d, ir), np.pad(ir, (0, 500)).reshape(-1, 1000).sum(0))
    assert np.allclose(apply_ir(x, ir), np.convolve(np.tile(x, 4), ir)[3000:4000])
