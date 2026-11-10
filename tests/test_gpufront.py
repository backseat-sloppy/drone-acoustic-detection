"""The batched torch frontend (entry 092) gives the numpy path's numbers."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dad import rangerun as rg  # noqa: E402
from dad import realrun as rr  # noqa: E402
from dad.config import FrontendConfig  # noqa: E402
from dad.features import LogMel  # noqa: E402
from dad.gpufront import GpuFrontend  # noqa: E402
from dad.train import spec_augment  # noqa: E402

N_OUT = 16_000
N_IN = int(np.ceil(N_OUT * 1.25)) + 32   # RangeSampler.n with speed 0.8-1.25
CFGS = {k: FrontendConfig(kind=k, n_fft=1024 if k == "linear" else 400) for k in ("mfe", "mfcc", "linear")}


def _audio(n_rows: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(N_IN) / 16_000
    tone = np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 2_000 * t)
    return (0.1 * rng.standard_normal((n_rows, N_IN)) + tone * rng.uniform(0.01, 1, (n_rows, 1))).astype(np.float32)


def test_cached_filter_is_resample_polys_own():
    from scipy.signal import resample_poly
    x = _audio(1)[0].astype(np.float64)
    for d in (80, 99, 101, 125):
        np.testing.assert_array_equal(rg.resample_down(x, d), resample_poly(x, 100, d).astype(np.float32))


def test_speed_change_matches_scipy():
    x = _audio(6)
    down = np.array([80, 99, 100, 101, 113, 125])
    got = GpuFrontend(CFGS["mfe"], "cpu").speed(torch.from_numpy(x), down, N_OUT).numpy()
    want = np.stack([rg.resample_down(r, d)[:N_OUT] for r, d in zip(x, down)])
    np.testing.assert_allclose(got, want, atol=2e-6)


@pytest.mark.parametrize("kind", ["mfe", "mfcc", "linear"])
def test_train_features_match_numpy(kind):
    cfg = CFGS[kind]
    x, down = _audio(4, 1), np.array([80, 100, 113, 125])
    fe = LogMel(cfg)
    eq = np.stack([rr._eq_curve(np.random.default_rng(i), fe._fb.shape[0]) for i in range(4)])
    want = np.stack([fe.frames(rg.resample_down(r, d)[:N_OUT]) for r, d in zip(x, down)])
    want = rr.to_model(rr.level_normalise(want + eq[:, :, None]), cfg)
    got = GpuFrontend(cfg, "cpu").train_features(x, down, eq).numpy()
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, atol=2e-3)


@pytest.mark.parametrize("kind", ["mfe", "mfcc", "linear"])
def test_inference_features_match_featurise(kind):
    cfg = CFGS[kind]
    x = _audio(3, 2)[:, :N_OUT]
    want = rr.featurise(x, rr.VARIANTS["buried"], LogMel(cfg))
    np.testing.assert_allclose(GpuFrontend(cfg, "cpu").features(x).numpy(), want, atol=2e-3)


def test_spec_augment_same_masks_as_the_loop():
    x = torch.randn(16, 64, 98)

    def old(x, rng, max_freq=8, max_time_frac=0.125):
        x = x.clone()
        b, m, t = x.shape
        max_time = max(1, int(t * max_time_frac))
        for i in range(b):
            fw = int(rng.integers(0, max_freq + 1))
            f0 = int(rng.integers(0, max(1, m - fw)))
            x[i, f0:f0 + fw, :] = 0.0
            tw = int(rng.integers(0, max_time + 1))
            t0 = int(rng.integers(0, max(1, t - tw)))
            x[i, :, t0:t0 + tw] = 0.0
        return x

    np.testing.assert_array_equal(spec_augment(x, np.random.default_rng(3)).numpy(),
                                  old(x, np.random.default_rng(3)).numpy())


class _RawSampler:
    """raw_batch like RangeSampler's, from noise and tones."""
    seed = 0

    def __init__(self, cfg):
        self.cfg, self.bands = cfg, LogMel(cfg)._fb.shape[0]

    def raw_batch(self, index, batch_size):
        rng = np.random.default_rng([0, index])
        y = (np.arange(batch_size) % 2).astype(np.int64)
        x = 0.1 * rng.standard_normal((batch_size, N_IN)).astype(np.float32)
        x[y == 1] += np.sin(2 * np.pi * 300 * np.arange(N_IN) / 16_000).astype(np.float32)
        down = rng.integers(80, 126, batch_size)
        eq = np.stack([rr._eq_curve(rng, self.bands) for _ in y])
        return x, down, eq, y


def test_netscorer_trains_and_scores_through_the_torch_frontend():
    cfg = CFGS["mfcc"]
    net = rr.NetScorer("dscnn", rr.VARIANTS["buried"], cfg, "cpu")
    net.torch_frontend = True
    net.fit(_RawSampler(cfg), n_windows=64, batch_size=8, workers=2, log=lambda s: None)
    assert net.mu.shape == (cfg.n_mfcc, 1)
    s = net.score(_audio(5, 4)[:, :N_OUT])
    assert s.shape == (5,) and np.all((s >= 0) & (s <= 1))


def test_pos_weight_raises_the_drone_score():
    # It9B (entry 160): weighting drone windows in the loss lifts every score on the same audio.
    cfg, x = CFGS["mfe"], _audio(16, 4)[:, :N_OUT]
    scores = {}
    for w in (1.0, 4.0):
        net = rr.NetScorer("dscnn", rr.VARIANTS["buried"], cfg, "cpu")
        net.torch_frontend = True
        net.fit(_RawSampler(cfg), n_windows=256, batch_size=16, workers=1, log=lambda s: None, pos_weight=w)
        scores[w] = net.score(x).mean()
    assert scores[4.0] > scores[1.0]


def test_bandpass_and_wiener_match_numpy():
    # Iteration 20 (entry 172): the torch frontend applies the same filters as the numpy one.
    cfg = FrontendConfig(bandpass=(80.0, 6000.0), wiener=True)
    x = _audio(4, 5)[:, :N_OUT]
    want = np.stack([LogMel(cfg).frames(a) for a in x])
    got = GpuFrontend(cfg, "cpu").logbands(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(got, want, atol=2e-4)
