import numpy as np
import pytest

from dad import realrun as rr
from dad.config import FrontendConfig
from dad.features import LogMel

SINE = np.sin(2 * np.pi * 1000.0 * np.arange(16_000) / 16_000).astype(np.float32)


def test_linear_frontend_keeps_bins_in_band_and_finds_the_tone():
    cfg = FrontendConfig(kind="linear", n_fft=1024)
    L = LogMel(cfg).frames(SINE)
    assert L.shape == (cfg.n_bands, 1 + (16_000 - 1024) // 160)
    assert cfg.n_bands == 509                                  # 15.625 Hz bins, 62.5-8000 Hz
    first = cfg.linear_bins()[0]
    assert np.argmax(L.mean(axis=1)) + first == 64             # 1000 Hz / 15.625 Hz


def test_mfcc_is_the_dct_of_the_level_normalised_log_mel():
    cfg = FrontendConfig(kind="mfcc", n_mfcc=20)
    X = rr.featurise(SINE[None], rr.VARIANTS["level"], LogMel(cfg))
    assert X.shape == (1, 20, 98)
    mfe = rr.featurise(SINE[None], rr.VARIANTS["level"], LogMel(FrontendConfig()))
    assert np.isclose(X[0, 0, 0], mfe[0, :, 0].sum() / np.sqrt(64), rtol=1e-4)   # c0 = scaled band sum


def test_checkpoint_remembers_its_frontend(tmp_path):
    pytest.importorskip("torch")
    from dad.models.zoo import build
    cfg = FrontendConfig(kind="linear", n_fft=1024)
    net = rr.NetScorer("dscnn", rr.VARIANTS["buried"], cfg)
    net.model = build("dscnn", cfg.n_bands)
    net.mu = np.zeros((cfg.n_bands, 1), np.float32)
    net.sigma = np.ones((cfg.n_bands, 1), np.float32)
    net.save(tmp_path / "m.pt")
    back = rr.NetScorer.load(tmp_path / "m.pt")                # default (log-mel) cfg passed in
    assert back.cfg.kind == "linear" and back.cfg.n_fft == 1024
    assert back.score(SINE[None]).shape == (1,)
