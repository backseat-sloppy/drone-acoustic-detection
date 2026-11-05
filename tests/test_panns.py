"""Iteration 13's PANNs wrapper: shapes and level handling, without the 359 MB weights."""
import numpy as np

from dad import panns


def test_level_scales_each_window_and_keeps_silence():
    a = np.stack([np.full(16000, 0.5, np.float32), np.zeros(16000, np.float32)])
    out = panns.level(a)
    assert abs(np.sqrt(np.mean(out[0] ** 2)) - panns.TARGET_RMS) < 1e-6
    assert np.all(out[1] == 0)


def test_cnn14_gives_a_2048_embedding_per_window():
    import torch
    net = panns._net().eval()
    with torch.no_grad():
        e = net(torch.randn(3, 16000) * 0.05)
    assert e.shape == (3, 2048) and torch.isfinite(e).all()


def test_finetuned_scorer_saves_and_loads(tmp_path):
    m = panns.PannsFtScorer()
    m._build(pretrained=False)
    x = np.random.default_rng(0).standard_normal((2, 16000)).astype(np.float32) * 0.05
    s = m.score(x)
    m.save(tmp_path / "x_pannsft.pt")
    assert np.allclose(panns.PannsFtScorer.load(tmp_path / "x_pannsft.pt").score(x), s, atol=1e-5)
    assert s.shape == (2,) and ((s > 0) & (s < 1)).all()
