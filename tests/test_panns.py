"""PANNs Cnn14_16k port (models/panns.py) and its scorer (pannsrun.py), branch 11.1.

The checkpoint tests run only where artifacts/pretrained/Cnn14_16k_mAP=0.438.pth exists (the server);
they are the ones that prove the port loads the published weights and computes the frontend those
weights were trained on. The rest run anywhere, on a randomly initialised Cnn14.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dad.models import panns as pn  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "artifacts" / "pretrained" / pn.CHECKPOINT_NAME
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason=f"{CKPT.name} not downloaded here")


def _tone_and_noise(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(16_000) / 16_000
    f0 = rng.uniform(120, 250, n)[:, None]
    tone = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 9)) * 0.05
    return (tone + 0.005 * rng.standard_normal(tone.shape)).astype(np.float32), \
        (0.05 * rng.standard_normal((n, 16_000))).astype(np.float32)


# ------------------------------------------------------------------ frontend

def test_filterbank_shape_and_band():
    W = pn.slaney_mel_filterbank()
    assert W.shape == (64, 257) and (W >= 0).all()
    f = np.linspace(0, 8000, 257)
    assert W[:, f < 50 - 31.25].sum() == 0          # nothing below fmin, beyond one bin of slack
    assert (W.sum(axis=1) > 0).all()                # every band has support


def test_filterbank_matches_librosa_if_installed():
    librosa = pytest.importorskip("librosa")        # not a repo dependency; checked where present
    ref = librosa.filters.mel(sr=16000, n_fft=512, n_mels=64, fmin=50, fmax=8000)
    np.testing.assert_allclose(pn.slaney_mel_filterbank(), ref, rtol=1e-5, atol=1e-7)


def test_spectrogram_matches_numpy_rfft():
    x = np.random.default_rng(1).standard_normal((2, 16_000)).astype(np.float32)
    with torch.no_grad():
        got = pn._Spectrogram()(torch.from_numpy(x))[:, 0].numpy()         # (N, frames, bins)
    xp = np.pad(x, ((0, 0), (256, 256)), mode="reflect")
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(512) / 512)
    fr = np.stack([xp[:, s:s + 512] for s in range(0, xp.shape[1] - 512 + 1, 160)], axis=1) * win
    ref = np.abs(np.fft.rfft(fr, axis=-1)) ** 2
    assert got.shape == ref.shape == (2, 101, 257)
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3 * ref.max())


def test_level_normalisation_removes_gain():
    m = pn.Cnn14().eval()
    with torch.no_grad():
        m.bn0.running_mean.fill_(-20.0)
    x = torch.from_numpy(_tone_and_noise(3)[0])
    with torch.no_grad():
        a, b = m.logmel(x), m.logmel(10.0 * x)
    np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-3)
    np.testing.assert_allclose(a.mean(dim=(1, 2, 3)).numpy(), -20.0, atol=1e-4)


# ------------------------------------------------------------------ Cnn14

def test_cnn14_shapes_on_one_second():
    m = pn.Cnn14().eval()
    with torch.no_grad():
        emb, logits = m(torch.zeros(2, 16_000).normal_())
    assert emb.shape == (2, 2048) and logits.shape == (2, 527)
    assert sum(p.numel() for p in m.parameters()) > 79_000_000                 # about 80 M, as published


def test_frontend_is_frozen():
    m = pn.Cnn14()
    frozen = {n for n, p in m.named_parameters() if not p.requires_grad}
    assert frozen == {"spectrogram_extractor.stft.conv_real.weight", "spectrogram_extractor.stft.conv_imag.weight",
                      "logmel_extractor.melW"}


@needs_ckpt
def test_checkpoint_loads_strictly_and_matches_our_frontend():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    m = pn.load_pretrained(CKPT)                                                 # strict=True inside
    np.testing.assert_allclose(sd["logmel_extractor.melW"].numpy(), pn.slaney_mel_filterbank().T, rtol=1e-4, atol=1e-6)
    re, im = pn.stft_kernels()
    np.testing.assert_allclose(sd["spectrogram_extractor.stft.conv_real.weight"].numpy(), re, atol=1e-5)
    np.testing.assert_allclose(sd["spectrogram_extractor.stft.conv_imag.weight"].numpy(), im, atol=1e-5)
    assert m.bn0.running_mean.shape == (64,)


# ------------------------------------------------------------------ scorer

@pytest.fixture()
def fake_pretrained(tmp_path):
    """A randomly initialised Cnn14 saved in the PANNs checkpoint layout, with plausible bn0 stats."""
    m = pn.Cnn14()
    with torch.no_grad():
        m.bn0.running_mean.fill_(-30.0); m.bn0.running_var.fill_(100.0)
    p = tmp_path / pn.CHECKPOINT_NAME
    torch.save({"model": m.state_dict()}, p)
    return p


class _FakeSampler:
    seed = 0

    def raw_batch(self, index, batch_size):
        from dad.rangerun import SPEED_UP
        pos, neg = _tone_and_noise(batch_size // 2, seed=index)
        y = (np.arange(batch_size) % 2).astype(np.int64)
        audio = np.empty((batch_size, 16_000), np.float32)
        audio[y == 1], audio[y == 0] = pos, neg
        return audio, np.full(batch_size, SPEED_UP), np.zeros((batch_size, 64), np.float32), y


def test_probe_then_ft_save_load_roundtrip(fake_pretrained, tmp_path):
    from dad import realrun as rr
    from dad.pannsrun import PannsScorer
    pos, neg = _tone_and_noise(8, seed=99)
    sc = PannsScorer(fake_pretrained).fit_probe(_FakeSampler(), 64, pos, neg, batch_size=16, workers=1,
                                                Cs=(1.0,), log=lambda m: None)
    assert sc.kind == "probe" and sc.info["probe"]["windows"] == 64
    s1 = sc.score(np.r_[pos, neg])
    sc.save(tmp_path / "probe.pt")
    with pytest.raises(FileNotFoundError):                                # a head-only checkpoint needs its encoder
        PannsScorer.load(tmp_path / "probe.pt", pretrained=tmp_path / "missing.pth")
    other = tmp_path / "other.pth"
    torch.save({"model": pn.Cnn14().state_dict()}, other)
    with pytest.raises(ValueError):                                       # ... and exactly that encoder
        PannsScorer.load(tmp_path / "probe.pt", pretrained=other)
    back = PannsScorer.load(tmp_path / "probe.pt", pretrained=fake_pretrained)
    np.testing.assert_allclose(back.score(np.r_[pos, neg]), s1, atol=1e-5)

    back.fit_finetune(_FakeSampler(), 32, pos, neg, batch_size=16, eval_every=16, workers=1, log=lambda m: None)
    assert back.kind == "ft" and len(back.history) == 2
    back.save(tmp_path / "ft.pt")
    ft = rr.NetScorer.load(tmp_path / "ft.pt")             # dispatches to PannsScorer; needs no encoder file
    assert type(ft).__name__ == "PannsScorer" and ft.kind == "ft" and ft.pretrained_md5 == sc.pretrained_md5
    np.testing.assert_allclose(ft.score(np.r_[pos, neg]), back.score(np.r_[pos, neg]), atol=1e-5)


def test_ft_refuses_without_probe(fake_pretrained):
    from dad.pannsrun import PannsScorer
    pos, neg = _tone_and_noise(2)
    with pytest.raises(ValueError):
        PannsScorer(fake_pretrained).fit_finetune(_FakeSampler(), 16, pos, neg, batch_size=16, workers=1)
