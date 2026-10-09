"""Tests for the laptop/GPU model zoo and the comparison harness.

Correctness of the plumbing, not model quality: shapes at every window length,
a training run that actually reduces loss, and a comparison that never lets a
test session leak into training.
"""

from __future__ import annotations

import numpy as np
import pytest

from dad.compare import Windows, false_alarms_at_recall, make_windows, run_comparison
from dad.config import FrontendConfig
from dad.synth import make_dataset

torch = pytest.importorskip("torch")

from dad.models.zoo import ARCHITECTURES, build  # noqa: E402
from dad.train import TorchDetector, TrainConfig  # noqa: E402


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
@pytest.mark.parametrize("window_s", [0.25, 1.0, 2.0])
def test_every_architecture_runs_at_every_window(arch, window_s):
    cfg = FrontendConfig(window_seconds=window_s)
    frames = 1 + (cfg.samples_per_window - cfg.n_fft) // cfg.hop_length
    model = build(arch, cfg.n_mels).eval()
    with torch.inference_mode():
        out = model(torch.zeros(3, 1, cfg.n_mels, frames))
    assert out.shape == (3, 2)
    assert torch.isfinite(out).all()


def test_torch_detector_learns_a_trivial_task():
    """Loud low bands versus loud high bands. Any working loop gets this."""
    rng = np.random.default_rng(0)
    n, m, t = 64, 64, 50
    X = rng.normal(size=(n, m, t)).astype(np.float32)
    y = np.arange(n) % 2
    X[y == 1, :16] += 3.0
    X[y == 0, -16:] += 3.0

    det = TorchDetector("dscnn", m, TrainConfig(epochs=8, device="cpu", spec_augment=False))
    det.fit(X[:48], y[:48], X[48:], y[48:])
    p = det.predict_proba(X[48:])
    assert p.shape == (16,)
    assert np.all((p >= 0) & (p <= 1))
    assert ((p > 0.5) == y[48:]).mean() >= 0.9
    assert det.predict_one(X[0]) == pytest.approx(det.predict_proba(X[:1])[0], abs=1e-5)


def test_make_windows_crops_and_shapes():
    cfg = FrontendConfig(window_seconds=0.5)
    clips = make_dataset(n_sessions=1, clips_per_session=4)
    w = make_windows(clips, cfg, crops=3)
    assert len(w.y) == 12
    assert w.audio.shape == (12, cfg.samples_per_window)
    assert w.logmel.shape[:2] == (12, cfg.n_mels)


def test_false_alarms_at_recall():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.65, 0.1, 0.2, 0.95])
    kind = np.array(["drone"] * 4 + ["mower", "wind", "wind", "machine"])
    fa = false_alarms_at_recall(y, s, kind, recall=1.0)
    # Threshold 0.6 keeps every drone; the mower (0.65) and machine (0.95) pass.
    assert fa["all"] == pytest.approx(0.5)
    assert fa["machine"] == 1.0 and fa["wind"] == 0.0


def test_comparison_smoke():
    logs: list[str] = []
    rows = run_comparison(
        windows=[0.5], models=["comb", "lgbm", "dscnn", "ensemble"],
        epochs=1, device="cpu", n_sessions=5, clips_per_session=8, log=logs.append,
    )
    assert [r.model for r in rows] == ["comb (no ML)", "mfcc + lightgbm", "dscnn", "ensemble (mean of ML)"]
    for r in rows:
        assert 0.0 <= r.auc <= 1.0
        assert 0.0 <= r.fa_at_95 <= 1.0
        assert r.infer_p50_ms > 0


def test_comparison_rejects_windows_longer_than_clips():
    with pytest.raises(ValueError):
        run_comparison(windows=[3.0], models=["comb"], n_sessions=5, clips_per_session=4)
