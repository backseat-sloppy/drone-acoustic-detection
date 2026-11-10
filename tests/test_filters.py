"""Iteration 20 (entry 172): the optional bandpass and Wiener stages of the frontend."""
import numpy as np

from dad.config import FrontendConfig
from dad.features import LogMel, StreamingLogMel, band_matrix, wiener

SR = 16_000


def test_off_by_default_leaves_the_frontend_unchanged():
    cfg = FrontendConfig()
    assert cfg.bandpass is None and not cfg.wiener
    spec = np.random.default_rng(0).random((5, cfg.n_freq_bins))
    assert wiener(spec, cfg) is spec


def test_bandpass_attenuates_outside_the_band_only():
    plain, bp = band_matrix(FrontendConfig()), band_matrix(FrontendConfig(bandpass=(80.0, 6000.0)))
    hz = np.linspace(0, SR / 2, plain.shape[1])
    gain = bp.sum(1) / plain.sum(1)
    centre = (plain * hz).sum(1) / plain.sum(1)
    assert np.all(gain[(centre > 300) & (centre < 3000)] > 0.95)   # passband untouched
    assert gain[np.argmin(centre)] <= 0.55 and gain[np.argmax(centre)] < 0.2   # lowest band sits on the 80 Hz corner


def test_wiener_keeps_harmonic_lines_and_lowers_broadband_noise():
    cfg = FrontendConfig(wiener=True)
    t = np.arange(SR) / SR
    noise = 0.05 * np.random.default_rng(1).standard_normal(SR)
    lines = sum(np.sin(2 * np.pi * 200 * k * t) for k in range(1, 6))
    fe_plain, fe_w = LogMel(FrontendConfig()), LogMel(cfg)
    drop = (fe_plain.frames(noise + lines) - fe_w.frames(noise + lines)).mean(1)   # dB-ish loss per band, ln power
    line_band = np.argmax(fe_plain.frames(lines).mean(1))
    assert drop[line_band] < 0.2                     # the strongest line keeps its power
    assert np.median(drop) > np.log(10 ** 0.3)       # white noise loses more than 3 dB (the median sits below the mean)


def test_streaming_matches_offline_with_filters():
    cfg = FrontendConfig(bandpass=(80.0, 6000.0), wiener=True)
    x = np.random.default_rng(2).standard_normal(2 * SR).astype(np.float32)
    st = StreamingLogMel(cfg)
    for i in range(0, 2 * SR, cfg.samples_per_hop):
        st.push(x[i:i + cfg.samples_per_hop])
    np.testing.assert_allclose(st.window(), LogMel(cfg).frames(x)[:, -cfg.frames_per_window:], atol=1e-4)
