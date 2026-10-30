"""Propagation: ISO 9613 absorption, spreading, pitch and Doppler (report entry 061)."""
import numpy as np

from dad import propagation as pr

FS = pr.FS


def tone(f, seconds=2.0):
    return np.sin(2 * np.pi * f * np.arange(int(seconds * FS)) / FS)


def peak_hz(x):
    w = x * np.hanning(len(x))
    return np.fft.rfftfreq(len(x), 1 / FS)[np.abs(np.fft.rfft(w)).argmax()]


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)))


def test_absorption_matches_iso_9613_2_table():
    # ISO 9613-2 Table 2, 10 degC, 70 % RH, dB/km
    for f, want in ((125, 0.4), (500, 1.9), (1000, 3.7), (2000, 9.7), (4000, 32.8), (8000, 117.0)):
        got = pr.air_absorption_db_per_m(f) * 1000
        assert abs(got / want - 1) < 0.05, (f, got, want)


def test_static_spreading_and_absorption():
    mid = slice(FS // 2, 3 * FS // 2)
    low, high = tone(100), tone(8000)
    assert abs(rms_db(pr.propagate(low, 10.0)[mid]) - rms_db(low[mid]) + 20) < 0.2       # 1 m -> 10 m: -20 dB
    loss = rms_db(high[mid]) - rms_db(pr.propagate(high, 200.0)[mid])
    want = 20 * np.log10(200) + pr.air_absorption_db_per_m(8000) * 199
    assert abs(loss - want) < 0.5, (loss, want)


def test_pitch_shift():
    assert abs(peak_hz(pr.pitch(tone(300), 12)) - 600) < 2
    assert abs(peak_hz(pr.pitch(tone(300), -5)) - 300 * 2 ** (-5 / 12)) < 2


def test_doppler_on_a_pass_by():
    f0, v, c = 200.0, 20.0, pr.speed_of_sound()
    x = tone(f0, 10.0)
    y = pr.propagate(x, pr.pass_by(len(x), v, 5.0, 5.0))
    near, far = peak_hz(y[: 2 * FS]), peak_hz(y[-2 * FS:])
    # approaching, then receding, almost along the line of travel at those times
    assert abs(near - f0 * c / (c - v * 0.99)) < 1.5, near
    assert abs(far - f0 * c / (c + v * 0.99)) < 1.5, far
    assert 4.5 < np.argmax(np.abs(y)) / FS < 5.5                                         # loudest at closest
