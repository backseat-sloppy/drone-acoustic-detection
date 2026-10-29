"""Outdoor sound propagation for the environment simulator (iteration 7).

A source recorded at r_ref (1 m for the stems' reference levels) is heard at
distance r through:

    spreading    spherical, gain r_ref / r (-6 dB per doubling)
    absorption   air absorption, ISO 9613-1: alpha(f) dB per metre, from
                 temperature, relative humidity and pressure. Treble goes
                 first, so a far explosion is a low rumble.
    Doppler      for a moving source: each emitted sample arrives r(t) / c
                 later, so the pitch rises while it approaches and falls
                 while it recedes.
    pitch        a fixed shift in semitones, by resampling (a faster engine,
                 a smaller insect); it changes the length too.

    y = propagate(x, 150.0)                                  # static, 150 m
    r = pass_by(len(x), speed=15, closest=20, t_closest=5)   # metres per sample
    y = propagate(x, r)                                      # passing, with Doppler

Left out: ground reflection and ground absorption, barriers, wind and
temperature gradients, turbulence. ISO 9613-2 has the ground term; add it if
far sources sound too clean.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import istft, resample_poly, stft

FS = 44100
NPERSEG, HOP = 2048, 512
# ISO 9613-2 table conditions, used as the default weather; calibration knobs, not constants
TEMP_C, RH_PCT, PA_KPA = 10.0, 70.0, 101.325


def speed_of_sound(temp_c=TEMP_C):
    return 331.3 * np.sqrt(1 + temp_c / 273.15)


def air_absorption_db_per_m(f, temp_c=TEMP_C, rh_pct=RH_PCT, pa_kpa=PA_KPA):
    """ISO 9613-1 pure-tone attenuation coefficient, dB per metre, at frequency f (Hz)."""
    f = np.asarray(f, dtype=float)
    T, T0, T01, pr = temp_c + 273.15, 293.15, 273.16, 101.325
    pa = pa_kpa / pr
    h = rh_pct * 10 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151) / pa       # molar concentration of water vapour
    fro = pa * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frn = pa * (T / T0) ** -0.5 * (9 + 280 * h * np.exp(-4.170 * ((T / T0) ** (-1 / 3) - 1)))
    return 8.686 * f ** 2 * (1.84e-11 / pa * (T / T0) ** 0.5 + (T / T0) ** -2.5 * (
        0.01275 * np.exp(-2239.1 / T) / (fro + f ** 2 / fro) + 0.1068 * np.exp(-3352.0 / T) / (frn + f ** 2 / frn)))


def pass_by(n, speed, closest, t_closest, fs=FS):
    """Distance per sample (m) of a source on a straight line: closest approach `closest` m at t_closest s."""
    t = np.arange(n) / fs
    return np.hypot(closest, speed * (t - t_closest))


def pitch(x, semitones, fs=FS):
    """Shift pitch by resampling: +12 doubles every frequency and halves the length."""
    k = Fraction(2 ** (semitones / 12)).limit_denominator(200)
    return x if k == 1 else resample_poly(x, k.denominator, k.numerator)


def propagate(x, r, r_ref=1.0, fs=FS, temp_c=TEMP_C, rh_pct=RH_PCT, pa_kpa=PA_KPA):
    """x recorded at r_ref, heard at r (m): a number, or one distance per sample of x (at emission).

    A moving source gets Doppler from its changing travel time; the delay common to all samples (the
    shortest travel time) is dropped, so sample 0 of the output is when the nearest point is heard
    minus its own lag, not a flash-to-bang wait. Output length = len(x) + the extra travel time.
    """
    r = np.broadcast_to(np.asarray(r, dtype=float), x.shape)
    if r.min() <= 0:
        raise ValueError("distance must be > 0")
    c = speed_of_sound(temp_c)
    y = x * r_ref / r                                                       # spreading, at emission
    if np.ptp(r) > 0:                                                       # Doppler: resample on arrival times
        arrive = np.arange(len(x)) / fs + (r - r.min()) / c                 # increasing while speed < c
        t = np.arange(int(np.ceil(arrive[-1] * fs)) + 1) / fs
        y = np.interp(t, arrive, y, left=0.0, right=0.0)
        r = np.interp(t, arrive, r)                                         # distance heard at each output sample
    f, _, Z = stft(y, fs, nperseg=NPERSEG, noverlap=NPERSEG - HOP, boundary="even")
    frame_r = r[np.clip(np.arange(Z.shape[1]) * HOP, 0, len(r) - 1)]
    alpha = air_absorption_db_per_m(f, temp_c, rh_pct, pa_kpa)
    Z *= 10 ** (-alpha[:, None] * (frame_r[None, :] - r_ref) / 20)
    _, out = istft(Z, fs, nperseg=NPERSEG, noverlap=NPERSEG - HOP, boundary=True)
    return out[: len(y)]
