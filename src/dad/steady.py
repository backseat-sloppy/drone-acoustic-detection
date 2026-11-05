"""Steady noise for training (iteration 11, entries 100-102).

The networks alarm on steady noise with a pink or brown tilt (entries 075, 100-102). No training
negative was steady and broadband after level normalisation; these generators supply some.

  slope_noise  (a) Gaussian noise with a spectral slope in dB/octave. The random smooth EQ of the
               box is the per-window EQ every training window already gets (realrun._eq_curve).
  phase_copy   (b) a window's FFT magnitude with random phases: the same spectrum, no time structure.
  wobble       a slow sinusoidal level change, so "perfectly flat level" is not a no-drone cue.
"""
from __future__ import annotations

import numpy as np

SLOPE_DB_PER_OCT = (-12.0, 3.0)
WOBBLE_HZ = (0.2, 2.0)
WOBBLE_DB = 3.0


def slope_noise(n: int, db_per_oct: float, rng: np.random.Generator, fs: int = 16_000) -> np.ndarray:
    """Unit-RMS Gaussian noise whose power falls by `db_per_oct` dB per octave (0 Hz removed)."""
    f = np.fft.rfftfreq(n, 1 / fs)
    g = np.zeros_like(f)
    g[1:] = (f[1:] / 1000.0) ** (db_per_oct / (20 * np.log10(2)))
    x = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * g, n)
    return (x / np.sqrt(np.mean(x ** 2))).astype(np.float32)


def phase_copy(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """x's magnitude spectrum with random phases, at x's RMS."""
    X = np.fft.rfft(np.asarray(x, dtype=np.float64))
    ph = np.exp(2j * np.pi * rng.random(len(X)))
    ph[0] = 1.0
    y = np.fft.irfft(np.abs(X) * ph, len(x))
    return (y * np.sqrt(np.mean(x.astype(np.float64) ** 2) / max(np.mean(y ** 2), 1e-30))).astype(np.float32)


def wobble(x: np.ndarray, rng: np.random.Generator, fs: int = 16_000) -> np.ndarray:
    """Multiply by a sinusoidal level change of 0.2-2 Hz and up to +-3 dB."""
    t = np.arange(len(x)) / fs
    db = rng.uniform(0.0, WOBBLE_DB) * np.sin(2 * np.pi * rng.uniform(*WOBBLE_HZ) * t + rng.uniform(0, 2 * np.pi))
    return (x * 10.0 ** (db / 20.0)).astype(np.float32)
