"""numpy port of `Pure-data+Script/EnviromentTest/DroneSound.pd`.

The patch, signal by signal (u = throttle / 127, throttle on the 0-127 slider):

    motors   4 sawtooth oscillators (phasor~, 0..1) at
             (150 + 180 u) * pscale + {0, +2.3, -3.7, +5.1} Hz,
             each pitch wandering by white noise through lop~ {0.4, 0.7, 0.5, 0.9} Hz
             times (200 + 600 u/127) Hz
    tone     sum of the four -> lop~ at 500 + 2000 u/127 Hz
    wash     white noise -> bp~ centre 500 + 1000 u Hz, Q 0.8 -> times line~ (0.2 - 0.15 u, 50 ms)
    body     tone + wash -> hip~ at 30 + 400 u/127 Hz -> times line~ (0.03 + 0.1 u, 50 ms)
    flutter  body times (1 + 4 * (white noise -> lop~ 8 Hz))

The three "u/127" are as the patch has them: objects 44, 46 and 49 divide the
already divided slider value (object 35) by 127 again, so those three barely
move with throttle (hip~ about 30 Hz, lop~ about 500-516 Hz, wander about
200-206 Hz). Mads confirmed it is an error, so by default the port uses the
intended mapping (u in place of u/127); PATCH_DOUBLE_DIVIDE = True reproduces
the patch, and only then does the port match a Pd render.

The patch also holds osc~ (objects 64-68) whose control input is not
connected: it runs at 0 Hz and adds a constant 0.01 before hip~, which removes
it. It is left out. writesf~'s 16-bit quantisation is left out as well.

Filters follow Pd's own d_filter.c formulas, so the spectra match a Pd render;
the noise does not match sample for sample (Pd's noise~ has its own generator),
so two renders agree in distribution, not in samples. Controls change once per
step (20 ms, as qlist delivers them in render.py), and time-varying filters
update at the step boundary.

    x = render(throttle, pscale=1.0)      # float64, 44.1 kHz, len(throttle) steps of 20 ms
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

FS = 44100
DETUNE = (0.0, 2.3, -3.7, 5.1)
WANDER_HZ = (0.4, 0.7, 0.5, 0.9)
TWO_PI = 2 * np.pi
# The patch's extra "/ 127" (see the docstring) is an error (Mads, 2026-10-26), so the default is the
# intended mapping. True reproduces the patch exactly, as a Pd render sounds.
PATCH_DOUBLE_DIVIDE = False


def _lop_coef(fc, fs):                      # Pd lop~: y += k (x - y), k = 2 pi fc / sr clipped to [0, 1]
    return min(max(TWO_PI * fc / fs, 0.0), 1.0)


def _qcos(f):                               # Pd's cosine approximation in bp~
    if -np.pi / 2 <= f <= np.pi / 2:
        g = f * f
        return ((g * g * g * (-1 / 720) + g * g * (1 / 24)) - g * 0.5) + 1
    return 0.0


# The three filters below run one step at a time with coefficients that change between steps. Their
# state is kept in Pd's own form (the last output, or hip~'s internal value), and handed to lfilter
# converted for the step's coefficients: a coefficient change then gives the same transient as in Pd.

def _lop(x, fcs, fs, n):
    out, y = np.empty_like(x), 0.0
    for i, fc in enumerate(fcs):
        k = _lop_coef(fc, fs)
        seg, _ = lfilter([k], [1, -(1 - k)], x[i * n:(i + 1) * n], zi=[(1 - k) * y])
        out[i * n:(i + 1) * n], y = seg, seg[-1]
    return out


def _hip(x, fcs, fs, n):
    """Pd hip~: w = x + c w[-1]; out = (1 + c)/2 (w - w[-1]), unity gain at Nyquist."""
    out, w_last = np.empty_like(x), 0.0
    for i, fc in enumerate(fcs):
        c = min(max(1 - TWO_PI * fc / fs, 0.0), 1.0)
        w, _ = lfilter([1], [1, -c], x[i * n:(i + 1) * n], zi=[c * w_last])
        out[i * n:(i + 1) * n] = 0.5 * (1 + c) * np.diff(w, prepend=w_last)
        w_last = w[-1]
    return out


def _bp(x, fcs, q, fs, n):
    """Pd bp~: y = x + c1 y[-1] + c2 y[-2]; out = gain y."""
    out, last, prev = np.empty_like(x), 0.0, 0.0
    for i, fc in enumerate(fcs):
        w = fc * TWO_PI / fs
        omr = 1.0 if q < 0.001 else min(w / q, 1.0)
        r = 1 - omr
        c1, c2, gain = 2 * _qcos(w) * r, -r * r, 2 * omr * (omr + r * w)
        y, _ = lfilter([1], [1, -c1, -c2], x[i * n:(i + 1) * n], zi=[c1 * last + c2 * prev, c2 * last])
        out[i * n:(i + 1) * n] = gain * y
        last, prev = y[-1], y[-2]
    return out


def _line(targets: np.ndarray, n: int, ramp: int, start: float = 0.0) -> np.ndarray:
    """Pd line~: on each step's new target, ramp linearly from the current value over `ramp` samples."""
    out = np.empty(len(targets) * n)
    v = start
    k = np.arange(1, n + 1)
    for i, t in enumerate(targets):
        seg = v + (t - v) * np.minimum(k / ramp, 1.0)
        out[i * n:(i + 1) * n] = seg
        v = seg[-1]
    return out


def render(throttle, pscale: float = 1.0, fs: int = FS, step_ms: float = 20.0, seed: int = 0) -> np.ndarray:
    """The patch's output for a throttle curve (one 0-127 value per step)."""
    rng = np.random.default_rng(seed)
    u = np.clip(np.asarray(throttle, float), 0, 127) / 127
    n = int(round(fs * step_ms / 1000))
    N = len(u) * n
    us = np.repeat(u, n)                                  # control value at every sample
    noise = lambda: rng.uniform(-1, 1, N)                 # noqa: E731  (Pd noise~ is uniform in [-1, 1))

    u2 = u / 127 if PATCH_DOUBLE_DIVIDE else u           # objects 44, 46, 49
    base = (150 + 180 * us) * pscale
    tone = np.zeros(N)
    for det, wf in zip(DETUNE, WANDER_HZ):
        k = _lop_coef(wf, fs)
        wander = lfilter([k], [1, -(1 - k)], noise()) * np.repeat(200 + 600 * u2, n)
        tone += np.mod(np.cumsum((base + det + wander) / fs), 1.0)
    tone = _lop(tone, 500 + 2000 * u2, fs, n)

    wash = _bp(noise(), 500 + 1000 * u, 0.8, fs, n)
    wash *= _line(0.2 - 0.15 * u, n, int(0.05 * fs))

    body = _hip(tone + wash, 30 + 400 * u2, fs, n)
    body *= _line(0.03 + 0.1 * u, n, int(0.05 * fs))

    k = _lop_coef(8, fs)
    flutter = 1 + 4 * lfilter([k], [1, -(1 - k)], noise())
    return body * flutter
