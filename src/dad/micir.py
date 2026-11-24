"""A measured microphone impulse response as the last step of the training audio (physical test prep).

The IRs in `data/ir/<session>/IR_mic_in*.wav` were measured on 2026-11-23 (48 kHz float, 2 s, aligned so
that the loopback channel peaks at 5 ms). They hold the whole playback chain the physical test uses:
loudspeaker, room at the measuring position, and microphone. Each is trimmed to start 1 ms before its
onset (the acoustic delay says nothing), cut to `IR_SECONDS`, resampled to the frontend's 16 kHz and scaled
to unit energy. The frontend's level normalisation removes the overall gain anyway.

`convolve_torch` filters a batch of windows (rows) with one of the IRs each, causally, keeping the first
n samples: what the microphone would have recorded had the window started from silence.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

IR_SECONDS = 0.1          # 99.9 % of the energy lies within 9 ms of the peak (entry 272)
ONSET_REL = 0.05          # onset: the first sample above 5 % of the peak
PRE_S = 0.001


def load_irs(folder: str | Path, sample_rate: int = 16_000, names=("IR_mic_in3", "IR_mic_in4")) -> np.ndarray:
    """(len(names), taps) float64, each trimmed, resampled and at unit energy.
    `folder` may also be a .npy that `save_irs` wrote (what the server gets: the WAVs are 384 kB each)."""
    if str(folder).endswith(".npy"):
        return np.load(folder).astype(np.float64)
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    out = []
    for n in names:
        x, sr = sf.read(str(Path(folder) / f"{n}.wav"), always_2d=True, dtype="float64")
        a = x[:, 0]
        onset = int(np.flatnonzero(np.abs(a) >= ONSET_REL * np.abs(a).max())[0])
        a = a[max(onset - int(PRE_S * sr), 0):][: int(IR_SECONDS * sr)]
        g = gcd(sample_rate, sr)
        h = resample_poly(a, sample_rate // g, sr // g)
        out.append(h / np.sqrt(np.sum(h ** 2)))
    taps = min(len(h) for h in out)
    return np.stack([h[:taps] for h in out])


def convolve_np(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Rows of x (or one row) filtered by h, first x.shape[-1] samples, float32."""
    from scipy.signal import fftconvolve
    x = np.asarray(x, dtype=np.float64)
    return fftconvolve(x, h.reshape((1,) * (x.ndim - 1) + (-1,)), axes=-1)[..., : x.shape[-1]].astype(np.float32)


def convolve_torch(x, irs: np.ndarray, which: np.ndarray):
    """x: (B, n) tensor; row i filtered by irs[which[i]]; first n samples, float32, on x's device."""
    import torch
    n, k = x.shape[1], irs.shape[1]
    nfft = 1 << (n + k - 1 - 1).bit_length()
    h = torch.from_numpy(irs[which]).to(x.device)
    y = torch.fft.irfft(torch.fft.rfft(x.double(), nfft) * torch.fft.rfft(h, nfft), nfft)[:, :n]
    return y.float()
