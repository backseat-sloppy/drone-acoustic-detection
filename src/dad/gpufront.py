"""The frontend for a whole batch at once in torch, on the GPU (report entry 092).

Profiling a training batch on the JupyterLab server (one thread) put 46% of
the time in the speed change (`scipy.signal.resample_poly`, half of it
redesigning the same filter for every window), 22% in the in-band power and
17% in the STFT, one window at a time. The networks waited for the CPU.

Here the speed change, STFT, bands, log, training EQ, level normalisation
and MFCC DCT run for all windows of a batch together. The numbers are the
numpy path's (`rangerun.RangeSampler.batch`, `realrun.featurise`) to float
rounding: the resampler is scipy's polyphase filter with the same design and
offset, the STFT runs in float64 as numpy's does. tests/test_gpufront.py
checks both against numpy.

    fe = GpuFrontend(cfg, "cuda")
    L = fe.train_features(audio, down, eq)   # RangeSampler.raw_batch output
    L = fe.features(audio)                   # inference, as realrun.featurise
"""
from __future__ import annotations

import functools
import math

import numpy as np
import torch
from scipy.fft import dct

from .config import FrontendConfig
from .features import _EPS, band_matrix
from .rangerun import SPEED_UP as UP, speed_filter
_FLOOR = math.log(1e-5)   # realrun._FLOOR


@functools.lru_cache(maxsize=None)
def poly_filter(down: int) -> tuple[int, int, int, np.ndarray]:
    """What scipy.signal.resample_poly(x, UP, down) does with its filter: reduced up, reduced down, samples
    dropped at the start, and the filter zero-padded in front and scaled by up."""
    g = math.gcd(UP, down)
    up, dn = UP // g, down // g
    h = speed_filter(down) * up
    half = (len(h) - 1) // 2
    pre = dn - half % dn
    return up, dn, (half + pre) // dn, np.concatenate([np.zeros(pre), h])


class GpuFrontend:
    def __init__(self, cfg: FrontendConfig, device) -> None:
        self.cfg, self.device = cfg, torch.device(device)
        self.fb = torch.tensor(band_matrix(cfg), device=self.device)                        # (bands, bins)
        self.win = torch.tensor(np.hanning(cfg.n_fft).astype(np.float32), device=self.device)
        self.dct = (torch.tensor(dct(np.eye(cfg.n_mels), type=2, axis=0, norm="ortho")[: cfg.n_mfcc],
                                 dtype=torch.float32, device=self.device) if cfg.kind == "mfcc" else None)
        self._taps: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _gather(self, down: int, n_in: int, n_out: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Input index and weight of every filter tap of every output sample: (taps, n_out) each.
        upfirdn's y[m] = sum_j x[j] h[m*dn - j*up], m counted after the samples resample_poly drops."""
        key = (down, n_in, n_out)
        if key not in self._taps:
            up, dn, drop, h = poly_filter(down)
            k = (np.arange(n_out) + drop) * dn
            j = k // up - np.arange(-(-len(h) // up))[:, None]                               # (taps, n_out)
            hi = k - j * up
            ok = (j >= 0) & (j < n_in) & (hi < len(h))
            w = np.where(ok, h[np.minimum(hi, len(h) - 1)], 0.0)
            self._taps[key] = (torch.tensor(np.where(ok, j, 0), device=self.device),
                               torch.tensor(w, device=self.device))
        return self._taps[key]

    def speed(self, x: torch.Tensor, down: np.ndarray, n_out: int) -> torch.Tensor:
        """Each row played UP/down[i] times faster (resample_poly(x, UP, down)), first n_out samples."""
        x = x.double()
        out = x[:, :n_out].clone()                                                            # down == UP: as is
        for d in np.unique(down):
            if d == UP:
                continue
            rows = torch.tensor(np.flatnonzero(down == d), device=self.device)
            j, w = self._gather(int(d), x.shape[1], n_out)
            out[rows] = (x[rows][:, j] * w).sum(dim=1)
        return out.float()   # speed_change returns float32 too

    def logbands(self, x: torch.Tensor) -> torch.Tensor:
        """(N, samples) -> (N, bands, frames), as features.LogMel.frames."""
        cfg = self.cfg
        fr = (x.float().unfold(1, cfg.n_fft, cfg.hop_length) * self.win).double()   # windowed in float32, FFT in float64, as numpy
        spec = torch.fft.rfft(fr, dim=-1).abs().square().float()
        return torch.log(spec @ self.fb.T + _EPS).transpose(1, 2)

    def finish(self, L: torch.Tensor, level_norm: bool = True) -> torch.Tensor:
        """realrun.level_normalise, then realrun.to_model."""
        if level_norm:
            ref = torch.log(torch.exp(L.double()).mean(dim=(1, 2), keepdim=True))
            L = torch.clamp(L - ref, min=_FLOOR).float()
        return L if self.dct is None else self.dct @ L

    def train_features(self, audio: np.ndarray, down: np.ndarray, eq: np.ndarray) -> torch.Tensor:
        """RangeSampler.raw_batch -> what RangeSampler.batch returns, on the device."""
        x = self.speed(torch.from_numpy(np.asarray(audio)).to(self.device), down, self.cfg.samples_per_window)
        L = self.logbands(x) + torch.from_numpy(eq).to(self.device)[:, :, None]
        return self.finish(L)

    def features(self, audio: np.ndarray, level_norm: bool = True) -> torch.Tensor:
        """realrun.featurise, on the device."""
        return self.finish(self.logbands(torch.from_numpy(np.asarray(audio)).to(self.device)), level_norm)
