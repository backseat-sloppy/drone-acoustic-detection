"""Log-mel frontend, with the streaming ring-buffer path that makes per-decision
feature cost scale with HOP rather than WINDOW.

Deliberately no librosa. The Skeptic's warning stands: a general-purpose mel
implementation can easily cost more than the network it feeds, and you cannot
report an honest latency budget for code you did not time. This is plain
numpy so every millisecond in the benchmark is attributable.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .config import DEFAULT, FrontendConfig

_EPS = 1e-10


def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    """HTK mel scale."""
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(cfg: FrontendConfig = DEFAULT) -> np.ndarray:
    """Triangular mel filterbank, shape (n_mels, n_freq_bins)."""
    edges_mel = np.linspace(hz_to_mel(cfg.fmin), hz_to_mel(cfg.fmax), cfg.n_mels + 2)
    edges_hz = mel_to_hz(edges_mel)
    bin_hz = np.linspace(0.0, cfg.sample_rate / 2.0, cfg.n_freq_bins)

    fb = np.zeros((cfg.n_mels, cfg.n_freq_bins), dtype=np.float32)
    for m in range(cfg.n_mels):
        lo, mid, hi = edges_hz[m], edges_hz[m + 1], edges_hz[m + 2]
        if mid <= lo or hi <= mid:
            continue
        rising = (bin_hz - lo) / (mid - lo)
        falling = (hi - bin_hz) / (hi - mid)
        fb[m] = np.clip(np.minimum(rising, falling), 0.0, None)

    # Slaney-style area normalisation keeps filter gain independent of width,
    # so low-frequency bands (where the blade-pass comb lives) are not
    # systematically quieter than high ones.
    widths = edges_hz[2:] - edges_hz[:-2]
    fb *= (2.0 / widths)[:, None].astype(np.float32)
    return fb


class LogMel:
    """Stateless full-window log-mel. Use for training and offline evaluation."""

    def __init__(self, cfg: FrontendConfig = DEFAULT) -> None:
        self.cfg = cfg
        self._fb = mel_filterbank(cfg)
        self._win = np.hanning(cfg.n_fft).astype(np.float32)

    def frames(self, x: np.ndarray) -> np.ndarray:
        """(n_samples,) -> (n_mels, n_frames) log-mel."""
        cfg = self.cfg
        x = np.asarray(x, dtype=np.float32)
        n = 1 + max(0, (len(x) - cfg.n_fft) // cfg.hop_length)
        if n == 0:
            return np.zeros((cfg.n_mels, 0), dtype=np.float32)
        # Strided view avoids materialising the overlapped copies.
        idx = np.lib.stride_tricks.as_strided(
            x,
            shape=(n, cfg.n_fft),
            strides=(x.strides[0] * cfg.hop_length, x.strides[0]),
        )
        spec = np.abs(np.fft.rfft(idx * self._win, axis=1)) ** 2
        mel = spec.astype(np.float32) @ self._fb.T
        return np.log(mel + _EPS).T.astype(np.float32)


class StreamingLogMel:
    """Incremental log-mel over a ring buffer of frames.

    This is the single highest-leverage implementation decision in the whole
    pipeline. Push one HOP of new audio and only the genuinely new frames are
    computed; the window is reassembled from retained frames. Per-decision
    feature cost therefore scales with hop (125 ms) instead of window (1 s),
    an 8x saving at the default settings.
    """

    def __init__(self, cfg: FrontendConfig = DEFAULT) -> None:
        self.cfg = cfg
        self._fb = mel_filterbank(cfg)
        self._win = np.hanning(cfg.n_fft).astype(np.float32)
        self._tail = np.zeros(0, dtype=np.float32)
        self._frames: deque[np.ndarray] = deque(maxlen=cfg.frames_per_window)

    def reset(self) -> None:
        self._tail = np.zeros(0, dtype=np.float32)
        self._frames.clear()

    def push(self, samples: np.ndarray) -> int:
        """Feed new audio. Returns how many new frames were computed."""
        cfg = self.cfg
        buf = np.concatenate([self._tail, np.asarray(samples, dtype=np.float32)])

        n_new = 1 + (len(buf) - cfg.n_fft) // cfg.hop_length if len(buf) >= cfg.n_fft else 0
        if n_new <= 0:
            self._tail = buf
            return 0

        idx = np.lib.stride_tricks.as_strided(
            buf,
            shape=(n_new, cfg.n_fft),
            strides=(buf.strides[0] * cfg.hop_length, buf.strides[0]),
        )
        spec = np.abs(np.fft.rfft(idx * self._win, axis=1)) ** 2
        mel = np.log((spec.astype(np.float32) @ self._fb.T) + _EPS).astype(np.float32)
        for row in mel:
            self._frames.append(row)

        # Retain only the un-consumed samples; the next frame starts exactly here.
        self._tail = buf[n_new * cfg.hop_length:].copy()
        return n_new

    @property
    def ready(self) -> bool:
        return len(self._frames) == self._frames.maxlen

    def window(self) -> np.ndarray:
        """(n_mels, frames_per_window). Raises if not yet filled."""
        if not self.ready:
            raise RuntimeError(
                f"window not full: {len(self._frames)}/{self._frames.maxlen} frames"
            )
        return np.stack(self._frames, axis=1)


def mfcc_from_logmel(logmel: np.ndarray, n_mfcc: int = 20) -> np.ndarray:
    """Orthonormal DCT-II along the mel axis. (n_mels, T) -> (n_mfcc, T)."""
    n_mels = logmel.shape[0]
    k = np.arange(n_mels)[None, :]
    n = np.arange(n_mels)[:, None]
    basis = np.cos(np.pi * (k + 0.5) * n / n_mels) * np.sqrt(2.0 / n_mels)
    basis[0] *= np.sqrt(0.5)
    return (basis[:n_mfcc].astype(np.float32) @ logmel).astype(np.float32)


def summary_stats(logmel: np.ndarray, n_mfcc: int = 20) -> np.ndarray:
    """Fixed-length vector for the classical baseline: MFCC mean/std plus
    delta mean/std. Shares the exact same frontend as the CNN, which is the
    point: the mel cost is common, so the baseline's speed advantage is only
    in the classifier head."""
    m = mfcc_from_logmel(logmel, n_mfcc)
    if m.shape[1] < 2:
        d = np.zeros_like(m)
    else:
        d = np.diff(m, axis=1)
    return np.concatenate(
        [m.mean(1), m.std(1), d.mean(1), d.std(1)]
    ).astype(np.float32)
