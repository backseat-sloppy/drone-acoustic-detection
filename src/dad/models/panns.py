"""PANNs Cnn14_16k: an AudioSet-pretrained encoder (proposed iteration 12, plan of 2026-11-06).

Kong, Cao, Iqbal, Wang, Wang and Plumbley, "PANNs: Large-Scale Pretrained Audio Neural Networks for
Audio Pattern Recognition", IEEE/ACM TASLP 28 (2020). Architecture ported from the reference code
(github.com/qiuqiangkong/audioset_tagging_cnn, MIT licence). Weights: `Cnn14_16k_mAP=0.438.pth`,
Zenodo record 3987831, CC BY 4.0 (per mirrors; the record itself was not checked from the Mac).

Why its own frontend. The checkpoint was pretrained on a different log-mel from the frozen contract in
`dad.config`: 512-sample (32 ms) periodic-Hann frames, reflect-padded and centred, power spectrum,
the librosa (Slaney) mel filterbank with Slaney area norm, and 10*log10 with amin 1e-10. The repo
frontend is 400-sample frames, the HTK mel scale and the natural log. Feeding the encoder the repo's
features would discard what pretraining learned, so the encoder carries the pretraining frontend,
written here in plain torch and numpy (no librosa, no torchlibrosa: the repo's no-librosa rule).
`tests/test_panns.py` checks the filterbank and the STFT kernels against the checkpoint's own frozen
copies (`logmel_extractor.melW`, `spectrogram_extractor.stft.conv_*`), so a mismatch cannot pass.

Level. Each window's mean log-mel (dB, over time and bands) is set to the mean of the checkpoint's
`bn0` running mean, the average AudioSet log-mel level. Loudness then says nothing about the label
(as the repo's "level" variant), and the encoder sees inputs at the level it was trained on.

Module names follow the reference code, so the checkpoint loads with strict=True.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 16_000
N_FFT = 512
HOP = 160
N_MELS = 64
FMIN, FMAX = 50.0, 8_000.0
AMIN = 1e-10
N_CLASSES = 527
EMBED_DIM = 2048
CHECKPOINT_NAME = "Cnn14_16k_mAP=0.438.pth"
CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1"


# ------------------------------------------------------------------ frontend (numpy references)

def hz_to_mel_slaney(f):
    """librosa.hz_to_mel(htk=False): linear below 1 kHz, logarithmic above."""
    f = np.asarray(f, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, math.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-12) / min_log_hz) / logstep, f / f_sp)


def mel_to_hz_slaney(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, math.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


def slaney_mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS,
                          fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    """librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax) with its defaults (htk=False, norm="slaney"):
    (n_mels, n_fft // 2 + 1), float32."""
    fftfreqs = np.linspace(0.0, sr / 2.0, 1 + n_fft // 2)
    mel_f = mel_to_hz_slaney(np.linspace(hz_to_mel_slaney(fmin), hz_to_mel_slaney(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    w = np.zeros((n_mels, len(fftfreqs)))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        w[i] = np.maximum(0.0, np.minimum(lower, upper))
    w *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]
    return w.astype(np.float32)


def stft_kernels(n_fft: int = N_FFT) -> tuple[np.ndarray, np.ndarray]:
    """The (bins, 1, n_fft) conv kernels of torchlibrosa's STFT: DFT rows times a periodic Hann window."""
    n = np.arange(n_fft)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * n / n_fft)          # scipy get_window("hann", fftbins=True)
    k = np.arange(n_fft // 2 + 1)[:, None]
    ang = 2 * np.pi * k * n[None, :] / n_fft
    return ((np.cos(ang) * win)[:, None, :].astype(np.float32),
            (-np.sin(ang) * win)[:, None, :].astype(np.float32))


# ------------------------------------------------------------------ frontend (torch modules)

class _STFT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        bins = N_FFT // 2 + 1
        self.conv_real = nn.Conv1d(1, bins, N_FFT, stride=HOP, bias=False)
        self.conv_imag = nn.Conv1d(1, bins, N_FFT, stride=HOP, bias=False)
        re, im = stft_kernels()
        with torch.no_grad():
            self.conv_real.weight.copy_(torch.from_numpy(re))
            self.conv_imag.weight.copy_(torch.from_numpy(im))


class _Spectrogram(nn.Module):
    """(N, samples) -> power spectrogram (N, 1, frames, bins), centred with reflect padding."""

    def __init__(self) -> None:
        super().__init__()
        self.stft = _STFT()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x[:, None, :], (N_FFT // 2, N_FFT // 2), mode="reflect")
        re, im = self.stft.conv_real(x), self.stft.conv_imag(x)           # (N, bins, frames)
        return (re ** 2 + im ** 2)[:, None].transpose(2, 3)


class _LogMel(nn.Module):
    """Power spectrogram -> 10*log10 mel power (N, 1, frames, mels)."""

    def __init__(self) -> None:
        super().__init__()
        self.melW = nn.Parameter(torch.from_numpy(slaney_mel_filterbank().T.copy()))   # (bins, mels)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return 10.0 * torch.log10(torch.clamp(spec @ self.melW, min=AMIN))


# ------------------------------------------------------------------ Cnn14

class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn1, self.bn2 = nn.BatchNorm2d(cout), nn.BatchNorm2d(cout)

    def forward(self, x: torch.Tensor, pool: tuple[int, int]) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool) if pool != (1, 1) else x


class Cnn14(nn.Module):
    """Cnn14 as in the reference code, from waveform to (embedding, AudioSet logits).

    forward(wave, eq_db=None, mask=None):
      wave  (N, samples) at 16 kHz;
      eq_db (N, 64) optional per-band gain in dB, added before level normalisation (training EQ);
      mask  optional callable on the (N, 64, frames) bn0 output (SpecAugment), applied in training only,
            where the reference code applies its own SpecAugmentation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.spectrogram_extractor = _Spectrogram()
        self.logmel_extractor = _LogMel()
        self.bn0 = nn.BatchNorm2d(N_MELS)
        chans = (1, 64, 128, 256, 512, 1024, 2048)
        for i in range(6):
            setattr(self, f"conv_block{i + 1}", _ConvBlock(chans[i], chans[i + 1]))
        self.fc1 = nn.Linear(EMBED_DIM, EMBED_DIM, bias=True)
        self.fc_audioset = nn.Linear(EMBED_DIM, N_CLASSES, bias=True)
        for m in (self.spectrogram_extractor, self.logmel_extractor):   # frozen, as in the reference
            for p in m.parameters():
                p.requires_grad_(False)

    def level_ref(self) -> torch.Tensor:
        """The mean AudioSet log-mel level (dB) the encoder was trained on: the mean of bn0's running mean."""
        return self.bn0.running_mean.mean()

    def logmel(self, wave: torch.Tensor, eq_db: torch.Tensor | None = None) -> torch.Tensor:
        """(N, samples) -> level-normalised log-mel (N, 1, frames, 64) in dB."""
        L = self.logmel_extractor(self.spectrogram_extractor(wave))
        if eq_db is not None:
            L = L + eq_db[:, None, None, :]
        return L - L.mean(dim=(1, 2, 3), keepdim=True) + self.level_ref()

    def forward(self, wave: torch.Tensor, eq_db: torch.Tensor | None = None, mask=None):
        x = self.logmel(wave, eq_db)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)                  # bn0 over the mel axis
        if self.training and mask is not None:
            x = mask(x[:, 0].transpose(1, 2)).transpose(1, 2)[:, None]
        for i in range(1, 7):
            x = getattr(self, f"conv_block{i}")(x, (2, 2) if i < 6 else (1, 1))
            x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)                                         # over mel
        x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2)                # over time
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        return F.dropout(x, p=0.5, training=self.training), self.fc_audioset(x)


def load_pretrained(path, device="cpu") -> Cnn14:
    """Cnn14 with the checkpoint's weights, strict: every key must match."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = Cnn14()
    model.load_state_dict(ck.get("model", ck), strict=True)
    return model.to(device).eval()


class PannsDetector(nn.Module):
    """Cnn14 encoder + a 2-logit head (index 1 = drone), the zoo models' output convention."""

    def __init__(self, encoder: Cnn14 | None = None) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else Cnn14()
        self.head = nn.Linear(EMBED_DIM, 2)

    def forward(self, wave, eq_db=None, mask=None) -> torch.Tensor:
        emb, _ = self.encoder(wave, eq_db, mask)
        return self.head(emb)
