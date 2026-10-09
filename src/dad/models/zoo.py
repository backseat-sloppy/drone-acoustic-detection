"""Bigger spectrogram models for a laptop or a GPU workstation.

Once the microcontroller requirement is dropped, the reasons dscnn.py gives
for staying tiny (INT8, kilobytes of RAM, CMSIS-NN kernels) no longer apply.
Compute is still a rounding error next to window fill: even a model ten times
slower than the DS-CNN is a few milliseconds against a 1000 ms window. So the
models here buy capacity, not speed, and each one tests a different idea of
what in the spectrogram separates a drone from a confuser:

  resnet       wider, deeper 2D convolutions with residual connections. Local
               time-frequency texture, more of it than the DS-CNN can hold.
  crnn         convolutions for texture, then a bidirectional GRU over time.
               Tests whether the TIME EVOLUTION (rotor wobble, Doppler) carries
               information the pooled CNN averages away. The README rejected
               this for streaming on an MCU, not on accuracy grounds.
  transformer  a small Audio-Spectrogram-Transformer-style encoder trained from
               scratch. Global attention across the whole window. Known to be
               data-hungry, so expect it to trail on small datasets; it is here
               to measure that, not because it is expected to win.

Every model takes (batch, 1, n_mels, n_frames) and returns two logits, and
every one accepts any number of frames, so the same architecture runs in the
window-length sweep without changes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .dscnn import DsCnn


class _ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
        self.skip = (
            nn.Identity()
            if stride == 1 and cin == cout
            else nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))
        )

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + self.skip(x))


class ResNetSmall(nn.Module):
    """Three residual stages, about 0.7M parameters at the default widths."""

    def __init__(self, n_mels: int = 64, n_classes: int = 2,
                 widths: tuple[int, ...] = (32, 64, 128), blocks_per_stage: int = 2,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        blocks: list[nn.Module] = []
        cin = widths[0]
        for i, w in enumerate(widths):
            for j in range(blocks_per_stage):
                stride = 2 if (i > 0 and j == 0) else 1
                blocks.append(_ResBlock(cin, w, stride))
                cin = w
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(cin, n_classes)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        return self.head(self.drop(self.pool(x).flatten(1)))


class Crnn(nn.Module):
    """Convolutions pool frequency only, so the GRU sees every frame."""

    def __init__(self, n_mels: int = 64, n_classes: int = 2,
                 channels: tuple[int, ...] = (32, 64, 128), hidden: int = 96,
                 dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        cin = 1
        for c in channels:
            layers += [
                nn.Conv2d(cin, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 1)),
            ]
            cin = c
        self.conv = nn.Sequential(*layers)
        f_out = n_mels // (2 ** len(channels))
        self.gru = nn.GRU(cin * f_out, hidden, batch_first=True, bidirectional=True)
        # Attention pooling over time: the model learns which frames matter,
        # rather than averaging a passing drone together with the silence
        # around it.
        self.attn = nn.Linear(2 * hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        x = self.conv(x)                                  # (B, C, F, T)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)    # (B, T, C*F)
        h, _ = self.gru(x)
        w = torch.softmax(self.attn(h), dim=1)
        return self.head(self.drop((w * h).sum(dim=1)))


def _sinusoidal(n: int, d: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10_000.0) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class SpecTransformer(nn.Module):
    """Each token is a group of consecutive frames across all mel bands.

    Sinusoidal positions rather than learned ones, so a model built once runs
    at any window length.
    """

    def __init__(self, n_mels: int = 64, n_classes: int = 2, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 4, frames_per_token: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.embed = nn.Conv1d(n_mels, d_model, kernel_size=frames_per_token, stride=frames_per_token)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.embed(x.squeeze(1)).transpose(1, 2)      # (B, tokens, d)
        x = x + _sinusoidal(x.shape[1], x.shape[2], x.device)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
        return self.head(self.norm(self.encoder(x)[:, 0]))


ARCHITECTURES: dict[str, type[nn.Module]] = {
    "dscnn": DsCnn,
    "resnet": ResNetSmall,
    "crnn": Crnn,
    "transformer": SpecTransformer,
}


def build(name: str, n_mels: int) -> nn.Module:
    try:
        return ARCHITECTURES[name](n_mels=n_mels)
    except KeyError:
        raise ValueError(f"unknown architecture {name!r}; choose from {sorted(ARCHITECTURES)}") from None


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
