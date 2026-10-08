"""Depthwise-separable CNN on log-mel: the recommended model.

The MLPerf Tiny DS-CNN class of network. The Researcher's decisive number: the
SAME DS-CNN measures 181.92 ms unoptimised and 0.39 ms optimised on the same
microcontroller class. A 465x spread WITHIN one architecture, which dwarfs any
gap between CNN, CRNN and transformer. So the architecture choice here is
nearly arbitrary among small convnets, and your latency work is quantisation
and kernel selection, not architecture search.

Torch is an optional dependency. Everything else in this package runs without
it so you can time the frontend and the baseline on a clean install.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    nn = object  # type: ignore


if TORCH_AVAILABLE:

    class _DsBlock(nn.Module):
        """Depthwise 3x3 then pointwise 1x1. The whole reason this is cheap."""

        def __init__(self, ch: int) -> None:
            super().__init__()
            self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
            self.bn1 = nn.BatchNorm2d(ch)
            self.pw = nn.Conv2d(ch, ch, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(ch)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.act(self.bn1(self.dw(x)))
            return self.act(self.bn2(self.pw(x)))

    class DsCnn(nn.Module):
        def __init__(self, n_mels: int = 64, n_classes: int = 2,
                     channels: int = 64, n_blocks: int = 4) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, channels, (10, 4), stride=(2, 2), padding=(5, 1), bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            self.blocks = nn.Sequential(*[_DsBlock(channels) for _ in range(n_blocks)])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(channels, n_classes)
            self.n_mels = n_mels

        def forward(self, x):
            # x: (B, 1, n_mels, n_frames)
            x = self.blocks(self.stem(x))
            return self.head(self.pool(x).flatten(1))

        def n_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

    def quantize_dynamic_int8(model: "DsCnn"):
        """INT8 on the Linear head only. torch's eager dynamic quantisation does
        not cover Conv2d, so this is a floor, not the real win. For the actual
        465x you need TFLite-Micro, CMSIS-NN or microTVM with full static INT8
        on the convolutions. Say so in the report rather than implying torch
        dynamic quantisation is embedded-grade."""
        return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

else:  # pragma: no cover

    class DsCnn:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError(
                "DS-CNN needs torch. Install the optional extra:  uv sync --extra dl"
            )
