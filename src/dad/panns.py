"""Iteration 13: a frozen pretrained audio network (PANNs Cnn14_16k) as the frontend (council, entry 053).

Kong, Cao, Iqbal, Wang, Wang, Plumbley, "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio
Pattern Recognition", 2020. Weights: Zenodo record 3987831, `Cnn14_16k_mAP=0.438.pth`, CC BY 4.0,
trained on AudioSet at 16 kHz (window 512, hop 160, 64 mel bands 50-8000 Hz). The checkpoint holds the
STFT kernels and the mel matrix of the original torchlibrosa frontend, so no audio library is needed.

`PannsScorer` embeds each 1 s window (the 2048-d layer before AudioSet's classifier) and scores it with
a logistic regression trained on the same sampled windows as the networks (`RangeSampler.raw_batch`:
mixing, simulated distance and speed change; the log-mel EQ is not applied, it acts on log-mel bands).
Every window is scaled to one RMS first, as the networks' frontend removes level.
"""
from __future__ import annotations

import pickle
import types
from pathlib import Path

import numpy as np

from .config import DEFAULT, FrontendConfig

WEIGHTS = Path(__file__).resolve().parents[2] / "artifacts" / "pretrained" / "Cnn14_16k_mAP=0.438.pth"
N_FFT, HOP = 512, 160
TARGET_RMS = 0.05           # every window is scaled to this RMS (about -26 dBFS) before the network


def _net():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ConvBlock(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
            self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
            self.bn1, self.bn2 = nn.BatchNorm2d(cout), nn.BatchNorm2d(cout)

        def forward(self, x, pool):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            return F.avg_pool2d(x, pool)

    class Cnn14(nn.Module):
        """Cnn14 as in PANNs `models.py`, in eval mode only; returns the 2048-d embedding."""

        def __init__(self):
            super().__init__()
            n = N_FFT // 2 + 1
            self.spectrogram_extractor = nn.Module()
            self.spectrogram_extractor.stft = nn.Module()
            self.spectrogram_extractor.stft.conv_real = nn.Conv1d(1, n, N_FFT, HOP, bias=False)
            self.spectrogram_extractor.stft.conv_imag = nn.Conv1d(1, n, N_FFT, HOP, bias=False)
            self.logmel_extractor = nn.Module()
            self.logmel_extractor.melW = nn.Parameter(torch.zeros(n, 64))
            self.bn0 = nn.BatchNorm2d(64)
            chans = [1, 64, 128, 256, 512, 1024, 2048]
            for i in range(6):
                setattr(self, f"conv_block{i + 1}", ConvBlock(chans[i], chans[i + 1]))
            self.fc1 = nn.Linear(2048, 2048)
            self.fc_audioset = nn.Linear(2048, 527)

        def forward(self, wav):                                   # (batch, samples)
            x = F.pad(wav[:, None, :], (N_FFT // 2, N_FFT // 2), mode="reflect")   # center=True
            st = self.spectrogram_extractor.stft
            p = st.conv_real(x) ** 2 + st.conv_imag(x) ** 2        # (batch, freq, time)
            mel = torch.matmul(p.transpose(1, 2), self.logmel_extractor.melW)     # (batch, time, mel)
            x = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))[:, None]          # ref 1, no top_db
            x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
            for i in range(5):
                x = getattr(self, f"conv_block{i + 1}")(x, (2, 2))
            x = self.conv_block6(x, (1, 1))
            x = x.mean(dim=3)                                     # over frequency
            x = x.max(dim=2).values + x.mean(dim=2)               # over time
            return F.relu(self.fc1(x))

    return Cnn14()


def load_cnn14(device: str = "cpu", weights: Path = WEIGHTS):
    import torch
    net = _net()
    state = torch.load(weights, map_location="cpu", weights_only=False)["model"]
    state = {k: v for k, v in state.items() if not k.startswith("logmel_extractor.") or k.endswith("melW")}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"Cnn14 weights missing: {missing}")
    return net.eval().to(device)


def level(audio: np.ndarray) -> np.ndarray:
    """Each window (rows) scaled to TARGET_RMS; silent windows stay silent."""
    a = np.asarray(audio, dtype=np.float32)
    rms = np.sqrt(np.mean(a.astype(np.float64) ** 2, axis=-1, keepdims=True))
    return (a * (TARGET_RMS / np.maximum(rms, 1e-8))).astype(np.float32)


class PannsScorer:
    name = "panns cnn14_16k + logistic regression"

    def __init__(self, cfg: FrontendConfig = DEFAULT, device: str = "cpu", C: float = 1.0) -> None:
        self.cfg, self.device, self.C = cfg, device, C
        self.net, self.clf = None, None

    def embed(self, audio: np.ndarray, batch_size: int = 256) -> np.ndarray:
        import torch
        if self.net is None:
            self.net = load_cnn14(self.device)
        out = []
        with torch.no_grad():
            for i in range(0, len(audio), batch_size):
                x = torch.from_numpy(level(audio[i:i + batch_size])).to(self.device)
                out.append(self.net(x).float().cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, 2048), np.float32)

    def fit(self, sampler, n_windows: int = 200_000, batch_size: int = 64, log=print) -> "PannsScorer":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        from .rangerun import resample_down
        from .realrun import prefetch
        n_out = self.cfg.samples_per_window
        X, y = [], []
        source = types.SimpleNamespace(batch=sampler.raw_batch)
        for b, (audio, down, _eq, t) in enumerate(prefetch(source, n_windows // batch_size, batch_size)):
            wav = np.stack([resample_down(a, d)[:n_out] for a, d in zip(audio, down)])
            X.append(self.embed(wav))
            y.append(t)
            if (b + 1) * batch_size % 50_048 < batch_size:
                log(f"    panns: {(b + 1) * batch_size} windows embedded")
        X, y = np.concatenate(X), np.concatenate(y)
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=2000))
        self.clf.fit(X, y)
        log(f"    panns: logistic regression fitted on {len(y)} windows, train accuracy "
            f"{self.clf.score(X, y):.3f}")
        return self

    def score(self, audio: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.embed(np.atleast_2d(audio)))[:, 1]

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(dict(C=self.C, clf=self.clf, weights=WEIGHTS.name), f)

    @classmethod
    def load(cls, path, cfg: FrontendConfig = DEFAULT, device: str = "cpu") -> "PannsScorer":
        """Unpickling runs code from the file: load only models you trained."""
        with open(path, "rb") as f:
            ck = pickle.load(f)
        m = cls(cfg, device, ck["C"])
        m.clf = ck["clf"]
        return m


class PannsFtScorer:
    """Iteration 14: Cnn14_16k fine-tuned end to end with a one-logit head (LP-FT, entry 126).

    The head trains alone for the first `probe_share` of the steps, then the whole network
    (backbone lr / 10). BatchNorm keeps AudioSet's running statistics: the net stays in eval mode.
    """
    name = "panns cnn14_16k fine-tuned"

    def __init__(self, cfg: FrontendConfig = DEFAULT, device: str = "cpu") -> None:
        self.cfg, self.device = cfg, device
        self.net, self.head = None, None

    def _build(self, pretrained: bool = True):
        import torch.nn as nn
        self.net = (load_cnn14(self.device) if pretrained else _net().eval().to(self.device))
        self.head = nn.Linear(2048, 1).to(self.device)

    def fit(self, sampler, n_windows: int = 200_000, batch_size: int = 32, lr: float = 1e-3,
            probe_share: float = 0.1, log=print) -> "PannsFtScorer":
        import torch
        import torch.nn.functional as F

        from .rangerun import resample_down
        from .realrun import prefetch
        torch.manual_seed(sampler.seed)
        self._build()
        n_out, steps = self.cfg.samples_per_window, n_windows // batch_size
        probe = int(probe_share * steps)
        opt = torch.optim.AdamW([{"params": self.net.parameters(), "lr": lr / 10},
                                 {"params": self.head.parameters(), "lr": lr}], weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        source = types.SimpleNamespace(batch=sampler.raw_batch)
        running = 0.0
        for step, (audio, down, _eq, t) in enumerate(prefetch(source, steps, batch_size), start=1):
            for p in self.net.parameters():
                p.requires_grad_(step > probe)
            wav = np.stack([resample_down(a, d)[:n_out] for a, d in zip(audio, down)])
            x = torch.from_numpy(level(wav)).to(self.device)
            y = torch.from_numpy(np.asarray(t, np.float32)).to(self.device)
            loss = F.binary_cross_entropy_with_logits(self.head(self.net(x))[:, 0], y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += float(loss.item())
            if step % 1000 == 0 or step == steps:
                log(f"    panns_ft: {step * batch_size} windows, loss {running / (step % 1000 or 1000):.4f}")
                running = 0.0
        return self

    def score(self, audio: np.ndarray, batch_size: int = 256) -> np.ndarray:
        import torch
        audio, out = np.atleast_2d(audio), []
        with torch.no_grad():
            for i in range(0, len(audio), batch_size):
                x = torch.from_numpy(level(audio[i:i + batch_size])).to(self.device)
                out.append(torch.sigmoid(self.head(self.net(x)))[:, 0].float().cpu().numpy())
        return np.concatenate(out) if out else np.empty(0, np.float32)

    def save(self, path) -> None:
        import torch
        torch.save(dict(net=self.net.state_dict(), head=self.head.state_dict()), path)

    @classmethod
    def load(cls, path, cfg: FrontendConfig = DEFAULT, device: str = "cpu") -> "PannsFtScorer":
        import torch
        m = cls(cfg, device)
        m._build(pretrained=False)
        ck = torch.load(path, map_location=device)
        m.net.load_state_dict(ck["net"])
        m.head.load_state_dict(ck["head"])
        return m
