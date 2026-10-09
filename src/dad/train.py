"""One training loop for every spectrogram model in models/zoo.py.

The DS-CNN shipped without one, so it could only ever be timed, never judged.
This loop is deliberately plain: AdamW, a one-cycle learning-rate schedule,
SpecAugment masking, and the epoch with the best validation AUC is kept
(ties broken by validation log-loss).
Nothing here is tuned per architecture, because the comparison is only fair
if every model gets the same recipe.

Normalisation statistics come from the training set only. Computing them over
train and test together is a small leak, but it is still a leak.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score

from .models.zoo import build, n_params


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 32
    lr: float = 2e-3
    weight_decay: float = 1e-2
    spec_augment: bool = True
    device: str = "auto"
    seed: int = 0


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def spec_augment(x: torch.Tensor, rng: np.random.Generator,
                 max_freq: int = 8, max_time_frac: float = 0.125) -> torch.Tensor:
    """One frequency mask and one time mask per example (Park et al., 2019).

    Masked cells are set to zero, which after normalisation is the band mean.
    The masks stop a model from leaning on one harmonic or one instant, which
    is exactly the brittleness that fails on real drones whose harmonics sit a
    little off where the training set put them.
    """
    x = x.clone()
    b, m, t = x.shape
    max_time = max(1, int(t * max_time_frac))
    for i in range(b):
        fw = int(rng.integers(0, max_freq + 1))
        f0 = int(rng.integers(0, max(1, m - fw)))
        x[i, f0:f0 + fw, :] = 0.0
        tw = int(rng.integers(0, max_time + 1))
        t0 = int(rng.integers(0, max(1, t - tw)))
        x[i, :, t0:t0 + tw] = 0.0
    return x


class TorchDetector:
    """Wraps a zoo architecture with fit / predict_proba / predict_one."""

    def __init__(self, arch: str, n_mels: int, cfg: TrainConfig | None = None) -> None:
        self.arch = arch
        self.n_mels = n_mels
        self.cfg = cfg or TrainConfig()
        self.device = resolve_device(self.cfg.device)
        self.model: torch.nn.Module | None = None
        self.val_auc_history: list[float] = []
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None

    @property
    def n_params(self) -> int:
        return n_params(self.model) if self.model is not None else n_params(build(self.arch, self.n_mels))

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return ((X - self._mu) / self._sigma).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> "TorchDetector":
        """X is log-mel windows, shape (N, n_mels, n_frames)."""
        c = self.cfg
        torch.manual_seed(c.seed)
        rng = np.random.default_rng(c.seed)

        # Per-mel-band statistics, from the training set only.
        self._mu = X.mean(axis=(0, 2))[:, None]
        self._sigma = X.std(axis=(0, 2))[:, None] + 1e-5

        Xn = torch.from_numpy(self._normalise(X))
        yt = torch.from_numpy(np.asarray(y, dtype=np.int64))

        self.model = build(self.arch, self.n_mels).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
        steps_per_epoch = math.ceil(len(Xn) / c.batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c.lr, total_steps=c.epochs * steps_per_epoch)

        can_validate = X_val is not None and y_val is not None and len(np.unique(y_val)) == 2
        # Ranked by AUC, ties broken by log-loss. AUC saturates at 1.0 on easy
        # data within an epoch or two, and keeping the first epoch to reach it
        # keeps an undertrained, unconfident model.
        best_key, best_state = (-math.inf, -math.inf), None

        for _ in range(c.epochs):
            self.model.train()
            perm = torch.from_numpy(rng.permutation(len(Xn)))
            for i in range(0, len(Xn), c.batch_size):
                idx = perm[i:i + c.batch_size]
                xb = Xn[idx]
                if c.spec_augment:
                    xb = spec_augment(xb, rng)
                logits = self.model(xb[:, None].to(self.device))
                loss = F.cross_entropy(logits, yt[idx].to(self.device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()

            self._recalibrate_batchnorm(Xn)
            if can_validate:
                p = self.predict_proba(X_val)
                auc = float(roc_auc_score(y_val, p))
                key = (auc, -float(log_loss(y_val, p, labels=[0, 1])))
                self.val_auc_history.append(auc)
                if key > best_key:
                    best_key, best_state = key, copy.deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def _recalibrate_batchnorm(self, Xn: torch.Tensor, batch_size: int = 256) -> None:
        """Recompute BatchNorm running statistics as an exact average over the
        clean training set.

        Training mode normalises with each batch's own statistics, but
        inference uses a slow exponential average of them. With few optimiser
        steps that average lags far behind the trained weights, and the model
        outputs near-constant probabilities even while it ranks perfectly. One
        extra forward pass with a cumulative average removes the lag.
        """
        bns = [m for m in self.model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        if not bns:
            return
        saved = [bn.momentum for bn in bns]
        for bn in bns:
            bn.reset_running_stats()
            bn.momentum = None  # cumulative moving average
        self.model.train()
        with torch.no_grad():
            for i in range(0, len(Xn), batch_size):
                self.model(Xn[i:i + batch_size, None].to(self.device))
        for bn, mom in zip(bns, saved):
            bn.momentum = mom
        self.model.eval()

    def predict_proba(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        self.model.eval()
        Xn = torch.from_numpy(self._normalise(X))
        out = []
        with torch.inference_mode():
            for i in range(0, len(Xn), batch_size):
                logits = self.model(Xn[i:i + batch_size, None].to(self.device))
                out.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    def predict_one(self, logmel: np.ndarray) -> float:
        """Single-window path, the one a live detector runs every hop."""
        x = torch.from_numpy(self._normalise(logmel[None]))[:, None].to(self.device)
        with torch.inference_mode():
            return float(torch.softmax(self.model(x), dim=1)[0, 1].item())
