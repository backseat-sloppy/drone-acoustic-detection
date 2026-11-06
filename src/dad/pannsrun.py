"""Scorers on the PANNs Cnn14_16k encoder (models/panns.py): P1 linear probe, P2 LP-FT.

Same interface as realrun.NetScorer (score / save / load), so score_glasgow.py, score_real_dev.py
and the GUIs take these checkpoints unchanged: NetScorer.load hands any checkpoint with
arch "panns_cnn14_16k" to PannsScorer.load.

Training batches come from train_clean.CleanSampler.raw_batch: audio, speed, EQ, label. The speed
change runs batched on the device (gpufront.GpuFrontend.speed, the numbers of resample_poly), the EQ
curve (natural-log units over 64 bands) is converted to dB and added to Cnn14's own 64 bands, and
the rest is Cnn14's pretraining frontend.

P1 (probe): the encoder frozen in eval mode, 2048-d embeddings of n_windows sampled training
windows, standardised, logistic regression with C picked by validation AUC. The scaler is folded
into the 2-logit head. The checkpoint stores only the head plus the encoder checkpoint's MD5.

P2 (LP-FT, Kumar et al., ICLR 2022): start from a P1 head, train everything (encoder frontend stays
frozen) with AdamW and a one-cycle schedule, SpecAugment after bn0 as in the reference code, and
keep the state with the best validation AUC (ties: log-loss), checked every eval_every windows, as
dad.train keeps the best validation epoch. The checkpoint stores the whole network (about 320 MB).
"""
from __future__ import annotations

import copy
import hashlib
import math
import time
import types
from pathlib import Path
from typing import Callable

import numpy as np

from .config import DEFAULT, FrontendConfig
from .models import panns as pn

ARCH = "panns_cnn14_16k"
_DB_PER_NEPER_POWER = 10.0 / math.log(10.0)     # natural log of power -> dB


def md5(path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


class PannsScorer:
    def __init__(self, pretrained: Path | None, device: str = "cpu", cfg: FrontendConfig = DEFAULT) -> None:
        """pretrained: the Cnn14_16k checkpoint. None builds an empty Cnn14 for a full (ft) state to fill."""
        import torch
        self.cfg, self.device = cfg, torch.device(device)
        if pretrained is None:
            self.pretrained, self.pretrained_md5, enc = Path(pn.CHECKPOINT_NAME), None, pn.Cnn14()
        else:
            self.pretrained = Path(pretrained)
            if not self.pretrained.exists():
                raise FileNotFoundError(f"encoder checkpoint {self.pretrained} not found; fetch it with "
                                        f"it11_1/run.sh fetch ({pn.CHECKPOINT_URL})")
            self.pretrained_md5, enc = md5(self.pretrained), pn.load_pretrained(self.pretrained)
        self.model = pn.PannsDetector(enc).to(self.device).eval()
        self.kind = "untrained"
        self.name = f"{ARCH} [{self.kind}]"
        self.history: list[dict] = []
        self.info: dict = {}
        self._gpufe = None

    # ------------------------------------------------------------ batches
    def _speed(self):
        if self._gpufe is None:
            from .gpufront import GpuFrontend
            self._gpufe = GpuFrontend(self.cfg, self.device)
        return self._gpufe

    def _inputs(self, b):
        """raw_batch (audio, down, eq, y) -> wave (N, 16000), eq_db (N, 64), y on the device."""
        import torch
        audio, down, eq, y = b
        wave = self._speed().speed(torch.from_numpy(np.asarray(audio)).to(self.device), down,
                                   self.cfg.samples_per_window)
        eq_db = torch.from_numpy(np.asarray(eq, np.float32) * _DB_PER_NEPER_POWER).to(self.device)
        return wave, eq_db, torch.from_numpy(np.asarray(y)).to(self.device)

    def embed(self, wave, eq_db=None):
        import torch
        self.model.eval()
        with torch.no_grad():
            return self.model.encoder(wave, eq_db)[0]

    # ------------------------------------------------------------ inference
    def score(self, audio: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """P(drone) per 1 s window, (N, 16000) float audio at 16 kHz."""
        import torch
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(audio), batch_size):
                w = torch.from_numpy(np.asarray(audio[i:i + batch_size], np.float32)).to(self.device)
                out.append(torch.softmax(self.model(w), dim=1)[:, 1].cpu().numpy())
        return np.concatenate(out) if out else np.empty(0)

    def audioset(self, audio: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """The 527 AudioSet clipwise probabilities per window (P0: no training)."""
        import torch
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(audio), batch_size):
                w = torch.from_numpy(np.asarray(audio[i:i + batch_size], np.float32)).to(self.device)
                out.append(torch.sigmoid(self.model.encoder(w)[1]).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, pn.N_CLASSES))

    # ------------------------------------------------------------ P1
    def fit_probe(self, sampler, n_windows: int, pos_val: np.ndarray, neg_val: np.ndarray,
                  batch_size: int = 64, workers: int = 3, Cs=(1e-4, 1e-3, 1e-2, 1e-1, 1.0),
                  log: Callable[[str], None] = print) -> "PannsScorer":
        import torch
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from .realrun import prefetch
        steps = n_windows // batch_size
        source = types.SimpleNamespace(batch=sampler.raw_batch)
        E, Y, t0 = [], [], time.time()
        for step, b in enumerate(prefetch(source, steps, batch_size, workers), start=1):
            wave, eq_db, y = self._inputs(b)
            E.append(self.embed(wave, eq_db).cpu().numpy().astype(np.float32)); Y.append(b[-1])
            if step % 200 == 0 or step == steps:
                log(f"    probe: embedded {step * batch_size} windows, {step * batch_size / (time.time() - t0):.0f}/s")
        E, Y = np.concatenate(E), np.concatenate(Y)
        ev = np.concatenate([self._embed_audio(pos_val), self._embed_audio(neg_val)])
        yv = np.r_[np.ones(len(pos_val)), np.zeros(len(neg_val))]
        mu, sd = E.mean(0), E.std(0) + 1e-6
        best = None
        for C in Cs:
            clf = LogisticRegression(C=C, max_iter=2000).fit((E - mu) / sd, Y)
            auc = float(roc_auc_score(yv, clf.decision_function((ev - mu) / sd)))
            log(f"    probe: C={C:g} validation AUC {auc:.4f}")
            if best is None or auc > best[0]:
                best = (auc, C, clf)
        auc, C, clf = best
        w, b0 = clf.coef_[0] / sd, float(clf.intercept_[0] - np.sum(clf.coef_[0] * mu / sd))
        with torch.no_grad():
            self.model.head.weight.zero_(); self.model.head.bias.zero_()
            self.model.head.weight[1] = torch.from_numpy(w.astype(np.float32))
            self.model.head.bias[1] = b0
        self.kind, self.name = "probe", f"{ARCH} [probe]"
        if C in (min(Cs), max(Cs)):
            log(f"    probe: WARNING C={C:g} is at the edge of the grid {list(Cs)}")
        self.info.update(probe=dict(C=C, val_auc=auc, windows=int(len(E)), Cs=list(Cs), C_at_grid_edge=C in (min(Cs), max(Cs)),
                                    seconds=round(time.time() - t0, 1)))
        return self

    def _embed_audio(self, audio: np.ndarray, batch_size: int = 256) -> np.ndarray:
        import torch
        out = []
        for i in range(0, len(audio), batch_size):
            w = torch.from_numpy(np.asarray(audio[i:i + batch_size], np.float32)).to(self.device)
            out.append(self.embed(w).cpu().numpy())
        return np.concatenate(out)

    # ------------------------------------------------------------ P2
    def fit_finetune(self, sampler, n_windows: int, pos_val: np.ndarray, neg_val: np.ndarray,
                     batch_size: int = 64, lr: float = 1e-4, weight_decay: float = 1e-2,
                     eval_every: int = 100_000, workers: int = 3,
                     log: Callable[[str], None] = print) -> "PannsScorer":
        import torch
        import torch.nn.functional as F
        from sklearn.metrics import log_loss, roc_auc_score
        from .realrun import prefetch
        from .train import spec_augment
        if self.kind != "probe":
            raise ValueError("LP-FT starts from a probe head: run fit_probe or load a probe checkpoint first")
        torch.manual_seed(sampler.seed)
        rng = np.random.default_rng(sampler.seed)
        mask = lambda x: spec_augment(x, rng)                                    # noqa: E731
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        steps = n_windows // batch_size
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
        yv = np.r_[np.ones(len(pos_val)), np.zeros(len(neg_val))]

        def validate() -> tuple[float, float]:
            p = np.clip(np.r_[self.score(pos_val), self.score(neg_val)], 1e-7, 1 - 1e-7)
            return float(roc_auc_score(yv, p)), -float(log_loss(yv, p, labels=[0, 1]))

        best_key = validate()
        best_state, best_at = copy.deepcopy(self.model.state_dict()), 0
        log(f"    ft: start (probe) validation AUC {best_key[0]:.4f}")
        source = types.SimpleNamespace(batch=sampler.raw_batch)
        self.model.train()
        t0, running, done, mark = time.time(), 0.0, 0, 0
        for step, b in enumerate(prefetch(source, steps, batch_size, workers), start=1):
            wave, eq_db, y = self._inputs(b)
            loss = F.cross_entropy(self.model(wave, eq_db, mask), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            running += float(loss.item()); done += 1
            seen = step * batch_size
            if seen // eval_every > mark or step == steps:
                mark = seen // eval_every
                key = validate()
                self.history.append(dict(windows=seen, loss=running / done, val_auc=key[0], val_neg_logloss=key[1],
                                         seconds=round(time.time() - t0, 1)))
                log(f"    ft: {seen:>7} windows, loss {running / done:.4f}, validation AUC {key[0]:.4f}, "
                    f"{seen / (time.time() - t0):.0f} windows/s")
                running, done = 0.0, 0
                if key > best_key:
                    best_key, best_state, best_at = key, copy.deepcopy(self.model.state_dict()), seen
                self.model.train()
        self.model.load_state_dict(best_state)
        self.model.eval()
        self.kind, self.name = "ft", f"{ARCH} [ft]"
        self.info.update(ft=dict(windows=n_windows, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                                 eval_every=eval_every, best_at_windows=best_at, best_val_auc=best_key[0],
                                 seconds=round(time.time() - t0, 1)))
        return self

    # ------------------------------------------------------------ persistence
    def save(self, path) -> None:
        import torch
        state = (self.model.state_dict() if self.kind == "ft"
                 else {k: v for k, v in self.model.state_dict().items() if k.startswith("head.")})
        torch.save(dict(arch=ARCH, kind=self.kind, pretrained=self.pretrained.name,
                        pretrained_md5=self.pretrained_md5, info=self.info, history=self.history,
                        state={k: v.cpu() for k, v in state.items()}), path)

    @classmethod
    def load(cls, path, cfg: FrontendConfig = DEFAULT, device: str = "cpu", pretrained: Path | None = None):
        """An ft checkpoint holds the whole network and loads on its own. A probe checkpoint holds only the
        head: it needs the encoder checkpoint (artifacts/pretrained/ by default), whose MD5 must match the
        one stored at training time."""
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("arch") != ARCH:
            raise ValueError(f"{path} is not a {ARCH} checkpoint")
        if ck["kind"] == "ft":
            sc = cls(None, device, cfg)
            sc.pretrained, sc.pretrained_md5 = Path(ck["pretrained"]), ck["pretrained_md5"]
        else:
            if pretrained is None:
                pretrained = Path(__file__).resolve().parents[2] / "artifacts" / "pretrained" / ck["pretrained"]
            sc = cls(pretrained, device, cfg)
            if sc.pretrained_md5 != ck["pretrained_md5"]:
                raise ValueError(f"encoder checkpoint {pretrained} has MD5 {sc.pretrained_md5}, "
                                 f"the scorer was trained on {ck['pretrained_md5']}")
        sc.model.load_state_dict(ck["state"], strict=ck["kind"] == "ft")
        sc.model.to(sc.device).eval()
        sc.kind, sc.info, sc.history = ck["kind"], ck.get("info", {}), ck.get("history", [])
        sc.name = f"{ARCH} [{sc.kind}]"
        return sc
