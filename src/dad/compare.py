"""Same detection task, different models, judged on identical windows.

This is the comparison ladder the council settled on, with the
microcontroller requirement dropped:

  1. comb          the no-ML harmonic detector from gate.py. The floor.
  2. lgbm          MFCC summary statistics into LightGBM. Classical ML.
  3. dscnn ...     spectrogram networks from models/zoo.py, bigger as you go.
  4. ensemble      the mean probability of every ML model in the run. Only
                   affordable because compute no longer matters.

Each rung must beat the one below it on held-out recording sessions, or it is
not earning its place. Every model sees exactly the same windows, cut from the
same clips, split the same way by session into train, validation and test.
Validation picks the epoch; test is touched once, for the reported number.

The headline metric is the false-alarm rate at 95% drone recall, not
accuracy. A detector is operated at a fixed miss rate, and what you then pay
is false alarms. AUC is reported alongside as the threshold-free summary.

Run it at several window lengths. The one way a better model can reduce
latency is by reaching the same false-alarm rate with a shorter window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from sklearn.metrics import roc_auc_score

from .config import FrontendConfig
from .features import LogMel, summary_stats
from .gate import HarmonicCombGate, decision_score
from .models.baseline import MfccBaseline
from .synth import Clip, make_dataset, session_split

CLIP_SECONDS = 2.0  # what synth.make_session generates
ML_TORCH = ("dscnn", "resnet", "crnn", "transformer")
ALL_MODELS = ("comb", "lgbm", *ML_TORCH, "ensemble")


@dataclass
class Windows:
    audio: np.ndarray    # (N, samples)
    logmel: np.ndarray   # (N, n_mels, frames)
    y: np.ndarray        # (N,) 1 = drone
    kind: np.ndarray     # (N,) source kind, for per-confuser false alarms


def make_windows(clips: list[Clip], cfg: FrontendConfig, crops: int = 1) -> Windows:
    """Cut `crops` evenly spaced windows from each clip.

    Several crops per training clip is cheap augmentation. Evaluation uses one
    centred crop, so each test clip counts once.
    """
    fe = LogMel(cfg)
    n = cfg.samples_per_window
    audio, ys, kinds = [], [], []
    for c in clips:
        if len(c.audio) < n:
            raise ValueError(f"window of {n} samples is longer than the {len(c.audio)}-sample clip")
        span = len(c.audio) - n
        starts = np.unique(np.linspace(0, span, crops).astype(int)) if crops > 1 else [span // 2]
        for s in starts:
            audio.append(c.audio[s:s + n])
            ys.append(c.label)
            kinds.append(c.kind)
    A = np.stack(audio).astype(np.float32)
    L = np.stack([fe.frames(a) for a in A])
    return Windows(A, L, np.asarray(ys), np.asarray(kinds))


class Detector(Protocol):
    name: str
    n_params: int | None

    def fit(self, train: Windows, val: Windows) -> "Detector": ...
    def predict(self, w: Windows) -> np.ndarray: ...
    def predict_one(self, audio: np.ndarray, logmel: np.ndarray) -> float: ...


class CombDetector:
    name = "comb (no ML)"
    n_params = 0

    def __init__(self, cfg: FrontendConfig) -> None:
        self.gate = HarmonicCombGate(cfg)

    def fit(self, train: Windows, val: Windows) -> "CombDetector":
        return self  # nothing to learn; the metrics below are threshold-free

    def predict(self, w: Windows) -> np.ndarray:
        return np.array([decision_score(self.gate(a)) for a in w.audio])

    def predict_one(self, audio: np.ndarray, logmel: np.ndarray) -> float:
        return decision_score(self.gate(audio))


class LgbmDetector:
    n_params = None

    def __init__(self) -> None:
        self.model = MfccBaseline()
        self.name = "mfcc + lightgbm"

    @staticmethod
    def _stats(logmel: np.ndarray) -> np.ndarray:
        return np.stack([summary_stats(m) for m in logmel])

    def fit(self, train: Windows, val: Windows) -> "LgbmDetector":
        self.model.fit(self._stats(train.logmel), train.y)
        return self

    def predict(self, w: Windows) -> np.ndarray:
        return self.model.predict_proba(self._stats(w.logmel))

    def predict_one(self, audio: np.ndarray, logmel: np.ndarray) -> float:
        return self.model.predict_one(summary_stats(logmel))


class NetDetector:
    def __init__(self, arch: str, n_mels: int, train_cfg) -> None:
        from .train import TorchDetector

        self.net = TorchDetector(arch, n_mels, train_cfg)
        self.name = arch

    @property
    def n_params(self) -> int:
        return self.net.n_params

    def fit(self, train: Windows, val: Windows) -> "NetDetector":
        self.net.fit(train.logmel, train.y, val.logmel, val.y)
        return self

    def predict(self, w: Windows) -> np.ndarray:
        return self.net.predict_proba(w.logmel)

    def predict_one(self, audio: np.ndarray, logmel: np.ndarray) -> float:
        return self.net.predict_one(logmel)


class EnsembleDetector:
    """Mean probability of already-trained ML members. Costs their sum."""

    name = "ensemble (mean of ML)"

    def __init__(self, members: list[Detector]) -> None:
        self.members = members

    @property
    def n_params(self) -> int | None:
        counts = [m.n_params for m in self.members if m.n_params]
        return sum(counts) if counts else None

    def fit(self, train: Windows, val: Windows) -> "EnsembleDetector":
        return self

    def predict(self, w: Windows) -> np.ndarray:
        return np.mean([m.predict(w) for m in self.members], axis=0)

    def predict_one(self, audio: np.ndarray, logmel: np.ndarray) -> float:
        return float(np.mean([m.predict_one(audio, logmel) for m in self.members]))


def false_alarms_at_recall(y: np.ndarray, scores: np.ndarray, kind: np.ndarray,
                           recall: float = 0.95) -> dict[str, float]:
    """Set the threshold that keeps `recall` of drones, then count false alarms.

    Returns the overall false-alarm rate and the rate per confuser kind. If too
    many drones score exactly zero (the comb on out-of-band input), the
    threshold drops to zero and every negative passes, which is the honest
    answer: that recall is unreachable without forwarding everything.
    """
    thr = float(np.quantile(scores[y == 1], 1.0 - recall))
    neg = y == 0
    out = {"all": float((scores[neg] >= thr).mean())}
    for k in np.unique(kind[neg]):
        sel = neg & (kind == k)
        out[str(k)] = float((scores[sel] >= thr).mean())
    return out


def _p50_ms(fn: Callable[[], object], n: int = 40, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return 1000.0 * float(np.median(ts))


@dataclass
class Row:
    window_s: float
    model: str
    n_params: int | None
    auc: float
    fa_at_95: float
    fa_by_kind: dict[str, float] = field(default_factory=dict)
    train_s: float = 0.0
    infer_p50_ms: float = 0.0


def run_comparison(
    windows: list[float],
    models: list[str],
    epochs: int = 15,
    device: str = "auto",
    n_sessions: int = 10,
    clips_per_session: int = 40,
    train_crops: int = 3,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> list[Row]:
    unknown = set(models) - set(ALL_MODELS)
    if unknown:
        raise ValueError(f"unknown models {sorted(unknown)}; choose from {list(ALL_MODELS)}")
    if any(w > CLIP_SECONDS for w in windows):
        raise ValueError(f"windows must be at most {CLIP_SECONDS} s, the synthetic clip length")
    if n_sessions < 5:
        raise ValueError("need at least 5 sessions: 2 test, 2 validation, 1+ train")

    train_cfg = None
    if any(m in ML_TORCH for m in models):
        from .train import TrainConfig

        train_cfg = TrainConfig(epochs=epochs, device=device, seed=seed)

    # Same clips and same session split at every window length, so rows across
    # windows differ only in the window.
    clips = make_dataset(n_sessions=n_sessions, clips_per_session=clips_per_session, seed=seed)
    rest, test_clips = session_split(clips, test_sessions=2, seed=seed)
    train_clips, val_clips = session_split(rest, test_sessions=2, seed=seed + 1)

    rows: list[Row] = []
    for window in windows:
        cfg = FrontendConfig(window_seconds=window)
        train = make_windows(train_clips, cfg, crops=train_crops)
        val = make_windows(val_clips, cfg)
        test = make_windows(test_clips, cfg)
        log(f"window {window:g} s: {len(train.y)} train / {len(val.y)} val / {len(test.y)} test windows")

        trained_ml: list[Detector] = []
        for name in models:
            if name == "ensemble":
                continue
            if name == "comb":
                det: Detector = CombDetector(cfg)
            elif name == "lgbm":
                det = LgbmDetector()
            else:
                det = NetDetector(name, cfg.n_mels, train_cfg)

            t0 = time.perf_counter()
            det.fit(train, val)
            train_s = time.perf_counter() - t0
            rows.append(_evaluate(det, test, window, train_s))
            log(f"  {rows[-1].model:<24} AUC {rows[-1].auc:.3f}  "
                f"false alarms @95% {rows[-1].fa_at_95:.3f}  ({train_s:.1f} s)")
            if name != "comb":
                trained_ml.append(det)

        if "ensemble" in models and len(trained_ml) >= 2:
            rows.append(_evaluate(EnsembleDetector(trained_ml), test, window, 0.0))
            log(f"  {rows[-1].model:<24} AUC {rows[-1].auc:.3f}  false alarms @95% {rows[-1].fa_at_95:.3f}")

    return rows


def _evaluate(det: Detector, test: Windows, window: float, train_s: float) -> Row:
    scores = det.predict(test)
    fa = false_alarms_at_recall(test.y, scores, test.kind)
    a, m = test.audio[0], test.logmel[0]
    return Row(
        window_s=window,
        model=det.name,
        n_params=det.n_params,
        auc=float(roc_auc_score(test.y, scores)),
        fa_at_95=fa.pop("all"),
        fa_by_kind=fa,
        train_s=train_s,
        infer_p50_ms=_p50_ms(lambda: det.predict_one(a, m)),
    )


def format_rows(rows: list[Row], device_label: str) -> str:
    bar, dash = "=" * 96, "-" * 96
    kinds = sorted({k for r in rows for k in r.fa_by_kind})
    head = (f"{'model':<24}{'params':>10}{'AUC':>7}{'FA@95':>8}"
            + "".join(f"{k[:7]:>9}" for k in kinds)
            + f"{'train s':>9}{'infer ms':>10}")
    lines = [bar, "MODEL COMPARISON  (synthetic data: the ranking is NOT a result)",
             f"inference device: {device_label}. FA@95 = false-alarm rate at 95% drone recall, lower is better.",
             "Per-kind columns are the false-alarm rate on each confuser at that same threshold.", bar]
    for window in sorted({r.window_s for r in rows}):
        lines += [f"window {window:g} s", head, dash]
        for r in (r for r in rows if r.window_s == window):
            params = "-" if r.n_params is None else ("0" if r.n_params == 0 else f"{r.n_params / 1e3:.0f}k")
            lines.append(
                f"{r.model:<24}{params:>10}{r.auc:>7.3f}{r.fa_at_95:>8.3f}"
                + "".join(f"{r.fa_by_kind.get(k, float('nan')):>9.3f}" for k in kinds)
                + f"{r.train_s:>9.1f}{r.infer_p50_ms:>10.2f}"
            )
        lines.append(dash)

    windows = sorted({r.window_s for r in rows})
    if len(windows) > 1:
        lines += ["", "FALSE ALARMS @95% RECALL vs WINDOW LENGTH",
                  "A model that matches a longer window's false-alarm rate at a shorter window buys latency.",
                  f"{'model':<24}" + "".join(f"{w:>9g}s" for w in windows), dash]
        for model in dict.fromkeys(r.model for r in rows):
            cells = {r.window_s: r.fa_at_95 for r in rows if r.model == model}
            lines.append(f"{model:<24}" + "".join(
                f"{cells[w]:>10.3f}" if w in cells else f"{'-':>10}" for w in windows))
    lines.append(bar)
    return "\n".join(lines)
