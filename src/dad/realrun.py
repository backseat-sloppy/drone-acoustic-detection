"""Training and evaluation on the real recordings.

The design follows the council decision in docs/REPORT.md, entry 010. Every
training positive comes from one recording chain (DDL) and every negative
from others (FSD50K), so a model can score well by recognising the chain.
Three things here exist only to make that harder and to make it visible:

* Level is removed inside the band the mel bank sees (`level_normalise`).
* Training windows can be mixed with real backgrounds and randomly
  equalised (`Variant.mix`, `Variant.eq`).
* Results are reported on DroneNoise, a different chain with different
  drones, next to the confounded validation contrast (`evaluate`).

DroneNoise is read on every run, so it is development data, not a test. The
sealed test sources (Halmstad, FSD50K-eval) are only touched by
`sealed_sets`, which the caller must ask for explicitly.
"""
from __future__ import annotations

import collections
import csv
import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import DEFAULT, FrontendConfig
from .features import LogMel, summary_stats
from .realdata import PROCESSED, SAMPLE_RATE, Recording, window_starts

# Held-out DDL sessions: both days, both drone models, near and far.
DDL_VAL_SESSIONS = frozenset({
    "290321-T005",                                              # Mini 2, day 1, 48-93 m
    "310321-T004", "310321-T009", "310321-T013", "310321-T015",  # Phantom 4 Pro, 12-199 m
    "310321-T018", "310321-T023", "310321-T026", "310321-T032",  # Mini 2, day 2, 23-150 m
})
PROBE_MICROPHONE = "M5"
THRESHOLD_MAX_RANGE_M = 100   # the operating point is set on drones close enough to hear
PROBE_MAX_DROP_DB = 20.0      # flyover windows this far below the pass-by peak are dropped
SILENCE_POWER = 1e-10         # in-band power below this is digital silence

_HIGHPASS = butter(2, 50.0, "highpass", fs=SAMPLE_RATE, output="sos")
_LN_PER_DB = math.log(10.0) / 10.0
_FLOOR = math.log(1e-5)       # 50 dB below the window's own mean in-band power


# ------------------------------------------------------------------ splits

@dataclass
class Splits:
    ddl_train: list[Recording]
    ddl_val: list[Recording]
    fsd_train: list[Recording]
    fsd_val: list[Recording]
    probe: list[Recording]
    dronenoise: list[Recording]  # every microphone; only for runs that train on DroneNoise
    halmstad: list[Recording]   # sealed
    fsd_eval: list[Recording]   # sealed


def make_splits(manifest: list[Recording], cfg: FrontendConfig = DEFAULT) -> Splits:
    """Assign every recording a role. Groups never straddle two roles."""
    n = cfg.samples_per_window
    long_enough = [r for r in manifest if r.n_samples >= n]  # short clips are dropped, not padded
    ddl = [r for r in long_enough if r.source == "ddl"]
    fsd = [r for r in long_enough if r.source == "fsd50k"]
    train_uploaders = {r.group for r in fsd if r.subset == "train"}
    return Splits(
        ddl_train=[r for r in ddl if r.recording_id not in DDL_VAL_SESSIONS],
        ddl_val=[r for r in ddl if r.recording_id in DDL_VAL_SESSIONS],
        fsd_train=[r for r in fsd if r.subset == "train"],
        # FSD50K's own validation clips, minus every uploader who also appears in training.
        fsd_val=[r for r in fsd if r.subset == "val" and r.group not in train_uploaders],
        probe=[r for r in long_enough if r.source == "dronenoise" and r.subset == PROBE_MICROPHONE],
        dronenoise=[r for r in long_enough if r.source == "dronenoise"],
        halmstad=[r for r in long_enough if r.source == "halmstad"],
        fsd_eval=[r for r in fsd if r.subset == "eval"],
    )


# ---------------------------------------------------------------- features

def inband_power(x: np.ndarray) -> float:
    """Mean power above 50 Hz. Wind below the mel bank's lower edge can sit
    15 dB above everything the model sees, so plain RMS is the wrong ruler."""
    y = sosfilt(_HIGHPASS, np.asarray(x, dtype=np.float64))
    return float(np.mean(y * y))


def level_db(logmel: np.ndarray) -> np.ndarray:
    """In-band level of each window, dB. logmel is (..., n_mels, frames)."""
    return 10.0 * np.log10(np.mean(np.exp(logmel.astype(np.float64)), axis=(-2, -1)))


def level_normalise(logmel: np.ndarray) -> np.ndarray:
    """Remove each window's own in-band level, then cap the dynamic range.

    Subtracting the log of the mean mel power is the same as scaling the
    waveform to a fixed in-band level. The floor stops digital silence, which
    only the int16 sources contain, from showing up as a cue of its own.
    """
    ref = np.log(np.mean(np.exp(logmel.astype(np.float64)), axis=(-2, -1), keepdims=True))
    return np.maximum(logmel - ref, _FLOOR).astype(np.float32)


@dataclass(frozen=True)
class Variant:
    """What is done to a window between the store and the model."""
    name: str
    level_norm: bool = True
    mix: bool = False            # sum with a real background (training only)
    eq: bool = False             # random spectral tilt and shape (training only)
    mix_prob_pos: float = 0.75
    snr_db: tuple[float, float] = (5.0, 20.0)       # drone over background, in-band
    mix_prob_neg: float = 0.5
    neg_rel_db: tuple[float, float] = (-10.0, 10.0)


VARIANTS = {
    "raw": Variant("raw", level_norm=False),
    "level": Variant("level"),
    "full": Variant("full", mix=True, eq=True),
    # Every drone window is buried in a background, down to 5 dB below it.
    "buried": Variant("buried", mix=True, eq=True, mix_prob_pos=1.0, snr_db=(-5.0, 15.0)),
}


def featurise(audio: np.ndarray, variant: Variant, fe: LogMel) -> np.ndarray:
    """(N, samples) -> (N, n_mels, frames), as the model sees it at inference."""
    L = np.stack([fe.frames(a) for a in audio])
    return level_normalise(L) if variant.level_norm else L


def _eq_curve(rng: np.random.Generator, n_mels: int) -> np.ndarray:
    """A random smooth gain across the mel bands, in natural-log power."""
    tilt = rng.uniform(-6.0, 6.0) * np.linspace(-0.5, 0.5, n_mels)
    shape = np.interp(np.arange(n_mels), np.linspace(0, n_mels - 1, 5), rng.normal(0.0, 2.0, 5))
    return ((tilt + shape) * _LN_PER_DB).astype(np.float32)


# ----------------------------------------------------------------- sampling

POOL_GRID = 1_600  # candidate window starts every 100 ms


class Pool:
    """Drone windows a sampler may draw: recordings, and where in each a window may start."""

    def __init__(self, name: str, recs: list[Recording], starts: list[np.ndarray], n: int) -> None:
        keep = [i for i, s in enumerate(starts) if len(s)]
        self.name, self.n = name, n
        self.recs = [recs[i] for i in keep]
        self.starts = [starts[i] for i in keep]
        self._cdf = np.cumsum([len(s) for s in self.starts], dtype=np.float64)
        if not self.recs:
            raise ValueError(f"pool {name!r} is empty")

    @property
    def seconds(self) -> float:
        """Roughly how much distinct audio the pool covers."""
        return float(self._cdf[-1]) * POOL_GRID / SAMPLE_RATE

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        i = int(np.searchsorted(self._cdf, rng.random() * self._cdf[-1], side="right"))
        rec, s = self.recs[i], self.starts[i]
        start = int(s[rng.integers(0, len(s))]) + int(rng.integers(-POOL_GRID // 2, POOL_GRID // 2))
        return rec.audio(min(max(start, 0), rec.n_samples - self.n), self.n)


def ddl_pool(recs: list[Recording], n: int, max_range_m: float | None = None, name: str = "ddl") -> Pool:
    """DDL windows. With `max_range_m`, only stretches whose range label is
    reliable and at most that far at both ends of the window."""
    ranges = _ddl_ranges() if max_range_m is not None else {}
    starts = []
    for r in recs:
        grid = np.arange(0, r.n_samples - n + 1, POOL_GRID, dtype=np.int64)
        if max_range_m is not None:
            at, rng_m = ranges[r.recording_id]
            first = rng_m[np.maximum(np.searchsorted(at, grid, side="right") - 1, 0)]
            last = rng_m[np.maximum(np.searchsorted(at, grid + n - 1, side="right") - 1, 0)]
            grid = grid[(first <= max_range_m) & (last <= max_range_m)]  # NaN fails both
        starts.append(grid)
    return Pool(name, recs, starts, n)


def dronenoise_pool(recs: list[Recording], n: int, max_drop_db: float = 10.0,
                    name: str = "dronenoise") -> Pool:
    """DroneNoise windows within `max_drop_db` of the loudest window of the same file."""
    starts = []
    for r in recs:
        y = sosfilt(_HIGHPASS, r.audio().astype(np.float64))
        total = np.concatenate([[0.0], np.cumsum(y * y)])
        grid = np.arange(0, r.n_samples - n + 1, POOL_GRID, dtype=np.int64)
        power = total[grid + n] - total[grid]
        starts.append(grid[power >= power.max() * 10.0 ** (-max_drop_db / 10.0)])
    return Pool(name, recs, starts, n)


# FSD50K classes that are steady, broadband or engine-like: the sounds a drone
# detector is most likely to mistake for a drone.
STEADY_NOISE = frozenset({
    "Engine", "Engine_starting", "Idling", "Accelerating_and_revving_and_vroom", "Vehicle",
    "Motor_vehicle_(road)", "Car", "Car_passing_by", "Bus", "Truck", "Motorcycle",
    "Race_car_and_auto_racing", "Aircraft", "Fixed-wing_aircraft_and_airplane", "Rail_transport",
    "Train", "Subway_and_metro_and_underground", "Boat_and_Water_vehicle", "Mechanical_fan",
    "Power_tool", "Drill", "Sawing", "Wind", "Rain", "Raindrop", "Thunder", "Thunderstorm",
    "Water", "Stream", "Ocean", "Waves_and_surf", "Fire", "Boiling", "Hiss", "Buzz",
})


class WindowSampler:
    """Draws training windows on the fly: half drone, half not."""

    def __init__(self, pools: list[Pool], negatives: list[Recording], cfg: FrontendConfig,
                 variant: Variant, seed: int = 0, pool_weights: list[float] | None = None,
                 steady_boost: float = 1.0) -> None:
        self.cfg, self.variant, self.seed = cfg, variant, seed
        self.n = cfg.samples_per_window
        self.fe = LogMel(cfg)
        self.pools = pools
        self._pool_cdf = np.cumsum(pool_weights or [1.0] * len(pools))
        self.neg = negatives
        # Backgrounds: one uploader holds 7% of the audio. Weighting each clip by
        # 1/sqrt(its uploader's clip count) keeps any one device from dominating.
        per_uploader = collections.Counter(r.group for r in self.neg)
        self._neg_cdf = np.cumsum([
            (steady_boost if STEADY_NOISE.intersection(r.labels.split(",")) else 1.0)
            / math.sqrt(per_uploader[r.group]) for r in self.neg])

    def _background(self, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        for _ in range(50):
            rec = self.neg[int(np.searchsorted(self._neg_cdf, rng.random() * self._neg_cdf[-1], side="right"))]
            x = rec.audio(int(rng.integers(0, rec.n_samples - self.n + 1)), self.n)
            p = inband_power(x)
            if p > SILENCE_POWER:
                return x, p
        raise RuntimeError("could not draw a non-silent background window")

    def positive(self, rng: np.random.Generator) -> np.ndarray:
        pool = self.pools[int(np.searchsorted(self._pool_cdf, rng.random() * self._pool_cdf[-1], side="right"))]
        d = pool.draw(rng)
        v = self.variant
        if v.mix and rng.random() < v.mix_prob_pos:
            b, pb = self._background(rng)
            snr = rng.uniform(*v.snr_db)
            d = d + b * math.sqrt(inband_power(d) / (pb * 10.0 ** (snr / 10.0)))
        return d

    def negative(self, rng: np.random.Generator) -> np.ndarray:
        a, pa = self._background(rng)
        v = self.variant
        # Half the negatives are sums too, so "being a mixture" says nothing about the label.
        if v.mix and rng.random() < v.mix_prob_neg:
            b, pb = self._background(rng)
            rel = rng.uniform(*v.neg_rel_db)
            a = a + b * math.sqrt(pa * 10.0 ** (rel / 10.0) / pb)
        return a

    def batch(self, index: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Batch `index` is the same every time for a given seed."""
        rng = np.random.default_rng([self.seed, index])
        y = (np.arange(batch_size) % 2).astype(np.int64)
        L = np.stack([self.fe.frames(self.positive(rng) if t else self.negative(rng)) for t in y])
        if self.variant.eq:
            L = L + np.stack([_eq_curve(rng, L.shape[1]) for _ in y])[:, :, None]
        return (level_normalise(L) if self.variant.level_norm else L), y


def prefetch(sampler: WindowSampler, n_batches: int, batch_size: int,
             workers: int = 3) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield batches 0..n_batches-1 in order, built on background threads."""
    todo: queue.Queue = queue.Queue()
    for i in range(n_batches):
        todo.put(i)
    done: queue.Queue = queue.Queue(maxsize=4 * workers)

    def work() -> None:
        while True:
            try:
                i = todo.get_nowait()
            except queue.Empty:
                return
            try:
                done.put((i, sampler.batch(i, batch_size)))
            except BaseException as e:  # surface worker errors in the consumer
                done.put((i, e))
                return

    for _ in range(workers):
        threading.Thread(target=work, daemon=True).start()
    held: dict[int, object] = {}
    for want in range(n_batches):
        while want not in held:
            i, item = done.get()
            held[i] = item
        item = held.pop(want)
        if isinstance(item, BaseException):
            raise item
        yield item


# ------------------------------------------------------------ evaluation sets

@dataclass
class EvalSet:
    name: str
    audio: np.ndarray      # (N, samples)
    y: np.ndarray          # (N,)
    kind: np.ndarray       # (N,) drone model for positives, sound class for negatives
    rec: np.ndarray        # (N,) recording id, the unit that is actually independent
    range_m: np.ndarray    # (N,) NaN where unknown or unreliable


def _ddl_ranges(root=PROCESSED) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per session: start sample of each kept file, and its range (NaN if unreliable)."""
    out: dict[str, tuple[list, list]] = collections.defaultdict(lambda: ([], []))
    with open(root / "ddl_clips.csv", newline="") as f:
        for c in csv.DictReader(f):
            if c["kept"] == "1":
                starts, ranges = out[c["session"]]
                starts.append(float(c["start_sample_16k"]))
                ranges.append(float(c["range_m"]) if c["range_reliable"] == "1" else math.nan)
    return {s: (np.asarray(a), np.asarray(b)) for s, (a, b) in out.items()}


def _windows(recs: list[Recording], n: int, hop: int, per_rec: int | None,
             kind: Callable[[Recording], str]) -> tuple[list, list, list, list, list]:
    audio, kinds, ids, starts, power = [], [], [], [], []
    for r in recs:
        s = window_starts(r.n_samples, n, hop)
        if per_rec is not None and len(s) > per_rec:
            s = s[np.linspace(0, len(s) - 1, per_rec).astype(int)]
        for st in s:
            x = r.audio(int(st), n)
            p = inband_power(x)
            if p <= SILENCE_POWER:
                continue
            audio.append(x); kinds.append(kind(r)); ids.append(r.recording_id)
            starts.append(int(st)); power.append(p)
    return audio, kinds, ids, starts, power


def _pack(name: str, label: int, w: tuple, range_m: np.ndarray | None = None) -> EvalSet:
    audio, kinds, ids, _, _ = w
    return EvalSet(name, np.stack(audio).astype(np.float32), np.full(len(audio), label),
                   np.asarray(kinds), np.asarray(ids),
                   np.full(len(audio), np.nan) if range_m is None else range_m)


def _ddl_set(name: str, recs: list[Recording], n: int) -> EvalSet:
    ranges = _ddl_ranges()
    w = _windows(recs, n, n, None, lambda r: r.labels)
    rng_m = np.array([
        ranges[rid][1][max(0, int(np.searchsorted(ranges[rid][0], st + n / 2, side="right")) - 1)]
        for rid, st in zip(w[2], w[3])
    ])
    return _pack(name, 1, w, rng_m)


def development_sets(splits: Splits, cfg: FrontendConfig = DEFAULT,
                     ddl_unseen: bool = False) -> dict[str, EvalSet]:
    """Validation and probe windows. Fixed: no randomness, no mixing.

    With `ddl_unseen`, every DDL session is added as one more set. Only valid
    for a model that was not trained on any DDL audio.
    """
    n = cfg.samples_per_window
    val_pos = _ddl_set("ddl_val", splits.ddl_val, n)
    val_neg = _pack("fsd_val", 0, _windows(splits.fsd_val, n, n, 3, lambda r: r.kind))

    # DroneNoise: a flyover is only loud around the pass-by. Windows far below
    # the file's own peak hold little drone and would be mislabelled positives.
    w = _windows(splits.probe, n, n, None, lambda r: r.attributes["model"])
    peak: dict[str, float] = collections.defaultdict(float)
    for rid, p in zip(w[2], w[4]):
        peak[rid] = max(peak[rid], p)
    keep = [i for i, (rid, p) in enumerate(zip(w[2], w[4]))
            if 10 * math.log10(p / peak[rid]) > -PROBE_MAX_DROP_DB]
    probe = _pack("dronenoise_probe", 1, tuple([c[i] for i in keep] for c in w))
    sets = {"val_pos": val_pos, "val_neg": val_neg, "probe": probe}
    if ddl_unseen:
        sets["ddl_all"] = _ddl_set("ddl_all", splits.ddl_train + splits.ddl_val, n)
    return sets


def sealed_sets(splits: Splits, cfg: FrontendConfig = DEFAULT) -> dict[str, EvalSet]:
    """The sealed test sources. Calling this is a test scoring: log it."""
    n = cfg.samples_per_window
    hal = _windows(splits.halmstad, n, n, None, lambda r: r.kind)
    idx_pos = [i for i, k in enumerate(hal[1]) if k == "drone"]
    idx_neg = [i for i, k in enumerate(hal[1]) if k != "drone"]
    return {
        "halmstad_pos": _pack("halmstad_drone", 1, tuple([c[i] for i in idx_pos] for c in hal)),
        "halmstad_neg": _pack("halmstad_other", 0, tuple([c[i] for i in idx_neg] for c in hal)),
        "fsd_eval": _pack("fsd_eval", 0, _windows(splits.fsd_eval, n, n, 2, lambda r: r.kind)),
    }


# ------------------------------------------------------------------ scorers

class LevelRule:
    """Quieter means drone. Not a detector: the size of the level shortcut."""
    name = "level rule"

    def __init__(self, cfg: FrontendConfig = DEFAULT) -> None:
        self.fe = LogMel(cfg)

    def score(self, audio: np.ndarray) -> np.ndarray:
        return -level_db(np.stack([self.fe.frames(a) for a in audio]))


class CombScorer:
    name = "comb (no ML)"

    def __init__(self, cfg: FrontendConfig = DEFAULT) -> None:
        from .gate import HarmonicCombGate
        self.gate = HarmonicCombGate(cfg)

    def score(self, audio: np.ndarray) -> np.ndarray:
        from .gate import decision_score
        return np.array([decision_score(self.gate(a)) for a in audio])


class LgbmScorer:
    def __init__(self, variant: Variant, cfg: FrontendConfig = DEFAULT) -> None:
        from .models.baseline import MfccBaseline
        self.variant, self.fe, self.model = variant, LogMel(cfg), MfccBaseline()
        self.name = f"mfcc + lightgbm [{variant.name}]"

    def fit(self, sampler: WindowSampler, n_windows: int = 40_000, batch_size: int = 64) -> "LgbmScorer":
        X, y = [], []
        for L, t in prefetch(sampler, n_windows // batch_size, batch_size):
            X.extend(summary_stats(m) for m in L)
            y.extend(t)
        self.model.fit(np.stack(X), np.asarray(y))
        return self

    def score(self, audio: np.ndarray) -> np.ndarray:
        L = featurise(audio, self.variant, self.fe)
        return self.model.predict_proba(np.stack([summary_stats(m) for m in L]))


class NetScorer:
    """A zoo architecture trained on streamed batches."""

    def __init__(self, arch: str, variant: Variant, cfg: FrontendConfig = DEFAULT, device: str = "cpu") -> None:
        import torch
        self.arch, self.variant, self.cfg = arch, variant, cfg
        self.fe = LogMel(cfg)
        self.name = f"{arch} [{variant.name}]"
        if device == "dml":
            import torch_directml
            self.device = torch_directml.device()
        else:
            self.device = torch.device(device)
        self.model = None
        self.mu = self.sigma = None
        self.history: list[dict] = []

    def fit(self, sampler: WindowSampler, n_windows: int = 200_000, batch_size: int = 64,
            lr: float = 2e-3, weight_decay: float = 1e-2, workers: int = 3,
            on_progress: Callable[["NetScorer", int], None] | None = None,
            progress_every: int = 50_000, log: Callable[[str], None] = print) -> "NetScorer":
        import torch
        import torch.nn.functional as F
        from .models.zoo import build
        from .train import spec_augment

        torch.manual_seed(sampler.seed)
        rng = np.random.default_rng(sampler.seed)
        steps = n_windows // batch_size

        # Per-band statistics from the first 64 training batches, then fixed.
        warm = np.concatenate([sampler.batch(i, batch_size)[0] for i in range(64)])
        self.mu = warm.mean(axis=(0, 2))[:, None].astype(np.float32)
        self.sigma = (warm.std(axis=(0, 2))[:, None] + 1e-5).astype(np.float32)

        self.model = build(self.arch, self.cfg.n_mels).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
        self.model.train()
        t0, running, mark = time.time(), 0.0, 0
        for step, (L, y) in enumerate(prefetch(sampler, steps, batch_size, workers), start=1):
            xb = spec_augment(torch.from_numpy((L - self.mu) / self.sigma), rng)
            loss = F.cross_entropy(self.model(xb[:, None].to(self.device)),
                                   torch.from_numpy(y).to(self.device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += float(loss.item())
            seen = step * batch_size
            if seen // progress_every > mark or step == steps:
                mark = seen // progress_every
                done = step - sum(h["steps"] for h in self.history)
                self.history.append(dict(windows=seen, steps=done, loss=running / max(done, 1),
                                         seconds=round(time.time() - t0, 1)))
                log(f"    {self.name}: {seen:>7} windows, loss {running / max(done, 1):.4f}, "
                    f"{seen / (time.time() - t0):.0f} windows/s")
                running = 0.0
                if on_progress is not None:
                    on_progress(self, seen)
                    self.model.train()
        self.model.eval()
        return self

    def score(self, audio: np.ndarray, batch_size: int = 256) -> np.ndarray:
        import torch
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(audio), batch_size):
                L = featurise(audio[i:i + batch_size], self.variant, self.fe)
                x = torch.from_numpy((L - self.mu) / self.sigma)[:, None].to(self.device)
                out.append(torch.softmax(self.model(x), dim=1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    def save(self, path) -> None:
        import torch
        torch.save(dict(arch=self.arch, variant=self.variant.name, mu=self.mu, sigma=self.sigma,
                        state={k: v.cpu() for k, v in self.model.state_dict().items()}), path)

    @classmethod
    def load(cls, path, cfg: FrontendConfig = DEFAULT, device: str = "cpu") -> "NetScorer":
        import torch
        from .models.zoo import build
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = cls(ck["arch"], VARIANTS[ck["variant"]], cfg, device)
        net.mu, net.sigma = ck["mu"], ck["sigma"]
        net.model = build(ck["arch"], cfg.n_mels)
        net.model.load_state_dict(ck["state"])
        net.model.to(net.device).eval()
        return net


# ------------------------------------------------------------------- metrics

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion. Holds up at small n and near 0 or 1."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def _ap(pos: np.ndarray, neg: np.ndarray) -> float:
    return float(average_precision_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def _fa_at_recall(pos: np.ndarray, neg: np.ndarray, recall: float = 0.95) -> float:
    return float((neg >= np.quantile(pos, 1.0 - recall)).mean())


def _recall_at_fa(pos: np.ndarray, neg: np.ndarray, fa: float = 0.05) -> float:
    return float((pos >= np.quantile(neg, 1.0 - fa)).mean())


def unseen_chain(pos: np.ndarray, neg: np.ndarray, recs: np.ndarray, recall: float = 0.95) -> dict:
    """How far a model is from the target on drones from a chain it never trained on.

    The threshold is that chain's own 95%-recall point, so this is the
    requirement's metric, and it is a ranking figure: it says nothing about
    whether one threshold would carry from one chain to another.
    """
    return dict(windows=len(pos), recordings=len(set(recs)), auc=_auc(pos, neg),
                fa_at_recall=_fa_at_recall(pos, neg, recall), recall_at_5pct_fa=_recall_at_fa(pos, neg))


def evaluate(scorer, sets: dict[str, EvalSet], recall: float = 0.95) -> dict:
    """Score the development sets. The threshold comes from validation only."""
    vp, vn, pr = sets["val_pos"], sets["val_neg"], sets["probe"]
    sp, sn, sq = scorer.score(vp.audio), scorer.score(vn.audio), scorer.score(pr.audio)

    audible = vp.range_m <= THRESHOLD_MAX_RANGE_M  # NaN compares false: unreliable ranges are left out
    thr = float(np.quantile(sp[audible], 1.0 - recall))
    out = dict(
        scorer=scorer.name, threshold=thr,
        n=dict(val_pos=len(sp), val_pos_audible=int(audible.sum()), val_neg=len(sn), probe=len(sq)),
        # The confounded contrast: DDL chain against FSD50K chains.
        val_auc=_auc(sp, sn), val_ap=_ap(sp, sn),
        val_recall=float((sp >= thr).mean()), val_fa=float((sn >= thr).mean()),
        # Transfer: drones from a chain and of models never trained on.
        transfer_auc=_auc(sq, sn), transfer_ap=_ap(sq, sn),
        transfer_recall=float((sq >= thr).mean()),
        transfer_fa_at_recall=_fa_at_recall(sq, sn, recall),
    )
    out["unseen"] = dict(dronenoise=unseen_chain(sq, sn, pr.rec, recall))
    if "ddl_all" in sets:
        da = sets["ddl_all"]
        sa = scorer.score(da.audio)
        for label, lo, hi in (("ddl 0-50 m", 0, 50), ("ddl 50-100 m", 50, 100), ("ddl 100-150 m", 100, 150),
                              ("ddl 150+ m", 150, 10_000), ("ddl 0-100 m", 0, 100)):
            sel = (da.range_m >= lo) & (da.range_m < hi)
            if sel.sum() >= 20:
                out["unseen"][label] = unseen_chain(sa[sel], sn, da.rec[sel], recall)
        near = da.range_m <= 50
        # Per session: adjacent windows are not independent, sessions are.
        out["ddl_sessions_0_50m"] = {
            s: dict(windows=int((near & (da.rec == s)).sum()),
                    recall_at_5pct_fa=_recall_at_fa(sa[near & (da.rec == s)], sn))
            for s in sorted(set(da.rec[near])) if (near & (da.rec == s)).sum() >= 10
        }
    out["transfer_by_model"] = {
        m: dict(n=int((pr.kind == m).sum()), auc=_auc(sq[pr.kind == m], sn),
                recall=float((sq[pr.kind == m] >= thr).mean()))
        for m in sorted(set(pr.kind))
    }
    out["val_by_model"] = {
        m: dict(n=int((vp.kind == m).sum()), auc=_auc(sp[vp.kind == m], sn),
                recall=float((sp[vp.kind == m] >= thr).mean()))
        for m in sorted(set(vp.kind))
    }
    # A drone detector should lose confidence with distance. A chain detector should not.
    out["val_by_range"] = {}
    for lo, hi in ((0, 50), (50, 100), (100, 150), (150, 1000)):
        sel = (vp.range_m >= lo) & (vp.range_m < hi)
        if sel.sum():
            out["val_by_range"][f"{lo}-{hi if hi < 1000 else '+'} m"] = dict(
                n=int(sel.sum()), mean_score=float(sp[sel].mean()),
                recall=float((sp[sel] >= thr).mean()), auc=_auc(sp[sel], sn))
    # The same background with and without a drone: the one contrast here
    # that the recording chain cannot explain.
    near, far = vp.range_m <= 50, vp.range_m >= 150
    out["paired"] = dict(
        dronenoise=paired_probe(scorer, vn.audio, pr.audio),
        ddl_near=paired_probe(scorer, vn.audio, vp.audio[near]),
        # Two controls. At 150 m and beyond a DDL window is mostly the site and
        # the recording chain, so a rise here that matches the near one is the
        # chain, not the drone. And adding a second background should not look
        # like a drone at all: about 0.5.
        ddl_far=paired_probe(scorer, vn.audio, vp.audio[far]),
        other_background=paired_probe(scorer, vn.audio, vn.audio[::-1]),
    )
    fa = collections.Counter(vn.kind[sn >= thr])
    tot = collections.Counter(vn.kind)
    out["val_fa_worst_kinds"] = {k: dict(n=tot[k], fa=fa[k] / tot[k])
                                 for k, _ in sorted(((k, fa[k] / tot[k]) for k in tot if tot[k] >= 20),
                                                    key=lambda t: -t[1])[:8]}
    return out


def paired_probe(scorer, backgrounds: np.ndarray, drones: np.ndarray,
                 snrs_db: tuple[float, ...] = (10.0, 0.0, -10.0), n_pairs: int = 1_500,
                 seed: int = 0) -> dict:
    """Can the scorer tell a background from the same background plus a drone?

    Each pair is one real background window, scored alone and again with a
    real drone window added at a set in-band signal-to-noise ratio. Both
    members of a pair share the background, its recording chain and its
    level, so nothing but the drone separates them. A scorer that has learned
    a recording chain gets about 0.5 here however well it validates.

    Reports, per ratio, the AUC of mixed against alone and the share of pairs
    in which adding the drone raised the score.
    """
    rng = np.random.default_rng(seed)
    bi = rng.choice(len(backgrounds), size=min(n_pairs, len(backgrounds)), replace=False)
    di = rng.integers(0, len(drones), size=len(bi))
    bg = backgrounds[bi]
    pb = np.array([inband_power(x) for x in bg])
    pd = np.array([inband_power(drones[j]) for j in di])
    alone = scorer.score(bg)
    out = {}
    for snr in snrs_db:
        gain = np.sqrt(pb * 10.0 ** (snr / 10.0) / pd).astype(np.float32)
        mixed = scorer.score(bg + gain[:, None] * drones[di])
        out[f"{snr:+.0f} dB"] = dict(auc=_auc(mixed, alone), raised=float((mixed > alone).mean()),
                                     mean_alone=float(alone.mean()), mean_mixed=float(mixed.mean()))
    return out


def evaluate_sealed(scorer, sets: dict[str, EvalSet], threshold: float, recall: float = 0.95) -> dict:
    """Score Halmstad and FSD50K-eval at a threshold frozen beforehand."""
    hp, hn, fe = sets["halmstad_pos"], sets["halmstad_neg"], sets["fsd_eval"]
    sp, sn, sf = scorer.score(hp.audio), scorer.score(hn.audio), scorer.score(fe.audio)

    def per_clip(es: EvalSet, s: np.ndarray, kinds=None) -> dict:
        # A clip counts once: its windows are not independent observations.
        clips = sorted(set(es.rec if kinds is None else es.rec[np.isin(es.kind, kinds)]))
        hit = sum(bool((s[es.rec == c] >= threshold).mean() >= 0.5) for c in clips)
        lo, hi = wilson(hit, len(clips))
        return dict(clips=len(clips), flagged=hit, rate=hit / len(clips), ci95=[lo, hi])

    clip_score = lambda es, s: np.array([s[es.rec == c].mean() for c in sorted(set(es.rec))])
    cp, cn = clip_score(hp, sp), clip_score(hn, sn)
    fa_fsd = int((sf >= threshold).sum())
    return dict(
        scorer=scorer.name, threshold=threshold,
        halmstad_clip_auc=_auc(cp, cn),
        halmstad_window_auc=_auc(sp, sn),
        halmstad_recall=per_clip(hp, sp),
        halmstad_fa_helicopter=per_clip(hn, sn, ["helicopter"]),
        halmstad_fa_background=per_clip(hn, sn, ["background"]),
        halmstad_fa_at_recall_own_threshold=_fa_at_recall(cp, cn, recall),
        fsd_eval_windows=len(sf), fsd_eval_fa=fa_fsd / len(sf),
        fsd_eval_fa_ci95=list(wilson(fa_fsd, len(sf))),
        fsd_eval_fa_by_kind={
            k: dict(n=int((fe.kind == k).sum()), fa=float((sf[fe.kind == k] >= threshold).mean()))
            for k in ("Engine", "Aircraft", "Fixed-wing_aircraft_and_airplane", "Motorcycle",
                      "Car", "Truck", "Mechanical_fan", "Power_tool", "Wind", "Rain", "Bird", "Insect")
            if (fe.kind == k).sum() >= 10
        },
    )
