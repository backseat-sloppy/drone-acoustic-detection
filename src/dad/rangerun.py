"""Iteration 3: train on rigs that recorded both classes, weighted by range.

The design is the council decision of docs/REPORT.md, entry 025.

* Two chains have drone and no-drone audio from one rig: Nightjar and d3set.
  Each leave-one-chain-out arm trains on one of them and tests on the other.
* Drone windows are labelled by slant range to the drone. Up to 200 m they
  are positives. From 200 to 300 m they are left out ("don't care"). Beyond
  300 m they are left out of training and kept as a control: a model that
  scores them above the same session's background has learned the session,
  not the drone.
* Positives are drawn from (chain, range bin) cells with fixed target shares
  per bin, so the 100-200 m band is not swamped by the many near windows.
  No cell is drawn more than `cap` times its natural rate.
* Each rig contributes as many negatives as positives, so the rig says
  nothing about the label. DDL and DroneNoise, which have no negatives of
  their own, are buried in FSD50K backgrounds as in iteration 2.

Recipes (`RECIPES`):
    near    positives up to 50 m only (A' in entry 025)
    range   plus real positives at 50-200 m, by the bin shares (C')
    simdist near drones summed into the training rig's own background at a
            signal-to-noise ratio of -10 to +5 dB, in place of real far
            positives (D'; gain only, as the LOCKED data rule allows)
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field

import numpy as np

from . import chains as ch
from . import realrun as rr
from .config import DEFAULT, FrontendConfig
from .features import LogMel
from .realdata import Recording

TRAIN_BINS = ((0, 50), (50, 100), (100, 150), (150, 200))
CONTROL_MIN_M = 300.0
NIGHTJAR_VAL = frozenset({"nightjar-flight-20260804-3"})
CONFUSER = frozenset({"nightjar-lawnmower1-20260805"})   # mower runs only while the drone flies
D3SET_VAL = "Validation"
# Lund 10_flower3: the ground truth runs 29 s past the end of the audio, so its
# clock cannot be trusted (report entry 024).
LUND_SUSPECT = frozenset({"10_flower3"})


# ------------------------------------------------------------------ roles

@dataclass
class Arm:
    name: str
    train_chain: str
    test_chain: str
    train: list[Recording]
    val: list[Recording]
    test: list[Recording]
    confuser: list[Recording]
    unseen: dict[str, list[Recording]]     # further chains scored for recall only


def _is_val(r: Recording) -> bool:
    return ch.session(r) in NIGHTJAR_VAL or (r.source == "d3set" and r.subset == D3SET_VAL)


def make_arm(manifest: list[Recording], test_chain: str) -> Arm:
    train_chain = {"nightjar": "d3set", "d3set": "nightjar"}[test_chain]
    by = collections.defaultdict(list)
    for r in manifest:
        by[r.source].append(r)
    tr = [r for r in by[train_chain] if ch.session(r) not in CONFUSER]
    return Arm(
        name=f"hold out {test_chain}", train_chain=train_chain, test_chain=test_chain,
        train=[r for r in tr if not _is_val(r)], val=[r for r in tr if _is_val(r)],
        test=[r for r in by[test_chain] if ch.session(r) not in CONFUSER],
        confuser=[r for r in by["nightjar"] if ch.session(r) in CONFUSER],
        unseen={c: [r for r in by[c] if ch.session(r) not in LUND_SUSPECT] for c in ("lund", "glasgow") if by[c]},
    )


UNITS = ("nightjar", "d3set", "fpv:ariel_25.02", "fpv:palmahim_10.02", "fpv:palmahim_27.05")


def _unit(r: Recording) -> str:
    return f"fpv:{ch.session(r)}" if r.source == "fpv" else r.source


def make_pool_arm(manifest: list[Recording], held: str) -> Arm:
    """Iteration 5: train on every rig with both classes except `held`.

    Units are Nightjar, d3set and the three FPV campaigns (the phones of one
    campaign recorded the same flights, entry 033, so a campaign is one unit).
    Glasgow always trains. Validation (the deployment threshold) comes from
    the Nightjar and d3set validation sessions that are in training. Lund is
    scored for recall only.

    `held="none"` trains on every unit and leaves nothing out (the final model
    for the real test, iteration 6); its test list is then empty.
    """
    if held != "none" and held not in UNITS:
        raise ValueError(f"unknown unit {held!r}; expected one of {UNITS} or 'none'")
    pool = [r for r in manifest if r.source in ("nightjar", "d3set", "fpv", "glasgow")
            and ch.session(r) not in CONFUSER]
    train = [r for r in pool if _unit(r) != held]
    by = collections.defaultdict(list)
    for r in manifest:
        by[r.source].append(r)
    return Arm(
        name=f"hold out {held}", train_chain="pool", test_chain=held,
        train=[r for r in train if not _is_val(r)], val=[r for r in train if _is_val(r)],
        test=[r for r in pool if _unit(r) == held],
        confuser=[r for r in by["nightjar"] if ch.session(r) in CONFUSER],
        unseen={"lund": [r for r in by["lund"] if ch.session(r) not in LUND_SUSPECT]},
    )


# ---------------------------------------------------------------- sampling

@dataclass(frozen=True)
class Recipe:
    name: str
    shares: dict = field(default_factory=dict)   # {(lo, hi): share of the real positive draws}
    sim_share: float = 0.0                       # share of draws that are simulated-distance positives
    sim_snr_db: tuple[float, float] = (-10.0, 5.0)
    cap: float = 8.0
    near_bury: bool = True                       # near positives buried in FSD50K, as iteration 2


RECIPES = {
    "near": Recipe("near", {(0, 50): 1.0}),
    "range": Recipe("range", {(0, 50): 0.20, (50, 100): 0.25, (100, 150): 0.30, (150, 200): 0.25}),
    "simdist": Recipe("simdist", {(0, 50): 1.0}, sim_share=0.80),
}


class Cell:
    """Windows a sampler may draw from one source and range bin."""

    def __init__(self, chain: str, lo: float, hi: float, draw, seconds: float) -> None:
        self.chain, self.lo, self.hi, self.draw, self.seconds = chain, lo, hi, draw, seconds
        self.weight = 0.0


def _chain_cell(cw: ch.ChainWindows, sel: np.ndarray, n: int, chain: str, lo: float, hi: float,
                grid: int) -> Cell | None:
    idx = np.flatnonzero(sel)
    if not len(idx):
        return None

    def draw(rng: np.random.Generator, idx=idx) -> np.ndarray:
        i = idx[rng.integers(0, len(idx))]
        rec = cw.recs[cw.rec_idx[i]]
        s = int(cw.start[i]) + int(rng.integers(-grid // 2, grid // 2 + 1))
        return rec.audio(min(max(s, 0), rec.n_samples - n), n)

    return Cell(chain, lo, hi, draw, len(idx) * grid / 16_000)


def _pool_cell(pool: rr.Pool, chain: str) -> Cell:
    return Cell(chain, 0, 50, pool.draw, pool.seconds)


def set_weights(cells: list[Cell], recipe: Recipe) -> None:
    """Target share per bin, split equally over the sources with audio in it.
    A cell above `cap` times its natural rate is cut back and the excess goes
    to the same source's next nearer bin."""
    total = sum(c.seconds for c in cells)
    real = 1.0 - recipe.sim_share
    for (lo, hi), share in recipe.shares.items():
        here = [c for c in cells if c.lo == lo and c.seconds > 0]
        for c in here:
            c.weight += real * share / len(here)
    for lo, hi in sorted(TRAIN_BINS, reverse=True):
        for c in [c for c in cells if c.lo == lo]:
            cap = recipe.cap * c.seconds / total
            if c.weight > cap and lo > 0:
                nearer = [d for d in cells if d.chain == c.chain and d.hi == lo]
                if nearer:
                    nearer[0].weight += c.weight - cap
                    c.weight = cap


class RangeSampler:
    """Training windows, half drone and half not, drawn on the fly.

    Same interface as realrun.WindowSampler (`batch`, `seed`), so the
    iteration 2 training loop and LightGBM wrapper are reused unchanged.

    `arm.train` may hold several rigs (iteration 5 pools them). Each rig
    supplies the share of negatives equal to its share of positives, so the
    rig says nothing about the label. Drone audio without range labels (the
    FPV campaigns) counts as near: it was labelled where the drone is heard.

    With `speed` (lo, hi), every window, drone or not, is resampled by one
    factor drawn from that range (entry 033): the drone's pitch and tempo
    change as for a faster or Doppler-shifted drone, and because negatives get
    the same treatment, the band a slowed window leaves empty near 8 kHz says
    nothing about the label.
    """

    def __init__(self, arm: Arm, splits: rr.Splits, recipe: Recipe, cfg: FrontendConfig = DEFAULT,
                 seed: int = 0, grid_s: float = 0.1, speed: tuple[float, float] | None = None) -> None:
        self.cfg, self.recipe, self.seed, self.speed = cfg, recipe, seed, speed
        self.n_out = cfg.samples_per_window
        # Windows are drawn long enough to be sped up by the largest factor.
        self.n = n = self.n_out if speed is None else int(math.ceil(self.n_out * speed[1])) + 32
        grid = int(grid_s * cfg.sample_rate)
        self.fe = LogMel(cfg)
        # Backgrounds and mixing as the iteration 2 "buried" variant.
        self.variant = rr.VARIANTS["buried"]
        long_cfg = FrontendConfig(window_seconds=n / cfg.sample_rate)
        fsd = [r for r in splits.fsd_train if r.n_samples >= n]   # long enough for the drawn length
        self.fsd = rr.WindowSampler([rr.ddl_pool(splits.ddl_train, n, 50.0)], fsd, long_cfg,
                                    self.variant, seed)

        by_src = collections.defaultdict(list)
        for r in arm.train:
            by_src[r.source].append(r)
        cells: list[Cell] = []
        self._rigs: dict[str, tuple[ch.ChainWindows, np.ndarray, np.ndarray]] = {}
        for src, recs in sorted(by_src.items()):
            cw = ch.chain_windows(recs, n, grid)
            pos = cw.y == 1
            unranged = pos & ~np.isfinite(cw.range_m)
            for lo, hi in TRAIN_BINS:
                if lo > 0 and recipe.shares.get((lo, hi), 0.0) == 0.0:
                    continue
                sel = pos & (cw.range_m >= lo) & (cw.range_m < hi)
                if lo == 0:
                    sel = sel | unranged
                c = _chain_cell(cw, sel, n, src, lo, hi, grid)
                if c is not None:
                    cells.append(c)
            # The rig's own background. Night recordings are drawn no more
            # often than night flights are, so time of day says nothing.
            neg = cw.y == 0
            if neg.any():
                cond = np.array([cw.recs[i].attributes.get("condition", "") for i in cw.rec_idx])
                pos_cond = collections.Counter(cond[pos & ~(cw.range_m >= 200)])
                neg_cond = collections.Counter(cond[neg])
                wn = np.array([pos_cond.get(c, 0) / max(sum(pos_cond.values()), 1) / neg_cond[c] if ok else 0.0
                               for c, ok in zip(cond, neg)])
                if wn.sum() == 0:
                    wn = neg.astype(float)
                idx = np.flatnonzero(wn > 0)
                self._rigs[src] = (cw, idx, np.cumsum(wn[idx] / wn[idx].sum()))
        cells.append(_pool_cell(rr.ddl_pool(splits.ddl_train, n, 50.0, name="ddl <=50 m"), "ddl"))
        cells.append(_pool_cell(rr.dronenoise_pool(splits.dronenoise, n), "dronenoise"))
        set_weights(cells, recipe)
        self.cells = [c for c in cells if c.weight > 0]
        w = np.array([c.weight for c in self.cells] + [recipe.sim_share])
        self._cell_cdf = np.cumsum(w / w.sum())
        self._grid = grid

        # Each rig with negatives: share of negatives = its share of positives,
        # plus an equal part of the simulated-distance share (those positives
        # are built on rig backgrounds).
        rigs = sorted(self._rigs)
        share = {s: sum(c.weight for c in self.cells if c.chain == s) + recipe.sim_share / max(len(rigs), 1)
                 for s in rigs}
        self._rig_names = rigs
        self._rig_share = np.array([share[s] / w.sum() for s in rigs])
        self.rig_neg_share = float(self._rig_share.sum())

    def describe(self) -> str:
        parts = [f"{c.chain} {c.lo}-{c.hi} m: {c.seconds / 60:.1f} min, share {c.weight / self._norm():.3f}"
                 for c in self.cells]
        if self.recipe.sim_share:
            parts.append(f"simulated distance: share {self.recipe.sim_share / self._norm():.3f}")
        for s, sh in zip(self._rig_names, self._rig_share):
            cw, idx, _ = self._rigs[s]
            parts.append(f"{s} negatives share {sh:.3f} ({len(idx) * self._grid / 16_000 / 60:.1f} min)")
        if self.speed:
            parts.append(f"speed {self.speed[0]:g}-{self.speed[1]:g}")
        return "; ".join(parts)

    def _norm(self) -> float:
        return sum(c.weight for c in self.cells) + self.recipe.sim_share

    def _rig_background(self, rng: np.random.Generator, src: str | None = None) -> tuple[np.ndarray, float]:
        if src is None:
            src = self._rig_names[int(rng.integers(0, len(self._rig_names)))]
        cw, idx, cdf = self._rigs[src]
        for _ in range(50):
            i = idx[int(np.searchsorted(cdf, rng.random(), side="right"))]
            rec = cw.recs[cw.rec_idx[i]]
            s = int(cw.start[i]) + int(rng.integers(-self._grid // 2, self._grid // 2 + 1))
            x = rec.audio(min(max(s, 0), rec.n_samples - self.n), self.n)
            p = rr.inband_power(x)
            if p > rr.SILENCE_POWER:
                return x, p
        raise RuntimeError("could not draw a non-silent rig background")

    def _bury(self, d: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        b, pb = self.fsd._background(rng)
        snr = rng.uniform(*self.variant.snr_db)
        return d + b * math.sqrt(rr.inband_power(d) / (pb * 10.0 ** (snr / 10.0)))

    def positive(self, rng: np.random.Generator) -> np.ndarray:
        k = int(np.searchsorted(self._cell_cdf, rng.random(), side="right"))
        if k == len(self.cells):            # simulated distance
            near = [c for c in self.cells if c.lo == 0]
            d = near[int(rng.integers(0, len(near)))].draw(rng)
            a, pa = self._rig_background(rng)
            snr = rng.uniform(*self.recipe.sim_snr_db)
            return a + d * math.sqrt(pa * 10.0 ** (snr / 10.0) / max(rr.inband_power(d), 1e-20))
        c = self.cells[k]
        d = c.draw(rng)
        if c.lo == 0 and self.recipe.near_bury:
            d = self._bury(d, rng)
        return d

    def negative(self, rng: np.random.Generator) -> np.ndarray:
        u = rng.random()
        if u < self.rig_neg_share:
            k = min(int(np.searchsorted(np.cumsum(self._rig_share), u, side="right")), len(self._rig_names) - 1)
            a, pa = self._rig_background(rng, self._rig_names[k])
        else:
            a, pa = self.fsd._background(rng)
        if rng.random() < self.variant.mix_prob_neg:
            b, pb = self.fsd._background(rng)
            rel = rng.uniform(*self.variant.neg_rel_db)
            a = a + b * math.sqrt(pa * 10.0 ** (rel / 10.0) / pb)
        return a

    def _finish(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.speed is None:
            return x[: self.n_out]
        return speed_change(x, rng.uniform(*self.speed))[: self.n_out]

    def batch(self, index: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng([self.seed, index])
        y = (np.arange(batch_size) % 2).astype(np.int64)
        L = np.stack([self.fe.frames(self._finish(self.positive(rng) if t else self.negative(rng), rng))
                      for t in y])
        L = L + np.stack([rr._eq_curve(rng, L.shape[1]) for _ in y])[:, :, None]
        return rr.level_normalise(L), y


def speed_change(x: np.ndarray, factor: float) -> np.ndarray:
    """Play `x` `factor` times faster: pitch and tempo both scale by `factor`.

    Polyphase resampling to len(x) / factor samples at the same rate, with the
    factor rounded to 1/100.
    """
    from scipy.signal import resample_poly
    down = int(round(100 * factor))
    if down == 100:
        return np.asarray(x, dtype=np.float32)
    return resample_poly(np.asarray(x, dtype=np.float64), 100, down).astype(np.float32)


# -------------------------------------------------------------- evaluation

@dataclass
class Scored:
    cw: ch.ChainWindows
    s: np.ndarray


def score_chain(scorer, recs: list[Recording], cfg: FrontendConfig, hop_s: float, batch: int = 512) -> Scored:
    n = cfg.samples_per_window
    cw = ch.chain_windows(recs, n, int(hop_s * cfg.sample_rate))
    keep, out = np.ones(len(cw), bool), []
    for i in range(0, len(cw), batch):
        js = range(i, min(i + batch, len(cw)))
        a = np.stack([cw.audio(j, n) for j in js])
        p = np.array([rr.inband_power(x) for x in a])
        keep[list(js)] = p > rr.SILENCE_POWER
        out.append(scorer.score(a))
    s = np.concatenate(out) if out else np.empty(0)
    return Scored(cw.subset(keep), s[keep])


def integrate(sc: Scored, k: int, hop_s: float) -> Scored:
    """Average the scores of k consecutive windows of one recording.

    The result is one decision per run of k windows, placed at the first of
    them: it covers (k - 1) hops plus one window of audio, and the alarm can
    only sound once all of it has arrived, so pass `window_s + (k - 1) * hop_s`
    to `ring_latency`. Runs broken by a skipped (silent) window are dropped.
    Range is taken at the middle window of the run.
    """
    if k <= 1:
        return sc
    cw, hop = sc.cw, int(round(hop_s * 16_000))
    order = np.lexsort((cw.start, cw.rec_idx))
    keep, val, mid = [], [], []
    for a in range(len(order) - k + 1):
        run = order[a:a + k]
        if cw.rec_idx[run[0]] != cw.rec_idx[run[-1]] or np.any(np.diff(cw.start[run]) != hop):
            continue
        keep.append(run[0]); mid.append(run[k // 2]); val.append(sc.s[run].mean())
    keep, mid = np.asarray(keep, int), np.asarray(mid, int)
    out = cw.subset(keep)
    out.range_m = cw.range_m[mid]
    return Scored(out, np.asarray(val))


def fa_threshold(neg: np.ndarray, fa: float) -> float:
    """The lowest threshold whose false-alarm rate on `neg` is at most `fa`.

    A plain quantile overshoots when many scores tie (the comb scores most
    windows exactly 0): every tied window would then count as an alarm.
    """
    thr = float(np.quantile(neg, 1.0 - fa))
    if (neg >= thr).mean() > fa:
        thr = float(np.nextafter(thr, np.inf))
    return thr


def _auc(pos, neg) -> float | None:
    return rr._auc(pos, neg) if len(pos) >= 5 and len(neg) >= 5 else None


def _recall_by_bin(sc: Scored, thr: float) -> dict:
    pos = sc.cw.y == 1
    out = {}
    if pos.sum() >= 10:
        out["all"] = dict(windows=int(pos.sum()), sessions=len(set(sc.cw.session[pos])),
                          recall=float((sc.s[pos] >= thr).mean()))
    for lo, hi in ch.RANGE_BINS + ((100, 200),):
        sel = pos & (sc.cw.range_m >= lo) & (sc.cw.range_m < hi)
        if sel.sum() >= 10:
            out[f"{lo}-{hi if hi < 100_000 else 'inf'} m"] = dict(
                windows=int(sel.sum()), sessions=len(set(sc.cw.session[sel])),
                recall=float((sc.s[sel] >= thr).mean()))
    return out


def bootstrap_recall(sc: Scored, thr: float, lo: float, hi: float, n_boot: int = 1000,
                     seed: int = 0) -> tuple[float, float] | None:
    """95% interval of recall in a range band, resampling sessions (flights),
    the unit that is actually independent."""
    sel = (sc.cw.y == 1) & (sc.cw.range_m >= lo) & (sc.cw.range_m < hi)
    sess = sorted(set(sc.cw.session[sel]))
    if len(sess) < 2:
        return None
    hits = {s: (sc.s[sel & (sc.cw.session == s)] >= thr) for s in sess}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(sess), len(sess))
        h = np.concatenate([hits[sess[i]] for i in pick])
        vals.append(h.mean())
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def ring_latency(sc: Scored, thr: float, window_s: float, lo: float = 100.0, hi: float = 200.0,
                 step_s: float = 1.0, label: int = 1) -> dict:
    """Time to the first alarm, from every moment the drone is at `lo`-`hi` m.

    From each start time t0 (every `step_s` while the drone is in the band),
    the alarm time is the end of the first window that starts at or after t0
    and scores at or above `thr`. The wait is censored, counted as a miss,
    once the drone is beyond `hi` or the recording ends.

    With `label=0` the same is measured on background windows, from every
    moment of the background, censored only at the end of a recording: the
    chance of an alarm within a given time when there is no drone at all.
    Detection times mean little without it.
    """
    cw, sr = sc.cw, 16_000
    waits = []
    for ri in sorted(set(cw.rec_idx[cw.y == label])):
        sel = np.flatnonzero((cw.rec_idx == ri) & (cw.y == label))
        order = sel[np.argsort(cw.start[sel])]
        t = cw.start[order] / sr
        hit = sc.s[order] >= thr
        inband = (cw.range_m[order] >= lo) & (cw.range_m[order] < hi) if label == 1 else np.ones(len(t), bool)
        stays = (cw.range_m[order] < hi) if label == 1 else np.ones(len(t), bool)
        if not len(t):
            continue
        for t0 in np.arange(t[0], t[-1], step_s):
            j0 = int(np.searchsorted(t, t0))
            if j0 >= len(t) or not inband[j0]:
                continue
            wait = math.inf
            for j in range(j0, len(t)):
                if not stays[j]:
                    break
                if hit[j]:
                    wait = t[j] + window_s - t0
                    break
            waits.append(wait)
    w = np.asarray(waits)
    if not len(w):
        return {}
    return dict(starts=len(w), median_s=float(np.median(w)) if np.isfinite(np.median(w)) else None,
                **{f"within_{k}s": float((w <= k).mean()) for k in (1, 2, 3, 5, 10)},
                never=float(np.isinf(w).mean()))


def evaluate_chain(sc: Scored, thr_deploy: float, window_s: float, fas=(0.05, 0.10, 0.20)) -> dict:
    """Held-out rig: recall by range at its own false-alarm budgets (ranking)
    and at the threshold frozen on validation (deployment)."""
    cw, s = sc.cw, sc.s
    pos, neg = cw.y == 1, cw.y == 0
    out = dict(windows=dict(drone=int(pos.sum()), background=int(neg.sum()),
                            sessions=len(set(cw.session))))
    if neg.sum() >= 20:
        out["own"] = {}
        for fa in fas:
            thr = fa_threshold(s[neg], fa)
            o = dict(threshold=thr, by_range=_recall_by_bin(sc, thr))
            o["recall_100_200_ci95"] = bootstrap_recall(sc, thr, 100, 200)
            o["latency_100_200"] = ring_latency(sc, thr, window_s)
            o["latency_background"] = ring_latency(sc, thr, window_s, label=0)
            # The scenario of entry 032: an inbound drone crosses 300 to 150 m
            # in the last 3 s before the 150 m line.
            o["latency_150_300"] = ring_latency(sc, thr, window_s, 150.0, 300.0)
            out["own"][f"{fa:.2f}"] = o
        # Range bins against the same sessions' background, and the control.
        out["auc_session"] = {}
        for lo, hi in ch.RANGE_BINS + ((100, 200),):
            sel = pos & (cw.range_m >= lo) & (cw.range_m < hi)
            sess = set(cw.session[sel]) & set(cw.session[neg])
            same = neg & np.isin(cw.session, list(sess))
            sel = sel & np.isin(cw.session, list(sess))
            a = _auc(s[sel], s[same])
            if a is not None:
                out["auc_session"][f"{lo}-{hi if hi < 100_000 else 'inf'} m"] = dict(
                    auc=a, pos=int(sel.sum()), neg=int(same.sum()), sessions=len(sess))
        out["deploy_fa"] = float((s[neg] >= thr_deploy).mean())
    out["deploy"] = dict(threshold=thr_deploy, by_range=_recall_by_bin(sc, thr_deploy),
                         recall_100_200_ci95=bootstrap_recall(sc, thr_deploy, 100, 200),
                         latency_100_200=ring_latency(sc, thr_deploy, window_s))
    return out


def _get(d, *keys):
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def headline(r: dict) -> dict:
    """The few numbers a run is judged by (entry 025)."""
    get = _get
    own = _get(r, "own", "0.10") or {}
    return dict(
        recall_100_200_at_10pct=get(own, "by_range", "100-200 m", "recall"),
        recall_0_50_at_10pct=get(own, "by_range", "0-50 m", "recall"),
        control_300plus_auc=get(r, "auc_session", "300-inf m", "auc"),
        deploy_recall_100_200=get(r, "deploy", "by_range", "100-200 m", "recall"),
        deploy_fa=r.get("deploy_fa"),
        latency_within_5s=get(own, "latency_100_200", "within_5s"),
        background_alarm_within_5s=get(own, "latency_background", "within_5s"),
        recall_all_at_10pct=get(own, "by_range", "all", "recall"),
        alarm_within_3s_150_300=get(own, "latency_150_300", "within_3s"),
        background_alarm_within_3s=get(own, "latency_background", "within_3s"),
    )
