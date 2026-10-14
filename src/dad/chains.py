"""The recording chains that have drone and no-drone audio from one rig.

scripts/prepare_new_data.py stores each Nightjar, d3set, Lund and Glasgow
recording whole and writes one manifest row per stretch of a single state
(drone, background, uncertain). This module turns those rows into windows,
each with the slant range to the drone at its centre, so that training can
weight by range and evaluation can report by range.

Nothing here changes the audio.
"""
from __future__ import annotations

import collections
import csv
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .realdata import PROCESSED, SAMPLE_RATE, Recording

CHAINS = ("nightjar", "d3set", "lund", "glasgow")
RANGE_BINS = ((0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 100_000))


@lru_cache(maxsize=1)
def range_tracks(root: str = str(PROCESSED)) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """recording_id -> (seconds from the segment start, slant range in metres)."""
    out: dict[str, tuple[list, list]] = collections.defaultdict(lambda: ([], []))
    with open(f"{root}/range_track.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t, d = out[r["recording_id"]]
            t.append(float(r["t_s"]))
            d.append(float(r["range_m"]))
    return {k: (np.asarray(t), np.asarray(d)) for k, (t, d) in out.items()}


def range_at(rec: Recording, starts: np.ndarray, n: int) -> np.ndarray:
    """Slant range at the centre of each window; NaN where unknown."""
    if rec.label != 1:
        return np.full(len(starts), np.nan)
    if "range_m" in rec.attributes:                  # Glasgow: fixed distance per clip
        return np.full(len(starts), float(rec.attributes["range_m"]))
    tr = range_tracks().get(rec.recording_id)
    if tr is None:
        return np.full(len(starts), np.nan)
    t, d = tr
    c = (np.asarray(starts) + n / 2) / SAMPLE_RATE
    r = np.interp(c, t, d)
    r[(c < t[0] - 0.2) | (c > t[-1] + 0.2)] = np.nan
    return r


def session(rec: Recording) -> str:
    return rec.attributes.get("session", rec.group)


@dataclass
class ChainWindows:
    """Candidate windows of one chain: where they start, and what is known about them."""
    recs: list[Recording]
    rec_idx: np.ndarray     # (N,) index into recs
    start: np.ndarray       # (N,) sample offset within the recording
    y: np.ndarray           # (N,) 1 drone, 0 background
    range_m: np.ndarray     # (N,) NaN for background or unknown
    session: np.ndarray     # (N,) str

    def __len__(self) -> int:
        return len(self.start)

    def audio(self, i: int, n: int) -> np.ndarray:
        return self.recs[self.rec_idx[i]].audio(int(self.start[i]), n)

    def subset(self, mask: np.ndarray) -> "ChainWindows":
        return ChainWindows(self.recs, self.rec_idx[mask], self.start[mask], self.y[mask],
                            self.range_m[mask], self.session[mask])


def chain_windows(recs: list[Recording], n: int, hop: int) -> ChainWindows:
    """Every full window, on a grid of `hop` samples, in the drone and background
    segments of `recs`. Uncertain segments are never used."""
    idx, st, y, rg, ss = [], [], [], [], []
    for i, r in enumerate(recs):
        if r.label not in (0, 1) or r.n_samples < n:
            continue
        s = np.arange(0, r.n_samples - n + 1, hop, dtype=np.int64)
        idx.append(np.full(len(s), i)); st.append(s); y.append(np.full(len(s), r.label))
        rg.append(range_at(r, s, n)); ss.append(np.full(len(s), session(r), dtype=object))
    if not idx:
        e = np.empty(0)
        return ChainWindows(recs, e.astype(int), e.astype(np.int64), e.astype(int), e, e.astype(object))
    return ChainWindows(recs, np.concatenate(idx), np.concatenate(st), np.concatenate(y),
                        np.concatenate(rg), np.concatenate(ss))


def range_bin(r: np.ndarray) -> np.ndarray:
    """Label of the RANGE_BINS bin of each range, '' where unknown."""
    out = np.full(len(r), "", dtype=object)
    for lo, hi in RANGE_BINS:
        out[(r >= lo) & (r < hi)] = f"{lo}-{hi if hi < 100_000 else 'inf'} m"
    return out


def describe(recs: list[Recording]) -> dict:
    """Minutes of drone audio by range bin and of background, per chain."""
    out: dict = {}
    for src in sorted({r.source for r in recs}):
        cw = chain_windows([r for r in recs if r.source == src], SAMPLE_RATE, SAMPLE_RATE // 10)
        b = range_bin(cw.range_m)
        d = {k: round(float(((b == k) & (cw.y == 1)).sum()) / 600, 1) for k in sorted(set(b[cw.y == 1]))}
        d["background"] = round(float((cw.y == 0).sum()) / 600, 1)
        out[src] = d
    return out


def finite_mean(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else math.nan
