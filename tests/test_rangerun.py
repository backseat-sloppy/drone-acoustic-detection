"""The pieces of iteration 3 that decide what a model trains on and how it is scored."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dad import chains as ch
from dad import rangerun as rg


def _cell(chain, lo, hi, seconds):
    return rg.Cell(chain, lo, hi, draw=None, seconds=seconds)


def test_bin_shares_split_equally_over_sources():
    cells = [_cell("rig", 0, 50, 100), _cell("ddl", 0, 50, 100), _cell("rig", 50, 100, 100),
             _cell("rig", 100, 150, 100), _cell("rig", 150, 200, 100)]
    rg.set_weights(cells, rg.Recipe("t", {(0, 50): 0.2, (50, 100): 0.25, (100, 150): 0.3, (150, 200): 0.25}))
    w = {(c.chain, c.lo): c.weight for c in cells}
    assert math.isclose(w[("rig", 0)], 0.1) and math.isclose(w[("ddl", 0)], 0.1)
    assert math.isclose(w[("rig", 150)], 0.25)
    assert math.isclose(sum(w.values()), 1.0)


def test_cap_moves_excess_to_the_nearer_bin_of_the_same_source():
    # 150-200 m holds 1% of the audio but is asked for 25% of the draws.
    cells = [_cell("rig", 0, 50, 79), _cell("rig", 50, 100, 10), _cell("rig", 100, 150, 10),
             _cell("rig", 150, 200, 1)]
    rg.set_weights(cells, rg.Recipe("t", {(0, 50): 0.2, (50, 100): 0.25, (100, 150): 0.3, (150, 200): 0.25},
                                    cap=8.0))
    w = {c.lo: c.weight for c in cells}
    assert math.isclose(w[150], 8 * 1 / 100)
    assert math.isclose(sum(w.values()), 1.0)
    assert w[100] <= 8 * 10 / 100 + 1e-12


def test_ring_latency_counts_window_fill_and_censors_beyond_the_band():
    # One drone segment, windows every second: in the band for 6 s, then out.
    n = 8
    cw = ch.ChainWindows(recs=[None], rec_idx=np.zeros(n, int), start=np.arange(n) * 16_000,
                         y=np.ones(n, int), range_m=np.array([150, 150, 150, 150, 150, 150, 250, 250.0]),
                         session=np.array(["s"] * n, dtype=object))
    s = np.array([0, 0, 1, 0, 0, 0, 1, 1.0])
    lat = rg.ring_latency(rg.Scored(cw, s), thr=0.5, window_s=1.0)
    # Starts at 0..5 s (the last start is before t[-1]=7). From 0, 1, 2 the first
    # alarm is the window at 2 s, ending at 3 s; from 3, 4, 5 the drone leaves
    # the band before any alarm.
    assert lat["starts"] == 6
    assert math.isclose(lat["within_1s"], 1 / 6)
    assert math.isclose(lat["within_5s"], 3 / 6)
    assert math.isclose(lat["never"], 3 / 6)


def test_range_bins_label_unknown_as_empty():
    b = ch.range_bin(np.array([10.0, 149.9, 150.0, np.nan, 1000.0]))
    assert list(b) == ["0-50 m", "100-150 m", "150-200 m", "", "300-inf m"]


def test_fa_threshold_never_exceeds_the_budget_when_scores_tie():
    neg = np.r_[np.zeros(90), np.linspace(0.1, 1.0, 10)]
    thr = rg.fa_threshold(neg, 0.20)
    assert (neg >= thr).mean() <= 0.20
    assert rg.fa_threshold(np.zeros(50), 0.10) > 0.0


def test_integrate_averages_consecutive_windows_of_one_recording_only():
    hop = 8_000                               # 0.5 s
    cw = ch.ChainWindows(recs=[None, None], rec_idx=np.array([0, 0, 0, 1, 1]),
                         start=np.array([0, hop, 3 * hop, 0, hop]), y=np.ones(5, int),
                         range_m=np.array([10.0, 20, 30, 40, 50]), session=np.array(["a"] * 5, dtype=object))
    out = rg.integrate(rg.Scored(cw, np.array([0.0, 1.0, 0.5, 0.2, 0.4])), 2, 0.5)
    # Recording 0: only (0, hop) is a run; the gap to 3*hop breaks the next one.
    assert list(out.s) == [0.5, pytest.approx(0.3)]
    assert list(out.cw.rec_idx) == [0, 1]
    assert list(out.cw.range_m) == [20.0, 50.0]


def test_background_wait_is_the_chance_of_an_alarm_without_a_drone():
    n = 6
    cw = ch.ChainWindows(recs=[None], rec_idx=np.zeros(n, int), start=np.arange(n) * 16_000,
                         y=np.zeros(n, int), range_m=np.full(n, np.nan),
                         session=np.array(["s"] * n, dtype=object))
    lat = rg.ring_latency(rg.Scored(cw, np.array([0, 0, 0, 1, 0, 0.0])), thr=0.5, window_s=1.0, label=0)
    # From 0..4 s: waits 4, 3, 2, 1, then none before the recording ends.
    assert lat["starts"] == 5 and math.isclose(lat["within_2s"], 2 / 5) and math.isclose(lat["never"], 1 / 5)
