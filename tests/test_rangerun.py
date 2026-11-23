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


def _rec(source, rid, label, subset="", session="s"):
    from dad.realdata import Recording
    return Recording(source=source, recording_id=rid, label=label, kind="drone" if label else "background",
                     group=f"{source}:{session}", subset=subset, n_samples=16_000, labels="",
                     attributes={"session": session})


def test_uav16_trains_in_the_pool_except_held_batches_and_all_but_the_plain_backgrounds():
    m = [_rec("nightjar", "n1", 1), _rec("uav16", "u1", 1, "train"), _rec("uav16", "rec-undated-0002", 0, "train"),
         _rec("uav16", "u3", 1, "held"), _rec("lund", "l1", 1), _rec("uav16", "rec-undated-0007", 0, "train"),
         _rec("uav16", "rec-20261012-0001", 0, "train")]
    arm = rg.make_pool_arm(m, "none")
    assert {r.recording_id for r in arm.train} == {"n1", "u1", "rec-undated-0002"}
    assert {k: [r.recording_id for r in v] for k, v in arm.unseen.items() if k != "lund"} == {
        "uav16": ["u3"], "uav16_confusers": ["rec-undated-0007"], "uav16_elsewhere": ["rec-20261012-0001"]}
    assert set(rg.make_pool_arm([m[0], m[4]], "none").unseen) == {"lund"}   # a manifest without it is unchanged
    without = rg.make_pool_arm(m, "none", uav16=False)                      # iteration 23: scored, not trained
    assert {r.recording_id for r in without.train} == {"n1"} and set(without.unseen) == set(arm.unseen)


def test_evaluate_chain_reports_false_alarms_for_a_set_without_drones():
    recs = [_rec("uav16", "rec-undated-0007", 0)]
    cw = ch.ChainWindows(recs, np.zeros(40, int), np.arange(40) * 8000, np.zeros(40, int), np.full(40, np.nan),
                         np.full(40, "s", dtype=object))
    out = rg.evaluate_chain(rg.Scored(cw, np.r_[np.full(10, 0.9), np.full(30, 0.1)]), 0.5, 1.0)
    assert out["deploy_fa"] == 0.25


def test_prepare_uav16_keeps_one_channel_and_skips_the_replays(tmp_path, monkeypatch):
    import csv
    import json
    import sys
    from pathlib import Path

    import soundfile as sf
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import prepare_data as pd_
    import prepare_new_data as pn

    root = tmp_path / "dataset"
    rows = [("rec-a", "ses-1", "self_collected", "true", "30.0", "40.0", "batch-20260522", "real_uav"),
            ("rec-b", "ses-2", "self_collected", "false", "", "", "batch-undated", "real_background"),
            ("rec-c", "ses-3", "controlled_replay", "false", "", "", "batch-20261013", "replayed_non_uav")]
    (root / "metadata").mkdir(parents=True)
    with open(root / "metadata" / "recordings.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["recording_id", "session_id", "data_origin", "has_uav", "horizontal_distance_m", "height_m",
                    "acquisition_batch_id", "scene_type", "qc_status", "channel_count", "azimuth_deg",
                    "nominal_condition_id"])
        for rid, ses, origin, uav, d, h, batch, folder in rows:
            w.writerow([rid, ses, origin, uav, d, h, batch, "scene", "pass", 16, "0.0" if d else "", "c"])
            p = root / "data" / folder / ses / rid
            p.mkdir(parents=True)
            sf.write(str(p / f"{rid}__ch-00.wav"), np.full(48_000, 0.01), 48_000, subtype="PCM_16")
    monkeypatch.setattr(pn, "U16_ROOT", root)
    monkeypatch.setattr(pd_, "OUT", tmp_path)
    manifest, summary = [], {}
    pn.prepare_uav16(manifest, [], summary)
    assert [(r["recording_id"], r["label"], r["subset"], r["n_samples"]) for r in manifest] == [
        ("rec-a", 1, "held", 16_000), ("rec-b", 0, "train", 16_000)]
    assert json.loads(manifest[0]["attributes"])["range_m"] == 50.0      # hypot(30, 40)
    assert summary["uav16"]["replay_skipped"] == 1
    assert (tmp_path / "uav16.i16").stat().st_size == 2 * 16_000 * 2
