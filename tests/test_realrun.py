"""The real-data training pieces, on a tiny store built in a temp folder."""
from __future__ import annotations

import csv

import numpy as np
import pytest

from dad import realrun as rr
from dad.config import DEFAULT
from dad.features import LogMel
from dad.realdata import load_manifest

SR = 16_000
FIELDS = ["source", "recording_id", "file", "dtype", "offset", "n_samples", "duration_s",
          "label", "kind", "group", "subset", "labels", "attributes"]


@pytest.fixture
def splits(tmp_path):
    rng = np.random.default_rng(0)
    rows, parts, offset = [], [], 0

    def add(source, rid, seconds, label, kind, group, subset, scale):
        nonlocal offset
        x = (rng.standard_normal(int(seconds * SR)) * scale).astype(np.float32)
        parts.append(x)
        rows.append(dict(source=source, recording_id=rid, file="a.f32", dtype="float32", offset=offset,
                         n_samples=len(x), duration_s=seconds, label=label, kind=kind, group=group,
                         subset=subset, labels=kind, attributes="{}"))
        offset += len(x)

    add("ddl", "290321-T004", 4, 1, "drone", "ddl:290321-T004", "2021-03-29", 0.002)
    add("ddl", "290321-T005", 3, 1, "drone", "ddl:290321-T005", "2021-03-29", 0.002)  # a validation session
    add("fsd50k", "1", 3, 0, "Wind", "fsd50k:ann", "train", 0.2)
    add("fsd50k", "2", 2, 0, "Rain", "fsd50k:bob", "train", 0.1)
    add("fsd50k", "3", 0.5, 0, "Bird", "fsd50k:bob", "train", 0.1)   # shorter than a window
    add("fsd50k", "4", 2, 0, "Engine", "fsd50k:ann", "val", 0.1)     # uploader also in training
    add("fsd50k", "5", 2, 0, "Engine", "fsd50k:cat", "val", 0.1)
    np.concatenate(parts).tofile(tmp_path / "a.f32")
    with open(tmp_path / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rr.make_splits(load_manifest(tmp_path))


def sampler(splits, variant, seed=0):
    pool = rr.ddl_pool(splits.ddl_train, DEFAULT.samples_per_window)
    return rr.WindowSampler([pool], splits.fsd_train, DEFAULT, variant, seed=seed)


def test_splits_keep_groups_apart_and_drop_short_clips(splits):
    assert [r.recording_id for r in splits.ddl_train] == ["290321-T004"]
    assert [r.recording_id for r in splits.ddl_val] == ["290321-T005"]
    assert [r.recording_id for r in splits.fsd_train] == ["1", "2"]
    # Clip 4 is a validation clip from an uploader who is also in training.
    assert [r.recording_id for r in splits.fsd_val] == ["5"]


def test_level_normalise_removes_gain():
    fe = LogMel(DEFAULT)
    x = np.random.default_rng(1).standard_normal(DEFAULT.samples_per_window).astype(np.float32) * 0.01
    a = rr.level_normalise(fe.frames(x)[None])
    b = rr.level_normalise(fe.frames(x * 100.0)[None])
    np.testing.assert_allclose(a, b, atol=1e-3)
    assert abs(rr.level_db(a)[0]) < 0.01


def test_level_normalise_hides_digital_silence():
    fe = LogMel(DEFAULT)
    x = np.random.default_rng(2).standard_normal(DEFAULT.samples_per_window).astype(np.float32)
    x[: len(x) // 2] = 0.0
    assert rr.level_normalise(fe.frames(x)[None]).min() >= rr._FLOOR


def test_inband_power_ignores_content_below_the_mel_bank():
    t = np.arange(SR) / SR
    rumble = np.sin(2 * np.pi * 8 * t)
    tone = 0.01 * np.sin(2 * np.pi * 400 * t)
    assert rr.inband_power(rumble + tone) < 20 * rr.inband_power(tone)


@pytest.mark.parametrize("variant", ["level", "full"])
def test_batches_are_reproducible_balanced_and_finite(splits, variant):
    s = sampler(splits, rr.VARIANTS[variant], seed=3)
    L1, y1 = s.batch(5, 8)
    L2, y2 = s.batch(5, 8)
    np.testing.assert_array_equal(L1, L2)
    assert y1.tolist() == [0, 1] * 4
    assert L1.shape == (8, DEFAULT.n_mels, 98) and np.isfinite(L1).all()
    assert not np.array_equal(L1, s.batch(6, 8)[0])


def test_mixing_sets_the_requested_signal_to_noise_ratio(splits):
    v = rr.Variant("fixed", mix=True, mix_prob_pos=1.0, snr_db=(10.0, 10.0))
    s = sampler(splits, v)
    clean = sampler(splits, rr.VARIANTS["level"])
    for i in range(5):
        d = clean.positive(np.random.default_rng(i))
        mixed = s.positive(np.random.default_rng(i))
        # White noise on both sides: the sum's power is drone plus background.
        ratio = rr.inband_power(d) / (rr.inband_power(mixed) - rr.inband_power(d))
        assert 10 * np.log10(ratio) == pytest.approx(10.0, abs=1.0)


def test_prefetch_returns_batches_in_order(splits):
    s = sampler(splits, rr.VARIANTS["level"])
    got = [L for L, _ in rr.prefetch(s, 6, 4, workers=3)]
    for i, L in enumerate(got):
        np.testing.assert_array_equal(L, s.batch(i, 4)[0])


def test_wilson_interval_is_sane_at_the_edges():
    lo, hi = rr.wilson(0, 59)
    assert lo == 0.0 and 0.05 < hi < 0.07   # zero false alarms in 59 clips bounds the rate near 6%
    lo, hi = rr.wilson(27, 30)
    assert 0.73 < lo < 0.76 and 0.96 < hi < 0.98


def test_paired_probe_separates_a_real_detector_from_a_blind_one():
    rng = np.random.default_rng(0)
    backgrounds = rng.standard_normal((40, SR)).astype(np.float32) * 0.1
    t = np.arange(SR) / SR
    drones = np.stack([np.sin(2 * np.pi * f * t) for f in (180.0, 200.0, 220.0)]).astype(np.float32)

    class HearsTone:  # energy near 200 Hz relative to the whole window
        def score(self, audio):
            spec = np.abs(np.fft.rfft(audio, axis=1)) ** 2
            return spec[:, 170:231].sum(axis=1) / spec.sum(axis=1)

    class HearsNothing:  # the same answer whatever it is given
        def score(self, audio):
            return np.full(len(audio), 0.5)

    seen = rr.paired_probe(HearsTone(), backgrounds, drones, snrs_db=(0.0,), n_pairs=40)["+0 dB"]
    blind = rr.paired_probe(HearsNothing(), backgrounds, drones, snrs_db=(0.0,), n_pairs=40)["+0 dB"]
    assert seen["auc"] > 0.99 and seen["raised"] == 1.0
    assert blind["auc"] == pytest.approx(0.5) and blind["raised"] == 0.0


def test_ddl_pool_without_a_range_limit_covers_every_start(splits):
    pool = rr.ddl_pool(splits.ddl_train, DEFAULT.samples_per_window)
    assert pool.seconds == pytest.approx(3.1, abs=0.1)   # a 4 s recording, 1 s windows, 100 ms grid
    assert pool.draw(np.random.default_rng(0)).shape == (DEFAULT.samples_per_window,)
