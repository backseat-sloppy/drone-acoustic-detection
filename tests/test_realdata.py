"""The processed-data reader, exercised on a tiny store built in a temp folder."""
from __future__ import annotations

import csv

import numpy as np
import pytest

from dad.realdata import load_manifest, window_starts

FIELDS = ["source", "recording_id", "file", "dtype", "offset", "n_samples", "duration_s",
          "label", "kind", "group", "subset", "labels", "attributes"]


@pytest.fixture
def store(tmp_path):
    a = np.linspace(-0.5, 0.5, 100, dtype=np.float32)
    b = np.arange(-50, 50, dtype=np.int16) * 300
    np.concatenate([np.zeros(7, np.float32), a]).tofile(tmp_path / "x.f32")
    b.tofile(tmp_path / "y.i16")
    rows = [
        dict(source="ddl", recording_id="s1", file="x.f32", dtype="float32", offset=7, n_samples=100,
             duration_s=0.00625, label=1, kind="drone", group="ddl:s1", subset="2021-03-29",
             labels="MINI", attributes='{"model": "DJI Mini 2"}'),
        dict(source="fsd50k", recording_id="42", file="y.i16", dtype="int16", offset=0, n_samples=100,
             duration_s=0.00625, label=0, kind="Wind", group="fsd50k:someone", subset="train",
             labels="Wind", attributes="{}"),
    ]
    with open(tmp_path / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return tmp_path, a, b


def test_reads_float_store_at_its_offset(store):
    root, a, _ = store
    rec = load_manifest(root)[0]
    assert rec.label == 1 and rec.attributes["model"] == "DJI Mini 2"
    np.testing.assert_array_equal(rec.audio(), a)
    np.testing.assert_array_equal(rec.audio(10, 5), a[10:15])


def test_int16_store_is_scaled_to_unit_range(store):
    root, _, b = store
    x = load_manifest(root)[1].audio()
    assert x.dtype == np.float32
    np.testing.assert_allclose(x, b / 32768.0)
    assert np.abs(x).max() < 1.0


def test_reading_past_the_recording_raises(store):
    root, _, _ = store
    with pytest.raises(ValueError):
        load_manifest(root)[0].audio(90, 20)


def test_window_starts_cover_only_full_windows():
    assert window_starts(10, 4, 3).tolist() == [0, 3, 6]
    assert window_starts(3, 4, 1).size == 0
