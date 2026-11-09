"""ITU ARIS must never reach training: every reader that feeds training stops on it."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

import dad.dronesynth as ds
from dad.realdata import load_manifest
from dad.sealed import SealedDataError, forbid_in_training

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
FIELDS = ["source", "recording_id", "file", "dtype", "offset", "n_samples", "duration_s",
          "label", "kind", "group", "subset", "labels", "attributes"]


def script(name: str):
    spec = importlib.util.spec_from_file_location(f"_sealed_test_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@pytest.mark.parametrize("name", [
    "data/extracted/ITU_ARIS_Lab/itu-aris-lab-acoustic-drone-dataset/audio/drone/x.wav",
    r"C:\data\extracted\sealed_peek\Drone_03.wav",
    "data/extracted/steady_peek/Steady_07.wav",
    r"data\extracted\All\ITU_ARIS_interval_18.wav",
    "data/extracted/All/interval_18.wav",
    "CleanDrone.wav",
    "aris_peek_03",
    "ITU ARIS",
    "aris",
])
def test_every_name_of_the_set_is_stopped(name):
    with pytest.raises(SealedDataError, match="never for training"):
        forbid_in_training(name, where="test")


@pytest.mark.parametrize("name", [
    "comparison", "Paris_street.wav", "variance", "ddl", "fsd50k", "nightjar", "halmstad",
    "data/extracted/nodrone/recordings/fpv/a.wav", "data/processed/stems/war/x.wav",
    "data/extracted/allowed/x.wav", "stem:background",
])
def test_other_sources_pass(name):
    forbid_in_training(name, where="test")


def test_load_manifest_stops_on_an_aris_row(tmp_path):
    row = dict.fromkeys(FIELDS, "")
    row.update(source="itu_aris", recording_id="r1", file="x.f32", dtype="float32", offset=0,
               n_samples=100, label=1, group="g", attributes="{}")
    write_csv(tmp_path / "manifest.csv", [row])
    with pytest.raises(SealedDataError):
        load_manifest(tmp_path)


def test_synth_stems_stop_on_an_aris_file(tmp_path, monkeypatch):
    write_csv(tmp_path / "manifest.csv",
              [dict(source="x", group="background", file="data/extracted/sealed_peek/Fugle.wav")])
    monkeypatch.setattr(ds, "STEMS", tmp_path)
    with pytest.raises(SealedDataError):
        ds._load_stems(("background",), 16_000)


def test_scene_stems_stop_on_an_aris_file(tmp_path, monkeypatch):
    make_scenes = script("make_scenes")
    write_csv(tmp_path / "manifest.csv",
              [dict(source="itu_aris", cls="wind", file="data/processed/stems/wind/a.wav")])
    monkeypatch.setattr(make_scenes, "STEMS", tmp_path / "manifest.csv")
    with pytest.raises(SealedDataError):
        make_scenes.load_stems(None)


def test_train_clean_stops_on_an_aris_drone_scene(tmp_path):
    train_clean = script("train_clean")
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "s.json").write_text(json.dumps(
        {"seed": 0, "events": [{"role": "drone", "onset": 0, "offset": 1, "file": "CleanDrone.wav"}]}))
    with pytest.raises(SealedDataError):
        train_clean.load_drones(tmp_path, 1800, None)


def test_train_clean_stops_on_an_aris_negative_even_past_the_limit(tmp_path):
    train_clean = script("train_clean")
    write_csv(tmp_path / "recordings.csv", [dict(wav="recordings/a.wav", source="fpv"),
                                            dict(wav="recordings/b.wav", source="fpv"),
                                            dict(wav="sealed_peek/Fugle.wav", source="fpv")])
    write_csv(tmp_path / "stems.csv", [dict(file="stems/a.wav", group="wind")])
    with pytest.raises(SealedDataError):  # limit 1 would only read the first two rows
        train_clean.load_nodrone(tmp_path, "nightjar", 1)
