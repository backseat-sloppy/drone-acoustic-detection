"""Real recordings from data/processed, as written by scripts/prepare_data.py.

The store is a handful of flat binary files plus manifest.csv. Nothing here
changes the audio: a `Recording` hands back exactly what was stored, as
float32 in [-1, 1). Level handling, mixing and every other training-time
choice lives with the training code, where it can be varied and logged.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
SAMPLE_RATE = 16_000


@lru_cache(maxsize=8)
def _store(path: str, dtype: str) -> np.memmap:
    return np.memmap(path, dtype=dtype, mode="r")


@dataclass(frozen=True)
class Recording:
    source: str        # ddl | dronenoise | halmstad | fsd50k
    recording_id: str
    label: int         # 1 = drone present
    kind: str          # drone, helicopter, background, or the FSD50K leaf label
    group: str         # everything sharing a group stays on one side of a split
    subset: str        # DDL: recording date; DroneNoise: microphone; FSD50K: train/val/eval
    n_samples: int
    labels: str
    attributes: dict = field(compare=False, hash=False)
    _file: str = field(repr=False, compare=False, default="")
    _dtype: str = field(repr=False, compare=False, default="float32")
    _offset: int = field(repr=False, compare=False, default=0)

    @property
    def seconds(self) -> float:
        return self.n_samples / SAMPLE_RATE

    def audio(self, start: int = 0, n: int | None = None) -> np.ndarray:
        """Samples [start, start + n) as float32. Out-of-range requests raise."""
        n = self.n_samples - start if n is None else n
        if start < 0 or n < 0 or start + n > self.n_samples:
            raise ValueError(f"{self.recording_id}: [{start}, {start + n}) is outside 0..{self.n_samples}")
        raw = _store(self._file, self._dtype)[self._offset + start: self._offset + start + n]
        x = np.asarray(raw, dtype=np.float32)
        return x / 32768.0 if self._dtype == "int16" else x


def load_manifest(root: Path | str = PROCESSED) -> list[Recording]:
    root = Path(root)
    out: list[Recording] = []
    with open(root / "manifest.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append(Recording(
                source=r["source"], recording_id=r["recording_id"], label=int(r["label"]),
                kind=r["kind"], group=r["group"], subset=r["subset"],
                n_samples=int(r["n_samples"]), labels=r["labels"],
                attributes=json.loads(r["attributes"] or "{}"),
                _file=str(root / r["file"]), _dtype=r["dtype"], _offset=int(r["offset"]),
            ))
    return out


def window_starts(n_samples: int, window: int, hop: int) -> np.ndarray:
    """Start indices of every full window. Empty if the recording is too short."""
    if n_samples < window:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, n_samples - window + 1, hop, dtype=np.int64)
