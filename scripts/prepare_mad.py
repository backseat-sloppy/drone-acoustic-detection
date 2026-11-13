"""Iteration 24: MAD's training videos as one float32 store for the `simdist_colour_mad` recipe.

    .venv/bin/python scripts/prepare_mad.py      # data/extracted/MAD/MAD_dataset -> data/processed/mad/

Split by video (the YouTube URL without its time stamp), never by clip. Every video with a clip in MAD's
own test.csv is held out (51 videos; 4 of them also have clips in training.csv), so no clip of a held-out
video is written here. The held-out list goes to heldout_videos.txt for scoring (scripts/mad_nodrone.py).
"""
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dad.sealed import forbid_in_training  # noqa: E402

SRC = ROOT / "data/extracted/MAD/MAD_dataset"
OUT = ROOT / "data/processed/mad"
CLASSES = ["communication", "shooting", "footsteps", "shelling", "vehicle", "helicopter", "fighter"]


def video(r):
    return r["youtube url"].split("&")[0]


rows = {s: list(csv.DictReader(open(SRC / f"{s}.csv", newline="", encoding="utf-8"))) for s in ("training", "test")}
held = {video(r) for r in rows["test"]}
train = [r for r in rows["training"] if video(r) not in held]
OUT.mkdir(parents=True, exist_ok=True)
off = 0
with open(OUT / "train.f32", "wb") as f, open(OUT / "train.csv", "w", newline="", encoding="utf-8") as m:
    w = csv.writer(m)
    w.writerow(["path", "class", "video", "offset", "n_samples"])
    for r in train:
        forbid_in_training(r["path"], r["youtube title"], where="prepare_mad")
        x, fs = sf.read(SRC / r["path"], dtype="float32", always_2d=True)
        assert fs == 16000, (r["path"], fs)
        x = np.ascontiguousarray(x[:, 0])
        f.write(x.tobytes())
        w.writerow([r["path"], CLASSES[int(r["label"])], video(r), off, len(x)])
        off += len(x)
(OUT / "heldout_videos.txt").write_text("\n".join(sorted(held)) + "\n", encoding="utf-8")
print(f"{len(train)} clips from {len({video(r) for r in train})} training videos, {off / 16000 / 3600:.2f} h; "
      f"{len(held)} videos held out")
