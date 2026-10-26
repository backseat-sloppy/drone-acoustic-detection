"""Write WAV clips from data/processed/ for upload to Edge Impulse.

Only sources whose licence allows upload to a third-party cloud are used:
Nightjar (CC BY), DroneNoise (CC BY) and Halmstad (CC0). The split into
training and testing is made here, by source and session, so Edge Impulse's
own random split is never needed (it would leak neighbouring clips):

    training   Nightjar sessions except the held-out ones (uploaded)
    validation one Nightjar session, kept local, to freeze the threshold
    testing   held-out Nightjar sessions, Halmstad, DroneNoise (one microphone per event)

Clips are cut from the drone (label 1) and no-drone (label 0) stretches of the
manifest; uncertain stretches (label -1, the guard bands) are skipped. The
label goes in front of the file name, as Edge Impulse's uploader expects:
`drone.<id>.wav`, `no_drone.<id>.wav`. Format only: 16 kHz mono int16.

Run after prepare_data.py (dronenoise halmstad) and prepare_new_data.py (nightjar):

    uv run python -I scripts/export_edge_impulse.py
"""
from __future__ import annotations

import collections
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16_000
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "edge_impulse"

CLIP_S = 10.0                # Edge Impulse windows each clip itself
MAX_S_PER_GROUP_LABEL = 900  # caps each (session, label) so no session dominates
NJ_TEST = {"nightjar:nightjar-flight-20260804-3",      # model 6's validation session
           "nightjar:nightjar-lawnmower1-20260805"}    # mower: a confuser never trained on
NJ_VAL = {"nightjar:nightjar-flight-20260806-1"}       # never uploaded; freezes the threshold
LABEL = {1: "drone", 0: "no_drone"}
CATEGORIES = ("training", "validation", "testing")


def category(r: dict) -> str | None:
    if r["source"] == "nightjar":
        return "testing" if r["group"] in NJ_TEST else "validation" if r["group"] in NJ_VAL else "training"
    if r["source"] == "halmstad":
        return "testing"
    if r["source"] == "dronenoise":
        # The nine microphones hear the same event; keep one so test clips stay independent.
        return "testing" if r["subset"] == "M5" else None
    return None


def clips(n: int, max_n: int) -> list[int]:
    """Start samples of whole CLIP_S clips in [0, n), evenly spread, at most max_n samples in all."""
    c = int(CLIP_S * SR)
    starts = list(range(0, n - c + 1, c))
    k = max(1, max_n // c)
    if len(starts) > k:
        starts = [starts[i] for i in np.linspace(0, len(starts) - 1, k).round().astype(int)]
    return starts


def main() -> None:
    with open(PROCESSED / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if category(r) and int(r["label"]) in LABEL]
    if OUT.exists():
        shutil.rmtree(OUT)
    for cat in CATEGORIES:
        (OUT / cat).mkdir(parents=True)

    by_key = collections.defaultdict(list)
    for r in rows:
        by_key[(r["group"], int(r["label"]))].append(r)

    stats = collections.Counter()
    for (group, lab), rs in sorted(by_key.items()):
        budget = int(MAX_S_PER_GROUP_LABEL * SR)
        total = sum(int(r["n_samples"]) for r in rs)
        for r in rs:
            n = int(r["n_samples"])
            share = budget * n // max(total, 1)          # spread the cap over the group's stretches
            dtype = np.int16 if r["dtype"] == "int16" else np.float32
            x = np.memmap(PROCESSED / r["file"], dtype=dtype, mode="r", offset=int(r["offset"]) * np.dtype(dtype).itemsize, shape=(n,))
            cat = category(r)
            for s in clips(n, share):
                y = np.asarray(x[s:s + int(CLIP_S * SR)])
                if dtype == np.float32:
                    y = np.clip(np.round(y * 32767), -32768, 32767).astype(np.int16)
                rid = r["recording_id"].replace("/", "_").replace("#", "_").replace(".", "_")
                path = OUT / cat / f"{LABEL[lab]}.{r['source']}_{rid}_{s // SR:05d}.wav"
                sf.write(path, y, SR, subtype="PCM_16")
                # Sidecar for `dad wav`, so the repo's models are scored on the same clips.
                drones = [{"start_s": 0.0, "end_s": CLIP_S}] if lab == 1 else []
                path.with_suffix(".json").write_text(json.dumps({"drones": drones, "source": r["source"], "kind": r["kind"]}))
                stats[(cat, r["source"], LABEL[lab])] += 1
    for k, v in sorted(stats.items()):
        print(*k, v, f"({v * CLIP_S / 60:.1f} min)")
    assert stats, "nothing exported: run the prepare scripts first"
    pack()


def pack(max_clips: int = 29) -> None:
    """Join clips of one session and label into files under 10 MB, for the browser uploader
    (one upload call takes at most 10 MB). Same audio; only the clip-level files are scored."""
    for cat in ("training", "testing"):
        out = OUT / "packed" / cat
        out.mkdir(parents=True)
        groups = collections.defaultdict(list)
        for p in sorted((OUT / cat).glob("*.wav")):
            label, rest = p.stem.split(".", 1)
            groups[(label, rest.rsplit("_", 1)[0].split("_uma16")[0])].append(p)
        for (label, key), ps in groups.items():
            for i in range(0, len(ps), max_clips):
                y = np.concatenate([sf.read(p, dtype="int16")[0] for p in ps[i:i + max_clips]])
                sf.write(out / f"{label}.{key}_part{i // max_clips:02d}.wav", y, SR, subtype="PCM_16")


if __name__ == "__main__":
    main()
    sys.exit(0)
