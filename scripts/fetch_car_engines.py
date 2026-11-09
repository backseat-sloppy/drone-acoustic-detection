"""The six engine recordings Pure-data+Script/CarDistractor/car_distractor.pd reads.

Each clip is read from the FSD50K dev archive on Zenodo by range request (only that clip is downloaded),
and treated as build_stems.py treats a distracter: at most the loudest 10 s, spectral gate unless it costs
more than 1 dB of the loudest frames, active RMS -26 dBFS. Written to
data/processed/stems/distracter/idling/fsd50k_dev_<id>.wav and added to manifest.csv.

    uv run python scripts/fetch_car_engines.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import build_stems as bs  # noqa: E402

fz = bs.fz
IDS = ["369049", "369053", "387676", "387677", "387679", "401552"]
OUT = bs.OUT / "distracter" / "idling"


def main():
    gt = {r["fname"]: r["labels"] for r in csv.DictReader(fz.open_meta("FSD50K.ground_truth/dev.csv"))}
    info = json.loads(fz.open_meta("FSD50K.metadata/dev_clips_info_FSD50K.json").read())
    cd = {Path(k).stem: v for k, v in fz.central_directory(fz.DEV_PARTS).items() if k.endswith(".wav")}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in IDS:
        meta = info[i]
        if not any(a in meta["license"] for a in fz.ALLOWED):
            raise SystemExit(f"{i}: licence {meta['license']} is not CC0 or CC BY")
        disk, off, csize, method, _ = cd[i]
        full = bs.read_mono(fz.read_member(disk, off, csize, method, fz.DEV_PARTS))
        profile = np.percentile(np.abs(bs.spec(full)[2]), 20, axis=1)     # from the whole clip, as build_stems
        x = bs.loudest(full, 10.0)
        before = bs.active_rms_db(x)
        y, _ = bs.gate(x, profile)
        removed = float(np.percentile(bs.frame_db(x), 10) - np.percentile(bs.frame_db(y), 10))
        kept, policy = bs.events_kept_db(x, y), "percentile"
        if kept < -1.0:                                      # a steady engine is its own floor
            policy, y, removed = "percentile->none", x, 0.0
        y, gain = bs.finish(y, -26.0)
        path = OUT / f"fsd50k_dev_{i}.wav"
        sf.write(path, y, bs.FS, subtype="PCM_24")
        rows.append(dict(group="distracter", cls="idling", source="fsd50k_dev", id=i, file=str(path.relative_to(bs.ROOT)),
                         seconds=round(len(y) / bs.FS, 2), denoise=policy, noise_removed_db=round(removed, 1),
                         content_kept_db=round(kept, 2), active_rms_in_dbfs=round(before, 1), gain_db=round(gain, 1),
                         active_rms_out_dbfs=round(bs.active_rms_db(y), 1),
                         peak_dbfs=round(20 * np.log10(np.abs(y).max()), 1), license=meta["license"],
                         uploader=meta["uploader"], title=meta["title"], labels=gt[i]))
        print(f"{i}: {csize / 1e6:.2f} MB read, {len(full) / bs.FS:.1f} s, {policy}, kept {kept:+.2f} dB, {meta['license']}, {meta['title']}")
    manifest = bs.OUT / "manifest.csv"
    with open(manifest, encoding="utf-8", newline="") as f:
        old = [r for r in csv.DictReader(f) if not (r["cls"] == "idling" and r["id"] in IDS)]
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in old + rows for k in r)))
        w.writeheader()
        w.writerows(old + rows)
    print(f"{len(rows)} stems -> {OUT}; manifest {len(old) + len(rows)} rows")


if __name__ == "__main__":
    main()
