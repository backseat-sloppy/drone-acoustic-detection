"""Iteration 22, probe B2, step 1 (this PC, no training): a compact 22.05 kHz store for the band network probe.

    uv run python -I scripts/it22_band_store.py      # writes artifacts/it22_band/windows.flac + windows.csv

Raw Nightjar (UMA-16 channel 0, 48 kHz) and FPV (channel 0, 44.1/48 kHz) rows of the manifest with label 0 or
1, read at the original rate through load_manifest's sealed-data guard, resampled to 22,050 Hz, cut into
non-overlapping 1 s windows, at most PER_ROW per row chosen at random. The windows are written back to back
into one FLAC file (int16) with one CSV row each: source, group, recording, label.
"""
import csv
import io
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dad.realdata import load_manifest  # noqa: E402

SR, PER_ROW, SEED = 22_050, 120, 22
OUT = ROOT / "artifacts" / "it22_band"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    z = zipfile.ZipFile(ROOT / "data" / "Acoustic_UAV_Field_Recordings" / "Fpv.zip")
    meta, n_win = [], 0
    with sf.SoundFile(str(OUT / "windows.flac"), "w", SR, 1, subtype="PCM_16") as out:
        for r in load_manifest():
            if r.source not in ("nightjar", "fpv") or r.label not in (0, 1):
                continue
            att = r.attributes
            if r.source == "nightjar":
                path = ROOT / "data" / "Nightjar_HF" / att["session"] / "audio" / att["parent"]
                sr = sf.info(str(path)).samplerate
                a0 = int(round(att["parent_start_s"] * sr))
                x = sf.read(str(path), start=a0, stop=a0 + int(r.seconds * sr), dtype="float32", always_2d=True)[0][:, 0]
            else:
                x, sr = sf.read(io.BytesIO(z.read(att["file"])), dtype="float32", always_2d=True)
                x = x[:, 0]
            g = math.gcd(sr, SR)
            y = resample_poly(x, SR // g, sr // g).astype(np.float32)
            n = len(y) // SR
            if n < 1:
                continue
            for i in np.sort(rng.choice(n, size=min(n, PER_ROW), replace=False)):
                out.write(np.clip(y[i * SR:(i + 1) * SR], -1.0, 1.0))
                meta.append((r.source, r.group, r.recording_id, r.label))
                n_win += 1
    with open(OUT / "windows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "group", "recording_id", "label"])
        w.writerows(meta)
    lab = np.array([m[3] for m in meta])
    print(f"{n_win} windows ({int(lab.sum())} drone), {len({m[1] for m in meta})} groups -> {OUT}")


if __name__ == "__main__":
    main()
