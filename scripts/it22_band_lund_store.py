"""Iteration 22, probe B3, step 1 (this PC, no training): Lund drone windows at 22.05 kHz with their range.

    uv run python -I scripts/it22_band_lund_store.py   # writes artifacts/it22_band/lund.flac + lund.csv

Lund is never trained on (rangerun: "scored for recall only"). Every Lund manifest row with label 1, except
the suspect recording 10_flower3, channel 0 of the raw 96 kHz WAV resampled to 22,050 Hz, up to PER_ROW
non-overlapping 1 s windows per row at random, each with its slant range from range_track.csv.
"""
import csv
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dad.chains import range_tracks  # noqa: E402
from dad.rangerun import LUND_SUSPECT  # noqa: E402
from dad.realdata import load_manifest  # noqa: E402
from prepare_new_data import LUND_CHANNEL, LUND_ZIP, _read_pcm_channel  # noqa: E402

SR, PER_ROW, SEED = 22_050, 150, 22
OUT = ROOT / "artifacts" / "it22_band"


def main() -> None:
    rng = np.random.default_rng(SEED)
    tracks = range_tracks()
    z = zipfile.ZipFile(LUND_ZIP)
    rows = [r for r in load_manifest() if r.source == "lund" and r.label == 1
            and r.attributes["session"] not in LUND_SUSPECT]
    raw: dict[str, tuple[np.ndarray, int]] = {}
    meta = []
    with sf.SoundFile(str(OUT / "lund.flac"), "w", SR, 1, subtype="PCM_16") as out:
        for r in rows:
            sess = r.attributes["session"]
            if sess not in raw:
                x, sr, _ = _read_pcm_channel(z, f"ljungbyhed_audio_dataset/{sess}/sound.wav", LUND_CHANNEL)
                g = math.gcd(sr, SR)
                raw = {sess: (resample_poly(x.astype(np.float64), SR // g, sr // g).astype(np.float32), SR)}
                print(f"{sess}: {len(x) / sr:.0f} s read, {len(meta)} windows so far", flush=True)
            y = raw[sess][0]
            a0 = int(round(r.attributes["parent_start_s"] * SR))
            seg = y[a0:a0 + int(r.seconds * SR)]
            n = len(seg) // SR
            t, d = tracks[r.recording_id]
            for i in np.sort(rng.choice(n, size=min(n, PER_ROW), replace=False)):
                out.write(np.clip(seg[i * SR:(i + 1) * SR], -1.0, 1.0))
                meta.append((sess, r.recording_id, round(float(np.interp(i + 0.5, t, d)), 1)))
    with open(OUT / "lund.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["session", "recording_id", "range_m"])
        w.writerows(meta)
    rng_m = np.array([m[2] for m in meta])
    print(f"{len(meta)} Lund drone windows from {len({m[0] for m in meta})} recordings, range "
          f"{rng_m.min():.0f}-{rng_m.max():.0f} m (median {np.median(rng_m):.0f})")


if __name__ == "__main__":
    main()
