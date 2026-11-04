"""Export every no-drone recording (label 0) of data/processed, plus the no-drone stems, as WAV + CSV.

    python export_nodrone.py REPO OUT

Samples are written exactly as stored (int16 stores -> 16-bit WAV, float32 stores -> 32-bit float WAV),
16 kHz mono, never rescaled. Drone stems (stems/drone) are left out.
"""
import csv
import shutil
import sys
import wave
from pathlib import Path

import numpy as np

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
P = repo / "data" / "processed"
SR = 16_000
out.mkdir(parents=True, exist_ok=True)


def write_wav(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    if x.dtype == np.int16:
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(x.tobytes())
    else:   # 32-bit IEEE float WAV (format tag 3), which the wave module cannot write
        data = x.astype("<f4").tobytes()
        hdr = (b"RIFF" + (4 + 26 + 12 + 8 + len(data)).to_bytes(4, "little") + b"WAVE"
               + b"fmt " + (18).to_bytes(4, "little") + (3).to_bytes(2, "little") + (1).to_bytes(2, "little")
               + SR.to_bytes(4, "little") + (SR * 4).to_bytes(4, "little") + (4).to_bytes(2, "little")
               + (32).to_bytes(2, "little") + (0).to_bytes(2, "little")
               + b"fact" + (4).to_bytes(4, "little") + (len(x)).to_bytes(4, "little")
               + b"data" + len(data).to_bytes(4, "little"))
        path.write_bytes(hdr + data)


rows = [r for r in csv.DictReader(open(P / "manifest.csv", newline="", encoding="utf-8")) if r["label"] == "0"]
stores, kept, seconds = {}, [], {}
for r in rows:
    f = P / r["file"]
    if not f.exists():
        print("missing store, skipped:", r["file"]); continue
    if r["file"] not in stores:
        stores[r["file"]] = np.memmap(f, dtype=r["dtype"], mode="r")
    o, n = int(r["offset"]), int(r["n_samples"])
    x = np.array(stores[r["file"]][o:o + n])
    name = "".join(c if c.isalnum() or c in "._-" else "_" for c in r["recording_id"]) + ".wav"
    rel = Path("recordings") / r["source"] / name
    write_wav(out / rel, x)
    kept.append({"wav": rel.as_posix(), **{k: v for k, v in r.items() if k not in ("file", "dtype", "offset")}})
    seconds[r["source"]] = seconds.get(r["source"], 0) + n / SR
with open(out / "recordings.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, kept[0].keys()); w.writeheader(); w.writerows(kept)
for s, t in sorted(seconds.items()):
    print(f"{s:<10} {sum(k['source'] == s for k in kept):>6} files {t / 60:8.1f} min")

S = P / "stems"
if S.exists():
    st = [r for r in csv.DictReader(open(S / "manifest.csv", newline="", encoding="utf-8")) if r["group"] != "drone"]
    for r in st:
        src = repo / r["file"]
        rel = Path("stems") / Path(r["file"]).relative_to("data/processed/stems")
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / rel)
        r["file"] = rel.as_posix()
    with open(out / "stems.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, st[0].keys()); w.writeheader(); w.writerows(st)
    shutil.copy2(S / "ATTRIBUTION.md", out / "stems_ATTRIBUTION.md")
    print(f"stems      {len(st):>6} files (drone stems left out)")
else:
    print("no stems folder")
