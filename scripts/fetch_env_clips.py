"""Fetch a handful of FSD50K eval clips without downloading the 6 GB archive.

FSD50K.eval_audio is a split zip on Zenodo (.z01 + .zip). The central
directory sits at the end of the .zip part, so HTTP range requests read it,
then read only the chosen members. The clips land in
data/extracted/fsd50k_env/ (derived data, CLAUDE.md section 6), together with
a sources.csv that records each clip's FSD50K id, labels, licence and
uploader.

    uv run python scripts/fetch_env_clips.py --list        # show candidates
    uv run python scripts/fetch_env_clips.py               # fetch the chosen set

The chosen set is CHOSEN below: one clip per environment category, CC0 or
CC BY only, labels limited to that category's branch of the ontology.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "FSD50K"
OUT = ROOT / "data" / "extracted" / "fsd50k_env"
URL = "https://zenodo.org/api/records/4060432/files/{}/content"
PARTS = ["FSD50K.eval_audio.z01", "FSD50K.eval_audio.zip"]   # disk 0, disk 1
DEV_PARTS = [f"FSD50K.dev_audio.z0{i}" for i in range(1, 6)] + ["FSD50K.dev_audio.zip"]
ALLOWED = ("publicdomain/zero", "licenses/by/")
BYTES_PER_S = 44100 * 2                                      # FSD50K: 44.1 kHz, 16-bit, mono

# category -> (wanted label, labels that may appear with it)
CATEGORIES = {
    "gunfire": ("Gunshot_and_gunfire", {"Explosion"}),
    "explosion": ("Explosion", {"Boom"}),
    "wind": ("Wind", set()),
    "birds": ("Bird_vocalization_and_bird_call_and_bird_song", {"Bird", "Animal", "Wild_animals", "Chirp_and_tweet"}),
    "car": ("Car_passing_by", {"Car", "Vehicle", "Motor_vehicle_(road)"}),
}
# fname per category, picked with --list (longest licence-clean clip that sounds like the category)
CHOSEN = {"gunfire": "404334", "explosion": "353421", "wind": "344887", "birds": "427369", "car": "336754"}


def get(name: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(URL.format(name), headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def size(name: str) -> int:
    rec = json.loads((RAW / "_source_record.json").read_text())
    return next(f["size"] for f in rec["files"] if f["key"] == name)


def central_directory(parts=PARTS) -> dict[str, tuple[int, int, int, int]]:
    """member name -> (disk, local header offset, compressed size, method)."""
    last = parts[-1]
    n = size(last)
    tail = get(last, n - 65536, n - 1)
    loc = tail.rfind(b"PK\x06\x07")                           # zip64 end-of-central-directory locator
    _, _, eocd64_off, _ = struct.unpack("<IIQI", tail[loc:loc + 20])
    rec = tail[eocd64_off - (n - 65536):] if eocd64_off >= n - 65536 else get(last, eocd64_off, eocd64_off + 55)
    cd_disk, = struct.unpack("<I", rec[20:24])
    cd_size, cd_off = struct.unpack("<QQ", rec[40:56])
    if parts[cd_disk] != last:
        raise SystemExit("central directory is not on the last part; not handled")
    cd = get(last, cd_off, cd_off + cd_size - 1)
    out, i = {}, 0
    while cd[i:i + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[i + 10:i + 12])
        csize, usize = struct.unpack("<II", cd[i + 20:i + 28])
        nlen, xlen, clen, disk = struct.unpack("<HHHH", cd[i + 28:i + 36])
        off, = struct.unpack("<I", cd[i + 42:i + 46])
        name = cd[i + 46:i + 46 + nlen].decode()
        extra = cd[i + 46 + nlen:i + 46 + nlen + xlen]
        j = 0
        while j < len(extra):                                # zip64 extra: only the fields set to 0xFFFF...
            tag, ln = struct.unpack("<HH", extra[j:j + 4])
            if tag == 1:
                vals, k = extra[j + 4:j + 4 + ln], 0
                if usize == 0xFFFFFFFF: usize, = struct.unpack("<Q", vals[k:k + 8]); k += 8
                if csize == 0xFFFFFFFF: csize, = struct.unpack("<Q", vals[k:k + 8]); k += 8
                if off == 0xFFFFFFFF: off, = struct.unpack("<Q", vals[k:k + 8]); k += 8
                if disk == 0xFFFF: disk, = struct.unpack("<I", vals[k:k + 4])
            j += 4 + ln
        out[name] = (disk, off, csize, method, usize)
        i += 46 + nlen + xlen + clen
    return out


def read_member(disk: int, off: int, csize: int, method: int, parts=PARTS) -> bytes:
    def span(start: int, length: int) -> bytes:              # read across the part boundary if needed
        d, buf = disk, b""
        while length:
            n = size(parts[d])
            if start >= n:
                start, d = start - n, d + 1
                continue
            take = min(length, n - start)
            buf += get(parts[d], start, start + take - 1)
            length, start = length - take, start + take
        return buf
    head = span(off, 30)
    nlen, xlen = struct.unpack("<HH", head[26:30])
    data = span(off + 30 + nlen + xlen, csize)
    return data if method == 0 else zlib.decompress(data, -15)


def candidates(cd: dict) -> dict[str, list[dict]]:
    gt = {r["fname"]: r["labels"].split(",") for r in csv.DictReader(open_meta("FSD50K.ground_truth/eval.csv"))}
    info = json.loads(open_meta("FSD50K.metadata/eval_clips_info_FSD50K.json").read())
    by_name = {Path(k).stem: v for k, v in cd.items() if k.endswith(".wav")}
    out = {}
    for cat, (want, ok) in CATEGORIES.items():
        rows = []
        for fname, labels in gt.items():
            meta = info[fname]
            if want not in labels or not set(labels) <= ok | {want} or not any(a in meta["license"] for a in ALLOWED):
                continue
            rows.append(dict(category=cat, fname=fname, seconds=round(by_name[fname][4] / BYTES_PER_S, 1),
                             labels=",".join(labels), license=meta["license"], uploader=meta["uploader"],
                             title=meta["title"]))
        out[cat] = sorted(rows, key=lambda r: -r["seconds"])
    return out


def open_meta(member: str):
    zname = "FSD50K.ground_truth.zip" if "ground_truth" in member else "FSD50K.metadata.zip"
    return io.TextIOWrapper(zipfile.ZipFile(RAW / zname).open(member), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print candidates per category and stop")
    args = ap.parse_args()
    cd = central_directory()
    cands = candidates(cd)
    if args.list:
        for cat, rows in cands.items():
            print(f"\n{cat}: {len(rows)} candidates")
            for r in rows[:12]:
                print(f"  {r['fname']:>7} {r['seconds']:5.1f}s  {r['title'][:50]:50} {r['license'].split('/')[-3]}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    by_name = {Path(k).stem: (k, v) for k, v in cd.items() if k.endswith(".wav")}
    rows = []
    for cat, fname in CHOSEN.items():
        key, (disk, off, csize, method, _) = by_name[fname]
        (OUT / f"{cat}.wav").write_bytes(read_member(disk, off, csize, method))
        rows.append(next(r for r in cands[cat] if r["fname"] == fname) | dict(member=key))
        print(f"{cat}: {fname} -> {OUT / (cat + '.wav')}")
    with open(OUT / "sources.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
