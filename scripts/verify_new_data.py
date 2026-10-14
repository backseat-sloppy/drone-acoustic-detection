"""Check the local copies of the 2026-10-12 datasets against the download log.

The files were fetched onto Google Drive by download_new_datasets.ps1, which
logged each one's size and checksum (MD5, SHA-256 or git SHA-1) after
matching it against the host. This re-hashes every local copy under data/
and compares.

    uv run python -I scripts/verify_new_data.py
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
LOG = DATA / "_download_tools" / "download_log.txt"
LINE = re.compile(r"\s OK\s+(?P<path>.+?)\s+(?P<size>\d+) bytes\s+(?P<alg>MD5|SHA256|GITSHA1) (?P<hash>[0-9a-f]+)")


def digest(path: Path, alg: str, size: int) -> str:
    h = hashlib.md5() if alg == "MD5" else hashlib.sha256() if alg == "SHA256" else hashlib.sha1()
    if alg == "GITSHA1":
        h.update(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    entries = {}
    for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE.search(line)
        if m:
            entries[m["path"]] = (int(m["size"]), m["alg"], m["hash"])
    bad = 0
    for i, (rel, (size, alg, want)) in enumerate(sorted(entries.items()), start=1):
        p = DATA / rel
        if not p.exists():
            print(f"MISSING  {rel}")
            bad += 1
            continue
        if p.stat().st_size != size:
            print(f"SIZE     {rel}: {p.stat().st_size} != {size}")
            bad += 1
            continue
        got = digest(p, alg, size)
        if got != want:
            print(f"HASH     {rel}: {alg} {got} != {want}")
            bad += 1
        if i % 50 == 0:
            print(f"  {i}/{len(entries)} checked", flush=True)
    print(f"{len(entries)} files in the log, {bad} problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
