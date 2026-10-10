"""Look inside the raw downloads without extracting or changing anything.

Prints what each archive actually contains: file counts, name patterns, WAV
header facts (sample rate, channels, duration, subtype). The dataset
descriptions disagree with each other in places, so nothing downstream should
be built on them before this has been run.

    uv run python -I scripts/inspect_raw.py
"""
from __future__ import annotations

import collections
import io
import sys
import zipfile
from pathlib import Path

import soundfile as sf

DATA = Path(__file__).resolve().parents[1] / "data"


def wav_facts(fileobj) -> tuple:
    info = sf.info(fileobj)
    return (info.samplerate, info.channels, info.subtype, round(info.duration, 4))


def tally(facts: list[tuple]) -> None:
    for fact, n in collections.Counter(facts).most_common(8):
        sr, ch, subtype, dur = fact
        print(f"    {n:>7} x  {sr} Hz, {ch} ch, {subtype}, {dur} s")


def tree(names: list[str], depth: int) -> None:
    counts = collections.Counter("/".join(n.split("/")[:depth]) for n in names)
    for prefix, n in sorted(counts.items())[:40]:
        print(f"    {n:>7}  {prefix}")
    if len(counts) > 40:
        print(f"    ... {len(counts) - 40} more")


def ddl() -> None:
    print("\n=== DDL_Surrey / MLSP_2022_Real_Data.zip")
    with zipfile.ZipFile(DATA / "DDL_Surrey" / "MLSP_2022_Real_Data.zip") as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        names = [i.filename for i in infos]
        print(f"  entries: {len(names)}, uncompressed {sum(i.file_size for i in infos) / 1e9:.2f} GB")
        exts = collections.Counter(Path(n).suffix.lower() for n in names)
        print(f"  extensions: {dict(exts)}")
        print("  folders:")
        tree(names, 2)
        wavs = [n for n in names if n.lower().endswith(".wav")]
        print("  sample names:")
        for n in wavs[:3] + wavs[len(wavs) // 2: len(wavs) // 2 + 2] + wavs[-2:]:
            print(f"    {n}")
        stems = [Path(n).stem for n in wavs]
        print(f"  stem lengths: {dict(collections.Counter(len(s) for s in stems))}")
        print(f"  class field [13:17]: {dict(collections.Counter(s[13:17] for s in stems))}")
        print(f"  type field [30]: {dict(collections.Counter(s[30:31] for s in stems))}")
        step = max(1, len(wavs) // 60)
        facts = [wav_facts(io.BytesIO(z.read(n))) for n in wavs[::step]]
        print(f"  WAV headers ({len(facts)} sampled evenly):")
        tally(facts)
        other = [n for n in names if not n.lower().endswith(".wav")]
        print(f"  non-WAV entries: {other[:20]}")


def dronenoise() -> None:
    print("\n=== DroneNoise_Salford")
    wavs = sorted((DATA / "DroneNoise_Salford").glob("*.wav"))
    print(f"  WAV files: {len(wavs)}")
    tally([wav_facts(str(p)) for p in wavs])
    events = [p for p in wavs if not p.name.startswith("Calib")]
    print(f"  calibration files: {len(wavs) - len(events)}, event files: {len(events)}")
    fields = [p.stem.split("_") for p in events]
    print(f"  field counts per name: {dict(collections.Counter(len(f) for f in fields))}")
    for idx in range(max(len(f) for f in fields)):
        values = collections.Counter(f[idx] for f in fields if len(f) > idx)
        print(f"    field {idx}: {dict(values.most_common(12))}")


def halmstad() -> None:
    print("\n=== Halmstad / Drone-detection-dataset-v1.0.0.zip")
    with zipfile.ZipFile(DATA / "Halmstad" / "Drone-detection-dataset-v1.0.0.zip") as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        names = [i.filename for i in infos]
        print(f"  entries: {len(names)}, uncompressed {sum(i.file_size for i in infos) / 1e9:.2f} GB")
        print(f"  extensions: {dict(collections.Counter(Path(n).suffix.lower() for n in names))}")
        print("  folders:")
        tree(names, 3)
        wavs = [n for n in names if n.lower().endswith(".wav")]
        print(f"  WAV files: {len(wavs)}")
        tally([wav_facts(io.BytesIO(z.read(n))) for n in wavs])
        print(f"  WAV name prefixes: {dict(collections.Counter(Path(n).name.split('_')[0] for n in wavs))}")
        print(f"  sample WAV names: {[Path(n).name for n in wavs[:6]]}")
        for n in names:
            low = Path(n).name.lower()
            if low.startswith(("license", "licence", "readme")):
                text = z.read(n).decode("utf-8", errors="replace")
                print(f"  --- {n} ({len(text)} chars) ---")
                print("    " + text[:2500].replace("\n", "\n    "))


def fsd50k_small() -> None:
    print("\n=== FSD50K (metadata, ground truth, doc)")
    for name in ("FSD50K.doc.zip", "FSD50K.ground_truth.zip", "FSD50K.metadata.zip"):
        with zipfile.ZipFile(DATA / "FSD50K" / name) as z:
            print(f"  {name}:")
            for i in z.infolist():
                if not i.is_dir():
                    print(f"    {i.file_size:>10}  {i.filename}")
    with zipfile.ZipFile(DATA / "FSD50K" / "FSD50K.ground_truth.zip") as z:
        for n in z.namelist():
            if n.endswith(".csv"):
                lines = z.read(n).decode("utf-8").splitlines()
                print(f"  {n}: {len(lines) - 1} rows; header: {lines[0]}; first: {lines[1]}")


if __name__ == "__main__":
    steps = {"ddl": ddl, "dronenoise": dronenoise, "halmstad": halmstad, "fsd50k": fsd50k_small}
    for key in sys.argv[1:] or list(steps):
        steps[key]()
