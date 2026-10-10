"""Turn the raw downloads into one uniform audio store plus a manifest.

Reads the raw archives in data/ (never modifies them) and writes to
data/processed/:

    ddl.f32, dronenoise.f32     flat float32 audio, 16 kHz mono
    halmstad.i16, fsd50k.i16    flat int16 audio, 16 kHz mono
    manifest.csv                one row per continuous recording
    ddl_clips.csv               one row per original DDL file, kept or dropped
    summary.json                counts and QA figures from this run

Only format is changed here: channel selection, resampling to 16 kHz, and
joining DDL's 100 ms files back into the continuous recordings they were cut
from. No gain change, no normalisation, no filtering, no mixing. Anything that
alters the signal belongs in training, where it can be varied and logged.

FSD50K audio must already be extracted to data/extracted/FSD50K (it is a
multi-disk zip, which Python's zipfile cannot read):

    7z x data\\FSD50K\\FSD50K.dev_audio.zip  -odata\\extracted\\FSD50K
    7z x data\\FSD50K\\FSD50K.eval_audio.zip -odata\\extracted\\FSD50K

Run:
    uv run python -I scripts/prepare_data.py            # everything
    uv run python -I scripts/prepare_data.py ddl fsd50k # chosen sources
"""
from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
import math
import multiprocessing as mp
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16_000
DATA = Path(__file__).resolve().parents[1] / "data"
OUT = DATA / "processed"

MANIFEST_FIELDS = [
    "source", "recording_id", "file", "dtype", "offset", "n_samples", "duration_s",
    "label", "kind", "group", "subset", "sr_orig", "channels_orig", "channel_used",
    "licence", "rms_dbfs", "peak", "labels", "attributes",
]


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return x
    g = math.gcd(SR, sr)
    return resample_poly(x, SR // g, sr // g)


def as_int16(x: np.ndarray) -> np.ndarray:
    return np.round(np.clip(x, -1.0, 32767 / 32768) * 32768).astype(np.int16)


def level(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
    return (round(20 * math.log10(rms), 2) if rms > 0 else -200.0, round(float(np.max(np.abs(x))), 5) if len(x) else 0.0)


class Store:
    """Appends recordings to one flat binary file and reports where each landed."""

    def __init__(self, name: str, dtype: str) -> None:
        self.dtype = dtype
        self.file = f"{name}.{'f32' if dtype == 'float32' else 'i16'}"
        self._fh = open(OUT / self.file, "wb")
        self.offset = 0

    def add(self, x: np.ndarray) -> tuple[int, int]:
        x = np.ascontiguousarray(x, dtype=self.dtype)
        x.tofile(self._fh)
        start, self.offset = self.offset, self.offset + len(x)
        return start, len(x)

    def close(self) -> None:
        self._fh.close()


def row(store: Store, offset: int, n: int, **kw) -> dict:
    base = dict.fromkeys(MANIFEST_FIELDS, "")
    base.update(file=store.file, dtype=store.dtype, offset=offset, n_samples=n,
                duration_s=round(n / SR, 3))
    base.update(kw)
    return base


# --------------------------------------------------------------------- DDL

# The host documents 13-character timestamps and separate flight and recording
# session IDs. The files actually carry 14-character timestamps and a single
# session ID, and a few have a negative altitude written as e.g. "0-1".
DDL_NAME = re.compile(
    r"^(?P<ts>\d{14})(?P<cls>[A-Z0-9]{4})(?P<bearing>\d{3})(?P<range>\d{3})"
    r"(?P<alt>[-\d]{3})(?P<temp>\d{4})(?P<type>[RS])(?P<session>\d{6}-T\d{3})-(?P<seq>\d{6})$"
)
DDL_MODEL = {"MINI": "DJI Mini 2", "PRO4": "DJI Phantom 4 Pro"}
DDL_CLIP_FRAMES = 9_600  # 100 ms at 96 kHz
# All eight channels agree within about 3 dB between 200 Hz and 2 kHz. Below
# 80 Hz they differ by up to 12 dB on 2021-03-29, which is wind hitting the
# capsules individually. Channel 0 carries the least of it.
DDL_CHANNEL = 0
WAV_HEADER = 44


def prepare_ddl(manifest: list[dict], summary: dict) -> None:
    store = Store("ddl", "float32")
    z = zipfile.ZipFile(DATA / "DDL_Surrey" / "MLSP_2022_Real_Data.zip")
    sessions: dict[str, list] = collections.defaultdict(list)
    for info in z.infolist():
        if info.is_dir():
            continue
        m = DDL_NAME.match(Path(info.filename).stem)
        if m is None:
            raise ValueError(f"unexpected DDL file name: {info.filename}")
        sessions[m["session"]].append((int(m["seq"]), info, m))

    seen: dict[str, tuple[str, int]] = {}
    clip_rows: list[dict] = []
    by_key: dict[tuple[str, int], dict] = {}
    stats = collections.Counter()
    for session in sorted(sessions):
        chunks: list[np.ndarray] = []
        pos = 0
        after_long = False
        classes = set()
        for seq, info, m in sorted(sessions[session], key=lambda t: t[0]):
            alt = m["alt"]
            clip = dict(
                session=session, seq=seq, name=Path(info.filename).stem, timestamp=m["ts"],
                cls=m["cls"], bearing_deg=int(m["bearing"]), range_m=int(m["range"]),
                altitude_m=(-1 if "-" in alt else 1) * int(alt.replace("-", "")),
                temperature_k=int(m["temp"]) / 10, frames_96k=0, kept=0, drop_reason="",
                start_sample_16k="", range_reliable=0, discontinuity_before=0,
            )
            clip_rows.append(clip)
            by_key[(session, seq)] = clip
            if info.file_size <= WAV_HEADER:
                clip["drop_reason"] = "empty"
                stats["empty"] += 1
                continue
            payload = z.read(info)[WAV_HEADER:]
            digest = hashlib.md5(payload).hexdigest()
            if digest in seen:
                clip["drop_reason"] = "duplicate of %s-%06d" % seen[digest]
                # Same audio, two different sets of labels: neither can be trusted.
                by_key[seen[digest]]["range_reliable"] = 0
                stats["duplicate"] += 1
                continue
            seen[digest] = (session, seq)
            frames = np.frombuffer(payload[: len(payload) // 32 * 32], dtype="<i4").reshape(-1, 8)
            x = frames[:, DDL_CHANNEL].astype(np.float64) / 2**31
            if len(x) > DDL_CLIP_FRAMES:
                # One file holding many clips' worth of audio under a single
                # label. Its labels, and those of everything after it in the
                # session, no longer line up with the audio.
                after_long = True
                stats["long_files"] += 1
            if chunks:
                near = np.abs(np.diff(np.concatenate([chunks[-1][-512:], x[:512]])))
                jump = abs(x[0] - chunks[-1][-1])
                usual = np.percentile(np.delete(near, min(511, len(chunks[-1]) - 1)), 99.5)
                if jump > 4 * usual + 1e-7:
                    clip["discontinuity_before"] = 1
                    stats["discontinuities"] += 1
            clip.update(kept=1, frames_96k=len(x), start_sample_16k=round(pos / 6, 2),
                        range_reliable=0 if after_long else 1)
            classes.add(m["cls"])
            chunks.append(x)
            pos += len(x)
        if len(classes) != 1:
            raise ValueError(f"session {session} mixes classes {classes}")
        cls = classes.pop()
        y = to_16k(np.concatenate(chunks), 96_000).astype(np.float32)
        offset, n = store.add(y)
        rms, peak = level(y)
        kept = [c for c in clip_rows if c["session"] == session and c["kept"]]
        ranges = [c["range_m"] for c in kept if c["range_reliable"]]
        manifest.append(row(
            store, offset, n, source="ddl", recording_id=session, label=1,
            kind="drone", group=f"ddl:{session}", subset=f"20{session[4:6]}-{session[2:4]}-{session[:2]}",
            sr_orig=96_000, channels_orig=8, channel_used=DDL_CHANNEL, licence="CC BY 4.0",
            rms_dbfs=rms, peak=peak, labels=cls,
            attributes=json.dumps(dict(
                model=DDL_MODEL[cls], files_kept=len(kept),
                range_m_min=min(ranges) if ranges else None,
                range_m_max=max(ranges) if ranges else None,
                range_reliable_fraction=round(len(ranges) / len(kept), 3),
            )),
        ))
        stats["sessions"] += 1
        stats["kept_files"] += len(kept)
        print(f"  ddl {session} {cls}: {n / SR:7.1f} s from {len(kept)} files", flush=True)
    store.close()

    with open(OUT / "ddl_clips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(clip_rows[0]))
        w.writeheader()
        w.writerows(clip_rows)
    summary["ddl"] = dict(stats, files_total=len(clip_rows),
                          range_reliable_files=sum(c["range_reliable"] for c in clip_rows))


# -------------------------------------------------------------- DroneNoise

# Codes decoded from DNM_SQM_FLYO_AESCT.xlsx, which ships with the dataset.
DN_MODEL = {
    "3p": "DJI Mini 3 Pro",
    "M3": "DJI Matrice 300 RTK",
    "Yn": "Yuneec H520E",
    # Not named by the host. The spreadsheet lists a 795 g gross weight for it.
    "Fp": "unconfirmed (795 g gross weight)",
}


def prepare_dronenoise(manifest: list[dict], summary: dict) -> None:
    store = Store("dronenoise", "float32")
    stats = collections.Counter()
    for path in sorted((DATA / "DroneNoise_Salford").glob("*.wav")):
        if path.name.startswith("Calib"):
            stats["calibration_files_skipped"] += 1
            continue
        site, drone, hagl, op, payload, start, wind, event, mic = path.stem.split("_")
        x, sr = sf.read(str(path), dtype="float64", always_2d=True)
        y = to_16k(x[:, 0], sr).astype(np.float32)
        offset, n = store.add(y)
        rms, peak = level(y)
        event_id = "_".join([site, drone, hagl, op, payload, start, wind, event])
        manifest.append(row(
            store, offset, n, source="dronenoise", recording_id=path.stem, label=1,
            kind="drone", group=f"dronenoise:{event_id}", subset=mic,
            sr_orig=sr, channels_orig=x.shape[1], channel_used=0, licence="CC BY 4.0",
            rms_dbfs=rms, peak=peak, labels=drone,
            attributes=json.dumps(dict(
                model=DN_MODEL[drone], height_agl_m=int(hagl),
                operation="flyover" if op[0] == "F" else "hover", speed_m_s=int(op[1:]),
                payload=payload == "Y", start=start, wind=wind, event=event, microphone=mic,
            )),
        ))
        stats["event_files"] += 1
    store.close()
    stats["distinct_events"] = len({r["group"] for r in manifest if r["source"] == "dronenoise"})
    summary["dronenoise"] = dict(stats)
    print(f"  dronenoise: {stats['event_files']} files, {stats['distinct_events']} distinct events", flush=True)


# ----------------------------------------------------------------- Halmstad

def prepare_halmstad(manifest: list[dict], summary: dict) -> None:
    store = Store("halmstad", "int16")
    stats = collections.Counter()
    seen: dict[str, str] = {}
    with zipfile.ZipFile(DATA / "Halmstad" / "Drone-detection-dataset-v1.0.0.zip") as z:
        for name in sorted(n for n in z.namelist() if n.lower().endswith(".wav")):
            stem = Path(name).stem
            cls = stem.split("_")[0].lower()
            payload = z.read(name)
            digest = hashlib.md5(payload).hexdigest()
            if digest in seen:
                # BACKGROUND_019 is a byte-for-byte copy of BACKGROUND_010.
                stats[f"duplicate_dropped:{stem}={seen[digest]}"] += 1
                continue
            seen[digest] = stem
            x, sr = sf.read(io.BytesIO(payload), dtype="float64", always_2d=True)
            y = as_int16(to_16k(x[:, 0], sr))
            offset, n = store.add(y)
            rms, peak = level(y / 32768)
            manifest.append(row(
                store, offset, n, source="halmstad", recording_id=stem,
                label=1 if cls == "drone" else 0, kind=cls,
                # The host does not say which clips were cut from the same
                # recording, so the whole class is one group.
                group=f"halmstad:{cls}", subset=cls,
                sr_orig=sr, channels_orig=x.shape[1], channel_used=0, licence="CC0 1.0",
                rms_dbfs=rms, peak=peak, labels=cls, attributes="{}",
            ))
            stats[cls] += 1
    store.close()
    summary["halmstad"] = dict(stats)
    print(f"  halmstad: {dict(stats)}", flush=True)


# ------------------------------------------------------------------- FSD50K

FSD_AUDIO = DATA / "extracted" / "FSD50K"
FSD_LICENCES = {
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0 1.0",
    "http://creativecommons.org/licenses/by/3.0/": "CC BY 3.0",
}
# FSD50K has no drone class, so an unmanned aircraft could sit among the
# negatives under another label. Titles, descriptions and tags are searched and
# every hit is left out. Most hits are the musical sense of "drone".
FSD_DRONE_WORDS = re.compile(
    r"drone|quad-?ro?copter|multi-?copter|multi-?rotor|hexa-?copter|octo-?copter|\buav\b|\buas\b|\bdji\b",
    re.IGNORECASE,
)


def _fsd_one(job: tuple[str, str]) -> tuple[str, np.ndarray | None, int, int]:
    fname, path = job
    try:
        x, sr = sf.read(path, dtype="float64", always_2d=True)
    except Exception:
        return fname, None, 0, 0
    return fname, as_int16(to_16k(x[:, 0], sr)), sr, x.shape[1]


def prepare_fsd50k(manifest: list[dict], summary: dict) -> None:
    zm = zipfile.ZipFile(DATA / "FSD50K" / "FSD50K.metadata.zip")
    zg = zipfile.ZipFile(DATA / "FSD50K" / "FSD50K.ground_truth.zip")
    stats = collections.Counter()
    jobs, meta = [], {}
    for part in ("dev", "eval"):
        info = json.loads(zm.read(f"FSD50K.metadata/{part}_clips_info_FSD50K.json"))
        truth = csv.DictReader(io.StringIO(zg.read(f"FSD50K.ground_truth/{part}.csv").decode("utf-8")))
        for t in truth:
            clip = info[t["fname"]]
            text = " ".join([clip["title"], clip["description"], " ".join(clip["tags"])])
            licence = FSD_LICENCES.get(clip["license"])
            if licence is None:
                stats["excluded_licence"] += 1
                continue
            if FSD_DRONE_WORDS.search(text):
                stats["excluded_drone_keyword"] += 1
                continue
            path = FSD_AUDIO / f"FSD50K.{part}_audio" / f"{t['fname']}.wav"
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing; extract FSD50K first (see module docstring)")
            subset = t.get("split", "eval")  # dev.csv carries train/val; eval.csv has no column
            meta[t["fname"]] = dict(labels=t["labels"], subset=subset, licence=licence,
                                    uploader=clip["uploader"])
            jobs.append((t["fname"], str(path)))

    store = Store("fsd50k", "int16")
    t0 = time.time()
    with mp.Pool(max(1, mp.cpu_count() - 2)) as pool:
        for i, (fname, y, sr, channels) in enumerate(pool.imap(_fsd_one, jobs, chunksize=64)):
            if y is None or len(y) == 0:
                stats["unreadable"] += 1
                continue
            m = meta[fname]
            offset, n = store.add(y)
            rms, peak = level(y / 32768)
            labels = m["labels"].split(",")
            manifest.append(row(
                store, offset, n, source="fsd50k", recording_id=fname, label=0,
                kind=labels[0],  # FSD50K lists the most specific label first
                # Grouping by uploader keeps one person's recordings, made on
                # one device, on one side of any split.
                group=f"fsd50k:{m['uploader']}", subset=m["subset"],
                sr_orig=sr, channels_orig=channels, channel_used=0, licence=m["licence"],
                rms_dbfs=rms, peak=peak, labels=m["labels"], attributes="{}",
            ))
            stats[f"kept_{m['subset']}"] += 1
            if i % 5000 == 0:
                print(f"  fsd50k {i}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    store.close()
    summary["fsd50k"] = dict(stats)
    print(f"  fsd50k: {dict(stats)}", flush=True)


STEPS = {
    "ddl": prepare_ddl,
    "dronenoise": prepare_dronenoise,
    "halmstad": prepare_halmstad,
    "fsd50k": prepare_fsd50k,
}


def main(argv: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = argv or list(STEPS)
    manifest_path, summary_path = OUT / "manifest.csv", OUT / "summary.json"
    manifest: list[dict] = []
    summary: dict = {}
    # Rerunning one source replaces only that source's rows.
    if manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            manifest = [r for r in csv.DictReader(f) if r["source"] not in wanted]
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    for name in wanted:
        t0 = time.time()
        print(f"[{name}]", flush=True)
        STEPS[name](manifest, summary)
        print(f"  done in {time.time() - t0:.0f} s", flush=True)

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(manifest)

    totals = collections.defaultdict(lambda: collections.Counter())
    for r in manifest:
        t = totals[r["source"]]
        t["recordings"] += 1
        t["seconds"] += float(r["duration_s"])
        t["positive_seconds" if int(r["label"]) else "negative_seconds"] += float(r["duration_s"])
    summary["totals"] = {s: {k: round(v, 1) for k, v in t.items()} for s, t in totals.items()}
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["totals"], indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
