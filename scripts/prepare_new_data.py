"""Turn the datasets added on 2026-10-12 into the same store as prepare_data.py.

Sources (raw, as downloaded, under data/):

    Nightjar_HF/                   DJI Mini 2, UMA-16 array, one US site
    D3set_IntRoLab/Flights/        Holybro X500, 16-mic ground array, two Canadian sites
    Ljungbyhed_Lund/*.zip          three drone types, 12-mic array, Swedish airfield
    extracted/Glasgow/             24 drones hovering at 1 m and 5 m, indoors

Glasgow must be extracted first (7z archive):

    7z x data\\Glasgow_Drone_Authentication\\ACSAC_2022_Drone_Authentication.7z -odata\\extracted\\Glasgow

Writes to data/processed/:

    nightjar.i16, d3set.i16, glasgow.i16, lund.f32   flat audio, 16 kHz mono
    manifest.csv       one row per *segment* for these sources (see below);
                       rows of other sources are kept as they are
    range_track.csv    slant range to the drone at 10 Hz for every drone segment
    summary.json       counts per source

Unlike the first four sources, these recordings change state over time: a
session starts with the drone on the ground, then flies, then lands. Each
source recording is therefore stored once, whole, and the manifest gets one
row per stretch of a single state, pointing into it:

    label  1  kind "drone"       motors on, per the drone's own log
    label  0  kind "background"  no drone: before the motors start or after they stop
    label -1  kind "uncertain"   guard margins around every change of state, and
                                 stretches whose state the logs do not settle

Only format is changed, exactly as in prepare_data.py: one channel,
polyphase resampling to 16 kHz. No gain, filtering or mixing.

Run:
    uv run python -I scripts/prepare_new_data.py                    # everything
    uv run python -I scripts/prepare_new_data.py nightjar d3set     # chosen sources
"""
from __future__ import annotations

import collections
import csv
import datetime as dt
import hashlib
import io
import json
import math
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import (DATA, MANIFEST_FIELDS, OUT, SR, Store, as_int16, level,  # noqa: E402
                          row, to_16k)

RANGE_HZ = 10           # rate of range_track.csv
GUARD_S = 10.0          # either side of a change of state: neither drone nor background
MIN_SEGMENT_S = 1.0     # shorter stretches are not worth a manifest row
EARTH_R = 6_371_000.0

TRACK_FIELDS = ["source", "recording_id", "t_s", "range_m", "horiz_m", "height_m"]

# The `pairs` step (entry 036) writes a second store per rig in which each
# sample is the mean of two channels of the same array, laid out exactly as
# the one-channel store, so a model can be scored on one or two microphones.
PAIR = False
STORE_SUFFIX = ""


def channel(x: np.ndarray, first: int, second: int) -> np.ndarray:
    """One channel, or with PAIR the mean of two, as float in [-1, 1)."""
    a = x[:, first].astype(np.float64)
    if PAIR:
        a = 0.5 * (a + x[:, second].astype(np.float64))
    return a / 32768


def enu(lat, lon, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """East and north offsets in metres from a reference point (flat earth; < 2 km here)."""
    x = np.radians(np.asarray(lon) - lon0) * EARTH_R * math.cos(math.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * EARTH_R
    return x, y


def segments_from_spans(n: int, drone_spans: list[tuple[float, float]], guard_s: float,
                        unknown_spans: list[tuple[float, float]] = ()) -> list[tuple[int, int, int]]:
    """Cut [0, n) samples into (start, end, label) runs.

    `drone_spans` are in seconds from the start of the recording. Everything
    within `guard_s` of a drone span's edge, and everything in
    `unknown_spans`, is labelled -1. The rest is background (0).
    """
    lab = np.zeros(n, dtype=np.int8)
    g = int(round(guard_s * SR))
    for a, b in drone_spans:
        a, b = int(round(a * SR)), int(round(b * SR))
        lab[max(0, a - g):max(0, min(n, b + g))] = -1
    for a, b in drone_spans:
        a, b = int(round(a * SR)), int(round(b * SR))
        if b - a > 2 * g:
            lab[max(0, a + g):max(0, min(n, b - g))] = 1
    for a, b in unknown_spans:
        lab[max(0, int(round(a * SR))):max(0, min(n, int(round(b * SR))))] = -1
    edges = np.flatnonzero(np.diff(lab)) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, n]
    return [(int(s), int(e), int(lab[s])) for s, e in zip(starts, ends) if e - s >= MIN_SEGMENT_S * SR]


KIND = {1: "drone", 0: "background", -1: "uncertain"}


def add_segments(manifest: list[dict], tracks: list[dict], store: Store, base: int, y: np.ndarray,
                 segs: list[tuple[int, int, int]], rid_prefix: str, range_fn, common: dict,
                 attrs: dict, stats: collections.Counter, scale: float = 1.0) -> None:
    """One manifest row per segment; range_fn(t_abs_s) -> (slant, horiz, height) or None."""
    for k, (s, e, lab) in enumerate(segs):
        rid = f"{rid_prefix}#{k:02d}"
        x = y[s:e].astype(np.float64) * scale
        rms, peak = level(x)
        a = dict(attrs, parent_start_s=round(s / SR, 3))
        if lab == 1 and range_fn is not None:
            t = np.arange(0.0, (e - s) / SR, 1.0 / RANGE_HZ)
            rng = range_fn(s / SR + t)
            if rng is not None:
                slant, horiz, height = rng
                ok = np.isfinite(slant)
                for ti, r1, h1, z1 in zip(t[ok], slant[ok], horiz[ok], height[ok]):
                    tracks.append(dict(source=common["source"], recording_id=rid, t_s=round(float(ti), 2),
                                       range_m=round(float(r1), 1), horiz_m=round(float(h1), 1),
                                       height_m=round(float(z1), 1)))
                if ok.any():
                    a.update(range_m_min=round(float(np.min(slant[ok])), 1),
                             range_m_median=round(float(np.median(slant[ok])), 1),
                             range_m_max=round(float(np.max(slant[ok])), 1))
                    for lo, hi in ((0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 1e9)):
                        stats[f"drone_s_{lo}-{hi if hi < 1e9 else 'inf'}m"] += float(
                            np.sum((slant[ok] >= lo) & (slant[ok] < hi))) / RANGE_HZ
        manifest.append(row(store, base + s, e - s, recording_id=rid, label=lab, kind=KIND[lab],
                            rms_dbfs=rms, peak=peak, attributes=json.dumps(a), **common))
        stats[f"{KIND[lab]}_s"] += (e - s) / SR
        stats[f"{KIND[lab]}_segments"] += 1


# ------------------------------------------------------------------ Nightjar

NJ_ROOT = DATA / "Nightjar_HF"
# Rig position from the series' own analysis (analysis/sb_fusion_d04/report.json,
# "rig"), surveyed on 2026-07-27. The author notes the rig moved on 2026-08-05
# by roughly 10 m, so ranges from the 08-05 and 08-06 sessions carry that error.
NJ_RIG = (41.534765, -75.595254)
NJ_RIG_HEIGHT_M = -0.5     # in the drone's take-off height frame (fit_el_datum, README)
NJ_CHANNEL = 0             # UMA-16; channel 9 is dead
NJ_SECOND = 8              # second microphone of the pair (entry 036); geometry not published here
FT = 0.3048                # OSD.height in the decoded CSV is in feet

NJ_CONDITION = {
    "nightjar-night-ambient-20260804": "night",
    "nightjar-night-ambient-20260805": "night",
    "nightjar-flight-20260804-night": "night",
    "nightjar-lawnmower1-20260805": "mower",
}


def _dji_track(path: Path) -> dict:
    """Absolute UTC time, slant range, horizontal range and height of one DJI log."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    fly = np.array([float(r["OSD.flyTime"] or 0) for r in rows])
    # Some logs carry few valid UTC stamps. Anchor on the first and count
    # forward with OSD.flyTime, as the dataset cards describe.
    i0 = next(i for i, r in enumerate(rows) if not r["CUSTOM.dateTime"].startswith("1970"))
    t0 = dt.datetime.fromisoformat(rows[i0]["CUSTOM.dateTime"].replace("Z", "+00:00")).timestamp()
    t = t0 + (fly - fly[i0])
    lat = np.array([float(r["OSD.latitude"]) for r in rows])
    lon = np.array([float(r["OSD.longitude"]) for r in rows])
    hgt = np.array([float(r["OSD.height"] or 0) for r in rows]) * FT
    x, y = enu(lat, lon, *NJ_RIG)
    horiz = np.hypot(x, y)
    z = hgt - NJ_RIG_HEIGHT_M
    order = np.argsort(t, kind="stable")
    return dict(t=t[order], slant=np.hypot(horiz, z)[order], horiz=horiz[order], height=z[order],
                on_ground=np.array([r["OSD.isOnGround"] == "true" for r in rows])[order],
                motor_on=np.array([r["OSD.isMotorOn"] == "true" for r in rows])[order],
                model=rows[i0].get("OSD.droneType", ""), log=path.name)


def prepare_nightjar(manifest: list[dict], tracks: list[dict], summary: dict) -> None:
    store = Store("nightjar" + STORE_SUFFIX, "int16")
    stats = collections.Counter()
    for sess in sorted(p for p in NJ_ROOT.iterdir() if p.is_dir()):
        logs = [_dji_track(p) for p in sorted((sess / "dji").glob("*.csv"))] if (sess / "dji").exists() else []
        motor_logs = [g for g in logs if g["motor_on"].any()]
        cond = NJ_CONDITION.get(sess.name, "day")
        for wav in sorted((sess / "audio").glob("uma16_*.wav")):
            t_file = int(wav.stem.split("_")[1])           # UTC epoch of the first sample
            x, sr = sf.read(str(wav), dtype="int16", always_2d=True)
            n_raw = x.shape[0]
            y = as_int16(to_16k(channel(x, NJ_CHANNEL, NJ_SECOND), sr))
            del x
            base, n = store.add(y)
            spans = []
            for g in motor_logs:
                on = g["t"][g["motor_on"]]
                a, b = on[0] - t_file, on[-1] - t_file
                if b > 0 and a < n / SR:
                    spans.append((a, b))

            def range_fn(t_rel, t_file=t_file):
                for g in motor_logs:
                    t_abs = t_file + t_rel
                    if t_abs[0] >= g["t"][0] - 1 and t_abs[-1] <= g["t"][-1] + 1:
                        return (np.interp(t_abs, g["t"], g["slant"]), np.interp(t_abs, g["t"], g["horiz"]),
                                np.interp(t_abs, g["t"], g["height"]))
                return None

            segs = segments_from_spans(n, spans, GUARD_S)
            common = dict(source="nightjar", group=f"nightjar:{sess.name}", subset=cond, sr_orig=sr,
                          channels_orig=16, channel_used=NJ_CHANNEL, licence="CC BY 4.0",
                          labels="DJI Mini 2")
            attrs = dict(session=sess.name, parent=wav.name, site="nightjar US", rig="UMA-16",
                         condition=cond, utc_start=t_file, logs=[g["log"] for g in motor_logs])
            add_segments(manifest, tracks, store, base, y, segs, f"{sess.name}/{wav.stem}",
                         range_fn, common, attrs, stats, scale=1 / 32768)
            stats["files"] += 1
            print(f"  nightjar {sess.name}/{wav.name}: {n_raw / sr:6.1f} s, "
                  f"{len(segs)} segments ({sum(1 for s in segs if s[2] == 1)} drone)", flush=True)
    store.close()
    summary["nightjar"] = {k: round(v, 1) if isinstance(v, float) else v for k, v in stats.items()}


# --------------------------------------------------------------------- d3set

D3_ROOT = DATA / "D3set_IntRoLab" / "Flights"
D3_CHANNEL = 0     # never listed as faulty; faulty channels were already removed by the authors
D3_MOTOR_ON_HZ = 20.0   # actuator.csv holds blade-pass frequency per motor; 0 when stopped


def _d3_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f, delimiter=";"))
    head = rows[0]
    out = {}
    for j, h in enumerate(head):
        col = []
        for r in rows[1:]:
            try:
                col.append(float(r[j]))
            except (ValueError, IndexError):
                col.append(math.nan)
        out[h] = np.asarray(col)
    return out


def _d3_seconds(ts: np.ndarray, t_start: float) -> np.ndarray:
    """Telemetry time to seconds from the start of the audio. Most files use Unix
    milliseconds; actuator.csv uses milliseconds from the start of the audio."""
    return ts / 1000 - t_start if np.nanmedian(ts) > 1e11 else ts / 1000


def prepare_d3set(manifest: list[dict], tracks: list[dict], summary: dict) -> None:
    store = Store("d3set" + STORE_SUFFIX, "int16")
    stats = collections.Counter()
    for d in sorted(p for p in D3_ROOT.glob("*/*") if p.is_dir()):
        split, flight = d.parent.name, d.name
        wav = d / "matrix.wav"
        if not wav.exists():
            stats["no_array_audio"] += 1
            continue
        info = sf.info(str(wav))
        if info.samplerate < SR:
            # "25 oct" was recorded at 8 kHz. Upsampling would leave the 4-8 kHz
            # band empty for this one flight, a cue no other recording has.
            stats[f"skipped_{info.samplerate}Hz"] += 1
            print(f"  d3set {split}/{flight}: skipped, {info.samplerate} Hz", flush=True)
            continue
        x, sr = sf.read(str(wav), dtype="int16", always_2d=True)
        y = as_int16(to_16k(channel(x, D3_CHANNEL, x.shape[1] - 1), sr))
        del x
        base, n = store.add(y)
        t_start = float(open(d / "StartTimestamp.csv").read().split()[-1].strip('"')) / 1000
        extra = json.loads((d / "additional_data.json").read_text()) if (d / "additional_data.json").exists() else {}
        spans, range_fn, unknown = [], None, []
        if (d / "actuator.csv").exists():
            act = _d3_csv(d / "actuator.csv")
            ta = _d3_seconds(act["TIMESTAMP"], t_start)
            bpf = np.nanmean(np.stack([act[f"MOTOR_{i}"] for i in range(1, 5)]), axis=0)
            on = np.flatnonzero(bpf > D3_MOTOR_ON_HZ)
            if len(on):
                spans = [(float(ta[on[0]]), float(ta[on[-1]]))]
        elif split == "SNR test":
            # Excerpts cut by the authors: "drone" is all drone, "background noise" is none.
            spans = [(0.0, n / SR)] if flight == "drone" else []
        else:
            unknown = [(0.0, n / SR)]
        if "reference_position" in extra and (d / "position.csv").exists():
            pos = _d3_csv(d / "position.csv")
            tp = _d3_seconds(pos["TIMESTAMP"], t_start)
            ref = extra["reference_position"]
            ex, ny = enu(pos["LATITUDE"], pos["LONGITUDE"], ref[0], ref[1])
            # GPS altitude jumps mid-flight (README); the range is approximate.
            z = pos["ALTITUDE"] - ref[2]
            ok = np.isfinite(ex) & np.isfinite(z) & np.isfinite(tp)
            tp, horiz, z = tp[ok], np.hypot(ex, ny)[ok], z[ok]

            def range_fn(t_rel, tp=tp, horiz=horiz, z=z):
                hz, zz = np.interp(t_rel, tp, horiz), np.interp(t_rel, tp, z)
                return np.hypot(hz, zz), hz, zz

        # Drone spans keep the full guard; the authors clap by hand before each flight.
        segs = segments_from_spans(n, spans, GUARD_S, unknown)
        if split == "SNR test":
            segs = [(0, n, 1 if flight == "drone" else 0)]
        common = dict(source="d3set", group=f"d3set:{split}/{flight}", subset=split, sr_orig=sr,
                      channels_orig=info.channels, channel_used=D3_CHANNEL, licence="GPL-3.0",
                      labels="Holybro X500")
        site = "unknown"
        if "reference_position" in extra:
            site = "d3set site A" if extra["reference_position"][0] < 46 else "d3set site B"
        attrs = dict(session=f"{split}/{flight}", parent=f"{split}/{flight}/matrix.wav", site=site,
                     rig="16-mic MEMS array", utc_start=round(t_start, 3))
        add_segments(manifest, tracks, store, base, y, segs, f"{split}/{flight}", range_fn, common,
                     attrs, stats, scale=1 / 32768)
        stats["flights"] += 1
        print(f"  d3set {split}/{flight}: {n / SR:6.1f} s, {len(segs)} segments", flush=True)
    store.close()
    summary["d3set"] = {k: round(v, 1) if isinstance(v, float) else v for k, v in stats.items()}


# ---------------------------------------------------------------------- Lund

LUND_ZIP = DATA / "Ljungbyhed_Lund" / "ljungbyhed_audio_dataset.zip"
LUND_CHANNEL = 0
# From info.md inside the zip. Recordings 06 to 13 are not described there;
# their drone type comes from the recording names only where it is stated.
LUND_DRONE = {
    "01_flower": "small_quad", "02_altitudes": "small_quad", "03_altitudes_runway": "small_quad",
    "04_diagonals": "small_quad", "05_runway_length": "small_quad",
    "11_fixed_flower": "fixed_wing (name)", "12_big_flower": "big_quad (name)",
    "13_big_altitudes": "big_quad (name)",
}


def _read_pcm_channel(z: zipfile.ZipFile, name: str, channel: int) -> tuple[np.ndarray, int, int]:
    """Stream one channel out of a 24- or 32-bit integer multichannel WAV inside the zip."""
    with z.open(name) as f:
        head = f.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError(f"{name}: not a WAV file")
        fmt = None
        while True:
            cid, size = f.read(4), int.from_bytes(f.read(4), "little")
            if cid == b"fmt ":
                body = f.read(size)
                ch, sr, bits = (int.from_bytes(body[2:4], "little"), int.from_bytes(body[4:8], "little"),
                                int.from_bytes(body[14:16], "little"))
                fmt = (ch, sr, bits)
            elif cid == b"data":
                break
            else:
                f.read(size)
        ch, sr, bits = fmt
        if bits not in (24, 32):
            raise ValueError(f"{name}: expected 24- or 32-bit PCM, found {bits}")
        width = bits // 8
        frame = width * ch
        block = frame * sr                  # one second at a time
        out, left = [], size - size % frame
        while left > 0:
            buf = f.read(min(block, left))
            if not buf:
                break
            left -= len(buf)
            buf = buf[: len(buf) - len(buf) % frame]
            if width == 4:
                v = np.frombuffer(buf, dtype="<i4").reshape(-1, ch)[:, channel]
                out.append((v / float(1 << 31)).astype(np.float32))
                continue
            b = np.frombuffer(buf, dtype=np.uint8).reshape(-1, ch, 3)[:, channel]
            v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
            v = np.where(v >= 1 << 23, v - (1 << 24), v)
            out.append((v / float(1 << 23)).astype(np.float32))
    return np.concatenate(out), sr, ch


def prepare_lund(manifest: list[dict], tracks: list[dict], summary: dict) -> None:
    store = Store("lund", "float32")
    stats = collections.Counter()
    z = zipfile.ZipFile(LUND_ZIP)
    root = "ljungbyhed_audio_dataset"
    recs = sorted({n.split("/")[1] for n in z.namelist() if n.endswith("/sound.wav")})
    for rec in recs:
        t0 = time.time()
        x, sr, ch = _read_pcm_channel(z, f"{root}/{rec}/sound.wav", LUND_CHANNEL)
        y = to_16k(x.astype(np.float64), sr).astype(np.float32)
        del x
        base, n = store.add(y)
        gt = list(csv.DictReader(io.StringIO(z.read(f"{root}/{rec}/ground_truth.csv").decode())))
        rx = list(csv.DictReader(io.StringIO(z.read(f"{root}/{rec}/receivers_gt.csv").decode())))
        centre = np.array([[float(r["X"]), float(r["Y"]), float(r["Z"])] for r in rx]).mean(axis=0)
        t = np.array([float(r["time"]) for r in gt])
        p = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in gt]) - centre
        horiz, zz = np.hypot(p[:, 0], p[:, 1]), p[:, 2]
        slant = np.hypot(horiz, zz)

        def range_fn(t_rel, t=t, horiz=horiz, zz=zz):
            hz, z1 = np.interp(t_rel, t, horiz), np.interp(t_rel, t, zz)
            return np.hypot(hz, z1), hz, z1

        # Ground truth covers the flight. Before and after it the drone stands at
        # its take-off point, 75-375 m away, and nothing says whether its motors
        # run: those stretches stay "uncertain" until checked.
        lo, hi = float(t.min()), float(t.max())
        segs = segments_from_spans(n, [(lo, hi)], 0.0, unknown_spans=[(0.0, lo), (hi, n / SR)])
        common = dict(source="lund", group=f"lund:{rec}", subset=rec[:2], sr_orig=sr, channels_orig=ch,
                      channel_used=LUND_CHANNEL, licence="none stated (internal use only)",
                      labels=LUND_DRONE.get(rec, "unknown"))
        attrs = dict(session=rec, parent=f"{rec}/sound.wav", site="Ljungbyhed airfield", rig="12-mic array",
                     takeoff_range_m=round(float(slant[0]), 1),
                     gt_drone_live_false=int(sum(r["drone_live"] != "True" for r in gt)))
        add_segments(manifest, tracks, store, base, y, segs, rec, range_fn, common, attrs, stats)
        stats["recordings"] += 1
        print(f"  lund {rec}: {n / SR:6.1f} s, ground truth {lo:.0f}-{hi:.0f} s, "
              f"{time.time() - t0:.0f} s", flush=True)
    store.close()
    summary["lund"] = {k: round(v, 1) if isinstance(v, float) else v for k, v in stats.items()}


# ------------------------------------------------------------------- Glasgow

GLA_ROOT = DATA / "extracted" / "Glasgow"
GLA_NAME = re.compile(r"_UAV_(?P<date>\d{8})_d(?P<drone>\d+)_(?P<state>[a-z]+)_(?P<dist>\d+)m_"
                      r"(?P<comb>[0-9A-Z]+)_(?P<batt>\d+)%_(?P<idx>\d+)_", re.IGNORECASE)


def prepare_glasgow(manifest: list[dict], tracks: list[dict], summary: dict) -> None:
    if not GLA_ROOT.exists():
        raise FileNotFoundError(f"{GLA_ROOT} is missing; extract the Glasgow 7z first (see module docstring)")
    store = Store("glasgow", "int16")
    stats = collections.Counter()
    for wav in sorted(GLA_ROOT.rglob("*.WAV")):
        info = sf.info(str(wav))
        x, sr = sf.read(str(wav), dtype="float64", always_2d=True)
        y = as_int16(to_16k(x[:, 0], sr))
        base, n = store.add(y)
        rms, peak = level(y / 32768)
        m = GLA_NAME.search(wav.name)
        if m:
            label, kind, rng = 1, "drone", int(m["dist"])
            attrs = dict(date=m["date"], drone=int(m["drone"]), state=m["state"], range_m=rng,
                         combination=m["comb"], battery_pct=int(m["batt"]), index=int(m["idx"]))
            labels = f"drone {int(m['drone'])}"
        else:
            label, kind, attrs, labels = 0, "background", dict(file=wav.name), "noise"
        manifest.append(row(store, base, n, source="glasgow", recording_id=wav.stem, label=label,
                            kind=kind, group="glasgow:all", subset=wav.parent.name, sr_orig=sr,
                            channels_orig=info.channels, channel_used=0, licence="CC BY 4.0",
                            rms_dbfs=rms, peak=peak, labels=labels, attributes=json.dumps(attrs)))
        stats[f"{kind}_files"] += 1
        stats[f"{kind}_s"] += n / SR
    store.close()
    summary["glasgow"] = {k: round(v, 1) if isinstance(v, float) else v for k, v in stats.items()}
    print(f"  glasgow: {dict(stats)}", flush=True)


# ----------------------------------------------------------------------- FPV

FPV_ZIP = DATA / "Acoustic_UAV_Field_Recordings" / "Fpv.zip"
# Folder names as published; the first carries a Hebrew vowel point.
FPV_CAMPAIGN = {"ariel_25.02": "ariel_25.02", "palmahaim_10.02": "palmahim_10.02"}


def prepare_fpv(manifest: list[dict], tracks: list[dict], summary: dict) -> None:
    """FPV campaigns of Acoustic UAV Field Recordings, from `labeled_data` only.

    The shipped `ProcessedAudioSplits` reuse one source segment across train,
    validate and test (entry 033), so they are not read. Every drone or
    background file is one manifest row. The phones of one campaign recorded
    the same flights at the same time, so the group is the campaign.
    """
    store = Store("fpv", "int16")
    stats = collections.Counter()
    seen: dict[str, str] = {}
    z = zipfile.ZipFile(FPV_ZIP)
    names = sorted(n for n in z.namelist() if "/labeled_data/" in n and "ProcessedAudioSplits" not in n
                   and n.lower().endswith(".wav"))
    for name in names:
        parts = name.split("/")
        camp_raw, device, cls = parts[2], parts[3], parts[-2]
        campaign = FPV_CAMPAIGN.get(camp_raw, "palmahim_27.05" if "27_5" in camp_raw else camp_raw)
        payload = z.read(name)
        if payload[:4] != b"RIFF":
            stats["not_wav_skipped"] += 1           # M4A saved with a .wav name
            continue
        digest = hashlib.md5(payload[44:]).hexdigest()
        if digest in seen:
            stats["duplicate_skipped"] += 1
            continue
        seen[digest] = name
        x, sr = sf.read(io.BytesIO(payload), dtype="float64", always_2d=True)
        if len(x) < sr:                              # under 1 s: no full window
            stats["too_short_skipped"] += 1
            continue
        y = as_int16(to_16k(x[:, 0], sr))
        offset, n = store.add(y)
        rms, peak = level(y / 32768)
        label = 1 if cls == "drone" else 0
        manifest.append(row(
            store, offset, n, source="fpv", recording_id=f"{campaign}/{device}/{cls}/{Path(name).stem}",
            label=label, kind="drone" if label else "background", group=f"fpv:{campaign}", subset=device,
            sr_orig=sr, channels_orig=x.shape[1], channel_used=0, licence="CC BY 4.0",
            rms_dbfs=rms, peak=peak, labels="FPV",
            attributes=json.dumps(dict(session=campaign, campaign=campaign, device=device, file=name,
                                       site=campaign, rig=device)),
        ))
        stats[f"{campaign}_{'drone' if label else 'background'}_s"] += n / SR
        stats["files"] += 1
    store.close()
    summary["fpv"] = {k: round(v, 1) if isinstance(v, float) else v for k, v in stats.items()}
    print(f"  fpv: {summary['fpv']}", flush=True)


def prepare_pairs(manifest: list[dict], tracks: list[dict], summary: dict) -> None:
    """Two-microphone stores for Nightjar and d3set; the manifest is not touched."""
    global PAIR, STORE_SUFFIX
    PAIR, STORE_SUFFIX = True, "_pair"
    try:
        for fn in (prepare_nightjar, prepare_d3set):
            fn([], [], {})
    finally:
        PAIR, STORE_SUFFIX = False, ""
    for name in ("nightjar", "d3set"):
        a, b = (OUT / f"{name}.i16").stat().st_size, (OUT / f"{name}_pair.i16").stat().st_size
        if a != b:
            raise RuntimeError(f"{name}: pair store has {b} bytes, one-channel store {a}")
    summary["pairs"] = dict(nightjar=f"channels {NJ_CHANNEL} and {NJ_SECOND}", d3set="first and last channel")


STEPS = {
    "pairs": prepare_pairs,
    "fpv": prepare_fpv,
    "nightjar": prepare_nightjar,
    "d3set": prepare_d3set,
    "lund": prepare_lund,
    "glasgow": prepare_glasgow,
}


def main(argv: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = argv or list(STEPS)
    manifest_path, summary_path, track_path = OUT / "manifest.csv", OUT / "summary.json", OUT / "range_track.csv"
    with open(manifest_path, newline="", encoding="utf-8") as f:
        manifest = [r for r in csv.DictReader(f) if r["source"] not in wanted]
    tracks: list[dict] = []
    if track_path.exists():
        with open(track_path, newline="", encoding="utf-8") as f:
            tracks = [r for r in csv.DictReader(f) if r["source"] not in wanted]
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for name in wanted:
        t0 = time.time()
        print(f"[{name}]", flush=True)
        STEPS[name](manifest, tracks, summary)
        print(f"  done in {time.time() - t0:.0f} s", flush=True)

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(manifest)
    with open(track_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRACK_FIELDS)
        w.writeheader()
        w.writerows(tracks)

    totals = collections.defaultdict(lambda: collections.Counter())
    for r in manifest:
        t = totals[r["source"]]
        t["recordings"] += 1
        t["seconds"] += float(r["duration_s"])
        lab = int(r["label"])
        t["positive_seconds" if lab == 1 else "negative_seconds" if lab == 0 else "uncertain_seconds"] += \
            float(r["duration_s"])
    summary["totals"] = {s: {k: round(v, 1) for k, v in t.items()} for s, t in totals.items()}
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({s: summary["totals"][s] for s in wanted if s in summary["totals"]}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
