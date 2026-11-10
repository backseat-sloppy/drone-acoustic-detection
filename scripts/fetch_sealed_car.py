"""The sealed car for The Final Evaluation: FSD50K eval clip 278188, read from Zenodo by range request.

Held apart on purpose: it is written to data/processed/stems/sealed_car/ and NOT added to stems/manifest.csv,
so no reader of the stems finds it, and dad.sealed.forbid_in_training stops it by name. It is treated as
scripts/fetch_car_engines.py treats the six development engines (at most the loudest 10 s, the same gate rule,
active RMS -26 dBFS), and its line of car_engines_sealed.txt is computed as make_tables.py computes
car_engines.txt. Nothing here listens to the clip or runs a detector on it.

    uv run python scripts/fetch_sealed_car.py
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
ID, NAME = "278188", "rally_car_idle_loop"
OUT = bs.OUT / "sealed_car"
CAR_DIR = bs.ROOT / "Pure-data+Script" / "CarDistractor"
SR, LOOP_S, ENGINE_HP_HZ, PITCH_45 = 44100, 2.5, 200.0, 45 * 31 / 800   # as make_tables.py


def steadiest(x):
    """As make_tables.py (which cannot be imported: it writes its tables when run)."""
    nf, hop = 8192, 2205
    w = np.hanning(nf)
    fr = np.array([np.abs(np.fft.rfft(x[k:k + nf] * w)) ** 2 for k in range(0, len(x) - nf, hop)])
    fq = np.fft.rfftfreq(nf, 1 / SR)
    m = (fq > 30) & (fq < 300)
    cen = (fr[:, m] * fq[m]).sum(1) / fr[:, m].sum(1)
    lev = 10 * np.log10(fr.sum(1))
    n = int(LOOP_S * SR / hop)
    if len(fr) <= n:
        return 0, len(x)
    sc = [np.std(np.log(cen[k:k + n])) * 100 + np.std(lev[k:k + n]) * 0.5 for k in range(len(fr) - n)]
    k = int(np.argmin(sc))
    return k * hop, min(int(LOOP_S * SR), len(x) - k * hop)


def a_corr(power, freqs):
    f2 = np.asarray(freqs, float) ** 2
    ra = 12194 ** 2 * f2 ** 2 / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194 ** 2))
    db = 20 * np.log10(np.maximum(ra, 1e-12)) + 2.0
    return 10 * np.log10(np.sum(power * 10 ** (db / 10)) / np.sum(power))


def main():
    gt = {r["fname"]: r["labels"] for r in csv.DictReader(fz.open_meta("FSD50K.ground_truth/eval.csv"))}
    info = json.loads(fz.open_meta("FSD50K.metadata/eval_clips_info_FSD50K.json").read())
    cd = {Path(k).stem: v for k, v in fz.central_directory(fz.PARTS).items() if k.endswith(".wav")}
    meta = info[ID]
    if not any(a in meta["license"] for a in fz.ALLOWED):
        raise SystemExit(f"{ID}: licence {meta['license']} is not CC0 or CC BY")
    disk, off, csize, method, _ = cd[ID]
    full = bs.read_mono(fz.read_member(disk, off, csize, method, fz.PARTS))
    profile = np.percentile(np.abs(bs.spec(full)[2]), 20, axis=1)
    x = bs.loudest(full, 10.0)
    before = bs.active_rms_db(x)
    y, _ = bs.gate(x, profile)
    removed = float(np.percentile(bs.frame_db(x), 10) - np.percentile(bs.frame_db(y), 10))
    kept, policy = bs.events_kept_db(x, y), "percentile"
    if kept < -1.0:                                          # a steady engine is its own floor
        policy, y, removed = "percentile->none", x, 0.0
    y, gain = bs.finish(y, -26.0)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"fsd50k_eval_{ID}.wav"
    sf.write(path, y, bs.FS, subtype="PCM_24")

    x2, sr = sf.read(path)
    st, ln = steadiest(x2)
    seg = x2[st:st + ln]
    P = np.abs(np.fft.rfft(seg)) ** 2
    fx = np.fft.rfftfreq(len(seg), 1 / sr) * PITCH_45
    keep = (fx > 0) & (fx < sr / 2)
    hp = ((fx[keep] / ENGINE_HP_HZ) ** 2 / (1 + (fx[keep] / ENGINE_HP_HZ) ** 2)) ** 2
    c = a_corr(P[keep] * hp, fx[keep]) + 10 * np.log10(np.sum(P[keep] * hp) / np.sum(P[keep])) + 20 * np.log10(np.std(seg)) + 26
    c_tyre = float((CAR_DIR / "car_engines.txt").read_text(encoding="utf-8").split("\n")[0].split()[3])
    line = f"../../data/processed/stems/sealed_car/{path.name} {NAME} {c:.2f} {c_tyre:.2f} {st} {ln} {ENGINE_HP_HZ:g}"
    (CAR_DIR / "car_engines_sealed.txt").write_text(line + "\n", encoding="utf-8")
    (OUT / "source.json").write_text(json.dumps(dict(
        source="fsd50k_eval", id=ID, title=meta["title"], uploader=meta["uploader"], license=meta["license"], labels=gt[ID],
        archive_bytes=csize, seconds=round(len(y) / bs.FS, 2), denoise=policy, noise_removed_db=round(removed, 1),
        content_kept_db=round(kept, 2), active_rms_in_dbfs=round(before, 1), gain_db=round(gain, 1)), indent=1), encoding="utf-8")
    print(f"{ID}: {csize / 1e6:.2f} MB read, {len(full) / bs.FS:.1f} s, kept {len(y) / bs.FS:.2f} s, {policy}, content kept {kept:+.2f} dB")
    print(f"licence {meta['license']}, uploader {meta['uploader']}, title {meta['title']}, labels {gt[ID]}")
    print(f"loop {st / SR:.2f}-{(st + ln) / SR:.2f} s, engine A-correction {c:.2f} dB, tyre {c_tyre:.2f} dB")
    print(line)


if __name__ == "__main__":
    main()
