"""All 35 ITU ARIS drone intervals, rebuilt as full continuous recordings.

ITU ARIS v1.1.0 (Zenodo 10.5281/zenodo.22682338, CC BY 4.0) publishes its drone audio as 4491
segments of 1 s at 22.05 kHz, cut from 35 hand-annotated intervals of one 3.12 h session with a
0.5 s hop (README; build_acolab_groups.py): an interval of D s has 2D - 1 segments, segment k
starting at 0.5 k s. Segments 0, 2, 4, ... 2D - 2 therefore tile the interval exactly, end to end.

For every interval in groups.json this script
    * takes its segments in file order and checks their count against groups.json n_segments
      and against 2D - 1;
    * checks every overlap: the second half of each segment equals the first half of the next
      (max abs difference must be 0 in the int16 samples), so the order and the tiling are right;
    * joins segments 0, 2, 4, ... (plus the last half-segment if a count is even, which does not
      occur in v1.1.0) and checks the result is D s long;
    * writes OUT/ITU_ARIS_interval_NN.wav (NN = groups.json group_id), 22.05 kHz mono PCM16 with
      the published samples unchanged, and OUT/index.json with each file's interval, source times,
      segments and checks. Nothing is resampled, scaled or filtered.

The 44.1 kHz session recording itself is not published; 22.05 kHz is the best available.
OUT must not exist yet or be a folder this script wrote (it holds index.json); raw data is only read.

    uv run python scripts/aris_intervals.py [--out data/extracted/All]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
ARIS = ROOT / "data/extracted/ITU_ARIS_Lab/itu-aris-lab-acoustic-drone-dataset"
FS = 22050


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "data/extracted/All")
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()) and not (a.out / "index.json").exists():
        raise SystemExit(f"{a.out} holds files this script did not write; choose another --out")

    g = json.loads((ARIS / "groups.json").read_text(encoding="utf-8"))
    segs: dict[str, list[str]] = {}
    for s in g["segments"]:
        if s["label"] == "drone":
            segs.setdefault(s["group"], []).append(s["file"])

    a.out.mkdir(parents=True, exist_ok=True)
    index, problems, total = [], [], 0.0
    for iv in g["intervals"]:
        gid = iv["group_id"]
        grp = f"drone_interval_{gid:02d}"
        files = sorted(segs.get(grp, []))
        D = iv["duration_s"]
        x = [sf.read(ARIS / f, dtype="int16")[0] for f in files]
        half = FS // 2
        bad_len = [f for f, s in zip(files, x) if len(s) != FS]
        diffs = [int(np.abs(x[k][half:].astype(np.int32) - x[k + 1][:half]).max()) for k in range(len(x) - 1)]
        parts = x[0::2] + ([x[-1][half:]] if len(x) % 2 == 0 else [])
        y = np.concatenate(parts)
        checks = dict(n_segments=len(files), n_segments_groups_json=iv["n_segments"], expected_2D_minus_1=int(2 * D - 1),
                      all_segments_1s=not bad_len, max_overlap_diff=max(diffs) if diffs else 0,
                      length_s=len(y) / FS, length_matches_duration=abs(len(y) / FS - D) < 1e-9)
        ok = (checks["n_segments"] == iv["n_segments"] == checks["expected_2D_minus_1"] and not bad_len
              and checks["max_overlap_diff"] == 0 and checks["length_matches_duration"])
        if not ok:
            problems.append((grp, checks))
        name = f"ITU_ARIS_interval_{gid:02d}.wav"
        sf.write(a.out / name, y, FS, subtype="PCM_16")
        total += len(y) / FS
        index.append(dict(file=name, group=grp, session_start_s=iv["start_s"], session_end_s=iv["end_s"],
                          duration_s=D, first_segment=files[0], last_segment=files[-1], checks=checks, ok=ok))
        print(f"{name}  {D:6.1f} s  session {iv['start_s']:8.1f}-{iv['end_s']:8.1f} s  "
              f"{len(files):3d} segments  overlap diff {checks['max_overlap_diff']}  {'ok' if ok else 'CHECK FAILED'}")

    (a.out / "index.json").write_text(json.dumps(dict(
        source="ITU ARIS Lab. Acoustic Drone Dataset v1.1.0, Muhaciroglu and Akgul (2026), Zenodo, "
               "doi 10.5281/zenodo.22682338, CC BY 4.0",
        built_by="scripts/aris_intervals.py", sample_rate_hz=FS, n_files=len(index),
        total_s=total, files=index), indent=1), encoding="utf-8")
    print(f"{len(index)} intervals, {total:.0f} s -> {a.out}")
    if problems:
        raise SystemExit(f"{len(problems)} intervals failed a check: {[p[0] for p in problems]}")


if __name__ == "__main__":
    main()
