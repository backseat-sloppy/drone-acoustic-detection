"""Report entry 230: iteration 23's 2 s colour networks on MAD (no drone), against it19 CRNN colour s1 and comb.

    .venv/bin/python scripts/mad_nodrone.py --device cuda      # writes artifacts/mad_nodrone/windows.csv, prints the table
    .venv/bin/python scripts/mad_nodrone.py --table             # the table again from the CSV

MAD (docs/datasets.md, 2026-11-12) is a test-only source: 7,466 clips of military-video audio, 16 kHz
mono, 1-10 s, labelled communication, shooting, footsteps, shelling, vehicle, helicopter, fighter.
None is assumed to hold a drone. Every clip is played at its own digital level (no dB SPL
calibration), and only clips of at least 2 s are scored so every detector sees the same clips.
Per detector and class: mean confidence over every window, and the share of windows over the
detector's own threshold (5% FA on validation, from its run file).
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import complete_test_engine as E  # noqa: E402

dwg = E.dwg
MAD = dwg.ROOT / "data/extracted/MAD/MAD_dataset"
CLASSES = ["communication", "shooting", "footsteps", "shelling", "vehicle", "helicopter", "fighter"]
IT24 = "--it24" in sys.argv   # iteration 24: add its networks, score only MAD's held-out videos (prepare_mad.py)
ITS = ("it23", "it24") if IT24 else ("it23",)
NAMES = ["it19 CRNN colour s1", *(f"{it} {a} 2s colour s{s}" for it in ITS for a in ("CNN", "CRNN") for s in (0, 1, 2)), "comb"]
for it, run in (("it23", "it23_2s_colour"), ("it24", "it24_2s_mad")):   # as drone_war_grid.MODELS, for checkouts without them
    for a, arch in (("CNN", "dscnn"), ("CRNN", "crnn")):
        for s in (0, 1, 2):
            rec = "simdist_colour_mad" if it == "it24" else "simdist_colour"
            dwg.MODELS.setdefault(f"{it} {a} 2s colour s{s}", (dwg.ROOT / f"artifacts/runs/{run}_s{s}.json",
                                                               dwg.MODEL_DIR / f"{run}_s{s}_none_{rec}_{arch}.pt"))
OUT = dwg.ROOT / ("artifacts/mad_nodrone_it24" if IT24 else "artifacts/mad_nodrone")
MIN_S = 2.0


def table():
    cell = defaultdict(list)
    for r in csv.DictReader(open(OUT / "windows.csv", newline="", encoding="utf-8")):
        arm = r["detector"].rsplit(" s", 1)[0] if r["detector"][:4] in ("it23", "it24") else r["detector"]
        for c in (r["class"], "all"):
            cell[(arm, c)].append((float(r["mean_conf"]), float(r["alarm_share"]), int(r["windows"])))
    arms = list(dict.fromkeys(k[0] for k in cell))
    print("mean confidence % / windows over threshold %, per clip, clips >= 2 s; it23/it24 three seeds pooled"
          + ("; MAD held-out videos only" if IT24 else ""))
    print(f"{'class':14}" + "".join(f"{a:>26}" for a in arms))
    for c in [*CLASSES, "all"]:
        line = f"{c:14}"
        for a in arms:
            v = cell[(a, c)]
            w = sum(x[2] for x in v)
            line += f"{100 * statistics.mean(x[0] for x in v):>16.1f} / {100 * sum(x[1] * x[2] for x in v) / w:5.1f}"
        print(line)


if "--table" not in sys.argv:
    device = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"
    rows = [r for s in ("training", "test") for r in csv.DictReader(open(MAD / f"{s}.csv", newline="", encoding="utf-8"))]
    if IT24:
        held = set((dwg.ROOT / "data/processed/mad/heldout_videos.txt").read_text(encoding="utf-8").split())
        rows = [r for r in rows if r["youtube url"].split("&")[0] in held]
    clips = []
    for r in rows:
        x, fs = sf.read(MAD / r["path"], dtype="float32", always_2d=True)
        if fs != E.SR:
            raise SystemExit(f"{r['path']}: {fs} Hz")
        if len(x) >= MIN_S * fs:
            clips.append((r["path"], CLASSES[int(r["label"])], x[:, 0]))
    print(f"{len(clips)} of {len(rows)} clips are at least {MIN_S:g} s", flush=True)
    thr, scorers = E.load_detectors(NAMES, device)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "windows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["detector", "path", "class", "windows", "mean_conf", "alarm_share"])
        for k, sc in scorers.items():
            for path, cls, x in clips:
                d = E.stream(E.Clip(path, x, E.SR, 1, "FLOAT"), sc, sc.cfg, thr[k])
                conf = dwg.squash(d.score, thr[k]) if k == "comb" else d.score
                w.writerow([k, path, cls, len(conf), float(np.mean(conf)), float(np.mean(d.drone))])
            print(k, "done", flush=True)
table()
