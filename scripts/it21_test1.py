"""Report entry 208: test 1's shape, `it19_colour` against iteration 21's CRNNs, three seeds each.

    for k in 0 1 2 3 4 5; do OMP_NUM_THREADS=1 .venv/bin/python scripts/it21_test1.py --shard $k 6 & done; wait
    .venv/bin/python scripts/it21_test1.py --table        # the comparison, from every shard

Without --shard the whole plan runs in one process; --quick is a 16-round trial. With --it23 (on every
call) the candidates are iteration 23's 2 s colour DS-CNN and CRNN, and the output is it23_test1/.
On the server: `CUDA_VISIBLE_DEVICES=$k ... --shard $k 3 --device cuda`, with the Mac's car render in
artifacts/complete_test/layers/ (no Pure Data there).

The ten ITU ARIS excerpts in data/extracted/sealed_peek at 60-100 dB SPL, against the war soundscape at
90 dB SPL and the cars layer (six development engines, 50 km/h, 30 m, as entry 179) at 20, 35 and
50 dB SPL, each layer also absent. ITU ARIS is only scored here. The six networks are registered for
this run only: iteration 21 failed its box, so they are not in drone_war_grid.MODELS or any GUI.
The cars and war layers are synthetic mixes: these figures are not detection performance.
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import complete_test_engine as E  # noqa: E402

dwg = E.dwg
RUNS = dwg.ROOT / "artifacts/runs"
NETS = {}
IT23 = "--it23" in sys.argv   # entry 211: the 2 s colour networks of iteration 23 against it19_colour
for s in (0, 1, 2):   # the reference's threshold comes from its re-score (entry 207), the same as its own run's
    if not IT23 or s == 1:   # with --it23 only the GUI's colour model, it19 CRNN colour s1 (entry 228)
        NETS[f"it19 colour s{s}"] = (RUNS / f"it22_ref_s{s}.json", dwg.MODEL_DIR / f"it19_colour_s{s}_none_simdist_colour_crnn.pt")
    if IT23:
        for arch in ("dscnn", "crnn"):
            NETS[f"it23 2s colour {arch} s{s}"] = (RUNS / f"it23_2s_colour_s{s}.json",
                                                   dwg.MODEL_DIR / f"it23_2s_colour_s{s}_none_simdist_colour_{arch}.pt")
    else:
        NETS[f"it21 uav16 s{s}"] = (RUNS / f"it21_uav16_s{s}.json", dwg.MODEL_DIR / f"it21_uav16_s{s}_none_simdist_colour_crnn.pt")
dwg.MODELS.update(NETS)
OUT = E.OUT_ROOT / ("it23_test1" if IT23 else "it21_test1")
quick = "--quick" in sys.argv


def table():
    rows = [r for f in sorted(OUT.glob("**/rounds.csv")) for r in csv.DictReader(open(f, newline="", encoding="utf-8"))]
    print(f"{len(rows)} rows from {len(list(OUT.glob('**/rounds.csv')))} file(s)")
    cell = defaultdict(list)
    for r in rows:
        arm = r["detector"].rsplit(" s", 1)[0]
        bg = f"war {r.get('war soundscape_db') or 'off'} / cars {r.get('cars_db') or 'off'}"
        if r["drone"] != "none":
            cell[(bg, arm, "drone", r["drone_level_db"])].append(float(r["mean_conf_drone"]))
        if r["mean_conf_no_drone"] not in ("", "nan"):
            cell[(bg, arm, "none", "drone round" if r["drone"] != "none" else "no-drone round")].append(float(r["mean_conf_no_drone"]))
    print("mean confidence %, three seeds pooled; n = rounds x seeds")
    for key in sorted(cell):
        print(f"{key[0]:28} {key[1]:22} {key[2]:5} {key[3]:>15}  {100 * statistics.mean(cell[key]):5.1f}  n={len(cell[key])}")


if "--table" not in sys.argv:
    drones = sorted(str(p) for p in (dwg.ROOT / "data/extracted/sealed_peek").glob("Drone_*.wav"))
    plan = E.Plan(kind="real", length_s=20.0, drones=drones[:1] if quick else drones,
                  drone_levels=[60.0] if quick else [60.0, 70.0, 80.0, 90.0, 100.0],
                  layers={"war soundscape": [90.0], "cars": [50.0] if quick else [20.0, 35.0, 50.0]},
                  detectors=list(NETS)[:2] if quick else list(NETS), step_s=5.0, no_drone_rounds=True,
                  also_without=["war soundscape", "cars"],
                  pd_settings=dict(wind=dict(E.PD_SETTINGS["wind"]), cars=dict(speed_kmh=50.0, dist_m=30.0)))
    problems = plan.check()
    if problems:
        raise SystemExit("\n".join(problems))
    print(plan.count(), flush=True)
    if quick:
        OUT = E.OUT_ROOT / "it21_test1_quick"
    shard = sys.argv[sys.argv.index("--shard") + 1:][:2] if "--shard" in sys.argv else None
    device = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"
    t = E.Test(plan, OUT / f"shard_{shard[0]}" if shard else OUT, device=device)
    if shard:
        t.rounds = t.rounds[int(shard[0])::int(shard[1])]
    t.run(progress=lambda s: print(s, flush=True))
    E.write_summary(t)
table()
