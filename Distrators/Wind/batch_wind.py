"""Render many wind wavs, each with random speed / resonance / volume picked from ranges you set.

Example:
  python batch_wind.py --n 20 --seconds 15 --speed 20 80 --reso 0 60 --vol 40 90 --outdir wind_dataset

All values are slider positions (0-127). A labels.csv is written next to the wavs, listing the
exact values used for each file.
"""
import argparse, csv, os, random
from concurrent.futures import ThreadPoolExecutor
from render_wind import render, HERE

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10, help="how many files")
ap.add_argument("--seconds", type=float, default=10, help="length of each file")
ap.add_argument("--speed", type=float, nargs=2, default=[0, 127], metavar=("MIN", "MAX"))
ap.add_argument("--reso", type=float, nargs=2, default=[0, 127], metavar=("MIN", "MAX"))
ap.add_argument("--vol", type=float, nargs=2, default=[30, 100], metavar=("MIN", "MAX"))
ap.add_argument("--outdir", default="wind_dataset", help="folder (relative = next to this script)")
ap.add_argument("--seed", type=int, default=None, help="set for a repeatable batch")
ap.add_argument("--workers", type=int, default=4, help="Pd instances rendering at the same time")
a = ap.parse_args()

outdir = a.outdir if os.path.isabs(a.outdir) else os.path.join(HERE, a.outdir)
os.makedirs(outdir, exist_ok=True)
rng = random.Random(a.seed)

jobs = []
for i in range(1, a.n + 1):
    jobs.append(dict(
        file=f"wind_{i:03d}.wav",
        speed=round(rng.uniform(*a.speed), 1),
        reso=round(rng.uniform(*a.reso), 1),
        vol=round(rng.uniform(*a.vol), 1),
        seconds=a.seconds,
    ))

def run(j):
    render(j["speed"], j["reso"], j["vol"], j["seconds"], os.path.join(outdir, j["file"]))
    print("done", j["file"], "speed", j["speed"], "reso", j["reso"], "vol", j["vol"], flush=True)
    return j

with ThreadPoolExecutor(max_workers=a.workers) as ex:
    results = list(ex.map(run, jobs))

with open(os.path.join(outdir, "labels.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file", "speed", "reso", "vol", "seconds"])
    w.writeheader(); w.writerows(results)
print("finished:", outdir)
