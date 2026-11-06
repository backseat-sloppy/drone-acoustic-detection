"""List iteration 11.2's trained models as gate-scorer arguments, one per line.

    python scripts/it11_2_models.py [--no-threshold] [artifacts/runs/it11_2_*.json ...]

Each line is "label=checkpoint@threshold" (or "label=checkpoint" with --no-threshold), from the run
JSONs of `run_range.py --rule143`. The label is <fold>_<recipe>_<model>_s<seed>, where the fold is the held-out rig.
The threshold is the run's deploy_threshold: entry 129's rule (5% FA, --deploy-fa 0.05) on the
validation sessions of the training rig only, so it never saw the held-out rig.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUFFIX = {"crnn": "crnn.pt", "panns_probe": "panns.pt", "dscnn": "dscnn.pt"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*")
    p.add_argument("--no-threshold", action="store_true")
    a = p.parse_args()
    runs = a.runs or sorted(glob.glob(str(REPO / "artifacts" / "runs" / "it11_2_*.json")))
    for f in runs:
        d = json.loads(Path(f).read_text())
        name, seed = d["args"]["name"], d["args"]["seed"]
        for r in d["results"]:
            if r["scorer"] not in SUFFIX or r.get("integrate", 1) != 1:
                continue
            held = r["arm"].rsplit(" ", 1)[-1]
            ck = f"artifacts/models/{name}_{held}_{r['recipe']}_{SUFFIX[r['scorer']]}"
            if not (REPO / ck).exists():
                continue
            label = f"{held}_{r['recipe']}_{r['scorer']}_s{seed}"
            print(f"{label}={ck}" + ("" if a.no_threshold else f"@{r['deploy_threshold']}"))


if __name__ == "__main__":
    main()
