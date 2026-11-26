"""Copy a Complete Test run folder without the rounds of one drone file, and write its summary.csv again.

    uv run python scripts/filter_run.py RUN [RUN ...] --drop Steady_10 [--drop Steady_07 ...] [--suffix _no_Steady_10]

Nothing is scored again: the kept rounds' rows, scores and SNRs are copied as they are, and summary.csv is
recomputed from them by the same code as any run (Complete_Test_Gui.load_test). The source folder is not touched,
and an existing output folder is never written over. config.json keeps the full plan (round numbers depend on it)
and gains a "filtered" note naming what was dropped.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def filter_run(src: Path, out: Path, drop: list[str]) -> tuple[int, int]:
    """The rounds dropped and kept (rounds whose drone name contains any of `drop`)."""
    with open(src / "rounds.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for d in drop:
        if not any(d in (r["drone"] or "") for r in rows):
            raise SystemExit(f"{src.name}: no round has a drone named like {d!r}")
    gone = {r["round"] for r in rows if any(d in (r["drone"] or "") for d in drop)}
    kept = [r for r in rows if r["round"] not in gone]
    out.mkdir(parents=True)                                  # fails if it exists: never written over
    with open(out / "rounds.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(kept)
    for f in ("scores.npz", "window_snr.npz", "layer_snr.npz"):
        z = np.load(src / f)
        np.savez_compressed(out / f, **{k: z[k] for k in z.files if k.split("|")[0] not in gone})
    live = json.loads((src / "live.json").read_text(encoding="utf-8"))
    if isinstance(live.get("rounds"), dict):
        live["rounds"] = {k: v for k, v in live["rounds"].items() if k not in gone}
    (out / "live.json").write_text(json.dumps(live, indent=1), encoding="utf-8")
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    cfg["filtered"] = dict(source=src.name, dropped_drones=drop, dropped_rounds=sorted(gone, key=int))
    (out / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return len(gone), len({r["round"] for r in kept})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--drop", required=True, action="append", help="part of a drone file name, e.g. Steady_10; repeat for more")
    ap.add_argument("--suffix", help="added to each folder name (default _no_<drop>_<drop>...)")
    a = ap.parse_args()
    import Complete_Test_Gui as G
    for src in (p.resolve() for p in a.runs):
        out = src.with_name(src.name + (a.suffix or "_no_" + "_".join(a.drop)))
        n_gone, n_kept = filter_run(src, out, a.drop)
        t = G.load_test(out)                                 # writes summary.csv from the kept rounds
        print(f"{out.name}: dropped {n_gone} rounds, kept {n_kept}; {len(t.results)} rounds with results")
