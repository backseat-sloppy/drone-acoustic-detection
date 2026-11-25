"""Combine the parts of a live plan played as several runs (the GUI's "play the plan as N runs") into one run folder
that Results and Conclusions open like any other. Nothing is scored again and the parts are left as they are.

    uv run python scripts/combine_live_runs.py PART1 PART2 ... [--out DIR]

The plans must be the same and every round may come from one part only. Not combined: the REAPER projects and their
playback WAVs (they stay in the parts). heard/ is hard-linked, so it takes no extra disk. `timeline_s` starts over in
each part, as each was its own REAPER project; `offset_s` (the place in the soundscape) goes on through all of them.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def combine(parts: list[Path], out: Path) -> int:
    """The combined folder; how many rounds it has."""
    cfgs = [json.loads((p / "config.json").read_text(encoding="utf-8")) for p in parts]
    lives = [json.loads((p / "live.json").read_text(encoding="utf-8")) for p in parts]
    for c in cfgs:                                           # what may differ: which part, and its REAPER project
        c["plan"]["live"].pop("part", None)
        c["plan"]["live"].pop("reaper_project", None)
    for p, c in zip(parts, cfgs):
        if c != cfgs[0]:
            raise SystemExit(f"{p.name}: not the same plan, thresholds or devices as {parts[0].name}")
    rounds = {k: v for j in lives for k, v in j["rounds"].items()}
    if len(rounds) != sum(len(j["rounds"]) for j in lives):
        raise SystemExit("the same round is in more than one part")
    csvs = [(p / "rounds.csv").read_text(encoding="utf-8").splitlines(keepends=True) for p in parts]
    if any(c[0] != csvs[0][0] for c in csvs):
        raise SystemExit("rounds.csv has different columns in the parts")

    (out / "heard").mkdir(parents=True)                      # an existing folder is never written over
    (out / "rounds.csv").write_text(csvs[0][0] + "".join(x for c in csvs for x in c[1:]), encoding="utf-8")
    for f in ("scores.npz", "window_snr.npz", "layer_snr.npz"):
        z = [np.load(p / f) for p in parts]
        np.savez_compressed(out / f, **{k: x[k] for x in z for k in x.files})
    for p in parts:
        for w in (p / "heard").glob("*.wav"):
            os.link(w, out / "heard" / w.name)
    cfgs[0]["combined_from"] = [p.name for p in parts]
    (out / "config.json").write_text(json.dumps(cfgs[0], indent=1), encoding="utf-8")
    live = {k: v for k, v in lives[0].items() if k != "reaper_project"}
    live.update(stopped=any(j["stopped"] for j in lives), wall_s=round(sum(j["wall_s"] for j in lives), 1),
                clipped_samples=sum(j.get("clipped_samples", 0) for j in lives),
                combined_from=[p.name for p in parts], rounds=rounds)
    (out / "live.json").write_text(json.dumps(live, indent=1), encoding="utf-8")
    return len(rounds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    parts = [p.resolve() for p in a.parts]
    out = (a.out or parts[0].with_name(parts[0].name + "_combined")).resolve()
    n = combine(parts, out)
    import Complete_Test_Gui as G                            # opens it as "Open results..." does; writes summary.csv
    t = G.load_test(out)
    print(f"{n} rounds in {out}; opened: {len(t.results)} rounds with results, {len(t.rounds)} in the plan")
