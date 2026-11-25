"""The complete test without the GUI, split over the GPUs the way the training runs are (entry 123):

    uv run python scripts/complete_test_run.py PLAN.json [--gpus N] [--out DIR]

PLAN.json is a plan saved with "Save plan..." in Complete_Test_Gui.py, or the config.json of an earlier
run (then its soundscape start is kept too). The rounds are dealt out round-robin to N worker processes;
worker k sees only GPU k (CUDA_VISIBLE_DEVICES=k) and scores the networks there, while mixing and comb
stay on its CPU. Worker k writes OUT/shard_k/; at the end the shards are merged into OUT/ (rounds.csv,
scores.npz, window_snr.npz, layer_snr.npz, summary.csv, config.json), which "Open results..." in the GUI opens like any other run.
A live plan (automatic, virtual) is scored on its one continuous timeline, as live_test_engine plays it: each
round's soundscape goes on from the round before (offset_s in live.json), but not in real time.
--gpus defaults to every GPU torch sees; --gpus 0 runs one worker on the CPU.

Drone files in the plan are found from their "data/" folder on (complete_test_engine.portable), so a
plan saved on the PC runs on the server, and the merged run opens back on the PC.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import complete_test_engine as E  # noqa: E402


def load_plan(path) -> tuple[E.Plan, float | None]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return E.Plan(**d["plan"]), d.get("soundscape_start_s")


LEAD_IN_S = 3.0                                             # live_test_engine's (not imported: sounddevice, no PortAudio here)


class Continuous(E.Test):
    """A live plan's rounds one after the other in the soundscape, as live_test_engine plays them."""

    def offset(self, r: dict) -> float:
        return self.start_s + LEAD_IN_S + (r["round"] - 1) * self.plan.length_s

    def rebuild(self, r: dict) -> dict:
        return E.mix_round(r, self.plan, self.layers_for(r), self.drones, self.offset(r))


def worker(a):
    plan, start = load_plan(a.plan)
    t = (Continuous if plan.live else E.Test)(plan, a.out / f"shard_{a.shard}", device=a.device)
    if start is not None:
        t.start_s = start
    t.rounds = t.rounds[a.shard::a.of]
    t.run(progress=lambda s: print(f"[worker {a.shard}] {s}", flush=True))


def merge(out: Path, n: int):
    shards = [out / f"shard_{k}" for k in range(n)]
    rows, fields, npz = [], None, {f: {} for f in ("scores", "window_snr", "layer_snr")}
    for s in shards:
        with open(s / "rounds.csv", newline="", encoding="utf-8") as fh:
            r = csv.DictReader(fh)
            fields = r.fieldnames
            rows += list(r)
        for f, got in npz.items():
            with np.load(s / f"{f}.npz") as d:
                got.update({k: d[k] for k in d.files})
    rows.sort(key=lambda r: int(r["round"]))                # stable: detectors keep their order within a round
    with open(out / "rounds.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        w.writerows(rows)
    for f, got in npz.items():
        np.savez_compressed(out / f"{f}.npz", **got)
    (out / "config.json").write_bytes((shards[0] / "config.json").read_bytes())
    plan, start = load_plan(out / "config.json")
    t = E.Test(plan, out)
    t.start_s, t.results = start, E.read_results(out)
    E.write_summary(t)
    if plan.live:                                            # where each round sits, for the GUI to rebuild it
        c = Continuous(plan, out)
        c.start_s = start
        (out / "live.json").write_text(json.dumps(dict(source="virtual", trigger="automatic", lead_in_s=LEAD_IN_S,
            stopped=False, rounds={str(r["round"]): dict(offset_s=c.offset(r)) for r in c.rounds}), indent=1), encoding="utf-8")
    want = len(plan.rounds())
    print(f"merged {n} shards: {len(t.results)} of {want} rounds -> {out}", flush=True)
    if len(t.results) != want:
        raise SystemExit("some rounds are missing; the shard folders are kept")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("plan", type=Path)
    ap.add_argument("--gpus", type=int, help="workers, one per GPU (default: every GPU; 0: one worker on the CPU)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--shard", type=int, help=argparse.SUPPRESS)  # set for the workers only
    ap.add_argument("--of", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--device", default="cpu", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.shard is not None:
        return worker(a)

    plan, _ = load_plan(a.plan)
    probs = plan.check()
    if plan.live and (plan.live["source"], plan.live["trigger"]) != ("virtual", "automatic"):
        probs.append("only an automatic, virtual live plan can be run here")
    if probs:
        raise SystemExit("\n".join(probs))
    if a.gpus is None:
        import torch
        a.gpus = torch.cuda.device_count()
    n = max(a.gpus, 1)
    out = a.out or E.OUT_ROOT / f"{time.strftime('%Y%m%d_%H%M%S')}_{plan.kind}"
    print(f"{plan.count()['total']} rounds x {len(plan.detectors)} detectors on {n} worker(s) "
          f"({'GPU' if a.gpus else 'CPU'}) -> {out}", flush=True)
    procs = []
    for k in range(n):
        env = {**os.environ, **({"CUDA_VISIBLE_DEVICES": str(k)} if a.gpus else {})}
        procs.append(subprocess.Popen([sys.executable, "-u", __file__, str(a.plan), "--out", str(out), "--shard", str(k),
                                       "--of", str(n), "--device", "cuda" if a.gpus else "cpu"], env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"worker exit codes {codes}; the shard folders in {out} are kept")
    merge(out, n)


if __name__ == "__main__":
    main()
