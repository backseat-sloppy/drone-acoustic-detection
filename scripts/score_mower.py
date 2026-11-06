"""Gate (e) of entry 135: false alarms on the mower-only windows of the Nightjar lawn-mower session.

    python -u scripts/score_mower.py --device cuda --out artifacts/it11_1/gates_mower.json \\
        "it9 CRNN=artifacts/models/final_crnn_none_simdist_crnn.pt@0.4828" "p1=artifacts/it11_1/p1_s0_panns.pt@<thr>"

The session is rangerun.CONFUSER (nightjar-lawnmower1-20260805): the mower runs through the drone
flight, with only about a minute of mower without drone (entry 025), so the mower-only
windows are few, and that is stated with every figure. It is kept out of every training pool
(rangerun.make_pool_arm). Per model:
- the false-alarm rate on mower-only windows at the model's frozen threshold ("label=path@threshold");
  the gate is at most 25% (model 6: 55%, entry 043);
- the AUC of drone against mower-only windows within the session (entry 053's within-rig check);
- recall on the session's drone windows at the frozen threshold.
Needs data/processed/manifest.csv, so it runs in the account that has the real data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dad import rangerun as rg  # noqa: E402
from dad import realrun as rr  # noqa: E402
from dad.config import DEFAULT  # noqa: E402
from dad.realdata import load_manifest  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", help='"label=checkpoint.pt@threshold"')
    p.add_argument("--device", default="cpu")
    p.add_argument("--hop", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=REPO / "artifacts" / "runs" / "gates_mower.json")
    a = p.parse_args()

    recs = rg.make_pool_arm(load_manifest(), "none").confuser
    if not recs:
        raise SystemExit(f"no recordings of {sorted(rg.CONFUSER)} in the manifest")
    scorers = [("comb", rr.CombScorer(DEFAULT), None)]
    for m in a.models:
        label, spec = m.split("=", 1)
        path, thr = spec.rsplit("@", 1)
        scorers.append((label, rr.NetScorer.load(REPO / path, DEFAULT, a.device), float(thr)))

    out = dict(session=sorted(rg.CONFUSER), hop_s=a.hop, gate_fa=0.25, models={})
    for label, sc, thr in scorers:
        s = rg.score_chain(sc, recs, DEFAULT, a.hop)
        y = s.cw.y
        pos, neg = s.s[y == 1], s.s[y == 0]
        row = dict(drone_windows=int(len(pos)), mower_only_windows=int(len(neg)),
                   auc_drone_vs_mower=rr._auc(pos, neg) if len(pos) and len(neg) else None)
        if thr is not None:
            fa = float((neg >= thr).mean()) if len(neg) else None
            row.update(frozen_threshold=thr, mower_fa=fa, mower_fa_wilson95=rr.wilson(int((neg >= thr).sum()), len(neg)),
                       drone_recall=float((pos >= thr).mean()) if len(pos) else None,
                       gate_e_pass=None if fa is None else fa <= 0.25)
        out["models"][label] = row
        print(label, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
