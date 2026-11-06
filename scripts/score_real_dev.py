"""Score networks on the real validation rigs it9 used: Nightjar flight 3 and d3set Validation (entry 123).

    .venv/bin/python scripts/score_real_dev.py --device cuda --out artifacts/runs/real_dev.json \\
        "it9 CRNN=artifacts/models/final_crnn_none_simdist_crnn.pt" "it11r CRNN 600k=artifacts/it11r_crnn600k/it11r_crnn600k_s0_crnn.pt"

Validation, not test: these sessions are rangerun.make_pool_arm(..., "none").val, which set it9's threshold.
it9 trained on other sessions of the same rigs, so it has an in-rig advantage here; the synthetic-trained
models never saw a real drone, nor Nightjar audio (train_clean's --val-rig), but did see d3set
no-drone audio. Per rig: AUC, and recall by range at a threshold set on that rig's own background
at 5 and 10% false alarms (rangerun.evaluate_chain), plus each model's own frozen threshold if given
as "label=path@threshold".
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
    p.add_argument("models", nargs="+", help='"label=checkpoint.pt" or "label=checkpoint.pt@threshold"')
    p.add_argument("--device", default="cpu")
    p.add_argument("--hop", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=REPO / "artifacts" / "runs" / "real_dev.json")
    a = p.parse_args()

    val = rg.make_pool_arm(load_manifest(), "none").val
    rigs = {src: [r for r in val if r.source == src] for src in sorted({r.source for r in val})}
    scorers = [("comb", rr.CombScorer(DEFAULT), None)]
    for m in a.models:
        label, spec = m.split("=", 1)
        path, thr = (spec.split("@") + [None])[:2]
        scorers.append((label, rr.NetScorer.load(REPO / path, DEFAULT, a.device), None if thr is None else float(thr)))

    out = {}
    for label, sc, thr in scorers:
        out[label] = {}
        for rig, recs in rigs.items():
            s = rg.score_chain(sc, recs, DEFAULT, a.hop)
            y = s.cw.y
            ev = rg.evaluate_chain(s, thr if thr is not None else float("inf"), 1.0, fas=(0.05, 0.10))
            row = dict(auc=rg._auc(s.s[y == 1], s.s[y == 0]), drone=int((y == 1).sum()), background=int((y == 0).sum()),
                       recall_all_5fa=rg._get(ev, "own", "0.05", "by_range", "all", "recall"),
                       recall_all_10fa=rg._get(ev, "own", "0.10", "by_range", "all", "recall"),
                       own=ev.get("own"))
            if thr is not None:
                row.update(frozen_threshold=thr, frozen_fa=ev.get("deploy_fa"),
                           frozen_recall_all=rg._get(ev, "deploy", "by_range", "all", "recall"))
            out[label][rig] = row
            print(f"{label:24s} {rig:9s} AUC {row['auc']:.3f}  recall@5%FA {row['recall_all_5fa']:.3f}  "
                  f"@10%FA {row['recall_all_10fa']:.3f}"
                  + (f"  frozen thr {thr:.3f}: FA {row['frozen_fa']:.3f} recall {row['frozen_recall_all']:.3f}"
                     if thr is not None else ""), flush=True)
        a.out.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
