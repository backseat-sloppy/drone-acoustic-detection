"""Entry 129: one threshold at 5% false alarms on the pooled real background of it9's validation sessions.

    python -u scripts/real_threshold.py --device cuda --out artifacts/it11_1/real_threshold.json \\
        "it9 CRNN=artifacts/models/final_crnn_none_simdist_crnn.pt" "p1=artifacts/it11_1/p1_s0_panns.pt"

Mads's script from his server account (~/e123_score/scripts/real_threshold.py, MD5
1fdbd6c6a47974e43827b7a8396826cb, 2026-11-05), which set the thresholds of entries 129 and 134, with the
model list and output path made arguments. The rule is unchanged:
- validation = rangerun.make_pool_arm(load_manifest(), "none").val (Nightjar and d3set validation sessions);
- scored with rangerun.score_chain at a 0.5 s hop;
- threshold = rangerun.fa_threshold(background windows, 0.05), over the background windows of both rigs
  pooled together.
The pooling is by window, so the rig with more background windows weighs more. Entry 131 described the
threshold as "pooled per rig, weighted, not by window count"; this script, and therefore entry 134, pool
by window (entry 139).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dad import rangerun as rg  # noqa: E402
from dad import realrun as rr  # noqa: E402
from dad.config import DEFAULT  # noqa: E402
from dad.realdata import load_manifest  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", help='"label=checkpoint.pt"')
    p.add_argument("--device", default="cuda")
    p.add_argument("--fa", type=float, default=0.05)
    p.add_argument("--out", type=Path, default=REPO / "artifacts" / "runs" / "real_threshold.json")
    a = p.parse_args()
    val = rg.make_pool_arm(load_manifest(), "none").val
    out = {}
    for m in a.models:
        label, path = m.split("=", 1)
        s = rg.score_chain(rr.NetScorer.load(REPO / path, DEFAULT, a.device), val, DEFAULT, 0.5)
        neg, pos = s.s[s.cw.y == 0], s.s[s.cw.y == 1]
        thr = rg.fa_threshold(neg, a.fa)
        out[label] = dict(threshold=thr, fa=float((neg >= thr).mean()), recall=float((pos >= thr).mean()),
                          drone=int(len(pos)), background=int(len(neg)), checkpoint=path)
        print("RES", label, out[label], flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
