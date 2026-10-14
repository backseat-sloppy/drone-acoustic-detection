"""Score saved models on the new same-rig chains, by range to the drone.

    .venv-dml\\Scripts\\python.exe scripts\\probe_chains.py --name probe_it2 \\
        --models it2_dronenoise_dscnn_buried it2_ddl_near_dscnn_buried --device dml

For each chain, drone windows are compared with three sets of negatives:

* `session`: background windows from the same session as the drone window
  (the same rig, site and day; the closest match in time the data allows),
* `chain`: every background window of that chain,
* `fsd`: the FSD50K validation windows of iterations 1 and 2.

Reported per range bin: AUC, mean score, and recall at the thresholds that
give 5% and 20% false alarms on the chain's own background.

A model that was never trained on a chain is an independent listener there:
whatever it finds at 150 m is evidence that a drone can be heard at 150 m on
that rig. Writes artifacts/runs/<name>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dad import chains as ch  # noqa: E402
from dad import realrun as rr  # noqa: E402
from dad.config import FrontendConfig  # noqa: E402
from dad.realdata import load_manifest  # noqa: E402


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    return rr._auc(pos, neg) if len(pos) >= 5 and len(neg) >= 5 else float("nan")


def score_windows(scorer, cw: ch.ChainWindows, n: int, batch: int = 512) -> np.ndarray:
    out = []
    for i in range(0, len(cw), batch):
        a = np.stack([cw.audio(j, n) for j in range(i, min(i + batch, len(cw)))])
        out.append(scorer.score(a))
    return np.concatenate(out) if out else np.empty(0)


def silent(cw: ch.ChainWindows, n: int) -> np.ndarray:
    return np.array([rr.inband_power(cw.audio(i, n)) <= rr.SILENCE_POWER for i in range(len(cw))])


def report(s: np.ndarray, cw: ch.ChainWindows, fsd: np.ndarray) -> dict:
    pos, neg = cw.y == 1, cw.y == 0
    sn = s[neg]
    thr = {fa: float(np.quantile(sn, 1 - fa)) for fa in (0.05, 0.20)} if neg.any() else {}
    bins = ch.range_bin(cw.range_m)
    out = dict(n_pos=int(pos.sum()), n_neg=int(neg.sum()), mean_neg=float(sn.mean()) if neg.any() else None,
               fsd_mean=float(fsd.mean()), chain_vs_fsd_neg_auc=auc(sn, fsd) if neg.any() else None,
               bins={})
    for b in [f"{lo}-{hi if hi < 100_000 else 'inf'} m" for lo, hi in ch.RANGE_BINS] + ["all"]:
        sel = pos & ((bins == b) if b != "all" else True)
        if sel.sum() < 10:
            continue
        sp = s[sel]
        # Same-session negatives: only sessions that have both drone and background windows.
        sess = set(cw.session[sel]) & set(cw.session[neg])
        same = neg & np.isin(cw.session, list(sess))
        sel_s = sel & np.isin(cw.session, list(sess))
        out["bins"][b] = dict(
            windows=int(sel.sum()), sessions=len(set(cw.session[sel])), mean=float(sp.mean()),
            auc_chain=auc(sp, sn), auc_fsd=auc(sp, fsd),
            auc_session=auc(s[sel_s], s[same]) if same.any() else None,
            session_pairs=dict(pos=int(sel_s.sum()), neg=int(same.sum())),
            recall_at_5pct=float((sp >= thr[0.05]).mean()) if thr else None,
            recall_at_20pct=float((sp >= thr[0.20]).mean()) if thr else None,
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--models", nargs="+", required=True, help="checkpoint stems under artifacts/models")
    p.add_argument("--chains", nargs="+", default=["nightjar", "d3set", "lund", "glasgow"])
    p.add_argument("--hop-s", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--window-seconds", type=float, default=1.0)
    args = p.parse_args()

    cfg = FrontendConfig(window_seconds=args.window_seconds)
    n, hop = cfg.samples_per_window, int(args.hop_s * cfg.sample_rate)
    manifest = load_manifest()
    splits = rr.make_splits(manifest, cfg)
    fsd_w = rr._windows(splits.fsd_val, n, n, 3, lambda r: r.kind)[0]
    fsd_audio = np.stack(fsd_w).astype(np.float32)
    sets = {}
    for c in args.chains:
        cw = ch.chain_windows([r for r in manifest if r.source == c], n, hop)
        if len(cw):
            cw = cw.subset(~silent(cw, n))
            sets[c] = cw
            print(f"{c}: {int((cw.y == 1).sum())} drone and {int((cw.y == 0).sum())} background windows", flush=True)
    results = {}
    for m in args.models:
        t0 = time.time()
        scorer = rr.NetScorer.load(REPO / "artifacts" / "models" / f"{m}.pt", cfg, args.device)
        fsd = scorer.score(fsd_audio)
        results[m] = {c: report(score_windows(scorer, cw, n), cw, fsd) for c, cw in sets.items()}
        print(f"\n{m} ({time.time() - t0:.0f} s)")
        for c, r in results[m].items():
            print(f"  {c}: background mean {r['mean_neg']:.3f}, FSD mean {r['fsd_mean']:.3f}")
            for b, v in r["bins"].items():
                sess = f"{v['auc_session']:.2f}" if v["auc_session"] is not None else "  - "
                print(f"    {b:>11}: n {v['windows']:5d} ({v['sessions']:2d} sess)  mean {v['mean']:.3f}  "
                      f"AUC chain {v['auc_chain']:.2f} session {sess} fsd {v['auc_fsd']:.2f}  "
                      f"recall@5%FA {v['recall_at_5pct']:.2f} @20%FA {v['recall_at_20pct']:.2f}", flush=True)
    out = REPO / "artifacts" / "runs" / f"{args.name}.json"
    out.write_text(json.dumps(dict(args=vars(args), results=results), indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
