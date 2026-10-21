"""Iteration 3: leave-one-rig-out training weighted by range (report entry 025).

    .venv-dml\\Scripts\\python.exe scripts\\run_range.py --name it3_screen \\
        --arms nightjar d3set --recipes near range simdist --models comb lgbm dscnn --device dml

For each arm (the rig named is held out), each recipe and each model: train
on the other rig plus DDL and DroneNoise near drones and FSD50K, freeze the
deployment threshold at 10% false alarms on the training rig's validation
sessions, then score the held-out rig, the lawn-mower session, Lund and
Glasgow. Writes artifacts/runs/<name>.json and checkpoints under
artifacts/models/ (`.pt` for networks, `.pkl` for LightGBM). The sealed sources (Halmstad, FSD50K-eval) are not touched.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dad import rangerun as rg  # noqa: E402
from dad import realrun as rr  # noqa: E402
from dad.config import FrontendConfig  # noqa: E402
from dad.realdata import load_manifest  # noqa: E402

NETS = ("dscnn", "resnet", "transformer")
HOP_S = {"glasgow": 2.0}   # 8.7 h of near-range hover; a coarser grid is plenty


def show(tag: str, r: dict) -> None:
    h = rg.headline(r)
    f = lambda v: "  -  " if v is None else f"{v:.3f}"
    print(f"    {tag:<34} recall 100-200 m @10%FA {f(h['recall_100_200_at_10pct'])}  "
          f"0-50 m {f(h['recall_0_50_at_10pct'])}  >300 m control AUC {f(h['control_300plus_auc'])}  "
          f"deploy: recall 100-200 {f(h['deploy_recall_100_200'])} FA {f(h['deploy_fa'])}  "
          f"<=5 s {f(h['latency_within_5s'])} (no drone {f(h['background_alarm_within_5s'])})  "
          f"all {f(h['recall_all_at_10pct'])}  3 s at 150-300 m {f(h['alarm_within_3s_150_300'])} "
          f"(no drone {f(h['background_alarm_within_3s'])})", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--arms", nargs="+", default=["nightjar", "d3set"], help="the rig held out")
    p.add_argument("--pool", action="store_true",
                   help="train on every rig but the held-out one (iteration 5); arms may then also be "
                        "fpv:ariel_25.02, fpv:palmahim_10.02, fpv:palmahim_27.05")
    p.add_argument("--pair", action="store_true",
                   help="read Nightjar and d3set from the two-microphone stores (entry 036); use with --load-from")
    p.add_argument("--speed", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                   help="resample every training window by a factor drawn from LO-HI")
    p.add_argument("--recipes", nargs="+", default=["near", "range", "simdist"], choices=list(rg.RECIPES))
    p.add_argument("--models", nargs="+", default=["comb", "lgbm", "dscnn"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--windows", type=int, default=200_000)
    p.add_argument("--lgbm-windows", type=int, default=40_000)
    p.add_argument("--window-seconds", type=float, default=1.0)
    p.add_argument("--eval-hop", type=float, default=0.5)
    p.add_argument("--deploy-fa", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--load-from", default=None, help="score checkpoints of this run name instead of training")
    p.add_argument("--integrate", nargs="+", type=int, default=[1],
                   help="also score the average of k consecutive windows, for each k given")
    args = p.parse_args()

    cfg = FrontendConfig(window_seconds=args.window_seconds)
    manifest = load_manifest()
    if args.pair:
        manifest = [dataclasses.replace(r, _file=r._file.replace(f"{r.source}.i16", f"{r.source}_pair.i16"))
                    if r.source in ("nightjar", "d3set") else r for r in manifest]
    splits = rr.make_splits(manifest, cfg)
    out_path = REPO / "artifacts" / "runs" / f"{args.name}.json"
    (REPO / "artifacts" / "models").mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    t_start = time.time()

    for held in args.arms:
        arm = rg.make_pool_arm(manifest, held) if args.pool else rg.make_arm(manifest, held)
        print(f"\n=== {arm.name}: train on {arm.train_chain} ({len(arm.train)} segments), "
              f"validate on {len(arm.val)} segments, test {len(arm.test)} segments", flush=True)

        def evaluate(scorer, tag: str, recipe: str, extra: dict) -> None:
            t0 = time.time()
            hop = args.eval_hop
            val = rg.score_chain(scorer, arm.val, cfg, hop)
            test = rg.score_chain(scorer, arm.test, cfg, hop)
            conf = rg.score_chain(scorer, arm.confuser, cfg, hop)
            unseen = {c: rg.score_chain(scorer, recs, cfg, HOP_S.get(c, hop)) for c, recs in arm.unseen.items()}
            for k in args.integrate:
                # k consecutive window scores averaged: one decision per k * hop of new audio
                span = args.window_seconds + (k - 1) * hop
                v = rg.integrate(val, k, hop)
                vneg = v.s[v.cw.y == 0]
                thr = rg.fa_threshold(vneg, args.deploy_fa)
                r = dict(arm=arm.name, recipe=recipe, scorer=tag, seed=args.seed, window_s=args.window_seconds,
                         integrate=k, decision_span_s=span, deploy_threshold=thr,
                         val_background_windows=int(len(vneg)), **extra)
                r["test"] = rg.evaluate_chain(rg.integrate(test, k, hop), thr, span)
                r["headline"] = rg.headline(r["test"])
                # Lawn-mower session: drone and mower together, mower alone after landing.
                c = rg.integrate(conf, k, hop)
                own10 = rg._get(r["test"], "own", "0.10", "threshold")
                cneg = c.s[c.cw.y == 0]
                r["confuser"] = dict(
                    background_windows=int(len(cneg)),
                    fa_deploy=float((cneg >= thr).mean()) if len(cneg) else None,
                    fa_own10=float((cneg >= own10).mean()) if len(cneg) and own10 is not None else None,
                    recall_own10=rg._recall_by_bin(c, own10) if own10 is not None else None)
                r["unseen"] = {}
                for name, sc in unseen.items():
                    if k == 1 or HOP_S.get(name, hop) == hop:
                        r["unseen"][name] = rg.evaluate_chain(rg.integrate(sc, k, hop), thr, span)
                r["eval_seconds"] = round(time.time() - t0, 1)
                results.append(r)
                show(f"{tag} [{recipe}] k={k}", r["test"])
                for name, u in r["unseen"].items():
                    show(f"  unseen {name}", u)
                out_path.write_text(json.dumps(dict(args=vars(args), results=results), indent=2))

        if "comb" in args.models:
            evaluate(rr.CombScorer(cfg), "comb (no ML)", "-", {})
        for rname in args.recipes:
            recipe = rg.RECIPES[rname]
            sampler = rg.RangeSampler(arm, splits, recipe, cfg, seed=args.seed,
                                      speed=tuple(args.speed) if args.speed else None)
            print(f"  recipe {rname}: {sampler.describe()}", flush=True)
            variant = rr.VARIANTS["buried"]
            if "lgbm" in args.models:
                ckpt = REPO / "artifacts" / "models" / f"{args.load_from or args.name}_{held.replace(':', '_')}_{rname}_lgbm.pkl"
                if args.load_from:
                    evaluate(rr.LgbmScorer.load(ckpt, cfg), "lgbm", rname, dict(loaded=ckpt.name))
                else:
                    t = time.time()
                    m = rr.LgbmScorer(variant, cfg).fit(sampler, args.lgbm_windows)
                    m.save(ckpt)
                    evaluate(m, "lgbm", rname, dict(train_seconds=round(time.time() - t, 1)))
            for arch in (a for a in args.models if a in NETS):
                ckpt = REPO / "artifacts" / "models" / f"{args.load_from or args.name}_{held.replace(':', '_')}_{rname}_{arch}.pt"
                if args.load_from:
                    net = rr.NetScorer.load(ckpt, cfg, args.device)
                    evaluate(net, arch, rname, dict(loaded=ckpt.name))
                    continue
                net = rr.NetScorer(arch, variant, cfg, args.device)
                t = time.time()
                net.fit(sampler, n_windows=args.windows)
                net.save(ckpt)
                evaluate(net, arch, rname, dict(train_seconds=round(time.time() - t, 1), history=net.history,
                                                windows=args.windows))
    print(f"done in {time.time() - t_start:.0f} s -> {out_path}")


if __name__ == "__main__":
    main()
