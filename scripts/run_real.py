"""Train and score detectors on the real recordings.

    .venv-dml\\Scripts\\python.exe scripts\\run_real.py --name gate \\
        --models level comb lgbm dscnn --variants level full --device dml

Writes artifacts/runs/<name>.json and, for neural models, a checkpoint under
artifacts/models/. Development data only, unless --sealed is given.

--sealed scores Halmstad and FSD50K-eval at each scorer's validation
threshold. That is a test scoring: it is appended to
artifacts/sealed_scorings.jsonl and must be recorded in docs/REPORT.md.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dad.config import FrontendConfig  # noqa: E402
from dad.realdata import load_manifest  # noqa: E402
from dad import realrun as rr  # noqa: E402

NETS = ("dscnn", "resnet", "crnn", "transformer")


def line(r: dict) -> str:
    by_range = " ".join(f"{v['mean_score']:.2f}" for v in r["val_by_range"].values())
    paired = "  ".join(f"{src} [" + " ".join(f"{v['auc']:.2f}" for v in d.values()) + "]"
                       for src, d in r["paired"].items())
    unseen = "  ".join(f"{k}: FA {v['fa_at_recall']:.3f}, recall@5%FA {v['recall_at_5pct_fa']:.2f}"
                       for k, v in r["unseen"].items())
    return (f"{r['scorer']:<28} val AUC {r['val_auc']:.3f}  transfer AUC {r['transfer_auc']:.3f}  "
            f"score by range [{by_range}]\n"
            f"{'':<28} paired AUC at +10/0/-10 dB: {paired}\n"
            f"{'':<28} at each chain's own 95% recall: {unseen}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--models", nargs="+", default=["level", "comb", "lgbm", "dscnn"])
    p.add_argument("--variants", nargs="+", default=["level", "full"], choices=list(rr.VARIANTS))
    p.add_argument("--positives", nargs="+", default=["ddl"], choices=["ddl", "ddl-near", "dronenoise"],
                   help="where training drones come from; several are sampled in equal shares")
    p.add_argument("--near-m", type=float, default=50.0, help="range limit for ddl-near")
    p.add_argument("--dn-holdout", default="none", choices=["none", "3p", "M3", "Yn", "Fp"],
                   help="DroneNoise drone model kept out of training and used as the probe")
    p.add_argument("--load-from", default=None, help="score saved checkpoints of this run name instead of training")
    p.add_argument("--snr", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                   help="override the variant's drone-over-background range, dB")
    p.add_argument("--steady-boost", type=float, default=1.0,
                   help="sampling weight for steady-noise backgrounds (engines, wind, water ...)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--windows", type=int, default=200_000, help="training windows per neural model")
    p.add_argument("--lgbm-windows", type=int, default=40_000)
    p.add_argument("--window-seconds", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sealed", action="store_true", help="also score the sealed test sources")
    args = p.parse_args()

    cfg = FrontendConfig(window_seconds=args.window_seconds)
    splits = rr.make_splits(load_manifest(), cfg)
    n = cfg.samples_per_window
    pools = []
    if "ddl" in args.positives:
        pools.append(rr.ddl_pool(splits.ddl_train, n))
    if "ddl-near" in args.positives:
        pools.append(rr.ddl_pool(splits.ddl_train, n, args.near_m, name=f"ddl up to {args.near_m:g} m"))
    probe_independent = True
    if "dronenoise" in args.positives:
        train_dn = [r for r in splits.dronenoise if r.labels != args.dn_holdout]
        pools.append(rr.dronenoise_pool(train_dn, n))
        # A model trained on DroneNoise can only be probed on a drone model it never heard.
        splits.probe = [r for r in splits.probe if r.labels == args.dn_holdout] or splits.probe
        probe_independent = args.dn_holdout != "none"
    print("training drones: " + ", ".join(f"{pl.name} ({pl.seconds / 60:.1f} min)" for pl in pools)
          + ("" if probe_independent else "  [DroneNoise probe is NOT independent in this run]"), flush=True)
    minutes = lambda recs: sum(r.seconds for r in recs) / 60
    print(f"splits: ddl train {len(splits.ddl_train)} sessions ({minutes(splits.ddl_train):.1f} min), "
          f"ddl val {len(splits.ddl_val)} ({minutes(splits.ddl_val):.1f} min), "
          f"fsd train {len(splits.fsd_train)} clips ({minutes(splits.fsd_train) / 60:.1f} h), "
          f"fsd val {len(splits.fsd_val)} clips ({minutes(splits.fsd_val) / 60:.1f} h), "
          f"probe {len(splits.probe)} files", flush=True)

    t0 = time.time()
    ddl_unseen = not any(a.startswith("ddl") for a in args.positives)
    sets = rr.development_sets(splits, cfg, ddl_unseen=ddl_unseen)
    print("development windows: " + ", ".join(f"{k} {len(v.y)}" for k, v in sets.items())
          + f" ({time.time() - t0:.0f} s)", flush=True)
    sealed = rr.sealed_sets(splits, cfg) if args.sealed else None

    out_dir = REPO / "artifacts" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (REPO / "artifacts" / "models").mkdir(parents=True, exist_ok=True)
    results, sealed_results = [], []

    def finish(scorer, train_seconds: float = 0.0, extra: dict | None = None) -> None:
        r = rr.evaluate(scorer, sets)
        r.update(train_seconds=round(train_seconds, 1), probe_independent=probe_independent,
                 positives=[pl.name for pl in pools], **(extra or {}))
        results.append(r)
        print(line(r), flush=True)
        if sealed is not None:
            s = rr.evaluate_sealed(scorer, sealed, r["threshold"])
            sealed_results.append(s)
            print(f"  SEALED {s['scorer']}: Halmstad clip AUC {s['halmstad_clip_auc']:.3f}, "
                  f"recall {s['halmstad_recall']['rate']:.2f}, "
                  f"helicopter FA {s['halmstad_fa_helicopter']['rate']:.2f}, "
                  f"background FA {s['halmstad_fa_background']['rate']:.2f}, "
                  f"FSD50K-eval FA {s['fsd_eval_fa']:.4f}", flush=True)
        (out_dir / f"{args.name}.json").write_text(json.dumps(
            dict(args=vars(args), results=results, sealed=sealed_results), indent=2))

    if "level" in args.models:
        finish(rr.LevelRule(cfg))
    if "comb" in args.models:
        finish(rr.CombScorer(cfg))
    for vname in args.variants:
        variant = rr.VARIANTS[vname]
        if args.snr:
            variant = dataclasses.replace(variant, snr_db=tuple(args.snr))
        sampler = rr.WindowSampler(pools, splits.fsd_train, cfg, variant, seed=args.seed,
                                   steady_boost=args.steady_boost)
        if "lgbm" in args.models:
            t = time.time()
            finish(rr.LgbmScorer(variant, cfg).fit(sampler, args.lgbm_windows), time.time() - t)
        for arch in (m for m in args.models if m in NETS):
            # DirectML has no GRU, so the CRNN always trains on the CPU.
            device = "cpu" if arch == "crnn" and args.device == "dml" else args.device
            ckpt = REPO / "artifacts" / "models" / f"{args.load_from or args.name}_{arch}_{vname}.pt"
            if args.load_from:
                finish(rr.NetScorer.load(ckpt, cfg, device), 0.0, dict(loaded=str(ckpt.name)))
                continue
            net = rr.NetScorer(arch, variant, cfg, device)
            curve = []

            def progress(n: rr.NetScorer, seen: int) -> None:
                r = rr.evaluate(n, sets)
                pd0 = r["paired"]["dronenoise"]["+0 dB"]["auc"]
                curve.append(dict(windows=seen, val_auc=r["val_auc"], transfer_auc=r["transfer_auc"],
                                  paired_dronenoise_0db=pd0))
                print(f"      at {seen}: val AUC {r['val_auc']:.3f}, transfer AUC {r['transfer_auc']:.3f}, "
                      f"paired DroneNoise 0 dB {pd0:.3f}", flush=True)

            t = time.time()
            net.fit(sampler, n_windows=args.windows, on_progress=progress)
            net.save(ckpt)
            finish(net, time.time() - t, dict(history=net.history, curve=curve, device=device,
                                              windows=args.windows))

    if sealed is not None:
        with open(REPO / "artifacts" / "sealed_scorings.jsonl", "a") as f:
            f.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), name=args.name,
                                    scorers=[s["scorer"] for s in sealed_results])) + "\n")
    print(f"done in {time.time() - t0:.0f} s -> artifacts/runs/{args.name}.json")


if __name__ == "__main__":
    main()
