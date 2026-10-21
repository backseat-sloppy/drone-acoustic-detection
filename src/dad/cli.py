"""Command line entry points.

    uv run dad bench            per-stage latency, the headline figure
    uv run dad sweep            accuracy vs end-to-end delay Pareto table
    uv run dad gate             tier-0 gate behaviour on synthetic clips
    uv run dad compare          every model on the same windows, laptop/GPU zoo
    uv run dad info             print the frozen frontend contract
    uv run dad wav FILES...     run a detector over WAV files, hop by hop
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .config import DEFAULT, FrontendConfig


def _cmd_info(args: argparse.Namespace) -> int:
    cfg = DEFAULT
    print("Frozen frontend contract")
    print("  " + cfg.describe())
    print(f"  frames per window : {cfg.frames_per_window}")
    print(f"  samples per window: {cfg.samples_per_window}")
    print(f"  samples per hop   : {cfg.samples_per_hop}")
    print(f"  mel filterbank    : {cfg.n_mels} x {cfg.n_freq_bins}")
    try:
        from .models.dscnn import TORCH_AVAILABLE

        print(f"  torch available   : {TORCH_AVAILABLE}")
    except Exception:
        print("  torch available   : False")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from .bench import benchmark, format_report

    cfg = FrontendConfig(window_seconds=args.window, hop_seconds=args.hop)
    results = benchmark(cfg, n_iter=args.iters)
    print(format_report(results))
    for key in ("baseline_error", "dscnn_error"):
        if key in results:
            print(f"\n[note] {key}: {results[key]}", file=sys.stderr)
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    from .gate import HarmonicCombGate, calibrate_threshold
    from .synth import make_dataset

    cfg = DEFAULT
    clips = make_dataset(n_sessions=4, clips_per_session=24, cfg=cfg)
    gate = HarmonicCombGate(cfg)

    negatives = [c.audio for c in clips if c.label == 0]
    positives = [c.audio for c in clips if c.label == 1]

    cal = calibrate_threshold(gate, positives, negatives, target_recall=args.recall)
    gate.threshold = cal["threshold"]

    print("TIER-0 HARMONIC GATE (synthetic data, scaffolding only)")
    print(f"  threshold at {args.recall:.0%} positive recall : {cal['threshold']:.3f}")
    print(f"  achieved recall                   : {cal['recall']:.3f}")
    print(f"  negative forward rate             : {cal['negative_forward_rate']:.3f}")
    print(f"  background rejected               : {100.0 * (1.0 - cal['negative_forward_rate']):.1f}%")
    print(f"  median score  positive / negative : {cal['pos_median']:.1f} / {cal['neg_median']:.1f}")

    if cal["useful"]:
        print(f"\n  VERDICT: cascade is worthwhile. The CNN runs on "
              f"{cal['negative_forward_rate']:.1%} of background windows.")
    else:
        print("\n  VERDICT: the cascade is NOT earning its place at this operating point.")
        print("  Forwarding most background to the CNN means you pay the CNN cost")
        print("  anyway. Either improve the gate or drop the tier and simplify.")

    if cal["recall_shortfall"] > 0.002:
        print(f"\n  RECALL {cal['recall']:.3f} against a {cal['target_recall']:.2f} target.")
        if cal["unreachable_frac"] > 0.005:
            print(f"  {cal['unreachable_frac']:.1%} of positives score zero: no in-band")
            print("  fundamental was found at all. Lowering the threshold cannot recover")
            print("  those. They need a longer window or a better frontend, which is the")
            print("  window-as-detection-range trade, not a gate tuning problem.")
        else:
            print(f"  No positive scores zero, so this is sample granularity, not a")
            print(f"  detection ceiling: with {cal['n_positives']} positives the")
            print("  1% quantile cannot be resolved more finely than one clip.")

    # Per-confuser breakdown: aggregate rates hide which source fools you.
    print("\n  score by source kind (median):")
    kinds: dict[str, list[float]] = {}
    for c in clips:
        kinds.setdefault(c.kind, []).append(gate(c.audio).score)
    for kind, scores in sorted(kinds.items(), key=lambda kv: -float(np.median(kv[1]))):
        passes = float(np.mean(np.array(scores) >= cal["threshold"]))
        print(f"    {kind:<10} median={np.median(scores):>8.1f}   pass rate={passes:.2f}")

    # The free RPM estimate.
    print("\n  blade-pass estimates on 5 positives:")
    for a in positives[:5]:
        e = gate(a)
        print(f"    f0 = {e.f0_hz:6.1f} Hz   rpm = {e.rpm:7.0f}   "
              f"score = {e.score:7.1f}   in-band={e.in_band}")
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    """Accuracy versus end-to-end delay. Three council members independently
    proposed this as the deliverable, so it is the figure to build the report
    around. On synthetic data the accuracy column is meaningless; the latency
    columns are real and hardware-specific."""
    from sklearn.metrics import roc_auc_score

    from .bench import latency_budget, range_penalty
    from .features import LogMel, summary_stats
    from .models.baseline import MfccBaseline
    from .synth import make_dataset, session_split

    rows = []
    for window in args.windows:
        cfg = FrontendConfig(window_seconds=window, hop_seconds=args.hop)
        clips = make_dataset(n_sessions=6, clips_per_session=20, cfg=cfg, seed=1)
        train, test = session_split(clips, test_sessions=2, seed=1)
        fe = LogMel(cfg)

        def featurise(cs):
            n = cfg.samples_per_window
            X = np.stack([summary_stats(fe.frames(c.audio[:n])) for c in cs])
            y = np.array([c.label for c in cs])
            return X, y

        Xtr, ytr = featurise(train)
        Xte, yte = featurise(test)
        model = MfccBaseline().fit(Xtr, ytr)
        auc = float(roc_auc_score(yte, model.predict_proba(Xte)))

        # Compute cost is measured, not assumed.
        import time

        xv = Xte[0]
        t0 = time.perf_counter()
        for _ in range(100):
            model.predict_one(xv)
        clf_ms = (time.perf_counter() - t0) / 100 * 1000

        budget = latency_budget(cfg, clf_ms)
        rng = range_penalty(window)
        rows.append((window, auc, clf_ms, budget["cold_onset_ms"],
                     budget["compute_share_pct"], rng["range_factor"]))

    print("ACCURACY vs END-TO-END DELAY   (synthetic data: AUC is NOT a result)")
    print("-" * 78)
    hdr = ("window".rjust(8) + "AUC".rjust(8) + "clf ms".rjust(10)
           + "cold-onset".rjust(13) + "compute%".rjust(11) + "range".rjust(9))
    print(hdr)
    print("-" * 78)
    for w, auc, clf, cold, share, rf in rows:
        print(f"{w:>7.3f}s {auc:>7.3f} {clf:>9.3f} {cold:>11.1f}ms {share:>10.3f}% {rf:>8.2f}x")
    print("-" * 78)
    print("Read the compute% column. That is the fraction of your latency that")
    print("architecture choice can possibly influence.")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from .compare import ML_TORCH, format_rows, run_comparison

    models = args.models
    if any(m in ML_TORCH for m in models):
        try:
            import torch
        except ImportError:
            print("Neural models need torch:  uv sync --extra dl", file=sys.stderr)
            return 2
        from .train import resolve_device

        dev = resolve_device(args.device)
        device_label = f"{dev} ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda" else "cpu"
    else:
        device_label = "cpu"

    rows = run_comparison(
        windows=args.windows,
        models=models,
        epochs=args.epochs,
        device=args.device,
        n_sessions=args.sessions,
        clips_per_session=args.clips,
        seed=args.seed,
    )
    print()
    print(format_rows(rows, device_label))
    return 0


def _cmd_wav(args: argparse.Namespace) -> int:
    import csv
    import json
    from pathlib import Path

    from .wavio import evaluate_file, load_scorer, rates, summarise, threshold_from_run

    files: list[Path] = []
    for p in map(Path, args.paths):
        files.extend(sorted(p.rglob("*.wav")) if p.is_dir() else [p])
    if not files:
        print("no WAV files found")
        return 1
    threshold = args.threshold
    if args.threshold_from:
        threshold = threshold_from_run(args.threshold_from, args.detector, args.window)
    cfg = FrontendConfig(window_seconds=args.window, hop_seconds=args.hop)
    scorer = load_scorer(args.detector, cfg, args.device)
    print(f"detector  : {args.detector}")
    print(f"frontend  : {cfg.describe()}")
    print(f"threshold : {threshold if threshold is not None else 'none (scores only, no True/False)'}")

    def pct(v):
        return "   -  " if v is None else f"{100 * v:5.1f}%"

    out = open(args.out, "w", newline="", encoding="utf-8") if args.out else None
    if out:
        hops = csv.DictWriter(out, fieldnames=["file", "t_end_s", "score", "drone", "truth"])
        hops.writeheader()
    results = []
    for f in files:
        r = evaluate_file(f, scorer, cfg, threshold)
        results.append(r)
        d = r["decisions"]
        note = f"  (channel 0 of {r['channels_orig']})" if r["channels_orig"] > 1 else ""
        line = f"  {f.name}: {r['seconds']:.1f} s, {r['sr_orig']} Hz {r['subtype']}{note}, {len(d.score)} decisions"
        if "counts" in r:
            q = rates(r["counts"], r["leads"])
            line += (f" | recall {pct(q['recall'])}  false pos {pct(q['false_positive_rate'])}"
                     f"  precision {pct(q['precision'])}")
            for x in r["leads"]:
                line += (" | no alarm before impact" if x["lead_s"] is None
                         else f" | alarm {x['lead_s']:.2f} s before impact")
        elif "truth" not in r:
            line += " | no sidecar .json, not scored"
        print(line)
        lab = r.get("labels")
        if out:
            hops.writerows(dict(file=str(f), t_end_s=round(float(d.t_end[i]), 4), score=float(d.score[i]),
                                drone="" if d.drone is None else bool(d.drone[i]),
                                truth="" if lab is None else int(lab[i])) for i in range(len(d.score)))

    summary = summarise(results)
    print(f"\nall files ({summary['files_with_truth']} of {summary['files']} with a sidecar):")
    print(f"  R2 recall              {pct(summary['recall'])}   (target 95%)")
    print(f"  R3 false-positive rate {pct(summary['false_positive_rate'])}   (target at most 5%)")
    print(f"  R4 precision           {pct(summary['precision'])}   (target 95%; drone share {pct(summary['drone_share'])})")
    print(f"  R5 alarm >= 3 s before impact  {pct(summary['approaches_alarm_3s_before_impact'])}"
          f"   ({summary['approaches']} approaches with an impact time)")
    print(f"  windows: {summary['counts']}")
    if out:
        out.close()
        print(f"per-hop decisions -> {args.out}")
    if args.summary:
        per_file = [dict({k: v for k, v in r.items() if k in ("file", "seconds", "sr_orig", "channels_orig",
                                                                 "subtype", "counts", "leads")},
                         meta=r["truth"].meta if "truth" in r else None) for r in results]
        Path(args.summary).write_text(json.dumps(dict(detector=args.detector, threshold=threshold,
                                                      frontend=cfg.describe(), summary=summary,
                                                      files=per_file), indent=2), encoding="utf-8")
        print(f"summary -> {args.summary}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dad", description="Acoustic drone detection, latency first")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="print the frozen frontend contract").set_defaults(fn=_cmd_info)

    b = sub.add_parser("bench", help="per-stage latency benchmark")
    b.add_argument("--window", type=float, default=DEFAULT.window_seconds)
    b.add_argument("--hop", type=float, default=DEFAULT.hop_seconds)
    b.add_argument("--iters", type=int, default=200)
    b.set_defaults(fn=_cmd_bench)

    g = sub.add_parser("gate", help="tier-0 gate calibration and RPM estimate")
    g.add_argument("--recall", type=float, default=0.99,
                   help="positive recall the gate must hold; forward rate is then measured")
    g.set_defaults(fn=_cmd_gate)

    s = sub.add_parser("sweep", help="accuracy vs end-to-end delay table")
    s.add_argument("--windows", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    s.add_argument("--hop", type=float, default=0.125)
    s.set_defaults(fn=_cmd_sweep)

    from .compare import ALL_MODELS

    c = sub.add_parser("compare", help="train every model on the same windows and compare")
    c.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=ALL_MODELS)
    c.add_argument("--windows", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    c.add_argument("--epochs", type=int, default=15)
    c.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    c.add_argument("--sessions", type=int, default=10)
    c.add_argument("--clips", type=int, default=40, help="clips per session")
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(fn=_cmd_compare)

    w = sub.add_parser("wav", help="run a detector over WAV files, hop by hop, and score against sidecars")
    w.add_argument("paths", nargs="+", help="WAV files, or folders searched for *.wav")
    w.add_argument("--detector", default="comb",
                   help="'comb' (no training), a .pt network or a .pkl LightGBM checkpoint")
    w.add_argument("--threshold", type=float, default=None, help="score at or above this means drone")
    w.add_argument("--threshold-from", default=None,
                   help="artifacts/runs/<run>.json: use the threshold that run froze for the checkpoint")
    w.add_argument("--window", type=float, default=DEFAULT.window_seconds,
                   help="must match the window the checkpoint was trained with")
    w.add_argument("--hop", type=float, default=DEFAULT.hop_seconds)
    w.add_argument("--device", default="cpu")
    w.add_argument("--out", default=None, help="CSV with every hop's score and answer")
    w.add_argument("--summary", default=None, help="JSON with the totals and per-file results")
    w.set_defaults(fn=_cmd_wav)

    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
