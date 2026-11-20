"""Threshold-free time before impact for finished virtual live tests: per run and detector, the highest
confidence in any no-drone window (the false-positive peak), and per drone round the earliest window in
the approach whose confidence is above that peak. Its time before impact is the warning a threshold set
just above the worst false alarm would have given, with zero false alarms in that run.

Read from the files each run wrote (scores.npz, window_snr.npz, rounds.csv, config.json); labels as
scripts/rescore_live.py (checked there against rounds.csv). comb on its squashed confidence (0-1).
Duplicated runs (identical scores) are counted once. Virtual runs only.

Also: the same with one false-positive peak per detector pooled over all runs (the worst case), and
where each run's false-positive peak sits (round, seconds into the round, before or after the drone,
the background levels), for finding what causes it.

    uv run python scripts/fp_margin_live.py [--out artifacts/complete_test/rescore_audible]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import drone_war_grid as dwg                                  # noqa: E402
from rescore_live import grid_t_end, labels, run_rounds, WIN_S  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/complete_test/rescore_audible")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    seen, per_round, peaks = set(), [], []
    for d in sorted((ROOT / "artifacts/complete_test").glob("*_live_virtual_*")):
        if not (d / "scores.npz").exists() or not (d / "rounds.csv").exists():
            continue
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        plan, thr, spans = cfg["plan"], cfg["thresholds"], cfg["drone_spans"]
        rounds = run_rounds(d)
        if not rounds:
            continue
        sc = np.load(d / "scores.npz")
        snr = np.load(d / "window_snr.npz")
        h = hashlib.sha1(b"".join(np.ascontiguousarray(sc[k]).tobytes() for k in sorted(sc.files))).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        buf = float(plan.get("buffer_after_s") or 0.0)
        dets = sorted({det for r in rounds for det in r["dets"]})
        for det in dets:
            fp_best = dict(conf=-1.0)
            drone_rounds = []
            for r in rounds:
                key = f"{r['round']}|{det}"
                if key not in sc.files:
                    continue
                s = np.asarray(sc[key], float)
                conf = dwg.squash(s, thr[det]) if det == "comb" else s
                t_end = grid_t_end(len(s))
                span = imp = None
                if r["drone"] != "none":
                    s0, s1, im = spans[r["drone"]]
                    st = float(r["start_s"])
                    span, imp = (st + s0, st + s1 + buf), st + im
                lab = labels(t_end, span)
                neg = np.flatnonzero((lab == 0) & np.isfinite(conf))
                if len(neg):
                    j = neg[np.argmax(conf[neg])]
                    if conf[j] > fp_best["conf"]:
                        row = r["dets"][det]
                        fp_best = dict(conf=float(conf[j]), round=r["round"], t_end=float(t_end[j]),
                                       where="no-drone round" if span is None else
                                       ("before drone" if t_end[j] <= span[0] else "after drone"),
                                       background={k: v for k, v in row.items() if k.endswith("_db") and k != "drone_db"
                                                   and v not in ("", "off")})
                if span is not None:
                    app = (t_end - WIN_S / 2 >= span[0]) & (t_end <= imp) & np.isfinite(conf)
                    w = snr[f"{r['round']}|1"][:len(s)] if f"{r['round']}|1" in snr.files else np.full(len(s), np.nan)
                    drone_rounds.append(dict(r=r, conf=conf, t_end=t_end, imp=imp, app=app, w=w))
            peaks.append(dict(run=d.name, detector=det, fp_peak=round(fp_best["conf"], 4) if fp_best["conf"] >= 0 else None,
                              **{k: fp_best.get(k) for k in ("round", "t_end", "where")},
                              background=json.dumps(fp_best.get("background", {}))))
            for x in drone_rounds:
                c, app = x["conf"], x["app"]
                p = fp_best["conf"] if fp_best["conf"] >= 0 else None
                above = np.flatnonzero(app & (c > p)) if p is not None else np.array([], int)
                peak_i = np.flatnonzero(app)[np.argmax(c[app])] if app.any() else None
                per_round.append(dict(
                    run=d.name, round=x["r"]["round"], drone=x["r"]["drone"], detector=det,
                    fp_peak=None if p is None else round(p, 4),
                    drone_peak=None if peak_i is None else round(float(c[peak_i]), 4),
                    drone_peak_before_impact_s=None if peak_i is None else round(float(x["imp"] - x["t_end"][peak_i]), 3),
                    gap=None if p is None or peak_i is None else round(float(c[peak_i] - p), 4),
                    first_above_fp_before_impact_s=round(float(x["imp"] - x["t_end"][above[0]]), 3) if len(above) else None,
                    snr_at_first_above_db=round(float(x["w"][above[0]]), 1) if len(above) and np.isfinite(x["w"][above[0]]) else None,
                    _c=c, _app=app, _t=x["t_end"], _imp=x["imp"]))
    # pooled: one false-positive peak per detector over every run
    pooled = {}
    for p in peaks:
        if p["fp_peak"] is not None:
            pooled[p["detector"]] = max(pooled.get(p["detector"], -1), p["fp_peak"])
    for x in per_round:
        P = pooled.get(x["detector"])
        above = np.flatnonzero(x["_app"] & (x["_c"] > P)) if P is not None else []
        x["pooled_fp_peak"] = P
        x["first_above_pooled_before_impact_s"] = round(float(x["_imp"] - x["_t"][above[0]]), 3) if len(above) else None
        # sustained: from this window on, every approach window up to impact stays above the pooled peak
        idx = np.flatnonzero(x["_app"])
        below = idx[x["_c"][idx] <= P] if P is not None else idx
        rest = idx[idx > below[-1]] if len(below) else idx
        x["stays_above_pooled_before_impact_s"] = round(float(x["_imp"] - x["_t"][rest[0]]), 3) if len(rest) else None
    keep = [k for k in per_round[0] if not k.startswith("_")]
    with open(a.out / "fp_margin_rounds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keep)
        w.writeheader()
        w.writerows([{k: x[k] for k in keep} for x in per_round])
    with open(a.out / "fp_peaks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(peaks[0]))
        w.writeheader()
        w.writerows(peaks)
    med = lambda v: f"{np.median(v):.2f}" if v else "-"       # noqa: E731
    print(f"{'detector':28s} rounds  fp_peak(med)  drone_peak(med)  gap(med)  above-own-FP  lead(med,s)  above-pooled  lead(med,s)  pooled_fp  stays-above(med,s)")
    for det in sorted({x["detector"] for x in per_round}):
        xs = [x for x in per_round if x["detector"] == det]
        own = [x["first_above_fp_before_impact_s"] for x in xs if x["first_above_fp_before_impact_s"] is not None]
        pol = [x["first_above_pooled_before_impact_s"] for x in xs if x["first_above_pooled_before_impact_s"] is not None]
        sty = [x["stays_above_pooled_before_impact_s"] for x in xs if x["stays_above_pooled_before_impact_s"] is not None]
        xs_own = [x for x in xs if x["fp_peak"] is not None]
        print(f"{det:28s} {len(xs):6d}  {med([x['fp_peak'] for x in xs if x['fp_peak'] is not None]):>12s}  "
              f"{med([x['drone_peak'] for x in xs if x['drone_peak'] is not None]):>15s}  {med([x['gap'] for x in xs if x['gap'] is not None]):>8s}  "
              f"{len(own):>5d}/{len(xs_own):<6d}  {med(own):>11s}  {len(pol):>5d}/{len(xs):<6d}  {med(pol):>11s}  "
              f"{pooled.get(det, float('nan')):9.3f}  {len(sty):>3d} {med(sty):>6s}")


if __name__ == "__main__":
    main()
