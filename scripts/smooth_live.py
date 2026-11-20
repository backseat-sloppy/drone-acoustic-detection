"""What a causal moving average of the confidence over N windows would do, on finished virtual live tests.

An analysis of a possible decision stage, not one: nothing in the detectors or the GUI changes, and
adopting one needs the council (CLAUDE.md section 4). For each N (1 = as now), each detector's per-hop
confidence (scores.npz; comb squashed) is replaced by the mean of the last N windows within its round
(windows with fewer than N before them in the round are left out, from drone and no-drone alike; a no-drone
window after a drone counts only once its whole average is past the drone phase, i.e. the buffer after
impact grows by the average's reach, (N - 1) x 0.125 s, so the drone's own tail is not a false alarm). Then,
over the distinct virtual runs (labels as scripts/rescore_live.py, checked there against rounds.csv):

    conf 3-7 s     REQUIREMENTS.md Test 2: mean confidence over 3 to 7 s before impact, each second equal
    no-drone       mean confidence of the no-drone windows, and their highest value over all runs (FP peak)
    AUC 3-7 s      how well the 3-7 s windows rank above the no-drone windows (threshold free; 0.5 = chance)
    first above    per drone round, the time before impact of the first approach window above the FP peak
    stays above    ... from which every approach window up to impact stays above it

N windows of 0.125 s: the average spans (N - 1) x 0.125 s + 1 s of audio and reacts about (N - 1) x 0.0625 s
later than one window. Virtual runs only; simulated drones from training sources: a tool, not a detection
result (CLAUDE.md section 5).

    uv run python scripts/smooth_live.py [--n 1 4 8 16 24] [--detectors "it9 CRNN" ...]
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


def causal_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Mean of the last n values at each index; NaN where fewer than n values (or any NaN) are there."""
    out = np.full(len(x), np.nan)
    if n <= 1:
        return x.astype(float)
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x, nan=0.0))])
    bad = np.concatenate([[0], np.cumsum(np.isnan(x))])
    i = np.arange(n, len(x) + 1)
    ok = (bad[i] - bad[i - n]) == 0
    out[i[ok] - 1] = (c[i[ok]] - c[i[ok] - n]) / n
    return out


def auc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv, kind="mergesort"), kind="mergesort") + 1.0
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    r = (sums / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[1, 4, 8, 16, 24])
    ap.add_argument("--detectors", nargs="+", default=["it9 CRNN", "it16 CRNN 600k s1", "it19 CRNN colour s1", "comb"])
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/complete_test/rescore_audible/smooth.csv")
    a = ap.parse_args()

    # load once: per detector, a list of rounds (conf, labels, t_end, impact or None)
    seen, data = set(), {d: [] for d in a.detectors}
    for d in sorted((ROOT / "artifacts/complete_test").glob("*_live_virtual_*")):
        if not (d / "scores.npz").exists() or not (d / "rounds.csv").exists():
            continue
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        rounds = run_rounds(d)
        if not rounds:
            continue
        sc = np.load(d / "scores.npz")
        h = hashlib.sha1(b"".join(np.ascontiguousarray(sc[k]).tobytes() for k in sorted(sc.files))).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        buf = float(cfg["plan"].get("buffer_after_s") or 0.0)
        for r in rounds:
            for det in a.detectors:
                key = f"{r['round']}|{det}"
                if key not in sc.files:
                    continue
                s = np.asarray(sc[key], float)
                conf = dwg.squash(s, cfg["thresholds"][det]) if det == "comb" else s
                t = grid_t_end(len(s))
                span = imp = None
                if r["drone"] != "none":
                    s0, s1, im = cfg["drone_spans"][r["drone"]]
                    st = float(r["start_s"])
                    span, imp = (st + s0, st + s1 + buf), st + im
                data[det].append(dict(conf=conf, lab=labels(t, span), t=t, span=span, imp=imp, run=d.name, rnd=r["round"]))

    rows = []
    for det in a.detectors:
        for n in a.n:
            neg, pos37, c37s, rounds = [], [], [], []
            for x in data[det]:
                c = causal_mean(x["conf"], n)
                # a no-drone window counts only if its whole average is clear of the drone phase: the phase is
                # stretched by the average's reach, (n - 1) hops (the detector's cooling, as the buffer, entry 229)
                lab = x["lab"] if x["span"] is None else labels(x["t"], (x["span"][0], x["span"][1] + (n - 1) * 0.125))
                neg.append(c[(lab == 0) & np.isfinite(c)])
                if x["imp"] is None:
                    continue
                tb = x["imp"] - x["t"]
                secs = [np.nanmean(c[(tb > k) & (tb <= k + 1)]) for k in range(3, 7)]
                if all(np.isfinite(secs)):
                    c37s.append(float(np.mean(secs)))
                band = (tb > 3) & (tb <= 7) & np.isfinite(c)
                pos37.append(c[band])
                app = (x["t"] - WIN_S / 2 >= x["span"][0]) & (x["t"] <= x["imp"]) & np.isfinite(c)
                rounds.append((c, app, tb))
            neg = np.concatenate(neg) if neg else np.zeros(0)
            pos37 = np.concatenate(pos37) if pos37 else np.zeros(0)
            peak = float(neg.max()) if len(neg) else None
            first, stay = [], []
            for c, app, tb in rounds:
                idx = np.flatnonzero(app)
                if peak is None or not len(idx):
                    continue
                above = idx[c[idx] > peak]
                if len(above):
                    first.append(float(tb[above[0]]))
                below = idx[c[idx] <= peak]
                rest = idx[idx > below[-1]] if len(below) else idx
                if len(rest):
                    stay.append(float(tb[rest[0]]))
            rows.append(dict(detector=det, n=n, span_s=round((n - 1) * 0.125 + 1.0, 3), drone_rounds=len(rounds),
                             conf_3_7s=round(float(np.mean(c37s)), 3) if c37s else None,
                             no_drone_mean=round(float(neg.mean()), 3) if len(neg) else None,
                             fp_peak=round(peak, 3) if peak is not None else None, no_drone_windows=len(neg),
                             auc_3_7s=round(auc(pos37, neg), 3) if auc(pos37, neg) is not None else None,
                             rounds_above=len(first), median_first_above_s=round(float(np.median(first)), 2) if first else None,
                             rounds_stay=len(stay), median_stays_above_s=round(float(np.median(stay)), 2) if stay else None))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    hdr = ["detector", "n", "span_s", "drone_rounds", "conf_3_7s", "no_drone_mean", "fp_peak", "auc_3_7s",
           "rounds_above", "median_first_above_s", "rounds_stay", "median_stays_above_s"]
    print(" | ".join(hdr))
    for r in rows:
        print(" | ".join(str(r[k]) for k in hdr))
    print(f"\n{len(seen)} distinct runs; {a.out}")


if __name__ == "__main__":
    main()
