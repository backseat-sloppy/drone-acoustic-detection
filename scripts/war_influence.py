"""Iteration 15: how much the war soundscape moves the drone score, and what the decision stage does about it.

Runs scripts/complete_test_engine.py (the engine of Complete_Test_Gui.py) over the ten ITU ARIS drone
excerpts (data/extracted/sealed_peek) mixed into stretches of the war track, plus the same drones with no
background, and scores each decision variant on the same hops:

    raw          a network's score per 1 s window (0.125 s hop), as the GUI shows it
    + median     the median of the last 8 scores (1 s), causal: no look-ahead
    mean of N    the mean of several networks' scores, threshold = mean of their frozen thresholds

Per variant and war level:
    (d_up / d_down: share of drone hops the war lifts / lowers by more than 0.1)
    war only     mean score, 99th percentile, share of hops alarmed (no drone anywhere in the round)
    coupling     d = score(drone + war) - score(drone alone) on drone windows: mean d, mean |d|,
                 share of hops with |d| > 0.1, and corr(d, war-only score at the same hop): how much the
                 war's own score spikes leak into the drone's score
    recall       drone windows at the frozen threshold; AUC and recall at 5% false alarms matched on the
                 war-only hops of the same war level

Readout, not detection performance (CLAUDE.md section 5): ten excerpts from one source (ARIS, opened as a
readout in entry 123) in a war track built from FSD50K stems. Nothing here is tuned on it.

    uv run python scripts/war_influence.py --stretches 4 --out artifacts/war_influence/it15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import complete_test_engine as cte  # noqa: E402

from dad.config import DEFAULT  # noqa: E402

DRONES = sorted(str(p) for p in (cte.dwg.ROOT / "data/extracted/sealed_peek").glob("Drone_*.wav"))
DRONE_DB = [50.0, 60.0]
WAR_DB = [55.0, 65.0]
LENGTH_S, STEP_S = 20.0, 2.0
REUSE = False
K_MED = 8                       # 1 s of 0.125 s hops; fixed before any run, not tuned
NETS = ["it9 CRNN", "it9 CNN", "it6 CNN"]
cte.dwg.MODELS["it6 lgbm"] = (cte.dwg.RUN, cte.dwg.MODEL_DIR / "it6_final_none_simdist_lgbm.pkl")
DETECTORS = NETS + ["it6 lgbm", "comb"]
VARIANTS = {                    # name -> (members, median)
    "it9 CRNN": (["it9 CRNN"], False),
    "it9 CRNN + median": (["it9 CRNN"], True),
    "mean CRNN, it9 CNN": (["it9 CRNN", "it9 CNN"], False),
    "mean CRNN, it9 CNN + median": (["it9 CRNN", "it9 CNN"], True),
    "mean of 3 nets": (NETS, False),
    "mean of 3 nets + median": (NETS, True),
    "it9 CNN": (["it9 CNN"], False),
    "it6 CNN": (["it6 CNN"], False),
    "lgbm (it6)": (["it6 lgbm"], False),
    "comb": (["comb"], False),
}


def causal_median(s: np.ndarray, k: int = K_MED) -> np.ndarray:
    p = np.concatenate([np.full(k - 1, np.nan), s])
    return np.nanmedian(np.lib.stride_tricks.sliding_window_view(p, k), axis=1)


def labels(n_hops: int, start_s: float | None) -> np.ndarray:
    """1 drone, 0 no drone, -1 straddling (dad.wavio.window_labels for one 10 s drone)."""
    b = (np.arange(n_hops) * DEFAULT.samples_per_hop + DEFAULT.samples_per_window) / DEFAULT.sample_rate
    a = b - DEFAULT.window_seconds
    lab = np.zeros(n_hops, int)
    if start_s is not None:
        e = start_s + cte.DRONE_S
        lab[(b > start_s) & (a < e)] = -1
        lab[(a >= start_s) & (b <= e)] = 1
    return lab


def run(layers: dict, start: float, out: Path, no_drone: bool) -> tuple[cte.Test, dict]:
    plan = cte.Plan(kind="real", length_s=LENGTH_S, drones=DRONES, drone_levels=DRONE_DB, layers=layers,
                    detectors=DETECTORS, step_s=STEP_S, no_drone_rounds=no_drone, soundscape_start_s=start)
    t = cte.Test(plan, out)
    if REUSE and (out / "scores.npz").exists():           # analysis only: the rounds are already scored
        t.prepare(progress=lambda m: None)
    else:
        t.run(progress=lambda m: print(m, flush=True) if m.startswith("round") and m.split()[1].endswith("0") else None)
    z = np.load(out / "scores.npz")
    return t, {k: z[k] for k in z.files}


def variant_score(sc: dict, rnd: int, members: list, med: bool, thr: dict) -> tuple[np.ndarray, float]:
    ss, ts = [], []
    for m in members:
        s = sc[f"{rnd}|{m}"]
        if m == "comb":                                  # comb runs 0 to hundreds: onto 0-1, threshold at 0.5
            s, t = cte.dwg.squash(s, thr[m]), 0.5
        else:
            t = thr[m]
        ss.append(s)
        ts.append(t)
    s = np.mean(ss, axis=0)
    return (causal_median(s) if med else s), float(np.mean(ts))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stretches", type=int, default=4, help="20 s stretches of the war track, seed 15")
    ap.add_argument("--out", type=Path, default=cte.dwg.ROOT / "artifacts/war_influence/it15")
    ap.add_argument("--reuse", action="store_true", help="score nothing; analyse the scores.npz already in --out")
    a = ap.parse_args()
    global REUSE
    REUSE = a.reuse
    starts = sorted(float(s) for s in np.random.default_rng(15).integers(0, 600 - int(LENGTH_S), a.stretches))
    print("war track stretches start at", starts, flush=True)

    alone_t, alone_sc = run({}, 0.0, a.out / "alone", False)
    thr = alone_t.thr
    alone = {(r["drone"], r["drone_level_db"], r["start_s"]): r["round"] for r in alone_t.rounds}

    # acc[variant][war_db] -> lists of per-hop arrays
    acc = {v: {w: dict(war=[], d=[], warhop=[], mix=[]) for w in WAR_DB} for v in VARIANTS}
    for i, st in enumerate(starts):
        t, sc = run({"war soundscape": WAR_DB}, st, a.out / f"war_{i}", True)
        nod = {r["background"]["war soundscape"]: r["round"] for r in t.rounds if r["drone"] is None}
        for v, (members, med) in VARIANTS.items():
            war_only = {w: variant_score(sc, nod[w], members, med, thr)[0] for w in WAR_DB}
            for w in WAR_DB:
                acc[v][w]["war"].append(war_only[w])
            for r in t.rounds:
                if r["drone"] is None:
                    continue
                w = r["background"]["war soundscape"]
                s_mix, _ = variant_score(sc, r["round"], members, med, thr)
                s_alone, _ = variant_score(alone_sc, alone[(r["drone"], r["drone_level_db"], r["start_s"])],
                                           members, med, thr)
                lab = labels(len(s_mix), r["start_s"])
                on = lab == 1
                acc[v][w]["d"].append(s_mix[on] - s_alone[on])
                acc[v][w]["warhop"].append(war_only[w][on])
                acc[v][w]["mix"].append(s_mix[on])

    rows = []
    for v, (members, med) in VARIANTS.items():
        t = variant_score(alone_sc, 1, members, med, thr)[1]
        for w in WAR_DB:
            g = {k: np.concatenate(x) for k, x in acc[v][w].items()}
            neg, pos, d = g["war"], g["mix"], g["d"]
            t5 = float(np.quantile(neg, 0.95))
            rows.append(dict(variant=v, war_db=w, threshold=t, war_mean=float(neg.mean()),
                             war_p99=float(np.quantile(neg, 0.99)), war_fa=float((neg >= t).mean()),
                             d_mean=float(d.mean()), d_abs=float(np.abs(d).mean()),
                             d_big=float((np.abs(d) > 0.1).mean()), d_up=float((d > 0.1).mean()),
                             d_down=float((d < -0.1).mean()),
                             corr_d_war=float(np.corrcoef(d, g["warhop"])[0, 1]),
                             recall=float((pos >= t).mean()), auc=auc(pos, neg),
                             recall_at_5fa=float((pos > t5).mean()), n_drone_hops=len(pos), n_war_hops=len(neg)))
    # drone alone: recall at the frozen threshold (no background)
    for v, (members, med) in VARIANTS.items():
        s = [variant_score(alone_sc, r["round"], members, med, thr) for r in alone_t.rounds]
        pos = np.concatenate([x[labels(len(x), r["start_s"]) == 1] for (x, _), r in zip(s, alone_t.rounds)])
        rows.append(dict(variant=v, war_db="off", threshold=s[0][1], recall=float((pos >= s[0][1]).mean()),
                         n_drone_hops=len(pos)))

    (a.out / "summary.json").write_text(json.dumps(dict(starts=starts, thresholds=thr, rows=rows), indent=1))
    cols = ["war_mean", "war_p99", "war_fa", "d_mean", "d_abs", "d_big", "d_up", "d_down", "corr_d_war", "recall", "auc", "recall_at_5fa"]
    print("\n| variant | war dB SPL | " + " | ".join(cols) + " |")
    for r in rows:
        print(f"| {r['variant']} | {r['war_db']} | " + " | ".join(
            f"{r[c]:.3f}" if c in r else "" for c in cols) + " |")


if __name__ == "__main__":
    main()
