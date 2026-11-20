"""Rescore finished virtual live tests (artifacts/complete_test/*_live_virtual_*) with a truth that only
counts the drone where it can be heard, from the files each run already wrote: no audio is rebuilt.

Why: a modified drone flies in a straight line for its whole file (complete_test_engine.approach), so a
10 s file at 100 km/h starts 278 m out. Its "80 dB" is at 1 m and falls as 1 / r, so for most of the
truth span it is far below the background, yet every window in the span is labelled drone. This script
reports, per detector:

    as run      the window labels of rounds.csv (rebuilt from the truth span and checked against it)
    audible     drone windows whose window SNR (window_snr.npz, 1 s) is below `floor` dB are left out:
                neither a hit nor a miss. No-drone windows are unchanged, so the false-positive rate is
                the same as run.
    distance    per drone round, how far away the drone was at the first alarm inside its approach
                (window middle at or after the drone's start, window end at or before the impact):
                hypot(closest_m, v * (impact - t_end)). Next to it, the distance at which the window SNR
                first reached the floor, i.e. where the drone became audible on this truth.

Every detector alarms at its own frozen threshold (config.json "thresholds"); a live alarm threshold the
tester set is used only for the check against rounds.csv. Runs with identical scores (the same plan run
again) are counted once. Virtual runs only: microphone runs did not hear the mix (mic_corr ~0.02).
Simulated drones from training sources: a tool, not a detection result (CLAUDE.md section 5).

    uv run python scripts/rescore_live.py [--floor -10 -5] [--out artifacts/complete_test/rescore_audible]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import drone_war_grid as dwg                                  # noqa: E402  (comb's squash)

HOP_S = 0.125
WIN_S = 1.0


def grid_t_end(n: int) -> np.ndarray:
    return np.arange(n) * HOP_S + WIN_S


def labels(t_end: np.ndarray, span: tuple[float, float] | None) -> np.ndarray:
    """dad.wavio.window_labels for one drone: 1 inside, -1 straddling, 0 outside."""
    lab = np.zeros(len(t_end), int)
    if span is None:
        return lab
    a, b = t_end - WIN_S, t_end
    inside = (a >= span[0]) & (b <= span[1])
    touch = (b > span[0]) & (a < span[1])
    lab[touch & ~inside] = -1
    lab[inside] = 1
    return lab


def speed_of(label: str) -> float:
    return float(re.search(r"@ ([\d.]+) km/h", label).group(1))


def run_rounds(d: Path) -> list[dict]:
    """rounds.csv rows grouped as {round: {detector: row}} plus the round's settings."""
    out: dict[int, dict] = {}
    with open(d / "rounds.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = out.setdefault(int(row["round"]), dict(round=int(row["round"]), drone=row["drone"],
                                                       start_s=row["start_s"], dets={}))
            r["dets"][row["detector"]] = row
    return [out[k] for k in sorted(out)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, nargs="+", default=[-10.0, -5.0])
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/complete_test/rescore_audible")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    used, dupes, empty, mismatches = [], [], [], []
    acc: dict[str, dict] = {}                                  # detector -> counters
    dist_rows = []
    curves: dict[str, dict[int, list[float]]] = {}            # detector -> {second before impact: [round means]}
    no_drone_conf: dict[str, list[float]] = {}                # detector -> confidence of every no-drone window
    for d in sorted((ROOT / "artifacts/complete_test").glob("*_live_virtual_*")):
        if not (d / "scores.npz").exists() or not (d / "rounds.csv").exists():
            empty.append(d.name)
            continue
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        plan, thr, spans = cfg["plan"], cfg["thresholds"], cfg["drone_spans"]
        rounds = run_rounds(d)
        if not rounds:
            empty.append(d.name)
            continue
        sc = np.load(d / "scores.npz")
        snr = np.load(d / "window_snr.npz")
        h = hashlib.sha1(b"".join(np.ascontiguousarray(sc[k]).tobytes() for k in sorted(sc.files))).hexdigest()
        if h in seen:
            dupes.append(d.name)
            continue
        seen.add(h)
        used.append(d.name)
        buf = float(plan.get("buffer_after_s") or 0.0)
        closest = float((plan.get("approach") or {}).get("closest_m", 1.0))
        alarm = (plan.get("live") or {}).get("threshold")
        for r in rounds:
            drone = r["drone"] != "none"
            for det, row in r["dets"].items():
                key = f"{r['round']}|{det}"
                if key not in sc.files:
                    continue
                s = np.asarray(sc[key], float)
                t_end = grid_t_end(len(s))
                span = imp = None
                if drone:
                    s0, s1, im = spans[r["drone"]]
                    st = float(r["start_s"])
                    span = (st + s0, st + s1 + buf)
                    imp = st + im
                lab = labels(t_end, span)
                # check against rounds.csv, at the threshold the run alarmed at
                conf = dwg.squash(s, thr[det]) if det == "comb" else s
                hit_run = (conf >= float(alarm)) if alarm is not None else (s >= thr[det])
                got = [int((hit_run & (lab == 1)).sum()), int((~hit_run & (lab == 1)).sum()),
                       int((hit_run & (lab == 0)).sum()), int((~hit_run & (lab == 0)).sum())]
                want = [int(row[k]) for k in ("tp", "fn", "fp", "tn")]
                if got != want:
                    mismatches.append(f"{d.name} round {r['round']} {det}: rebuilt {got}, rounds.csv {want}")
                # the rescoring, at the frozen threshold
                hit = s >= thr[det]
                c = acc.setdefault(det, dict(tp=0, fn=0, fp=0, tn=0, **{f"a{f:g}": [0, 0] for f in a.floor},
                                             rounds=0, detected=0))
                c["tp"] += int((hit & (lab == 1)).sum()); c["fn"] += int((~hit & (lab == 1)).sum())
                c["fp"] += int((hit & (lab == 0)).sum()); c["tn"] += int((~hit & (lab == 0)).sum())
                no_drone_conf.setdefault(det, []).extend(conf[lab == 0].tolist())
                if not drone:
                    continue
                w = np.full(len(s), np.nan)
                ws = snr[f"{r['round']}|1"] if f"{r['round']}|1" in snr.files else np.full(len(s), np.nan)
                w[:min(len(s), len(ws))] = ws[:len(s)]
                for f in a.floor:
                    pos = (lab == 1) & (w >= f)
                    c[f"a{f:g}"][0] += int((hit & pos).sum()); c[f"a{f:g}"][1] += int(pos.sum())
                v = speed_of(r["drone"]) / 3.6
                approach = (t_end - WIN_S / 2 >= span[0]) & (t_end <= imp)
                dist = np.hypot(closest, v * np.clip(imp - t_end, 0, None))
                first = np.flatnonzero(hit & approach)
                aud = np.flatnonzero(approach & (w >= min(a.floor)))
                c["rounds"] += 1
                c["detected"] += bool(len(first))
                # REQUIREMENTS.md Test 2: mean confidence over 3 to 7 s before impact, each second equally
                tb = imp - t_end                                  # time before impact at each window's end
                conf = dwg.squash(s, thr[det]) if det == "comb" else s
                secs = [np.nanmean(conf[(tb > k) & (tb <= k + 1)]) for k in range(3, 7)
                        if ((tb > k) & (tb <= k + 1)).any()]
                c37 = float(np.mean(secs)) if len(secs) == 4 else None
                band = (tb > 3) & (tb <= 7)
                snr37 = float(np.nanmedian(w[band])) if np.isfinite(w[band]).any() else None
                curve = curves.setdefault(det, {})
                for k in range(0, 10):
                    m = (tb > k) & (tb <= k + 1)
                    if m.any():
                        curve.setdefault(k, []).append(float(np.nanmean(conf[m])))
                dist_rows.append(dict(run=d.name, round=r["round"], drone=r["drone"], detector=det,
                                      threshold=round(thr[det], 4),
                                      conf_3_7s=None if c37 is None else round(c37, 4),
                                      median_snr_3_7s_db=None if snr37 is None else round(snr37, 1),
                                      approach_s=round(float(imp - t_end[approach][0]), 3) if approach.any() else None,
                                      audible_lead_s=round(float(imp - t_end[aud[0]]), 3) if len(aud) else None,
                                      lead_s=round(float(imp - t_end[first[0]]), 3) if len(first) else None,
                                      start_dist_m=round(float(dist[approach][0]), 1) if approach.any() else None,
                                      first_alarm_dist_m=round(float(dist[first[0]]), 1) if len(first) else None))

    ratio = lambda x, y: f"{x / y:.3f}" if y else ""          # noqa: E731
    with open(a.out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["detector", "drone_rounds", "recall_as_run", "drone_windows_as_run",
                    *[x for fl in a.floor for x in (f"recall_snr_ge_{fl:g}", f"drone_windows_snr_ge_{fl:g}")],
                    "false_positive_rate", "no_drone_windows", "share_rounds_detected",
                    "median_lead_s", "median_audible_lead_s",
                    "mean_conf_3_7s", "rounds_with_3_7s", "median_snr_3_7s_db", "mean_conf_no_drone"])
        for det, c in acc.items():
            rows = [x for x in dist_rows if x["detector"] == det]
            c37 = [x["conf_3_7s"] for x in rows if x["conf_3_7s"] is not None]
            s37 = [x["median_snr_3_7s_db"] for x in rows if x["median_snr_3_7s_db"] is not None]
            nd = no_drone_conf.get(det, [])
            tail = [f"{np.mean(c37):.3f}" if c37 else "", len(c37), f"{np.median(s37):.1f}" if s37 else "",
                    f"{np.nanmean(nd):.3f}" if nd else ""]
            fa = [x["lead_s"] for x in rows if x["lead_s"] is not None]
            au = [x["audible_lead_s"] for x in rows if x["audible_lead_s"] is not None]
            w.writerow([det, c["rounds"], ratio(c["tp"], c["tp"] + c["fn"]), c["tp"] + c["fn"],
                        *[x for fl in a.floor for x in (ratio(*c[f"a{fl:g}"]), c[f"a{fl:g}"][1])],
                        ratio(c["fp"], c["fp"] + c["tn"]), c["fp"] + c["tn"], ratio(c["detected"], c["rounds"]),
                        f"{np.median(fa):.1f}" if fa else "", f"{np.median(au):.1f}" if au else "", *tail])
    with open(a.out / "curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["detector", *[f"{k}-{k + 1}s" for k in range(10)]])
        for det, cv in curves.items():
            w.writerow([det, *[f"{np.mean(cv[k]):.3f}" if k in cv else "" for k in range(10)]])
    with open(a.out / "distance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(dist_rows[0]))
        w.writeheader()
        w.writerows(dist_rows)
    (a.out / "runs.json").write_text(json.dumps(dict(used=used, duplicates=dupes, empty_or_stopped=empty,
                                                     mismatches=mismatches, floors=a.floor), indent=1), encoding="utf-8")
    print(f"runs used {len(used)}, duplicates {len(dupes)}, empty/stopped {len(empty)}, label mismatches {len(mismatches)}")
    for m in mismatches[:10]:
        print("  ", m)
    print((a.out / "summary.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
