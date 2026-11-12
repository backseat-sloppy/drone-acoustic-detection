"""Iteration 22, probe B3, step 2 (server, GPU): does the 8-11 kHz gain of probe B2 survive at range, on Lund?

    PYTHONPATH=$PWD/src .venv/bin/python -u scripts/it22_band_lund.py   # artifacts/it22_band/{windows,lund}.*

Frontends A (16 kHz, 64 bands to 8 kHz) and B (22.05 kHz, 74 bands to 11 kHz) as it22_band_net.py, the same
CRNN and training. Training: the Nightjar and FPV windows of three of the four GroupKFold(4) folds; the
fourth fold (by group) is held out to set each arm's threshold at 1% false alarms on its no-drone windows.
Lund drone windows (never trained on) are scored at that threshold and binned by slant range.
Pass (entry 216, fixed before the run): B's detection rate, averaged over the bins from 100 m on, at least
3 points above A's; B no more than 1 point below A in any bin; both on at least 2 of the 3 seeds.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import it22_band_net as B  # noqa: E402

SEEDS, FA, BINS = (0, 1, 2), 0.01, (0, 50, 100, 150, 1e9)


def load(name: str) -> tuple[np.ndarray, list[dict]]:
    rows = list(csv.DictReader(open(B.DIR / f"{name}.csv", encoding="utf-8")))
    audio, sr = sf.read(str(B.DIR / f"{name}.flac"), dtype="float32")
    assert sr == B.SR and len(audio) == len(rows) * B.SR, (name, sr, len(audio), len(rows))
    x = audio.reshape(len(rows), B.SR)
    F = {"A": [], "B": []}
    for j in np.array_split(np.arange(len(x)), max(1, len(x) // 1000)):
        f = B.feats(x[j])
        F["A"].append(f["A"]); F["B"].append(f["B"])
    return {k: np.concatenate(v) for k, v in F.items()}, rows


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F, rows = load("windows")
    L, lrows = load("lund")
    y = np.array([int(r["label"]) for r in rows])
    grp = np.array([r["group"] for r in rows])
    tr, va = next(iter(GroupKFold(4).split(y, y, grp)))
    neg = va[y[va] == 0]
    rng_m = np.array([float(r["range_m"]) for r in lrows])
    edges = list(zip(BINS[:-1], BINS[1:]))
    print(f"train {len(tr)}, threshold on {len(neg)} no-drone windows of {sorted(set(grp[va]))}; Lund {len(lrows)} "
          f"windows, per bin " + ", ".join(f"{lo:g}-{hi:g} m: {int(((rng_m >= lo) & (rng_m < hi)).sum())}" for lo, hi in edges),
          flush=True)
    out = {"seeds": []}
    for s in SEEDS:
        row = {"seed": s}
        for k in ("A", "B"):
            p = B.train_score(F[k][tr], y[tr], np.concatenate([F[k][neg], L[k]]), s, dev, save=B.DIR / f"probe_{k}_s{s}.pt")
            pn, pl = p[:len(neg)], p[len(neg):]
            thr = float(np.quantile(pn, 1 - FA))
            row[k] = dict(threshold=thr, bins=[float(np.mean(pl[(rng_m >= lo) & (rng_m < hi)] > thr)) for lo, hi in edges],
                          all=float(np.mean(pl > thr)), mean_conf=float(np.mean(pl)))
        far = [i for i, (lo, _) in enumerate(edges) if lo >= 100]
        d = np.array(row["B"]["bins"]) - np.array(row["A"]["bins"])
        row["pass"] = bool(np.mean(d[far]) >= 0.03 and d.min() >= -0.01)
        print(f"seed {s}: A {np.round(row['A']['bins'], 3).tolist()} B {np.round(row['B']['bins'], 3).tolist()} "
              f"(far B-A {np.mean(d[far]):+.3f}, worst bin {d.min():+.3f}) -> {'pass' if row['pass'] else 'no'}", flush=True)
        out["seeds"].append(row)
    out["bins_m"] = edges
    out["pass"] = sum(r["pass"] for r in out["seeds"]) >= 2
    print("B3", "PASSES" if out["pass"] else "FAILS")
    (B.DIR / "lund_result.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
