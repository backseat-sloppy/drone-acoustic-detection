"""Iteration 22, probe B2, step 2 (server, GPU): does a CRNN gain from 8-11 kHz, everything else equal?

    PYTHONPATH=$PWD/src .venv/bin/python -u scripts/it22_band_net.py   # reads artifacts/it22_band/windows.*

The same windows (it22_band_store.py: Nightjar and FPV at 22.05 kHz) through two frontends:
  A: resampled to 16 kHz, the frozen frontend (64 log-mel 50-8000 Hz, 25/10 ms);
  B: at 22.05 kHz, 74 log-mel 50-11000 Hz (the same mel spacing extended), 25/10 ms (551/220 samples).
Both level-normalised as realrun.featurise does, standardised per band on the training fold.
The CRNN of the zoo, Adam 1e-3, batch 64, EPOCHS epochs, a random smooth EQ per window (as in training),
no tuning: nothing is chosen on the held-out fold. 4 folds grouped by session (GroupKFold), 2 seeds.
B is worth a store rebuild if its mean held-out AUC beats A's by at least 0.01 and it wins in at least 3
of the 4 folds (seed means) - fixed in report entry 214 before the run.
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dad import realrun as rr  # noqa: E402
from dad.config import FrontendConfig  # noqa: E402
from dad.features import LogMel  # noqa: E402
from dad.models.zoo import build  # noqa: E402

SR, EPOCHS, SEEDS, BATCH = 22_050, 12, (0, 1), 64
CFG_A = FrontendConfig()
CFG_B = FrontendConfig(sample_rate=SR, n_fft=551, hop_length=220, n_mels=74, fmin=50.0, fmax=11_000.0)
DIR = ROOT / "artifacts" / "it22_band"


def feats(x: np.ndarray) -> dict[str, np.ndarray]:
    a = resample_poly(x, 320, 441, axis=1).astype(np.float32)          # 22050 -> 16000
    out = {}
    for k, cfg, audio in (("A", CFG_A, a), ("B", CFG_B, x)):
        fe = LogMel(cfg)
        out[k] = rr.featurise(audio, rr.VARIANTS["level"], fe)
    return out


def train_score(Xtr, ytr, Xte, seed: int, dev) -> np.ndarray:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu, sd = Xtr.mean(axis=(0, 2), keepdims=True), Xtr.std(axis=(0, 2), keepdims=True) + 1e-5
    net = build("crnn", Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(EPOCHS):
        net.train()
        for i in np.array_split(rng.permutation(len(ytr)), max(1, len(ytr) // BATCH)):
            eq = np.stack([rr._eq_curve(rng, Xtr.shape[1]) for _ in i])[:, :, None]
            xb = torch.from_numpy(((Xtr[i] + eq - mu) / sd).astype(np.float32)[:, None]).to(dev)
            yb = torch.from_numpy(ytr[i].astype(np.int64)).to(dev)
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(xb), yb).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return np.concatenate([torch.softmax(net(torch.from_numpy(((Xte[j] - mu) / sd).astype(np.float32)[:, None]).to(dev)),
                                             dim=1)[:, 1].cpu().numpy()
                               for j in np.array_split(np.arange(len(Xte)), max(1, len(Xte) // 512))])


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = list(csv.DictReader(open(DIR / "windows.csv", encoding="utf-8")))
    audio, sr = sf.read(str(DIR / "windows.flac"), dtype="float32")
    assert sr == SR and len(audio) == len(rows) * SR, (sr, len(audio), len(rows))
    x = audio.reshape(len(rows), SR)
    y = np.array([int(r["label"]) for r in rows])
    grp = np.array([r["group"] for r in rows])
    F = {"A": [], "B": []}
    for j in np.array_split(np.arange(len(x)), max(1, len(x) // 1000)):
        f = feats(x[j])
        F["A"].append(f["A"]); F["B"].append(f["B"])
    F = {k: np.concatenate(v) for k, v in F.items()}
    print(f"{len(y)} windows ({int(y.sum())} drone), {len(set(grp))} groups; A {F['A'].shape[1:]}, B {F['B'].shape[1:]}",
          flush=True)
    res = []
    for fold, (tr, te) in enumerate(GroupKFold(4).split(x, y, grp)):
        row = dict(fold=fold, test_groups=sorted(set(grp[te])), n_test=int(len(te)), drone_test=int(y[te].sum()))
        for k in ("A", "B"):
            row[k] = [float(roc_auc_score(y[te], train_score(F[k][tr], y[tr], F[k][te], s, dev))) for s in SEEDS]
        print(f"fold {fold} {row['test_groups']}: A {np.mean(row['A']):.3f} {row['A']}  B {np.mean(row['B']):.3f} {row['B']}",
              flush=True)
        res.append(row)
    a = np.array([np.mean(r["A"]) for r in res]); b = np.array([np.mean(r["B"]) for r in res])
    out = dict(folds=res, mean_A=float(a.mean()), mean_B=float(b.mean()), folds_B_wins=int((b > a).sum()),
               worth_rebuild=bool(b.mean() - a.mean() >= 0.01 and (b > a).sum() >= 3))
    print(f"mean A {a.mean():.3f}  B {b.mean():.3f}  (B - A {b.mean() - a.mean():+.3f}), B wins {(b > a).sum()} of 4 "
          f"-> {'worth a rebuild' if out['worth_rebuild'] else 'not worth a rebuild'}", flush=True)
    (DIR / "result.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
