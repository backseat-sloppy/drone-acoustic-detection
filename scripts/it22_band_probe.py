"""Iteration 22, probe B: does 8-11 kHz add drone / no-drone information to the 64 log-mel bands at 8 kHz?

    uv run python -I scripts/it22_band_probe.py      # writes artifacts/it22_band_probe.json

Raw Nightjar (UMA-16, 48 kHz) and FPV (phones, 44.1/48 kHz) recordings, the two training sources that hold
drone and no-drone audio from the same devices, so a band edge cannot stand in for the label. Up to 40
one-second windows per manifest row with label 0 or 1, read at the original rate.

Features per window: A = the frozen frontend (16 kHz, 64 log-mel 50-8000 Hz), level-normalised;
B = A plus 12 log-mel bands 8-11 kHz from the same audio at 22.05 kHz, with the same level reference.
Per band, the mean and the standard deviation over time. A small LightGBM is trained on one source and
scored on the other, and in 5-fold cross-validation grouped by session. B is "worth a GPU arm" if its
AUC beats A's by at least 0.01 in both transfer directions (fixed in the box before running).
"""
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dad.config import FrontendConfig  # noqa: E402
from dad.features import LogMel  # noqa: E402
from dad.realdata import load_manifest  # noqa: E402  (the sealed-data guard)

PER_ROW, SEED = 40, 22
A_CFG = FrontendConfig()
HI_CFG = FrontendConfig(sample_rate=22_050, n_fft=551, hop_length=220, n_mels=12, fmin=8_000.0, fmax=11_000.0)
FPV_ZIP = ROOT / "data" / "Acoustic_UAV_Field_Recordings" / "Fpv.zip"
NJ = ROOT / "data" / "Nightjar_HF"


def resample(x: np.ndarray, sr: int, to: int) -> np.ndarray:
    g = math.gcd(sr, to)
    return resample_poly(x, to // g, sr // g).astype(np.float32)


def windows(x: np.ndarray, sr: int, rng: np.random.Generator) -> list[np.ndarray]:
    n = len(x) // sr
    if n < 1:
        return []
    picks = np.sort(rng.choice(n, size=min(n, PER_ROW), replace=False))
    return [x[i * sr:(i + 1) * sr] for i in picks]


def features(w: np.ndarray, sr: int, fa: LogMel, fh: LogMel) -> tuple[np.ndarray, np.ndarray]:
    a = fa.frames(resample(w, sr, 16_000))
    h = fh.frames(resample(w, sr, 22_050))
    ref = np.log(np.mean(np.exp(a.astype(np.float64))))
    a, h = np.maximum(a - ref, math.log(1e-5)), np.maximum(h - ref, math.log(1e-5))
    fa_ = np.concatenate([a.mean(1), a.std(1)])
    return fa_, np.concatenate([fa_, h.mean(1), h.std(1)])


def main() -> None:
    rng = np.random.default_rng(SEED)
    fa, fh = LogMel(A_CFG), LogMel(HI_CFG)
    z = zipfile.ZipFile(FPV_ZIP)
    XA, XB, y, src, grp = [], [], [], [], []
    for r in load_manifest():
        if r.source not in ("nightjar", "fpv") or r.label not in (0, 1):
            continue
        att = r.attributes
        if r.source == "nightjar":
            path = NJ / att["session"] / "audio" / att["parent"]
            sr = sf.info(str(path)).samplerate
            a0 = int(round(att["parent_start_s"] * sr))
            x = sf.read(str(path), start=a0, stop=a0 + int(r.seconds * sr), dtype="float32", always_2d=True)[0][:, 0]
        else:
            x, sr = sf.read(io.BytesIO(z.read(att["file"])), dtype="float32", always_2d=True)
            x = x[:, 0]
        for w in windows(x, sr, rng):
            a, b = features(w, sr, fa, fh)
            XA.append(a); XB.append(b); y.append(r.label); src.append(r.source); grp.append(r.group)
    XA, XB, y, src, grp = map(np.asarray, (XA, XB, y, src, grp))
    print(f"{len(y)} windows: " + ", ".join(f"{s} {l}: {int(np.sum((src == s) & (y == l)))}"
                                            for s in ("nightjar", "fpv") for l in (0, 1)), flush=True)

    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    def auc(X, tr, te) -> float:
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
        m.fit(X[tr], y[tr])
        return float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))

    out = {}
    for a, b in (("nightjar", "fpv"), ("fpv", "nightjar")):
        tr, te = src == a, src == b
        out[f"{a} -> {b}"] = {"A": auc(XA, tr, te), "B": auc(XB, tr, te)}
    folds = {"A": [], "B": []}
    for tr, te in GroupKFold(5).split(XA, y, grp):
        if len(set(y[te])) == 2:
            folds["A"].append(auc(XA, tr, te)); folds["B"].append(auc(XB, tr, te))
    out["grouped 5-fold"] = {k: float(np.mean(v)) for k, v in folds.items()}
    out["grouped 5-fold per fold"] = folds
    for k, v in out.items():
        if "per fold" not in k:
            print(f"{k:>16}: A {v['A']:.3f}  B {v['B']:.3f}  (B - A {v['B'] - v['A']:+.3f})", flush=True)
    gain = [out[k]["B"] - out[k]["A"] for k in ("nightjar -> fpv", "fpv -> nightjar")]
    out["worth_gpu_arm"] = bool(min(gain) >= 0.01)
    print("B worth a GPU arm:", out["worth_gpu_arm"])
    p = ROOT / "artifacts" / "it22_band_probe.json"
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
