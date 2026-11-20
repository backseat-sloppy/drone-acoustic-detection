"""Report entries 250-253: score the crossed runs (A trained with Mathias's clips, B with ITU ARIS clips) and the
reference CRNN on both clip sets, mixed into dev_test's war backgrounds exactly as dev_test mixes a drone.

    uv run python -u scripts/exp251_cross.py [--models NAME ...]

Every clip set is scored by every model, so each model has a set it trained on (a memorisation ceiling) and one
it did not. None of this is a detection result: run B has trained on ITU ARIS (entry 250).
Writes artifacts/dev_test/exp251/<model>.npz, rounds.json and cross.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dev_test", ROOT / "scripts" / "dev_test.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)

ARIS = ROOT / "data/extracted/sealed_peek"
SETS = {"mathias": sorted(D.RECORDED.glob("*.wav")),
        "aris": [ARIS / f"Drone_{i:02d}.wav" for i in range(1, 11)]}   # CleanDrone.wav is on the server only
for _k, _run in (("exp251 A mathias", "exp251_A_mathias_s1"), ("exp251 B aris", "exp251_B_aris_s1")):
    _pt = D.BRANCH_MODELS / f"{_run}_none_simdist_colour_crnn.pt"
    if _pt.exists():
        D.dwg.MODELS[_k] = (ROOT / f"artifacts/runs/{_run}.json", _pt)
MODELS = ["exp251 A mathias", "exp251 B aris", "it23 CRNN relabel colour s1", "comb", "it6 lgbm"]
OUT = D.OUT / "exp251"


def rounds() -> list[tuple[dict, np.ndarray]]:
    bgs, out = D.backgrounds(), []
    for (s, bg), x in bgs.items():
        out.append((dict(kind="none", stretch=s, bg="+".join(bg)), x.astype(np.float32)))
    for name, files in SETS.items():
        for j, f in enumerate(files):
            x = D.read_wav(f, D.DEFAULT).audio
            for s in (j % len(D.WAR_STARTS), (j + 3) % len(D.WAR_STARTS)):
                for bg in D.WAR_BGS:
                    for db in D.DRONE_DBS:
                        out.append((dict(kind="drone", set=name, file=f.stem, stretch=s, bg="+".join(bg), drone_db=db),
                                    (D.at_level(x, db) + bgs[s, bg]).astype(np.float32)))
    return out


def readout(rs: list[dict], s: list[np.ndarray]) -> dict:
    def hops(f):
        return np.concatenate([v for r, v in zip(rs, s) if f(r)])
    out = {}
    none_war, none_all = hops(lambda r: r["kind"] == "none" and r["bg"] == "war"), \
        hops(lambda r: r["kind"] == "none" and r["bg"] in {"+".join(b) for b in D.WAR_BGS})
    out["none war"] = 100 * none_war.mean()
    for name in SETS:
        dw = hops(lambda r: r["kind"] == "drone" and r["set"] == name and r["bg"] == "war")
        da = hops(lambda r: r["kind"] == "drone" and r["set"] == name)
        out[f"{name}: drone war"] = 100 * dw.mean()
        out[f"{name}: gap war"] = 100 * (dw.mean() - none_war.mean())
        out[f"{name}: auc war"] = D._auc(dw, none_war)
        out[f"{name}: gap all-war"] = 100 * (da.mean() - none_all.mean())
        out[f"{name}: auc all-war"] = D._auc(da, none_all)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=MODELS)
    a = p.parse_args()
    rs = rounds()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = [r for r, _ in rs]
    (OUT / "rounds.json").write_text(json.dumps(dict(audio_md5=D.audio_md5(rs), rounds=meta)), encoding="utf-8")
    thr, scorers = D.E.load_detectors(a.models, "cpu")
    res = {}
    for m in a.models:
        sc = scorers[m]
        s = [D.stream(D.Clip("r", x, D.SR, 1, "FLOAT"), sc, sc.cfg, thr[m]).score for _, x in rs]
        if m == "comb":
            s = [D.dwg.squash(v, thr[m]) for v in s]
        np.savez_compressed(OUT / f"{m}.npz", *s)
        res[m] = readout(meta, s)
        print(m, json.dumps({k: round(float(v), 3) for k, v in res[m].items()}), flush=True)
    old = json.loads((OUT / "cross.json").read_text(encoding="utf-8")) if (OUT / "cross.json").exists() else {}
    (OUT / "cross.json").write_text(json.dumps(old | res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
