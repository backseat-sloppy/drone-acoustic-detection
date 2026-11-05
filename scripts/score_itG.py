"""Score runs A and B of report entry 106 once on its test sets, at each run's frozen threshold.

    uv run python scripts/score_itG.py --runs artifacts/itG/A_s0 artifacts/itG/A_s1 ... --out artifacts/itG/test.json

Each run is the <out>/<name> prefix of scripts/train_clean.py (<name>.json, _dscnn.pt, _lgbm.pkl);
comb uses the threshold of the first run's JSON. Test sets (entry 106, fixed before any run):

    aris        the 10 sealed_peek ARIS excerpts with the war track at 0 dB, exactly as detect_gui
                saved them (artifacts/detect_gui/mixes/Drone_*); share of 1 s windows alarmed at a
                0.125 s hop, per clip and mean (no frame labels: the drone is taken to be present)
    war_only    the Dummy mix and the war track alone at the GUI's start (120.2 s, 10 s): false alarms
    room        the whole Glasgow room recording (60.5 min): false alarms (the leak check)
    fsd_eval    every fsd50k_eval stem of data/processed/stems (no drone in any): false alarms
    glasgow     data/processed/scenes/itG/test_drone (drones d21-d24) against .../test_nodrone, at a
                0.5 s hop as train_clean.py validates: AUC, recall and false alarms at the threshold

The rule: success if the mean ARIS share is >= 0.45 and false alarms are <= 0.05 on war_only, room and
fsd_eval; failure if the ARIS share is <= 0.30; leak if room false alarms > 0.05.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

from dad import realrun as rr  # noqa: E402
from dad.config import DEFAULT  # noqa: E402
from dad.wavio import Clip, read_wav, stream  # noqa: E402
import train_clean as tc  # noqa: E402

MIXES = REPO / "artifacts/detect_gui/mixes"
ROOM = REPO / "data/extracted/Glasgow/noise/_uav_2022noise_n_noise_1m_.WAV"
WAR = REPO / "data/processed/war/war_track.wav"
SCENES = REPO / "data/processed/scenes/itG"


def share(scorer, thr, audio) -> float:
    d = stream(Clip("", np.asarray(audio, np.float32), DEFAULT.sample_rate, 1, ""), scorer, DEFAULT, thr)
    return float(d.drone.mean()) if len(d.score) else float("nan")


def sets():
    aris = {p.name.split("_war")[0]: read_wav(p).audio for p in sorted(MIXES.glob("Drone_*_snr+0dB_seed1000.wav"))}
    war = read_wav(WAR).audio
    s = int(120.2 * DEFAULT.sample_rate)
    war_only = [read_wav(next(MIXES.glob("Dummy_*_snr+0dB_seed1000.wav"))).audio, war[s:s + 10 * DEFAULT.sample_rate]]
    fsd = [read_wav(REPO / r["file"]).audio for r in csv.DictReader((REPO / "data/processed/stems/manifest.csv").open())
           if r["source"] == "fsd50k_eval"]
    pos = tc.load_drones(SCENES / "test_drone", 0, None)[1]       # val_seed 0: every scene lands in "val"
    neg = tc.load_drones(SCENES / "test_nodrone", 0, None)[3]
    hop = int(0.5 * DEFAULT.sample_rate)
    return dict(aris=aris, war_only=war_only, room=read_wav(ROOM).audio, fsd=fsd,
                g_pos=tc.eval_windows(pos, hop), g_neg=tc.eval_windows(neg, hop))


def evaluate(scorer, thr, S) -> dict:
    aris = {k: share(scorer, thr, a) for k, a in S["aris"].items()}
    sp, sn = scorer.score(S["g_pos"]), scorer.score(S["g_neg"])
    out = dict(threshold=thr, aris=aris, aris_mean=float(np.mean(list(aris.values()))),
               war_only=float(np.mean([share(scorer, thr, a) for a in S["war_only"]])),
               room=share(scorer, thr, S["room"]),
               fsd_eval=float(np.mean(np.concatenate([stream(Clip("", a, DEFAULT.sample_rate, 1, ""), scorer, DEFAULT,
                                                             thr).drone for a in S["fsd"]]))),
               glasgow=dict(auc=rr._auc(sp, sn), recall=float((sp >= thr).mean()), false_alarms=float((sn >= thr).mean()),
                            drone_windows=len(sp), nodrone_windows=len(sn)))
    ok_fa = max(out["war_only"], out["room"], out["fsd_eval"]) <= 0.05
    out["verdict"] = ("leak" if out["room"] > 0.05 else "success" if out["aris_mean"] >= 0.45 and ok_fa
                      else "failure" if out["aris_mean"] <= 0.30 else "neither")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, required=True, help="train_clean output prefixes")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    S = sets()
    print(f"aris clips {len(S['aris'])}, fsd_eval stems {len(S['fsd'])}, glasgow test windows "
          f"{len(S['g_pos'])} drone / {len(S['g_neg'])} no-drone", flush=True)
    res = {}
    first = json.loads(a.runs[0].with_suffix(".json").read_text())
    res["comb"] = evaluate(rr.CombScorer(DEFAULT), first["models"]["comb"]["threshold_5pct_fa"], S)
    meta = {run: json.loads(run.with_suffix(".json").read_text())["models"] for run in a.runs}
    # LightGBM before any network: on macOS, LightGBM after torch has loaded segfaults (two OpenMP
    # runtimes in one process, report entry 110); train_clean.py has the same order.
    for run, m in meta.items():
        res[f"{run.name}/lgbm"] = evaluate(rr.LgbmScorer.load(f"{run}_lgbm.pkl", DEFAULT), m["lgbm"]["threshold_5pct_fa"], S)
    for run, m in meta.items():
        res[f"{run.name}/dscnn"] = evaluate(rr.NetScorer.load(f"{run}_dscnn.pt", DEFAULT, "cpu"),
                                            m["dscnn"]["threshold_5pct_fa"], S)
    for k, v in res.items():
        print(f"{k:16} aris {v['aris_mean']:.3f}  war {v['war_only']:.3f}  room {v['room']:.3f}  fsd {v['fsd_eval']:.3f}  "
              f"glasgow auc {v['glasgow']['auc']:.3f} rec {v['glasgow']['recall']:.3f} fa {v['glasgow']['false_alarms']:.3f}"
              f"  -> {v['verdict']}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
