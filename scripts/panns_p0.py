"""P0: no training. How well do the 527 AudioSet outputs of the pretrained Cnn14_16k separate drone
from no-drone windows on the validation data? (branch 11.1, plan step P0)

    .venv/bin/python scripts/panns_p0.py --device cuda --out artifacts/it11_1/p0.json \\
        --drones data/processed/scenes/it11s_valdrone --nodrone ~/nodrone

Each comparison is scored where its data exists on this machine, and skipped with a note otherwise:
  - "val_synthetic": train_clean's validation split, Pd drone scenes (seed >= --val-seed) against
    the --val-rig real no-drone clips (needs --drones and --nodrone);
  - "real_<rig>": real drones against the same rig's background, the validation sessions it9 used
    (rangerun.make_pool_arm(manifest, "none").val; needs data/processed/manifest.csv);
  - "mower": drone windows against mower-only windows of nightjar-lawnmower1-20260805 (the CONFUSER
    session; entry 053's within-rig drone-against-mower check);
  - "glasgow_room": Glasgow d17-d20 hover against the second half of the room recording, and
    "glasgow_war": war-track windows as the negative (needs data/extracted/Glasgow).
Per comparison: the AUC of every AudioSet class (higher score = drone), the 15 best classes, and
named a-priori classes. Choosing classes by these AUCs is choosing on validation data, so P0 is a
diagnostic, never a result. AudioSet's "Drone" class is the musical drone, not a UAV.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from dad import realrun as rr  # noqa: E402
from dad.config import DEFAULT  # noqa: E402
from dad.models import panns as pn  # noqa: E402
from dad.pannsrun import PannsScorer  # noqa: E402

N = DEFAULT.samples_per_window
A_PRIORI = ("Aircraft", "Aircraft engine", "Propeller, airscrew", "Helicopter", "Fixed-wing aircraft, airplane",
            "Engine", "Light engine (high frequency)", "Lawn mower", "Buzz", "Insect", "Mosquito",
            "Fly, housefly", "Bee, wasp, etc.", "Hum", "Mechanical fan", "Drone", "Silence")


def labels() -> list[str] | None:
    f = REPO / "artifacts" / "pretrained" / "class_labels_indices.csv"
    if not f.exists():
        return None
    return [r["display_name"] for r in csv.DictReader(f.open())]


def per_class_auc(sp: np.ndarray, sn: np.ndarray) -> np.ndarray:
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(sp)), np.zeros(len(sn))]
    S = np.r_[sp, sn]
    return np.array([roc_auc_score(y, S[:, k]) if np.ptp(S[:, k]) > 0 else 0.5 for k in range(S.shape[1])])


def summarise(name, auc, names, n_pos, n_neg) -> dict:
    order = np.argsort(-auc)
    lab = (lambda k: names[k]) if names else (lambda k: str(k))
    row = dict(drone_windows=int(n_pos), nodrone_windows=int(n_neg), auc_all=[round(float(v), 4) for v in auc],
               top15=[(lab(k), round(float(auc[k]), 4)) for k in order[:15]])
    if names:
        row["a_priori"] = {c: round(float(auc[names.index(c)]), 4) for c in A_PRIORI if c in names}
    print(f"{name}: {n_pos} drone / {n_neg} no-drone windows; best {row['top15'][:5]}", flush=True)
    return row


class _Collect:
    """A scorer for rangerun.score_chain that keeps the 527 AudioSet columns of the windows score_chain
    keeps: it drops windows whose in-band power is at most realrun.SILENCE_POWER, after scoring."""

    def __init__(self, sc: PannsScorer) -> None:
        self.sc, self.rows = sc, []

    def score(self, audio):
        P = self.sc.audioset(audio)
        keep = np.array([rr.inband_power(x) > rr.SILENCE_POWER for x in audio], bool)
        self.rows.append(P[keep])
        return P[:, 0]

    def matrix(self, s) -> np.ndarray:
        P = np.concatenate(self.rows)
        if len(P) != len(s.s):
            raise RuntimeError(f"kept {len(P)} AudioSet rows but score_chain kept {len(s.s)} windows")
        return P


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=REPO / "artifacts" / "it11_1" / "p0.json")
    p.add_argument("--pretrained", type=Path, default=REPO / "artifacts" / "pretrained" / pn.CHECKPOINT_NAME)
    p.add_argument("--hop", type=float, default=0.5)
    p.add_argument("--drones", type=Path, nargs="*", default=[])
    p.add_argument("--nodrone", type=Path, default=None)
    p.add_argument("--val-seed", type=int, default=1800)
    p.add_argument("--val-rig", default="nightjar")
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sc = PannsScorer(a.pretrained, a.device)
    names = labels()
    out = dict(pretrained_md5=sc.pretrained_md5, hop_s=a.hop, labels=names is not None, comparisons={}, skipped={})
    hop = int(a.hop * DEFAULT.sample_rate)

    def save():
        a.out.write_text(json.dumps(out, indent=1))

    # train_clean's validation split
    if a.drones and a.nodrone:
        from train_clean import eval_windows, load_drones, load_nodrone
        d_val = sum((load_drones(r, a.val_seed, None)[1] for r in a.drones), [])
        n_val = load_nodrone(a.nodrone, a.val_rig, None)[1]
        if d_val and n_val:
            sp, sn = sc.audioset(eval_windows(d_val, hop)), sc.audioset(eval_windows(n_val, hop))
            out["comparisons"]["val_synthetic"] = summarise("val_synthetic", per_class_auc(sp, sn), names, len(sp), len(sn))
            save()
    else:
        out["skipped"]["val_synthetic"] = "needs --drones and --nodrone"

    # real validation rigs and the mower session
    if (REPO / "data" / "processed" / "manifest.csv").exists():
        from dad import rangerun as rg
        from dad.realdata import load_manifest
        arm = rg.make_pool_arm(load_manifest(), "none")
        for rig in sorted({r.source for r in arm.val}):
            col = _Collect(sc)
            s = rg.score_chain(col, [r for r in arm.val if r.source == rig], DEFAULT, a.hop)
            P = col.matrix(s)
            y = s.cw.y
            out["comparisons"][f"real_{rig}"] = summarise(f"real_{rig}", per_class_auc(P[y == 1], P[y == 0]), names,
                                                           (y == 1).sum(), (y == 0).sum())
            save()
        if arm.confuser:
            col = _Collect(sc)
            s = rg.score_chain(col, arm.confuser, DEFAULT, a.hop)
            P = col.matrix(s)
            y = s.cw.y
            out["comparisons"]["mower"] = summarise("mower", per_class_auc(P[y == 1], P[y == 0]), names,
                                                    (y == 1).sum(), (y == 0).sum())
            save()
    else:
        out["skipped"]["real_rigs_and_mower"] = "no data/processed/manifest.csv in this account"

    # Glasgow
    gl = REPO / "data" / "extracted" / "Glasgow"
    if gl.exists():
        from score_glasgow import VAL_DRONES, windows
        from train_clean import read16k
        pos = np.concatenate([windows(read16k(f), hop) for f in sorted(gl.glob("*/_UAV_*_hover_*.WAV"))
                              if int(re.search(r"_d(\d+)_hover_", f.name).group(1)) in VAL_DRONES])
        room = read16k(next((gl / "noise").glob("*.WAV")))
        sp = sc.audioset(pos)
        sn = sc.audioset(windows(room[len(room) // 2:], hop))
        out["comparisons"]["glasgow_room"] = summarise("glasgow_room", per_class_auc(sp, sn), names, len(sp), len(sn))
        war = REPO / "data" / "processed" / "war" / "war_track.wav"
        if war.exists():
            sw = sc.audioset(windows(read16k(war), hop))
            out["comparisons"]["glasgow_war"] = summarise("glasgow_war", per_class_auc(sp, sw), names, len(sp), len(sw))
        save()
    else:
        out["skipped"]["glasgow"] = "no data/extracted/Glasgow in this account"
    save()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
