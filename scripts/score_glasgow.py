"""Score networks on real Glasgow hover drones d17-d20 against the Glasgow room recording (entry 125).

    uv run python scripts/score_glasgow.py --out artifacts/runs/e125_glasgow.json \\
        "it9 CRNN=artifacts/models/final_crnn_none_simdist_crnn.pt@0.4828" ...

d17-d20 are the validation drones of make_scenes.py (entry 108); d21-d24 are test and are not read here.
Validation, with caveats: it9 trained on Glasgow drones (rangerun.make_pool_arm: "Glasgow always
trains"), so its figures are optimistic; and the room recording was made at another gain or setting than
the background under the drones (entry 108), so it is a same-lab, not a same-take, negative. Each window's
level is removed before the model (variant "level"), so the gain itself says nothing.
Per model: AUC, recall at 5 and 10% false alarms on the room recording, and at the model's frozen
threshold ("label=path@threshold") the room false-alarm rate and the recall, also per distance.
"""
from __future__ import annotations

import argparse
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
from train_clean import read16k  # noqa: E402

GLASGOW = REPO / "data" / "extracted" / "Glasgow"
VAL_DRONES = (17, 18, 19, 20)
N = DEFAULT.samples_per_window


def windows(x: np.ndarray, hop: int) -> np.ndarray:
    w = np.stack([x[s:s + N] for s in range(0, len(x) - N + 1, hop)])
    return w[[rr.inband_power(a) > rr.SILENCE_POWER for a in w]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", help='"label=checkpoint.pt@threshold"')
    p.add_argument("--device", default="cpu")
    p.add_argument("--hop", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=REPO / "artifacts" / "runs" / "e125_glasgow.json")
    p.add_argument("--room-second-half", action="store_true",
                   help="room negatives from the second half only (the first half trains entry 133's arm)")
    p.add_argument("--war-snr", type=float, nargs="*", default=[],
                   help="also mix every drone window into a random war-track window at these in-band SNRs (dB), "
                        "against war-track windows alone; recall at 1%% FA on those (entry 133's gate d)")
    a = p.parse_args()
    hop = int(a.hop * DEFAULT.sample_rate)

    pos, dist = [], []
    for f in sorted(GLASGOW.glob("*/_UAV_*_hover_*.WAV")):
        m = re.search(r"_d(\d+)_hover_(\d)m_", f.name)
        if int(m.group(1)) in VAL_DRONES:
            w = windows(read16k(f), hop)
            pos.append(w); dist += [m.group(2) + " m"] * len(w)
    pos, dist = np.concatenate(pos), np.array(dist)
    room = read16k(next((GLASGOW / "noise").glob("*.WAV")))
    neg = windows(room[len(room) // 2:] if a.room_second_half else room, hop)
    print(f"drone windows {len(pos)} (d{VAL_DRONES[0]}-d{VAL_DRONES[-1]}), room windows {len(neg)}", flush=True)

    scorers = [("comb", rr.CombScorer(DEFAULT), None)]
    for spec in a.models:
        label, rest = spec.split("=", 1)
        path, thr = rest.split("@")
        scorers.append((label, rr.NetScorer.load(REPO / path, DEFAULT, a.device), float(thr)))

    out = dict(drone_windows=len(pos), room_windows=len(neg), hop_s=a.hop, drones=list(VAL_DRONES),
               room_second_half=a.room_second_half, models={})
    if a.war_snr:
        war = windows(read16k(REPO / "data/processed/war/war_track.wav"), hop)
        rng = np.random.default_rng(133)
        pick = war[rng.integers(0, len(war), len(pos))]
        pw = np.array([rr.inband_power(x) for x in pick]); pd = np.array([rr.inband_power(x) for x in pos])
        mixes = {snr: pick + pos * np.sqrt(pw * 10 ** (snr / 10) / np.maximum(pd, 1e-20))[:, None] for snr in a.war_snr}
        out.update(war_windows=len(war), war_snr=a.war_snr)
    for label, sc, thr in scorers:
        sp, sn = sc.score(pos), sc.score(neg)
        row = dict(auc=rr._auc(sp, sn), recall_at_5fa=rr._recall_at_fa(sp, sn))
        t10 = float(np.quantile(sn, 0.90))
        row["recall_at_10fa"] = float((sp > t10).mean())
        for d in ("1 m", "5 m"):
            row[f"auc_{d}"] = rr._auc(sp[dist == d], sn)
        if thr is not None:
            row.update(frozen_threshold=thr, frozen_room_fa=float((sn >= thr).mean()),
                       frozen_recall=float((sp >= thr).mean()),
                       **{f"frozen_recall_{d}": float((sp[dist == d] >= thr).mean()) for d in ("1 m", "5 m")})
        if a.war_snr:
            sw = sc.score(war); t1 = float(np.quantile(sw, 0.99))
            row["war_fa_frozen"] = float((sw >= thr).mean()) if thr is not None else None
            for snr, m in mixes.items():
                sm = sc.score(m)
                row[f"war{snr:+g}_auc"] = rr._auc(sm, sw)
                row[f"war{snr:+g}_recall_1fa"] = float((sm > t1).mean())
        out["models"][label] = row
        print(label, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        a.out.write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
