"""P1 (frozen PANNs Cnn14_16k + linear probe) and P2 (LP-FT) on train_clean.py's data (branch 11.1).

    # P1, the e133 compliant data:
    .venv/bin/python scripts/train_panns.py --mode probe --name p1_s0 --device cuda --windows 100000 \\
        --drones data/processed/scenes/glasgow_d1_16 data/processed/scenes/it11s_valdrone \\
        data/processed/scenes/it11s/nodrone --nodrone ~/nodrone --speed 0.8 1.25 --eq --mix-real 0.8
    # P2 from that probe, one seed per GPU:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_panns.py --mode ft --init <out>/p1_s0_panns.pt \\
        --name p2_s0 --seed 0 --device cuda --windows 600000 ...same data flags...

Data, split, augmentation and validation are train_clean.py's, by import, so a P1/P2 run differs
from the e133 runs (entries 133-134) only in the network. The validation windows are train_clean's
(drone scenes with seed >= --val-seed against the --val-rig no-drone clips), and the threshold in the
JSON is at 5% false alarms on them. comb is scored as a sanity check; the lgbm baseline on the same
data and split is in the e133 JSONs. Nothing here is a test result: test sources are excluded by
train_clean.load_nodrone, and sealed data is not read.

Writes <out>/<name>.json and <name>_panns.pt (probe: head only, about 20 KB; ft: whole network).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from dad import realrun as rr  # noqa: E402
from dad.config import DEFAULT  # noqa: E402
from dad.models import panns as pn  # noqa: E402
from dad.pannsrun import PannsScorer  # noqa: E402
from train_clean import CleanSampler, FS, eval_windows, load_drones, load_nodrone, scores  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("probe", "ft"), required=True)
    p.add_argument("--drones", type=Path, nargs="+", required=True)
    p.add_argument("--nodrone", type=Path, default=None)
    p.add_argument("--name", required=True)
    p.add_argument("--out", type=Path, default=REPO / "artifacts" / "it11_1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--pretrained", type=Path, default=REPO / "artifacts" / "pretrained" / pn.CHECKPOINT_NAME)
    p.add_argument("--init", type=Path, default=None, help="ft: the probe checkpoint to start from (LP-FT)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--windows", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4, help="ft: one-cycle peak learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--eval-every", type=int, default=100_000, help="ft: validate every this many windows")
    p.add_argument("--select", choices=("last", "best"), default="last",
                   help="ft: keep the final state (default, as the e133 CRNNs) or the best validation state (entry 140)")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--val-seed", type=int, default=1800)
    p.add_argument("--val-rig", default="nightjar")
    p.add_argument("--eval-hop", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None, help="load only this many scenes (smoke tests)")
    p.add_argument("--speed", nargs=2, type=float, default=None, metavar=("LO", "HI"))
    p.add_argument("--eq", action="store_true")
    p.add_argument("--mix-real", type=float, default=0.0)
    p.add_argument("--mix-snr", nargs=2, type=float, default=(-10.0, 5.0))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)  # noqa: E731
    if a.mode == "ft" and a.init is None:
        raise SystemExit("--mode ft needs --init <probe checkpoint> (LP-FT starts from the probe head)")

    d_train, d_val, w_train, w_val = (sum(z, []) for z in zip(*(load_drones(r, a.val_seed, a.limit) for r in a.drones)))
    n_train, n_val = load_nodrone(a.nodrone, a.val_rig, a.limit) if a.nodrone else ([], [])
    n_train, n_val = n_train + w_train, n_val + w_val
    if not (d_train and d_val and n_train and n_val):
        raise SystemExit(f"empty split: drones {len(d_train)}/{len(d_val)}, no-drone {len(n_train)}/{len(n_val)}")
    hop = int(a.eval_hop * FS)
    pos, neg = eval_windows(d_val, hop), eval_windows(n_val, hop)
    log(f"train: {len(d_train)} drone scenes, {len(n_train)} no-drone clips; "
        f"validation: {len(pos)} drone and {len(neg)} no-drone windows")
    sampler = CleanSampler(d_train, n_train, a.seed, a.speed, a.eq, a.mix_real, tuple(a.mix_snr))

    res = dict(name=a.name, mode=a.mode, seed=a.seed, arch="panns_cnn14_16k", window_s=1.0, eval_hop_s=a.eval_hop,
               windows=a.windows, batch_size=a.batch_size, argv=sys.argv[1:],
               frontend="PANNs Cnn14_16k pretraining log-mel: 16 kHz, 512/160 samples, 64 Slaney mel 50-8000 Hz, "
                        "10*log10, window mean level set to the bn0 mean",
               augment=dict(speed=a.speed, eq=a.eq, mix_real=a.mix_real, mix_snr=list(a.mix_snr)),
               train=dict(drone_scenes=len(d_train), nodrone_clips=len(n_train),
                          nodrone_groups=sorted({g for _, g in n_train})),
               val=dict(drone_scenes=len(d_val), drone_windows=len(pos), nodrone_clips=len(n_val),
                        nodrone_windows=len(neg), rig=a.val_rig if a.nodrone else None), models={})
    res["models"]["comb"] = scores(rr.CombScorer(DEFAULT), pos, neg)
    log(f"comb: {res['models']['comb']}")

    t0 = time.time()
    if a.mode == "probe":
        sc = PannsScorer(a.pretrained, a.device).fit_probe(sampler, a.windows, pos, neg, a.batch_size,
                                                             a.workers, log=log)
    else:
        sc = PannsScorer.load(a.init, device=a.device, pretrained=a.pretrained)
        res["init"] = str(a.init)
        sc.fit_finetune(sampler, a.windows, pos, neg, a.batch_size, a.lr, a.weight_decay, a.eval_every,
                        a.workers, select=a.select, log=log)
    res["pretrained_md5"] = sc.pretrained_md5
    ck = a.out / f"{a.name}_panns.pt"
    sc.save(ck)
    key = f"panns_{a.mode}"
    res["models"][key] = dict(scores(sc, pos, neg), train_seconds=round(time.time() - t0, 1),
                              info=sc.info, history=sc.history, checkpoint=ck.name)
    log(f"{key}: { {k: v for k, v in res['models'][key].items() if k not in ('history', 'info')} }")
    (a.out / f"{a.name}.json").write_text(json.dumps(res, indent=1))
    log(f"wrote {a.out / (a.name + '.json')}")


if __name__ == "__main__":
    main()
