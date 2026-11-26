"""Iteration 25: the clean-drone world simulator (report entries 239-).

    uv run python scripts/run_worldsim.py --name it25_worldsim_s0 --models comb lgbm crnn dscnn --device mps

Trains on `SynthSampler` with real Glasgow drones (every drone but the validation four) through
`dronesynth.real_drone`, over FSD50K-dev "train" stems, with Glasgow's noise recording as chain-twin
negatives and ladder families of 1 + 3. Validation is the same simulator on the four held-out drones,
the "val" stems and the last 20 % of the noise recording: a fixed set of families drawn with seed
VAL_SEED, the same for every model. The deployment threshold is frozen at --deploy-fa false alarms
on the validation no-drone windows. Reference networks (`--reference`) are scored on the same
windows; they never trained on this simulator.

Nothing here reads ITU ARIS, FSD50K eval, the war track or the scenes; every training reader calls
`forbid_in_training` (dronesynth). Writes artifacts/runs/<name>.json and artifacts/models/<name>_<arch>.pt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dad import dronesynth as ds  # noqa: E402
from dad import realrun as rr  # noqa: E402
from dad.config import DEFAULT  # noqa: E402

VAL_DRONES = (6, 12, 18, 24)      # one in six, fixed before any number (entry 239)
VAL_SEED = 25_000
BEDS = ("background", "wind", "war")    # war is test 1's bed, so it is a bed here too, not only a confuser
SNR_BINS = ((0.0, 10.0), (-10.0, 0.0), (-20.0, -10.0), (-30.0, -20.0))   # the last only in the faint probe
IR_FILE = REPO / "Microphone_Impulse_response/ir_measurements/20261123_120401/IR_mic_in3.wav"   # the array room, 2026-11-23
FAINT_SEED, FAINT_SNR = 31_000, (-30.0, -20.0)   # iteration 31's faint probe: test 1's quietest drone is ~-30 dB


def sources(split: str, glasgow_ids, uav16: bool) -> tuple[list, list]:
    """(drones, twins) for a split: Glasgow, plus with `uav16` its closest held/train hovers and plain noise."""
    drones, twins = ds.glasgow_drones(glasgow_ids), [ds.glasgow_noise(split)]
    if uav16:
        drones, twins = drones + ds.uav16_drones(split), twins + [ds.uav16_noise(split)]
    return drones, twins


def validation_set(n_families: int, uav16: bool = False, p_quiet: float = 0.0, p_vehicle: float = 0.0) -> dict:
    drones, twins = sources("val", VAL_DRONES, uav16)
    s = ds.SynthSampler([], seed=VAL_SEED, drones=drones, twin=twins, ladder=3, stem_split="val", bed_groups=BEDS,
                        p_quiet=p_quiet, p_vehicle=p_vehicle)
    print(f"validation: {s.describe()}", flush=True)
    rng = np.random.default_rng(VAL_SEED)
    w = [m for _ in range(n_families) for m in s.family(rng)]
    return dict(audio=np.stack([a for a, *_ in w]).astype(np.float32), y=np.array([t for _, t, *_ in w]),
                snr=np.array([v for *_, v, _ in w]), bed=np.array([i["bed"] for *_, i in w]),
                confuser=np.array([i["confuser"] for *_, i in w]),
                twin=np.array([bool(i.get("twin")) for *_, i in w]))


def faint_probe(n_families: int, p_quiet: float = 0.0, p_vehicle: float = 0.0) -> dict:
    """Iteration 31: validation families with the drone at FAINT_SNR, a readout only (never the threshold).
    Validation's sampler (held-out drones, val stems, twins) with another seed and the drone range moved down."""
    drones, twins = sources("val", VAL_DRONES, False)
    s = ds.SynthSampler([], seed=FAINT_SEED, drones=drones, twin=twins, ladder=3, stem_split="val", bed_groups=BEDS,
                        p_quiet=p_quiet, p_vehicle=p_vehicle, snr_db=FAINT_SNR)
    print(f"faint probe: {s.describe()}", flush=True)
    rng = np.random.default_rng(FAINT_SEED)
    w = [m for _ in range(n_families) for m in s.family(rng)]
    return dict(audio=np.stack([a for a, *_ in w]).astype(np.float32), y=np.array([t for _, t, *_ in w]),
                snr=np.array([v for *_, v, _ in w]), bed=np.array([i["bed"] for *_, i in w]),
                confuser=np.array([i["confuser"] for *_, i in w]),
                twin=np.array([bool(i.get("twin")) for *_, i in w]))


PROBE_SEED = 29_000    # iteration 29's steady probe: disjoint from VAL_SEED, same held-out drones and val stems


def steady_probe(n_families: int) -> dict:
    """Iteration 29 (entry 257): validation families over steady scenes, a readout only (never the threshold).
    Half the beds are a steady noise (coloured noise or synthetic wind, dronesynth.steady); every confuser
    is one. No chain twins, so the steady sound is the only thing that differs from a plain scene. The other
    beds are the val stems, so real steady FSD50K sounds (rain, stream, insects, wind) are read out too."""
    drones, _ = sources("val", VAL_DRONES, False)
    s = ds.SynthSampler([], seed=PROBE_SEED, drones=drones, twin=None, ladder=3, stem_split="val", bed_groups=BEDS,
                        p_steady=0.5, p_steady_confuser=1.0)
    print(f"steady probe: {s.describe()}", flush=True)
    rng = np.random.default_rng(PROBE_SEED)
    w = [m for _ in range(n_families) for m in s.family(rng)]
    return dict(audio=np.stack([a for a, *_ in w]).astype(np.float32), y=np.array([t for _, t, *_ in w]),
                snr=np.array([v for *_, v, _ in w]), bed=np.array([i["bed"] for *_, i in w]),
                confuser=np.array([i["confuser"] for *_, i in w]),
                twin=np.zeros(len(w), dtype=bool))


NIGHTJAR_WINDOWS = 20_000   # iteration 32: a second real readout, at most this many 1 s windows, drawn with NIGHTJAR_SEED
NIGHTJAR_SEED = 32_000


def nightjar_real() -> dict | None:
    """Real Nightjar field audio as 1 s windows (no overlap), relabelled by ear (labels/nightjar_relabel.csv, CLAUDE.md
    section 6), without the lawn-mower session. Never trained on in this simulator. None if the store is not here."""
    from dad import chains as ch
    from dad import rangerun as rg
    from dad import realdata
    if not (realdata.PROCESSED / "manifest.csv").exists():
        return None
    m = realdata.relabel(realdata.load_manifest(), REPO / "labels" / "nightjar_relabel.csv")
    recs = [r for r in m if r.source == "nightjar" and ch.session(r) not in rg.CONFUSER]
    n = DEFAULT.samples_per_window
    cw = ch.chain_windows(recs, n, n)
    keep = np.sort(np.random.default_rng(NIGHTJAR_SEED).permutation(len(cw))[:NIGHTJAR_WINDOWS])
    cw = cw.subset(keep)
    return dict(audio=np.stack([cw.audio(i, n) for i in range(len(cw))]).astype(np.float32), y=np.asarray(cw.y),
                range_m=np.nan_to_num(np.asarray(cw.range_m, float), nan=1.0))   # unranged drones count as near


def uav16_real() -> dict:
    """Real, unsimulated UAV16 audio as 1 s windows (no overlap): every hover of the two held batches
    (5-70 m) as drone, and the 24 backgrounds that are not plain environment (fans, exhausts, a car,
    other places; never trained on) as no drone. Same array as the uav16 training drones, so it
    favours a network that trained on them."""
    n, win, y, rng_m, rid = DEFAULT.samples_per_window, [], [], [], []
    for r in ds._u16_rows(ds.UAV16):
        drone = r["has_uav"] == "true"
        if (drone and r["acquisition_batch_id"] not in ds.U16_HELD) or r["recording_id"] in ds.U16_PLAIN:
            continue
        x = ds._u16_audio(r, ds.UAV16).astype(np.float32) / 32768
        k = len(x) // n
        win.append(x[:k * n].reshape(k, n))
        y += [int(drone)] * k
        rng_m += [float(np.hypot(float(r["horizontal_distance_m"]), float(r["height_m"]))) if drone else np.nan] * k
    return dict(audio=np.concatenate(win), y=np.array(y), range_m=np.array(rng_m))


def real_readout(s: np.ndarray, u: dict, thr: float) -> dict:
    pos, neg = u["y"] == 1, u["y"] == 0
    r = dict(auc=rr._auc(s[pos], s[neg]), recall_at_thr=float((s[pos] >= thr).mean()),
             fa_at_thr=float((s[neg] >= thr).mean()), conf_drone=float(s[pos].mean()),
             conf_no_drone=float(s[neg].mean()))
    for lo, hi in ((0, 20), (20, 40), (40, 80)):
        m = pos & (u["range_m"] >= lo) & (u["range_m"] < hi)
        r[f"recall_range_{lo}_{hi}"] = float((s[m] >= thr).mean())
    return r


def readout(s: np.ndarray, v: dict, thr: float) -> dict:
    pos, neg = v["y"] == 1, v["y"] == 0
    r = dict(threshold=thr, auc=rr._auc(s[pos], s[neg]),
             recall_at_thr=float((s[pos] >= thr).mean()), fa_at_thr=float((s[neg] >= thr).mean()),
             conf_no_drone=float(s[neg].mean()),
             conf_no_drone_plain=float(s[neg & ~v["twin"]].mean()),
             conf_no_drone_twin=float(s[neg & v["twin"]].mean()),
             fa_twin_at_thr=float((s[neg & v["twin"]] >= thr).mean()),
             auc_drone_vs_twin=rr._auc(s[pos], s[neg & v["twin"]]))
    empty = neg & (v["bed"] == "silence") & (v["confuser"] == "none") & ~v["twin"]
    if empty.any():   # exact zeros, as the Complete Test GUI's empty layers (entry 246)
        r["conf_empty_silence"] = float(s[empty].mean())
    for lo, hi in SNR_BINS:
        m = pos & (v["snr"] >= lo) & (v["snr"] < hi)
        if m.any():
            r[f"conf_drone_snr_{lo:g}_{hi:g}"] = float(s[m].mean())
            r[f"recall_snr_{lo:g}_{hi:g}"] = float((s[m] >= thr).mean())
    for key in ("bed", "confuser"):
        for g in np.unique(v[key]):
            m = v[key] == g
            r[f"{key}_{g}"] = dict(n_pos=int((m & pos).sum()), n_neg=int((m & neg).sum()),
                                   auc=rr._auc(s[m & pos], s[m & neg]),
                                   conf_drone=float(s[m & pos].mean()), conf_no_drone=float(s[m & neg].mean()),
                                   fa_at_thr=float((s[m & neg] >= thr).mean()))
    return r


def show(tag: str, r: dict) -> None:
    print(f"  {tag:<28} AUC {r['auc']:.3f}  @thr {r['threshold']:.3f}: recall {r['recall_at_thr']:.3f} "
          f"FA {r['fa_at_thr']:.3f} (twin {r['fa_twin_at_thr']:.3f})  conf no drone {r['conf_no_drone']:.3f}  "
          f"drone by SNR " + " / ".join(f"{r[f'conf_drone_snr_{lo:g}_{hi:g}']:.3f}" for lo, hi in SNR_BINS
                                       if f"conf_drone_snr_{lo:g}_{hi:g}" in r)
          + "  war bed AUC " + (f"{r['bed_war']['auc']:.3f}" if "bed_war" in r else "-")
          + "  war confuser AUC " + (f"{r['confuser_war']['auc']:.3f}" if "confuser_war" in r else "-")
          + "".join(f"  {g} conf no drone {r[f'bed_{g}']['conf_no_drone']:.3f} FA {r[f'bed_{g}']['fa_at_thr']:.3f}"
                    for g in ("quiet", "silence") if f"bed_{g}" in r)
          + (f"  vehicle conf no drone {r['confuser_vehicle']['conf_no_drone']:.3f} FA {r['confuser_vehicle']['fa_at_thr']:.3f}"
             if "confuser_vehicle" in r else "")
          + (f"  empty silence conf {r['conf_empty_silence']:.3f}" if "conf_empty_silence" in r else ""), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--models", nargs="+", default=["comb", "lgbm", "crnn"])
    p.add_argument("--device", default="mps")
    p.add_argument("--windows", type=int, default=200_000)
    p.add_argument("--lgbm-windows", type=int, default=40_000)
    p.add_argument("--val-families", type=int, default=1500)
    p.add_argument("--deploy-fa", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--uav16", action="store_true",
                   help="iteration 26: UAV16's closest hovers as drones and its plain noise as twins (both splits)")
    p.add_argument("--p-vehicle", type=float, default=0.0,
                   help="iteration 28: share of confusers that are a car, truck or motorcycle driving past the microphone")
    p.add_argument("--p-quiet", type=float, default=0.0,
                   help="iteration 27: share of scenes that are near-empty (silence or a faint floor, often one confuser alone)")
    p.add_argument("--p-steady", type=float, default=0.0,
                   help="iteration 29: share of scenes whose bed is a steady noise; a quarter of confusers are then steady too")
    p.add_argument("--ladder", type=int, default=3,
                   help="training families of 1 scene + LADDER drone copies (3: 75%% drone windows; 1: 50%%, iteration 30). "
                        "Validation stays ladder 3")
    p.add_argument("--p-ir", type=float, default=0.0,
                   help="iteration 30.5: share of training windows heard through the measured array-room IR (IR_FILE); "
                        "> 0 also reads out validation through the IR")
    p.add_argument("--p-room", type=float, default=0.0,
                   help="iteration 32: share of families with random paths on drone, confuser and twin and a real floor")
    p.add_argument("--snr-lo", type=float, default=-20.0,
                   help="iteration 31: training drones from SNR-LO to +10 dB (default -20, as before). Validation unchanged")
    p.add_argument("--faint-families", type=int, default=0,
                   help="iteration 31: families in the faint probe, drone at -30 to -20 dB (0: none)")
    p.add_argument("--probe-families", type=int, default=500, help="iteration 29: families in the steady probe (0: none)")
    p.add_argument("--reference", nargs="*", default=[], metavar="CKPT",
                   help="checkpoints under artifacts/models scored on the same validation windows")
    args = p.parse_args()

    out = REPO / "artifacts" / "runs" / f"{args.name}.json"
    (REPO / "artifacts" / "models").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    val = validation_set(args.val_families, args.uav16, args.p_quiet, args.p_vehicle)
    probe = steady_probe(args.probe_families) if args.probe_families else None
    ir = ds.load_ir(IR_FILE) if args.p_ir else None
    val_ir = dict(val, audio=np.stack([ds.apply_ir(a, ir) for a in val["audio"]])) if ir is not None else None
    faint = faint_probe(args.faint_families, args.p_quiet, args.p_vehicle) if args.faint_families else None
    u16 = uav16_real() if ds.UAV16.exists() else None
    nj = nightjar_real()
    if nj is not None:
        print(f"nightjar real: {int(nj['y'].sum())} drone, {int((nj['y'] == 0).sum())} no-drone windows", flush=True)
    if u16 is not None:
        print(f"uav16 real: {int(u16['y'].sum())} drone, {int((u16['y'] == 0).sum())} no-drone windows", flush=True)
    neg = val["y"] == 0
    print(f"validation windows: {len(val['y'])} ({int(neg.sum())} no drone, {int(neg.sum() - val['twin'].sum())} "
          f"plain, {int(val['twin'].sum())} twin) in {time.time() - t0:.0f} s", flush=True)
    train_ids = [d for d in range(1, 25) if d not in VAL_DRONES]
    drones, twins = sources("train", train_ids, args.uav16)
    sampler = ds.SynthSampler([], seed=args.seed, drones=drones, twin=twins, ladder=args.ladder, stem_split="train",
                              bed_groups=BEDS, p_quiet=args.p_quiet, p_vehicle=args.p_vehicle, p_steady=args.p_steady,
                              snr_db=(args.snr_lo, 10.0), ir=ir, p_ir=args.p_ir, p_room=args.p_room)
    print(f"training: {sampler.describe()}", flush=True)
    results = []

    def evaluate(scorer, tag: str, extra: dict) -> None:
        s = scorer.score(val["audio"])
        thr = float(np.quantile(s[neg], 1.0 - args.deploy_fa))
        r = dict(scorer=tag, seed=args.seed, window_s=DEFAULT.window_seconds, deploy_threshold=thr,
                 val=readout(s, val, thr), **extra)
        if u16 is not None:
            r["uav16_real"] = real_readout(scorer.score(u16["audio"]), u16, thr)
        if nj is not None:
            r["nightjar_real"] = real_readout(scorer.score(nj["audio"]), nj, thr)
        if probe is not None:
            r["steady_probe"] = readout(scorer.score(probe["audio"]), probe, thr)
        if val_ir is not None:   # the same windows and threshold, heard through the array room
            r["val_ir"] = readout(scorer.score(val_ir["audio"]), val_ir, thr)
        if faint is not None:
            r["faint_probe"] = readout(scorer.score(faint["audio"]), faint, thr)
        results.append(r)
        show(tag, r["val"])
        if val_ir is not None:
            show("  through the room IR", r["val_ir"])
        if faint is not None:
            q = r["faint_probe"]
            print(f"  {'':<28} faint probe (-30 to -20 dB) @thr: drone conf {q['conf_drone_snr_-30_-20']:.3f} "
                  f"recall {q['recall_snr_-30_-20']:.3f}  AUC {q['auc']:.3f}  no drone {q['conf_no_drone']:.3f}", flush=True)
        if probe is not None:
            q = r["steady_probe"]
            print(f"  {'':<28} steady probe @thr: " + "  ".join(
                f"{k} no drone {q[k]['conf_no_drone']:.3f} FA {q[k]['fa_at_thr']:.3f}"
                for k in sorted(q) if k.startswith(("bed_", "confuser_")) and k != "confuser_none")
                + f"  drone by SNR " + " / ".join(f"{q[f'conf_drone_snr_{lo:g}_{hi:g}']:.3f}" for lo, hi in SNR_BINS
                                                 if f"conf_drone_snr_{lo:g}_{hi:g}" in q), flush=True)
        if u16 is not None:
            q = r["uav16_real"]
            print(f"  {'':<28} uav16 real: AUC {q['auc']:.3f} recall {q['recall_at_thr']:.3f} (<20 m "
                  f"{q['recall_range_0_20']:.3f}, 20-40 {q['recall_range_20_40']:.3f}, 40-80 {q['recall_range_40_80']:.3f})"
                  f" FA {q['fa_at_thr']:.3f}", flush=True)
        if nj is not None:
            q = r["nightjar_real"]
            print(f"  {'':<28} nightjar real: AUC {q['auc']:.3f} recall {q['recall_at_thr']:.3f} FA {q['fa_at_thr']:.3f} "
                  f"conf drone {q['conf_drone']:.3f} no drone {q['conf_no_drone']:.3f}", flush=True)
        out.write_text(json.dumps(dict(args=vars(args), val_drones=VAL_DRONES, train_drones=train_ids,
                                       results=results), indent=2))

    if "comb" in args.models:
        evaluate(rr.CombScorer(DEFAULT), "comb (no ML)", {})
    for ck in args.reference:
        net = rr.NetScorer.load(REPO / "artifacts" / "models" / ck, DEFAULT, args.device)
        evaluate(net, f"reference {ck}", dict(loaded=ck))
    variant = rr.VARIANTS["buried"]
    if "lgbm" in args.models:
        t = time.time()
        m = rr.LgbmScorer(variant, DEFAULT).fit(sampler, args.lgbm_windows)
        m.save(REPO / "artifacts" / "models" / f"{args.name}_lgbm.pkl")
        evaluate(m, "lgbm", dict(train_seconds=round(time.time() - t, 1), windows=args.lgbm_windows))
    for arch in (a for a in args.models if a in ("dscnn", "crnn")):
        net = rr.NetScorer(arch, variant, DEFAULT, args.device)
        t = time.time()
        net.fit(sampler, n_windows=args.windows, workers=args.workers)
        net.save(REPO / "artifacts" / "models" / f"{args.name}_{arch}.pt")
        evaluate(net, arch, dict(train_seconds=round(time.time() - t, 1), windows=args.windows, history=net.history))
    print(f"done in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
