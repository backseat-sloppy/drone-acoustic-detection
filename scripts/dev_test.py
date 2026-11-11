"""A development copy of test 1 (REQUIREMENTS.md), without ITU ARIS: report entries 206-207.

    uv run python scripts/dev_test.py run  TAG --half tune --models "it9 CRNN" "it16 CRNN 600k s0" ...
    uv run python scripts/dev_test.py judge TAG --ref "it16 CRNN 600k"

Drones: Lund (the 10 s nearest the array in each recording) and Halmstad drone clips; no model in
drone_war_grid.MODELS trained on either. Split by file into a tune and a confirm half: Lund's nearest
stretch against its second nearest, Halmstad's even against odd clips.
Backgrounds, with complete_test_engine's own layers and levels: the war soundscape at 90 dB SPL, alone and
with wind (Pd, 10 m/s, 50 dB SPL) and cars (Pd, 50 km/h, 30 m, 50 dB SPL); wind alone and cars alone without a
drone. Six 10 s stretches: war track starts that no logged test run used, each paired with one car's pass-by.
Every drone is mixed over the whole 10 s at 100, 90, 80, 70 and 60 dB SPL, into two of the stretches; the
same stretch without the drone is the no-drone round, so the two are paired. A Halmstad helicopter in the
drone's place (80 dB SPL, alone and in war) is a no-drone round too.
`run` keeps every hop's score; `judge` prints mean confidence, the AUC of drone-in-war hops against no-drone
hops, and the same after a pin: one constant on each model's logit so its drone confidence in war equals
the reference's. Development figures, never detection performance on ITU ARIS.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import complete_test_engine as E  # noqa: E402
import drone_war_grid as dwg  # noqa: E402

from dad.realdata import load_manifest  # noqa: E402
from dad.wavio import Clip, stream  # noqa: E402

OUT = dwg.ROOT / "artifacts/dev_test"
SR, N = E.SR, int(10 * E.SR)
WAR_DB, WIND_DB, CARS_DB, HELI_DB = 90.0, 50.0, 50.0, 80.0
DRONE_DBS = (100.0, 90.0, 80.0, 70.0, 60.0)
# Logged runs used 0-20, 30-50, 114-154, 150-162, 401-429, 473-493 and 540-560 s of the war track.
WAR_STARTS = (60.0, 180.0, 250.0, 320.0, 360.0, 450.0)
PD = dict(wind_speed_ms=10.0, car_speed_kmh=50.0, car_dist_m=30.0)
WAR_BGS = (("war",), ("war", "wind"), ("war", "cars"), ("war", "wind", "cars"))
NO_DRONE_BGS = WAR_BGS + (("wind",), ("cars",))
LUND_SUSPECT = "10_flower3"
ALLOW_TYRES_ONLY = False    # --tyres-only: run although the engine recordings are missing (the cars are then tyre noise)


def picks(half: str) -> dict[str, list]:
    """{"drone": [(name, audio)], "helicopter": [...]} of one half, 10 s each at 16 kHz."""
    man = load_manifest()
    track = collections.defaultdict(list)
    with open(dwg.ROOT / "data/processed/range_track.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["source"] == "lund":
                track[r["recording_id"]].append(float(r["range_m"]))
    out = {"drone": [], "helicopter": []}
    for r in sorted((r for r in man if r.source == "lund" and r.label == 1), key=lambda r: r.recording_id):
        if LUND_SUSPECT in r.recording_id:
            continue
        rng = np.array(track[r.recording_id])                       # 10 Hz, from the recording's start
        mean = np.convolve(rng, np.ones(100) / 100, "valid")        # mean range of every 10 s
        ok = np.arange(len(mean)) * SR // 10 + N <= r.n_samples
        mean = np.where(ok, mean, np.inf)
        a = int(np.argmin(mean))
        far = np.where(np.abs(np.arange(len(mean)) - a) >= 200, mean, np.inf)   # 20 s away or more
        i = a if half == "tune" else int(np.argmin(far))
        out["drone"].append((f"lund {r.recording_id} {i / 10:.0f}s {mean[i]:.0f}m", r.audio(i * SR // 10, N)))
    for kind in ("drone", "helicopter"):
        clips = sorted((r for r in man if r.source == "halmstad" and r.kind == kind), key=lambda r: r.recording_id)
        for r in clips[0 if half == "tune" else 1::2]:
            out[kind].append((f"halmstad {r.recording_id}", r.audio(0, min(N, r.n_samples))))
    return out


def backgrounds() -> dict:
    """{(stretch, layers): 10 s of background}, layers a tuple out of war, wind, cars."""
    missing = [p.name for p in E._car_engines() if not p.exists()]
    kept = E._cars_file(E.pd_config("cars", PD)).exists()           # a render made where the engines were
    if missing and not kept and not ALLOW_TYRES_ONLY:               # Pd renders silently without them: tyre noise only
        raise SystemExit(f"car engines missing ({', '.join(missing)}): run scripts/fetch_car_engines.py")
    layers = E.make_layers()
    war = layers["war soundscape"]
    wind = E.Layer("wind (Pd)", "", True, lambda: E._wind_pd(E.pd_config("wind (Pd)", PD)))
    cars = E.Layer("cars", "", True, lambda: E._cars(E.pd_config("cars", PD)))
    per_car = len(cars.signal()[0]) / SR / len(E.car_names())       # one pass-by per engine, closest halfway
    out = {}
    for s, w0 in enumerate(WAR_STARTS):
        part = dict(war=war.window(N, w0, WAR_DB), wind=wind.window(N, 7.0 * s, WIND_DB),
                    cars=cars.window(N, (s + 0.5) * per_car - 5.0, CARS_DB))
        for bg in NO_DRONE_BGS:
            out[s, bg] = np.sum([part[k] for k in bg], axis=0)
    return out


def at_level(x: np.ndarray, db: float) -> np.ndarray:
    x = np.pad(x.astype(np.float64), (0, N - len(x)))
    return x * (E.rms_for(db) / (float(np.sqrt(np.mean(x ** 2))) or 1.0))


def rounds(half: str) -> list[tuple[dict, np.ndarray]]:
    """Every round as (what it is, its 10 s mix)."""
    p, bgs = picks(half), backgrounds()
    out = [(dict(kind="none", stretch=s, bg="+".join(bg)), x) for (s, bg), x in bgs.items()]
    for j, (name, x) in enumerate(p["drone"]):
        for s in (j % len(WAR_STARTS), (j + 3) % len(WAR_STARTS)):
            for bg in WAR_BGS:
                for db in DRONE_DBS:
                    d = at_level(x, db)
                    snr = 10 * np.log10(np.mean(d ** 2) / np.mean(bgs[s, bg] ** 2))
                    out.append((dict(kind="drone", file=name, stretch=s, bg="+".join(bg), drone_db=db, snr_db=round(float(snr), 1)),
                                d + bgs[s, bg]))
    for j, (name, x) in enumerate(p["helicopter"]):
        h, s = at_level(x, HELI_DB), j % len(WAR_STARTS)
        out.append((dict(kind="helicopter", file=name, stretch=s, bg="alone"), h))
        out.append((dict(kind="helicopter", file=name, stretch=s, bg="war"), h + bgs[s, ("war",)]))
    return out


def run(a) -> None:
    rs = rounds(a.half)
    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "rounds.json").write_text(json.dumps(dict(half=a.half, rounds=[r for r, _ in rs]), indent=0), encoding="utf-8")
    thr, scorers = E.load_detectors(a.models, a.device)
    for m in a.models:
        sc = scorers[m]
        s = [stream(Clip("r", x.astype(np.float32), SR, 1, "FLOAT"), sc, sc.cfg, thr[m]).score for _, x in rs]
        s = [dwg.squash(v, thr[m]) for v in s] if m == "comb" else s
        np.savez_compressed(out / f"{m}.npz", *s)
        print(f"{m}: {len(rs)} rounds, {sum(map(len, s))} hops", flush=True)


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _auc(pos, neg) -> float:
    r = np.argsort(np.argsort(np.concatenate([pos, neg]), kind="stable"), kind="stable") + 1.0
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def readout(rs: list[dict], s: list[np.ndarray], shift: float = 0.0) -> dict:
    """Mean confidence (%) per condition, each round counted equally; AUC over hops."""
    conf = [1 / (1 + np.exp(-(_logit(v) + shift))) for v in s]
    mean = lambda sel: 100 * float(np.mean([conf[i].mean() for i, r in enumerate(rs) if sel(r)]))   # noqa: E731
    hops = lambda sel: np.concatenate([conf[i] for i, r in enumerate(rs) if sel(r)])                # noqa: E731
    o = {"drone war": mean(lambda r: r["kind"] == "drone" and r["bg"] == "war"),
         "drone war low": mean(lambda r: r["kind"] == "drone" and r["bg"] == "war" and r["drone_db"] <= 70),
         "drone lund": mean(lambda r: r["kind"] == "drone" and r["bg"] == "war" and r["file"].startswith("lund")),
         "drone halmstad": mean(lambda r: r["kind"] == "drone" and r["bg"] == "war" and r["file"].startswith("halmstad")),
         "drone all": mean(lambda r: r["kind"] == "drone")}
    for bg in ("war", "wind", "cars", "war+wind", "war+cars", "war+wind+cars"):
        o[f"none {bg}"] = mean(lambda r, bg=bg: r["kind"] == "none" and r["bg"] == bg)
    o["none with cars"] = mean(lambda r: r["kind"] == "none" and "cars" in r["bg"])
    o["helicopter"] = mean(lambda r: r["kind"] == "helicopter")
    o["auc war"] = _auc(hops(lambda r: r["kind"] == "drone" and r["bg"] == "war"),
                        hops(lambda r: r["kind"] == "none" and r["bg"] == "war"))
    o["auc all"] = _auc(hops(lambda r: r["kind"] == "drone"), hops(lambda r: r["kind"] != "drone"))
    return o


def pin(rs, s, target: float) -> float:
    """The constant on the logit that brings drone confidence in war to `target` (bisection)."""
    lo, hi = -20.0, 20.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if readout_drone_war(rs, s, mid) < target else (lo, mid)
    return (lo + hi) / 2


def readout_drone_war(rs, s, shift) -> float:
    return 100 * float(np.mean([(1 / (1 + np.exp(-(_logit(v) + shift)))).mean()
                                for r, v in zip(rs, s) if r["kind"] == "drone" and r["bg"] == "war"]))


def judge(a) -> None:
    out = OUT / a.tag
    rs = json.loads((out / "rounds.json").read_text(encoding="utf-8"))["rounds"]
    res = {}
    for f in sorted(out.glob("*.npz")):
        z = np.load(f)
        res[f.stem] = [z[k] for k in z.files]
    raw = {m: readout(rs, s) for m, s in res.items()}
    cols = list(next(iter(raw.values())))
    fmt = lambda o: " ".join(f"{o[c]:6.3f}" if c.startswith("auc") else f"{o[c]:6.1f}" for c in cols)   # noqa: E731
    print("columns:", " | ".join(cols))
    print("\nunpinned")
    for m, o in raw.items():
        print(f"{m:24} {fmt(o)}")
    ref = [m for m in raw if m.startswith(a.ref)] if a.ref else []
    if ref:
        target = float(np.mean([raw[m]["drone war"] for m in ref]))
        print(f"\npinned: drone confidence in war = {target:.1f} (mean of {', '.join(ref)}); last column: the constant")
        pinned = {}
        for m, s in res.items():
            c = pin(rs, s, target)
            pinned[m] = readout(rs, s, c) | {"shift": c}
            print(f"{m:24} {fmt(pinned[m])} {c:+6.2f}")
        (out / "judge.json").write_text(json.dumps(dict(unpinned=raw, pinned=pinned, target=target), indent=1), encoding="utf-8")
    else:
        (out / "judge.json").write_text(json.dumps(dict(unpinned=raw), indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("tag")
    r.add_argument("--half", choices=("tune", "confirm"), default="tune")
    r.add_argument("--models", nargs="+", required=True)
    j = sub.add_parser("judge")
    j.add_argument("tag")
    j.add_argument("--ref", default="", help="model-name prefix whose mean drone confidence in war is the pin's target")
    r.add_argument("--device", default="cpu")
    r.add_argument("--tyres-only", action="store_true", help="run without the car engine recordings")
    a = ap.parse_args()
    ALLOW_TYRES_ONLY = getattr(a, "tyres_only", False)
    run(a) if a.cmd == "run" else judge(a)
