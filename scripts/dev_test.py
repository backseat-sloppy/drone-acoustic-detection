"""A development copy of test 1 (REQUIREMENTS.md), without ITU ARIS: report entries 206-207 and 243.

    uv run python scripts/dev_test.py run  TAG --half tune --models "it9 CRNN" "it16 CRNN 600k s0" ...
    uv run python scripts/dev_test.py judge TAG --ref "it16 CRNN 600k"

Drones: Lund (the 10 s nearest the array in each recording) and Halmstad drone clips; no model in
drone_war_grid.MODELS trained on either, but Lund recall has steered recipe choices (entry 243). Split by
file into a tune and a confirm half: Lund's nearest stretch against its second nearest, Halmstad's even
against odd clips. The halves share Lund's flights, so the confirm half is no independent test.
Backgrounds, with complete_test_engine's own layers and levels: the 600 s war track (pinned, entry 242) at
90 dB SPL, alone and with wind (Pd, 10 m/s, 50 dB SPL) and cars (Pd, 50 km/h, 30 m, 50 dB SPL); wind alone
and cars alone without a drone. Six 10 s stretches: war track starts that no logged test run used, each
paired with one car's pass-by; 36 more war-only stretches are no-drone rounds.
Every drone is mixed over the whole 10 s at 100, 90, 80, 70 and 60 dB SPL, into two of the six stretches;
the same stretch without the drone is the no-drone round, so the two are paired. A Halmstad helicopter in
the drone's place (80 dB SPL, alone and in war) is a no-drone round too, and so is a Halmstad background
clip mixed exactly like a drone: the rig's own ambience, a matched negative for Halmstad's drones.
A third drone source (entry 246): Mathias's recordings in data/recordedDroneClipped, mixed like the others,
split by recording (by file name) into the halves; a recording, not a clip, is the bootstrap's unit. Their
rounds come after the others, so `run --extend TAG` reuses an earlier tag's scores and scores only them.
`run` keeps every hop's score and the MD5 of the audio; `judge` uses only scores made on the same audio. It
prints Mads's criterion, the gap between mean confidence with a drone and without (in war; then with the
distractors), mean confidence per condition and the AUCS, the same after a pin (one constant on each model's
logit so its drone confidence in war equals the reference's), and a paired two-way cluster bootstrap of the
main figures. Development figures, never detection performance on ITU ARIS.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import complete_test_engine as E  # noqa: E402
import drone_war_grid as dwg  # noqa: E402

from dad.config import DEFAULT  # noqa: E402
from dad.realdata import load_manifest  # noqa: E402
from dad.wavio import Clip, read_wav, stream  # noqa: E402

OUT = dwg.ROOT / "artifacts/dev_test"
SR, N = E.SR, int(10 * E.SR)
WAR_DB, WIND_DB, CARS_DB, HELI_DB = 90.0, 50.0, 50.0, 80.0
DRONE_DBS = (100.0, 90.0, 80.0, 70.0, 60.0)
# Logged test runs used these stretches of the war track (s); the dev copy keeps clear of them.
LOGGED = ((0, 20), (30, 50), (114, 154), (150, 162), (401, 429), (473, 493), (540, 560))
WAR_STARTS = (60.0, 180.0, 250.0, 320.0, 360.0, 450.0)
# War-only no-drone stretches (entry 243): every other free 10 s slot of the 600 s track, so that the
# no-drone side of AUC war has 42 stretches, not 6. They are left out of AUC all, which keeps entry 207's set.
EXTRA_STARTS = tuple(float(t) for t in range(0, 600, 10) if all(
    t + 10 <= a or t >= b for a, b in LOGGED + tuple((w, w + 10) for w in WAR_STARTS)))
# The 600 s track WAR_STARTS were chosen on. Since 2026-11-11 the engine's default is the 38.5 s
# WarSoundScape.wav, where these starts wrap into test 1's own excerpt (entry 242), so it is pinned here.
DEV_WAR = E.WAR_TRACK
PD = dict(wind_speed_ms=10.0, car_speed_kmh=50.0, car_dist_m=30.0)
WAR_BGS = (("war",), ("war", "wind"), ("war", "cars"), ("war", "wind", "cars"))
NO_DRONE_BGS = WAR_BGS + (("wind",), ("cars",))
LUND_SUSPECT = "10_flower3"
RECORDED = dwg.ROOT / "data/recordedDroneClipped"     # Mathias's drone recordings, 10 s clips (entry 246)
SOURCES = ("lund", "halmstad", "recorded")            # drone sources, the first word of a drone round's file
FA = 0.05                   # judge's detection rate at this false-alarm rate: an oracle point of the ROC curve
DET = f"det at {FA:.0%} FA"
LOWER_BETTER = {"none with cars"}
# the lgbm baseline (CLAUDE.md 5), registered here only: Mads dropped it from the GUI lists on 2026-10-30
dwg.MODELS.setdefault("it6 lgbm", (dwg.RUN, dwg.MODEL_DIR / "it6_final_none_simdist_lgbm.pkl"))
# Networks versioned in git only on branch iteration-21-uav16 (entry 245), copied with `git cat-file` into this
# git-ignored folder (their run files into artifacts/runs/); registered here only, so the GUI lists stay as they are.
BRANCH_MODELS = dwg.ROOT / "artifacts/dev_test/models"
for _s in (0, 1, 2):
    for _name, _run, _tail in ((f"it24 CNN 2s war s{_s}", f"it24_2s_mad_s{_s}", "simdist_colour_mad_dscnn"),
                               (f"it24 CRNN 2s war s{_s}", f"it24_2s_mad_s{_s}", "simdist_colour_mad_crnn"),
                               (f"it23 CNN 2s colour s{_s}", f"it23_2s_colour_s{_s}", "simdist_colour_dscnn"),
                               (f"it23 CRNN 2s colour s{_s}", f"it23_2s_colour_s{_s}", "simdist_colour_crnn"),
                               (f"it21 CRNN uav16 s{_s}", f"it21_uav16_s{_s}", "simdist_colour_crnn")):
        if (BRANCH_MODELS / f"{_run}_none_{_tail}.pt").exists():
            dwg.MODELS[_name] = (dwg.ROOT / f"artifacts/runs/{_run}.json", BRANCH_MODELS / f"{_run}_none_{_tail}.pt")
ALLOW_TYRES_ONLY = False    # --tyres-only: run although the engine recordings are missing (the cars are then tyre noise)


def picks(half: str) -> dict[str, list]:
    """{"drone": [(name, audio)], "helicopter": [...], "background": [...], "recorded": [...]} of one half,
    10 s each at 16 kHz."""
    man = load_manifest()
    track = collections.defaultdict(list)
    with open(dwg.ROOT / "data/processed/range_track.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["source"] == "lund":
                track[r["recording_id"]].append(float(r["range_m"]))
    out = {"drone": [], "helicopter": [], "background": []}
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
    for kind in ("drone", "helicopter", "background"):
        clips = sorted((r for r in man if r.source == "halmstad" and r.kind == kind), key=lambda r: r.recording_id)
        for r in clips[0 if half == "tune" else 1::2]:
            out[kind].append((f"halmstad {r.recording_id}", r.audio(0, min(N, r.n_samples))))
    clips = sorted(RECORDED.glob("*.wav"))
    takes = sorted({c.stem.rsplit("_", 1)[0] for c in clips})[0 if half == "tune" else 1::2]   # by recording
    out["recorded"] = [(f"recorded {c.stem}", read_wav(c, DEFAULT).audio) for c in clips
                       if c.stem.rsplit("_", 1)[0] in takes]   # as the complete test reads a drone file
    return out


def backgrounds() -> dict:
    """{(stretch, layers): 10 s of background}, layers a tuple out of war, wind, cars."""
    missing = [p.name for p in E._car_engines() if not p.exists()]
    kept = E._cars_file(E.pd_config("cars", PD)).exists()           # a render made where the engines were
    if missing and not kept and not ALLOW_TYRES_ONLY:               # Pd renders silently without them: tyre noise only
        raise SystemExit(f"car engines missing ({', '.join(missing)}): run scripts/fetch_car_engines.py")
    war = E.make_layers(str(DEV_WAR))["war soundscape"]
    wind = E.Layer("wind (Pd)", "", True, lambda: E._wind_pd(E.pd_config("wind (Pd)", PD)))
    cars = E.Layer("cars", "", True, lambda: E._cars(E.pd_config("cars", PD)))
    per_car = len(cars.signal()[0]) / SR / len(E.car_names())       # one pass-by per engine, closest halfway
    out = {}
    for s, w0 in enumerate(WAR_STARTS):
        part = dict(war=war.window(N, w0, WAR_DB), wind=wind.window(N, 7.0 * s, WIND_DB),
                    cars=cars.window(N, (s + 0.5) * per_car - 5.0, CARS_DB))
        for bg in NO_DRONE_BGS:
            out[s, bg] = np.sum([part[k] for k in bg], axis=0)
    for s, w0 in enumerate(EXTRA_STARTS, len(WAR_STARTS)):
        out[s, ("war",)] = war.window(N, w0, WAR_DB)
    return out


def core(r: dict) -> bool:
    """A round on one of the six stretches of entry 207 (every round but the extra war-only ones)."""
    return r["stretch"] < len(WAR_STARTS)


def _dw(r: dict) -> bool:
    return r["kind"] == "drone" and r["bg"] == "war"


def _none_war(r: dict) -> bool:
    return r["kind"] == "none" and r["bg"] == "war"


# Each AUC over hops: (positive rounds, negative rounds). "auc rig" sets Halmstad's drones against Halmstad's
# background clips mixed the same way; AUC all keeps entry 207's no-drone rounds (no extra or rig rounds).
AUCS = {"auc war": (_dw, _none_war),
        **{f"auc war {src}": (lambda r, src=src: _dw(r) and r["file"].startswith(src), _none_war) for src in SOURCES},
        "auc rig": (lambda r: _dw(r) and r["file"].startswith("halmstad"), lambda r: r["kind"] == "rig"),
        "auc all": (lambda r: r["kind"] == "drone", lambda r: r["kind"] in ("none", "helicopter") and core(r))}


def at_level(x: np.ndarray, db: float) -> np.ndarray:
    x = x.astype(np.float64)[:N]
    x = np.pad(x, (0, N - len(x)))
    return x * (E.rms_for(db) / (float(np.sqrt(np.mean(x ** 2))) or 1.0))


def rounds(half: str) -> list[tuple[dict, np.ndarray]]:
    """Every round as (what it is, its 10 s mix in float32, as scored and hashed)."""
    p, bgs = picks(half), backgrounds()
    out = []
    add = lambda r, x: out.append((r, x.astype(np.float32)))   # noqa: E731

    def drones(clips, unit=None):
        for j, (name, x) in enumerate(clips):
            u = {} if unit is None else dict(unit=unit(name))
            for s in (j % len(WAR_STARTS), (j + 3) % len(WAR_STARTS)):
                for bg in WAR_BGS:
                    for db in DRONE_DBS:
                        d = at_level(x, db)
                        snr = 10 * np.log10(np.mean(d ** 2) / np.mean(bgs[s, bg] ** 2))
                        add(dict(kind="drone", file=name, stretch=s, bg="+".join(bg), drone_db=db,
                                 snr_db=round(float(snr), 1)) | u, d + bgs[s, bg])

    for (s, bg), x in bgs.items():
        add(dict(kind="none", stretch=s, bg="+".join(bg)), x)
    drones(p["drone"])
    for j, (name, x) in enumerate(p["helicopter"]):
        h, s = at_level(x, HELI_DB), j % len(WAR_STARTS)
        add(dict(kind="helicopter", file=name, stretch=s, bg="alone"), h)
        add(dict(kind="helicopter", file=name, stretch=s, bg="war"), h + bgs[s, ("war",)])
    for j, (name, x) in enumerate(p["background"]):         # mixed exactly as a drone in war alone
        for s in (j % len(WAR_STARTS), (j + 3) % len(WAR_STARTS)):
            for db in DRONE_DBS:
                add(dict(kind="rig", file=name, stretch=s, bg="war", drone_db=db), at_level(x, db) + bgs[s, ("war",)])
    drones(p["recorded"], unit=lambda name: name.rsplit("_", 1)[0])   # last, so entry 244's rounds come first
    return out


def audio_md5(rs) -> str:
    """One MD5 over every round's mix as scored: `judge` only compares score files made on the same audio."""
    h = hashlib.md5()
    for _, x in rs:
        h.update(x.astype(np.float32).tobytes())
    return h.hexdigest()


def scored_on(tag_dir: Path, m: str, md5: str) -> bool:
    """Whether model m's score file in tag_dir was made on the audio with this MD5."""
    side = tag_dir / f"{m}.json"
    return (tag_dir / f"{m}.npz").exists() and side.exists() and \
        json.loads(side.read_text(encoding="utf-8")).get("audio_md5") == md5


def run(a) -> None:
    rs = rounds(a.half)
    meta = dict(half=a.half, war_file=DEV_WAR.relative_to(dwg.ROOT).as_posix(), audio_md5=audio_md5(rs))
    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "rounds.json").write_text(json.dumps(meta | dict(rounds=[r for r, _ in rs]), indent=0), encoding="utf-8")
    k = 0
    if a.extend:                  # an earlier tag whose rounds are this tag's first k: its scores are reused
        base = OUT / a.extend
        b = json.loads((base / "rounds.json").read_text(encoding="utf-8"))
        k = len(b["rounds"])
        if b["rounds"] != [r for r, _ in rs[:k]] or b.get("audio_md5") != audio_md5(rs[:k]):
            raise SystemExit(f"{a.extend}'s rounds are not this tag's first {k} rounds")
        if missing := [m for m in a.models if not scored_on(base, m, b["audio_md5"])]:
            raise SystemExit(f"not scored on {a.extend}'s audio: {', '.join(missing)}")
        rs = [(r, None) for r, _ in rs[:k]] + rs[k:]      # their audio is not needed again: the memory goes
    thr, scorers = E.load_detectors(a.models, a.device)
    for m in a.models:
        sc = scorers[m]
        z = np.load(OUT / a.extend / f"{m}.npz") if k else {}
        new = [stream(Clip("r", x, SR, 1, "FLOAT"), sc, sc.cfg, thr[m]).score for _, x in rs[k:]]
        s = [z[f"arr_{i}"] for i in range(k)] + ([dwg.squash(v, thr[m]) for v in new] if m == "comb" else new)
        np.savez_compressed(out / f"{m}.npz", *s)
        (out / f"{m}.json").write_text(json.dumps(meta), encoding="utf-8")
        print(f"{m}: {len(rs)} rounds ({k} from {a.extend or '-'}), {sum(map(len, s))} hops", flush=True)


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _wauc(ip, ineg, k: int, wp, wn) -> float:
    """AUC on a shared grid of k unique scores (ip, ineg: grid index of each positive and negative hop), each
    hop weighted: the chance a positive scores above a negative, ties counting half (Mann-Whitney)."""
    n_at = np.bincount(ineg, weights=wn, minlength=k)
    below = np.cumsum(n_at) - n_at
    with np.errstate(invalid="ignore", divide="ignore"):   # a resample that drew none of a set: nan
        return float(np.dot(wp, below[ip] + 0.5 * n_at[ip]) / (wp.sum() * wn.sum()))


def _wdet(ip, ineg, k: int, wp, wn, fa: float = FA) -> float:
    """Share of the positive weight at or above the lowest grid score that at most `fa` of the negative
    weight reaches: a point of the ROC curve, the detection rate at that false-alarm rate."""
    at_or_above = np.bincount(ineg, weights=wn, minlength=k)[::-1].cumsum()[::-1]
    ok = at_or_above <= fa * wn.sum()
    j = int(np.argmax(ok)) if ok.any() else k               # none: the threshold sits above every score
    return float(np.bincount(ip, weights=wp, minlength=k)[j:].sum() / wp.sum())


def _auc(pos, neg) -> float:
    u, inv = np.unique(np.concatenate([pos, neg]), return_inverse=True)
    return _wauc(inv[: len(pos)], inv[len(pos):], len(u), np.ones(len(pos)), np.ones(len(neg)))


def readout(rs: list[dict], s: list[np.ndarray], shift: float = 0.0) -> dict:
    """Mean confidence (%) per condition, each round counted equally; the AUCS over hops."""
    conf = [1 / (1 + np.exp(-(_logit(v) + shift))) for v in s]
    srcs = [src for src in SOURCES if any(_dw(r) and r["file"].startswith(src) for r in rs)]
    mean = lambda sel: 100 * float(np.mean([conf[i].mean() for i, r in enumerate(rs) if sel(r)]))   # noqa: E731
    hops = lambda sel: np.concatenate([conf[i] for i, r in enumerate(rs) if sel(r)] or [np.empty(0)])   # noqa: E731
    o = {"drone war": mean(lambda r: r["kind"] == "drone" and r["bg"] == "war"),
         "drone war low": mean(lambda r: r["kind"] == "drone" and r["bg"] == "war" and r["drone_db"] <= 70),
         **{f"drone {src}": mean(lambda r, src=src: _dw(r) and r["file"].startswith(src)) for src in srcs},
         "drone all": mean(lambda r: r["kind"] == "drone")}
    for bg in ("war", "wind", "cars", "war+wind", "war+cars", "war+wind+cars"):
        o[f"none {bg}"] = mean(lambda r, bg=bg: r["kind"] == "none" and r["bg"] == bg)
    o["none with cars"] = mean(lambda r: r["kind"] == "none" and "cars" in r["bg"])
    o["helicopter"] = mean(lambda r: r["kind"] == "helicopter")
    if any(r["kind"] == "rig" for r in rs):
        o["rig"] = mean(lambda r: r["kind"] == "rig")
    for k, (pos, neg) in AUCS.items():
        pos, neg = hops(pos), hops(neg)
        if len(pos) and len(neg):
            o[k] = _auc(pos, neg)
    # Mads's criterion (entry 245): the gap between confidence with a drone and without, in points
    gaps = {"gap war": o["drone war"] - o["none war"],
            **{f"gap war {src}": o[f"drone {src}"] - o["none war"] for src in srcs},
            "gap all": o["drone all"] - mean(lambda r: r["kind"] in ("none", "helicopter") and core(r))}
    return gaps | o


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


def boot_weights(rs: list[dict], n: int, seed: int = 0) -> np.ndarray:
    """(n + 1, rounds) round weights: row 0 all ones, rows 1..n a two-way cluster bootstrap (Owen's pigeonhole
    bootstrap, which errs on the wide side). The rounds are not independent: a drone file's 40 rounds share
    its audio, and a background stretch is reused by every round on it. So the two crossed factors are drawn
    with replacement at once, the source files within their stratum (Lund drones, Halmstad drones,
    helicopters, rig clips; recorded drones by recording, their `unit`) and the stretches within theirs (the
    six of entry 207, the extra ones), and a round weighs the product of its file's and its stretch's counts
    (a helicopter alone has no stretch)."""
    rng = np.random.default_rng(seed)
    draw = lambda k: rng.multinomial(k, np.full(k, 1 / k), size=n)            # noqa: E731
    w = np.ones((n + 1, len(rs)))
    for kept in (True, False):                     # the six stretches of entry 207, and the extra ones
        ss = sorted({r["stretch"] for r in rs if core(r) == kept})
        ms = draw(len(ss)) if ss else None
        for i, r in enumerate(rs):
            if r["bg"] != "alone" and core(r) == kept:
                w[1:, i] *= ms[:, ss.index(r["stretch"])]
    files = collections.defaultdict(lambda: collections.defaultdict(list))   # stratum -> file -> rounds
    for i, r in enumerate(rs):
        if r["kind"] != "none":
            files["drone " + r["file"].split()[0] if r["kind"] == "drone" else r["kind"]][r.get("unit", r["file"])].append(i)
    for st in sorted(files):
        names = sorted(files[st])
        mf = draw(len(names))
        for k, name in enumerate(names):
            w[1:, files[st][name]] *= mf[:, [k]]
    return w


def boot(rs: list[dict], res: dict, n: int, refs=(), seed: int = 0) -> dict:
    """Each figure per model (Mads's two gaps, the two means of entry 182's rule, the AUCS, DET): the
    estimate, a 95 % percentile interval, how often the model is best and in the top 3 over the resamples,
    and its difference to each of `refs` (models named before scoring) with a 95 % interval. Every model gets
    the same resamples (boot_weights), so the differences are paired."""
    wr = boot_weights(rs, n, seed)
    pick = lambda f: np.flatnonzero([f(r) for r in rs])    # noqa: E731
    srcs = [src for src in SOURCES if any(_dw(r) and r["file"].startswith(src) for r in rs)]
    means = {"drone war": pick(_dw), "none with cars": pick(lambda r: r["kind"] == "none" and "cars" in r["bg"]),
             **{f"drone war {src}": pick(lambda r, src=src: _dw(r) and r["file"].startswith(src)) for src in srcs},
             "none war": pick(_none_war), "drone all": pick(lambda r: r["kind"] == "drone"),
             "none all": pick(lambda r: r["kind"] in ("none", "helicopter") and core(r))}
    aucs = {k: (pick(p), pick(q)) for k, (p, q) in AUCS.items()}
    out = {}
    for m, s in res.items():
        mu, lens = np.array([v.mean() for v in s]), np.array([len(v) for v in s])
        with np.errstate(invalid="ignore"):          # a resample that drew no round of a set: nan, left out below
            mean = {k: 100 * (wr[:, i] @ mu[i]) / wr[:, i].sum(1) for k, i in means.items()}
        f = {"gap war": mean["drone war"] - mean["none war"],
             **{f"gap war {src}": mean[f"drone war {src}"] - mean["none war"] for src in srcs},
             "gap all": mean["drone all"] - mean["none all"],
             "drone war": mean["drone war"], "none with cars": mean["none with cars"]}
        for k, (p, q) in aucs.items():
            if not (len(p) and len(q)):
                continue
            ps, qs = np.concatenate([s[i] for i in p]), np.concatenate([s[i] for i in q])
            u, inv = np.unique(np.concatenate([ps, qs]), return_inverse=True)
            ip, iq, rp, rq = inv[: len(ps)], inv[len(ps):], np.repeat(p, lens[p]), np.repeat(q, lens[q])
            f[k] = np.array([_wauc(ip, iq, len(u), wr[b, rp], wr[b, rq]) for b in range(n + 1)])
            if k == "auc all":
                f[DET] = np.array([_wdet(ip, iq, len(u), wr[b, rp], wr[b, rq]) for b in range(n + 1)])
        out[m] = f
    names, table = list(out), {}
    for fig in out[names[0]]:
        sign = -1 if fig in LOWER_BETTER else 1
        rank = np.argsort(np.argsort(-sign * np.array([out[m][fig][1:] for m in names]), 0), 0)   # 0: best
        table[fig] = {}
        for k, m in enumerate(names):      # nan: a resample that drew no round of a set (rare); left out
            v = out[m][fig]
            lo, hi = np.nanpercentile(v[1:], [2.5, 97.5])
            row = dict(est=float(v[0]), lo=float(lo), hi=float(hi),
                       p_best=float(np.mean(rank[k] == 0)), p_top3=float(np.mean(rank[k] < 3)))
            for ref in refs:
                row[f"vs {ref}"] = [float(x) for x in np.nanpercentile(v[1:] - out[ref][fig][1:], [2.5, 50, 97.5])]
            table[fig][m] = row
    return table

def judge(a) -> None:
    out = OUT / a.tag
    meta = json.loads((out / "rounds.json").read_text(encoding="utf-8"))
    rs, res = meta["rounds"], {}
    for f in sorted(out.glob("*.npz")):
        if not scored_on(out, f.stem, meta.get("audio_md5")):
            print(f"left out {f.stem}: not scored on this tag's audio (no audio_md5, or another); score it again")
            continue
        z = np.load(f)
        res[f.stem] = [z[k] for k in z.files]
    if not res:
        raise SystemExit(f"no score files to judge in {out}")
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
    else:
        pinned = target = None
    refs = [m for m in a.vs if m in res]
    b = boot(rs, res, a.boot, refs) if a.boot else {}
    for fig, rows in b.items():
        sign, f = (-1 if fig in LOWER_BETTER else 1), (".3f" if fig.startswith(("auc", "det")) else ".1f")
        print(f"\n{fig}: estimate [95 % CI], share of resamples best / in the top 3, difference to "
              f"{', '.join(refs) or 'no reference'} [95 % CI] ({a.boot} paired two-way cluster resamples)")
        for m, v in sorted(rows.items(), key=lambda kv: -sign * kv[1]["est"]):
            vs = "  ".join(f"{v[f'vs {r}'][0]:+{f}} to {v[f'vs {r}'][2]:+{f}}" for r in refs)
            print(f"{m:34} {v['est']:{f}} [{v['lo']:{f}}, {v['hi']:{f}}]  {v['p_best']:4.0%} {v['p_top3']:4.0%}  {vs}")
    (out / "judge.json").write_text(json.dumps(dict(meta=meta | dict(rounds=len(rs)), unpinned=raw, pinned=pinned,
                                                    target=target, boot=b), indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("tag")
    r.add_argument("--half", choices=("tune", "confirm"), default="tune")
    r.add_argument("--models", nargs="+", required=True)
    r.add_argument("--extend", default="", help="an earlier tag whose rounds are this tag's first: its scores are reused")
    j = sub.add_parser("judge")
    j.add_argument("tag")
    j.add_argument("--ref", default="", help="model-name prefix whose mean drone confidence in war is the pin's target")
    j.add_argument("--boot", type=int, default=1000, help="bootstrap resamples for the 95 %% intervals (0: none)")
    j.add_argument("--vs", nargs="*", default=["it9 CRNN", "it6 lgbm"],
                   help="references named before scoring: each model's paired difference to them (entry 243)")
    r.add_argument("--device", default="cpu")
    r.add_argument("--tyres-only", action="store_true", help="run without the car engine recordings")
    a = ap.parse_args()
    ALLOW_TYRES_ONLY = getattr(a, "tyres_only", False)
    run(a) if a.cmd == "run" else judge(a)
