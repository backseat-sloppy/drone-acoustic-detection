"""Soundscape test: every drone in drone_only/ mixed into every 20 s noise clip, scored by model 6.

For each noise clip k the loop runs 1 + N_DRONES tests:
  - the noise alone (a negative, for false alarms),
  - the noise + drone i, with the drone starting at a fixed onset that rotates with k
    (onset = ((i + k) mod 10) s), so speed and onset are not tied together.
The drone sound (seed) also rotates with k: clip k uses the (k mod n)-th of the n seeds found in drone_only/.
Each test: write noise WAV -> mix with the drone WAV -> write mix WAV + sidecar JSON ->
score with dad.wavio.evaluate_file (dscnn, lgbm, comb at their frozen it6 thresholds) ->
append one JSON line -> delete the WAVs. Tests already in the results file are skipped.

Synthetic scaffolding (CLAUDE.md section 5): never report these numbers as detection
performance, and never choose a threshold from them.

    uv run python soundscape_test.py --clips 10
    uv run python soundscape_test.py --clips 10 --noise silence     # the drones alone
    uv run python soundscape_test.py --clips 10 --gain-db -20      # level sensitivity
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from dad.config import DEFAULT
from dad.wavio import evaluate_file, load_scorer, threshold_from_run
from drone_sim import FULL_SCALE_DB, P_REF, C

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
DRONES = HERE / "drone_only"
RUN = ROOT / "artifacts/runs/it6_final.json"
MODELS = ROOT / "artifacts/models"
COMB_THRESHOLD = 1.9136685717165247        # it6_final.json, "comb (no ML)", hold out none
CLIP_S, FS = 20.0, 44100
N_ONSETS = 10                              # onsets 0, 1, ..., 9 s: the 10 s drone always ends inside the clip
SCALE = P_REF * 10 ** (FULL_SCALE_DB / 20)  # pascals at full scale 1.0


def placeholder_noise(k):
    """shortcut: scene.py's background + wind + bangs + vehicles, until the colleague's generator exists."""
    from scene import add_events, to_spl, wind
    rng = np.random.default_rng(10_000 + k)
    n = int(CLIP_S * FS)
    ev, log = add_events(rng, n, FS, 6.0)
    x = to_spl(rng.standard_normal(n), 40.0) + wind(rng, n, FS, 45.0) + ev
    return x / SCALE, dict(kind="placeholder", seed=10_000 + k, bg_db=40.0, wind_db=45.0, event_rate=6.0, events=log)


def silence(k):
    return np.zeros(int(CLIP_S * FS)), dict(kind="silence")


def drone_range(t, onset, speed, start_dist):
    """Slant range (m) of the drone whose sound reaches the mic at clip time t (straight approach,
    impact at onset + start_dist/speed). None before the first sound arrives."""
    T = start_dist / speed
    tau = (t - onset - speed * T / C) / (1 - speed / C)      # emission time of the sound heard at t
    return None if tau < 0 else round(float(max(speed * (T - tau), 1.0)), 1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=10, help="number of noise clips")
    ap.add_argument("--noise", choices=["placeholder", "silence"], default="placeholder")
    ap.add_argument("--gain-db", type=float, default=0.0, help="added to the whole mix, for level sensitivity")
    ap.add_argument("--out", default=str(ROOT / "artifacts/soundscape/results.jsonl"))
    ap.add_argument("--keep", action="store_true", help="keep the WAVs (to listen to them)")
    a = ap.parse_args()

    assert drone_range(0.0, 0.0, 30.0, 300.0) is None
    assert abs(drone_range(300 / C, 0.0, 30.0, 300.0) - 300.0) < 0.5    # first sound left at 300 m
    assert drone_range(10.0 + 1 / C, 0.0, 30.0, 300.0) == 1.0           # impact

    make_noise = placeholder_noise if a.noise == "placeholder" else silence
    by_seed = {}                                                 # seed -> [(path, params, audio)] in speed order
    for p in sorted(DRONES.glob("drone_s??_??.wav")):
        x, fs = sf.read(p, dtype="float64")
        assert fs == FS and len(x) <= CLIP_S * FS / 2 + 1, p   # 44.1 kHz, at most 10 s
        m = json.loads(p.with_suffix(".json").read_text())
        by_seed.setdefault(m["seed"], []).append((p, m, x))
    assert by_seed, f"no drones in {DRONES}: run drone_only.py first"
    seeds = sorted(by_seed)
    assert len({len(v) for v in by_seed.values()}) == 1, "every seed needs the same number of drones"

    scorers = {}
    for name, spec in [("dscnn", MODELS / "it6_final_none_simdist_dscnn.pt"),
                       ("lgbm", MODELS / "it6_final_none_simdist_lgbm.pkl"), ("comb", "comb")]:
        thr = COMB_THRESHOLD if spec == "comb" else threshold_from_run(RUN, str(spec), DEFAULT.window_seconds)
        scorers[name] = (load_scorer(str(spec), DEFAULT, "cpu"), thr, "comb" if spec == "comb" else sha(spec))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout.strip())

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / "work"
    work.mkdir(exist_ok=True)
    done = {json.loads(l)["id"] for l in out.open(encoding="utf-8")} if out.exists() else set()

    for k in range(a.clips):
        seed = seeds[k % len(seeds)]                            # drone sound rotates with the clip
        drones = by_seed[seed]
        tests = [(None, None)] + [(i, ((i + k) % N_ONSETS) * 1.0) for i in range(len(drones))]
        noise = None
        for i, onset in tests:
            tid = f"{a.noise}_g{a.gain_db:+.0f}_n{k:05d}_" + ("none" if i is None else f"s{seed:02d}_d{i + 1:02d}")
            if tid in done:
                continue
            if noise is None:                                   # the colleague's step: a noise WAV + its params
                x, nparams = make_noise(k)
                noise_wav = work / f"noise_{k:05d}.wav"
                sf.write(noise_wav, x.astype(np.float32), FS, subtype="FLOAT")
                noise, _ = sf.read(noise_wav, dtype="float64")
            mix = noise.copy()
            truth = dict(drones=[])
            if i is not None:
                _, m, d = drones[i]
                j = int(onset * FS)
                mix[j:j + len(d)] += d
                start = onset + m["start_dist"] / C               # first drone sound reaches the mic
                impact = onset + len(d) / FS
                truth["drones"] = [dict(start_s=round(start, 4), end_s=round(impact, 4), impact_s=round(impact, 4))]
            mix *= 10 ** (a.gain_db / 20)
            wav = work / f"{tid}.wav"
            sf.write(wav, mix.astype(np.float32), FS, subtype="FLOAT")
            wav.with_suffix(".json").write_text(json.dumps(truth))

            row = dict(id=tid, clip=k, noise=nparams, gain_db=a.gain_db, onset_s=onset, commit=commit, dirty=dirty,
                       full_scale_db_spl=FULL_SCALE_DB, frontend=DEFAULT.describe(), truth=truth["drones"],
                       drone=None if i is None else dict(file=drones[i][0].name, **{key: drones[i][1][key] for key in
                                                         ("seed", "speed", "start_dist", "elev_deg", "az_deg", "L_1m", "pscale")}))
            for name, (scorer, thr, ckpt) in scorers.items():
                r = evaluate_file(wav, scorer, DEFAULT, thr)
                dec = r["decisions"]
                if "t_end_s" not in row:
                    row["t_end_s"] = [round(float(t), 3) for t in dec.t_end]
                    row["label"] = [int(v) for v in r["labels"]]
                    if i is not None:
                        row["range_m"] = [drone_range(float(t), onset, m["speed"], m["start_dist"])
                                          for t in dec.t_end]
                row[name] = dict(checkpoint=ckpt, threshold=thr, counts=r["counts"], leads=r["leads"],
                                 score=[round(float(s), 4) for s in dec.score])
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if not a.keep:
                wav.unlink(); wav.with_suffix(".json").unlink()
            lead = row["dscnn"]["leads"][0]["lead_s"] if i is not None else None
            print(tid, "dscnn", row["dscnn"]["counts"], "" if i is None else f"lead {lead}")
        if noise is not None and not a.keep:
            noise_wav.unlink()


if __name__ == "__main__":
    main()
