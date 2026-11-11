"""Iteration 22, gate A: does per-window PCEN keep the drone's footprint in what the network sees?

    uv run python -I scripts/it22_pcen_gate.py       # writes artifacts/it22_pcen_gate.json

300 near drones (the sampler's 0-50 m cells) each summed into a rig background at -10, 0 and +5 dB in-band
SNR (as it20_gates.py, training sources only). For each frontend the features of background + drone and of
the background alone are standardised per band over all windows (as NetScorer.fit does); the footprint is
the mean absolute difference between the two, in sigma. Frontends: the frozen log-mel, and PCEN with
T = 0.06, 0.4 and 2 s. PCEN passes if, for some T, its footprint at 0 dB is at least 0.8 of log-mel's and at
-10 dB at least 0.8 too; the T with the largest 0 dB footprint is the one trained (fixed in the box).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dad import rangerun as rg  # noqa: E402
from dad import realrun as rr  # noqa: E402
from dad.config import FrontendConfig  # noqa: E402
from dad.features import LogMel  # noqa: E402
from dad.realdata import load_manifest  # noqa: E402

N, SEED = 300, 22
SNRS_DB = (-10.0, 0.0, 5.0)
CFGS = {"log-mel": FrontendConfig(), **{f"PCEN T={t:g}": FrontendConfig(pcen=True, pcen_t=t) for t in (0.06, 0.4, 2.0)}}


def main() -> None:
    plain = FrontendConfig()
    manifest = load_manifest()
    splits = rr.make_splits(manifest, plain)
    sampler = rg.RangeSampler(rg.make_pool_arm(manifest, "none"), splits, rg.RECIPES["simdist"], plain, seed=SEED)
    rng = np.random.default_rng(SEED)
    near = [c for c in sampler.cells if c.lo == 0]
    n = plain.samples_per_window
    bgs, drs = [], []
    for _ in range(N):
        drs.append(near[int(rng.integers(0, len(near)))].draw(rng)[:n])
        bgs.append(sampler._rig_background(rng)[0][:n])
    out = {}
    for name, cfg in CFGS.items():
        fe = LogMel(cfg)
        out[name] = {}
        for snr in SNRS_DB:
            mix = [b + d * math.sqrt(rr.inband_power(b) * 10 ** (snr / 10) / max(rr.inband_power(d), 1e-20))
                   for b, d in zip(bgs, drs)]
            F = rr.featurise(np.stack(mix + bgs), rr.VARIANTS["level"], fe)
            F = (F - F.mean(axis=(0, 2), keepdims=True)) / (F.std(axis=(0, 2), keepdims=True) + 1e-5)
            out[name][f"{snr:+g} dB"] = float(np.mean(np.abs(F[:N] - F[N:])))
        print(name, {k: round(v, 3) for k, v in out[name].items()}, flush=True)
    ref = out["log-mel"]
    ratios = {k: {s: v[s] / ref[s] for s in v} for k, v in out.items() if k != "log-mel"}
    passing = [k for k, r in ratios.items() if r["+0 dB"] >= 0.8 and r["-10 dB"] >= 0.8]
    out["ratio_to_log_mel"] = ratios
    out["pass"] = bool(passing)
    out["chosen"] = max(passing, key=lambda k: ratios[k]["+0 dB"]) if passing else None
    print("ratios:", {k: {s: round(x, 2) for s, x in r.items()} for k, r in ratios.items()})
    print("PCEN gate", "PASSES with " + out["chosen"] if passing else "FAILS")
    p = ROOT / "artifacts" / "it22_pcen_gate.json"
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
