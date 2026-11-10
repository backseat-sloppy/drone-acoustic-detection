"""Iteration 20 (report entry 172): the two CPU gates that decide which filter arms are trained.

    uv run python -I scripts/it20_gates.py        # writes artifacts/it20_gates.json

Gate A: does an 80 Hz - 6 kHz order-4 bandpass change what the network sees, once each version is
standardised per mel band as NetScorer.fit does? 2,000 windows: half near drones in a rig background at the
recipe's simulated-distance SNR, half rig backgrounds (FSD50K is not on this PC).
No-op if the RMS difference is under 0.05 sigma and under 2% of cells move more than 0.25 sigma.

Gate B: does the Wiener gain keep the drone's harmonic lines? Near drones (the sampler's 0-50 m cells)
summed into a rig background at -10, 0 and +5 dB in-band SNR; the box judges 0 dB. Line bins: where the
drone alone, averaged over the window, stands >= 6 dB above its own median across frequency. Pass if >= 80%
of line bins keep a mean gain above 0.5 and the background alone loses >= 6 dB of in-band power.

Training sources only: the pool arm `none` (no rig held out), through load_manifest's sealed-data guard.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dad import rangerun as rg  # noqa: E402
from dad import realrun as rr  # noqa: E402
from dad.config import FrontendConfig  # noqa: E402
from dad.features import LogMel, wiener  # noqa: E402
from dad.gate import HarmonicCombGate  # noqa: E402
from dad.realdata import load_manifest  # noqa: E402

N_A, N_B, SEED = 2_000, 300, 20
SNRS_DB = (-10.0, 0.0, 5.0)
PLAIN = FrontendConfig()
BP = FrontendConfig(bandpass=(80.0, 6000.0))
W = FrontendConfig(wiener=True)


def power_spec(x: np.ndarray, cfg: FrontendConfig) -> np.ndarray:
    """(frames, bins) STFT power, framed as features.LogMel.frames."""
    fr = np.lib.stride_tricks.sliding_window_view(x.astype(np.float32), cfg.n_fft)[:: cfg.hop_length]
    return np.abs(np.fft.rfft(fr * np.hanning(cfg.n_fft).astype(np.float32), axis=1)) ** 2


def standardised(fe: LogMel, audio: list[np.ndarray]) -> np.ndarray:
    L = rr.featurise(np.stack(audio), rr.VARIANTS["level"], fe)
    return (L - L.mean(axis=(0, 2), keepdims=True)) / (L.std(axis=(0, 2), keepdims=True) + 1e-5)


def main() -> None:
    manifest = load_manifest()
    splits = rr.make_splits(manifest, PLAIN)
    sampler = rg.RangeSampler(rg.make_pool_arm(manifest, "none"), splits, rg.RECIPES["simdist"], PLAIN, seed=SEED)
    rng = np.random.default_rng(SEED)
    out = {}

    # ---- Gate A. FSD50K is not on this PC, so no FSD backgrounds: positives are near drones in a rig
    # background at the recipe's simulated-distance SNR, negatives are rig backgrounds.
    near = [c for c in sampler.cells if c.lo == 0]

    def sim_positive() -> np.ndarray:
        dr = near[int(rng.integers(0, len(near)))].draw(rng)
        bg, pb = sampler._rig_background(rng)
        snr = rng.uniform(*sampler.recipe.sim_snr_db)
        return bg + dr * math.sqrt(pb * 10 ** (snr / 10) / max(rr.inband_power(dr), 1e-20))

    audio = [sim_positive() if i % 2 else sampler._rig_background(rng)[0] for i in range(N_A)]
    d = standardised(LogMel(BP), audio) - standardised(LogMel(PLAIN), audio)
    rms, moved = float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d) > 0.25))
    band_rms = np.sqrt(np.mean(d ** 2, axis=(0, 2)))
    out["gate_a"] = dict(rms_sigma=rms, share_cells_over_0_25=moved, worst_bands=np.argsort(band_rms)[-5:][::-1].tolist(),
                         worst_band_rms=np.sort(band_rms)[-5:][::-1].round(3).tolist(),
                         no_op=bool(rms < 0.05 and moved < 0.02))
    print(f"Gate A: RMS {rms:.4f} sigma, {moved:.2%} of cells move > 0.25 sigma -> "
          f"{'no-op' if out['gate_a']['no_op'] else 'NOT a no-op'}", flush=True)

    # ---- Gate B
    hz = np.linspace(0, W.sample_rate / 2, W.n_freq_bins)
    inband = (hz >= W.fmin) & (hz <= W.fmax)
    gate = HarmonicCombGate()
    survive = {s: [] for s in SNRS_DB}     # per window: (f0, kept line bins, line bins)
    bg_loss_db = []
    for _ in range(N_B):
        dr = near[int(rng.integers(0, len(near)))].draw(rng)[: W.samples_per_window]
        bg, pb = sampler._rig_background(rng)
        bg = bg[: W.samples_per_window]
        Sd = power_spec(dr, W).mean(0)
        line = inband & (10 * np.log10(np.maximum(Sd, 1e-20) / np.maximum(median_filter(Sd, W.wiener_bins, mode="nearest"), 1e-20)) >= 6.0)
        f0 = gate(dr).f0_hz
        for snr in SNRS_DB:
            x = bg + dr * math.sqrt(rr.inband_power(bg) * 10 ** (snr / 10) / max(rr.inband_power(dr), 1e-20))
            P = power_spec(x, W)
            G = (wiener(P, W) / np.maximum(P, 1e-20)).mean(0)
            survive[snr].append((f0, int(np.sum(G[line] > 0.5)), int(line.sum())))
        Pb = power_spec(bg, W)[:, inband]
        bg_loss_db.append(10 * math.log10(Pb.sum() / max(wiener(power_spec(bg, W), W)[:, inband].sum(), 1e-20)))

    out["gate_b"] = {}
    for snr, rows in survive.items():
        f0 = np.array([r[0] for r in rows]); kept = np.array([r[1] for r in rows]); n = np.array([r[2] for r in rows])
        by_f0 = {}
        for lo, hi in ((0, 150), (150, 250), (250, 1e9)):
            m = (f0 >= lo) & (f0 < hi)
            by_f0[f"{lo:g}-{hi:g} Hz"] = dict(windows=int(m.sum()), kept=float(kept[m].sum() / max(n[m].sum(), 1)))
        out["gate_b"][f"{snr:+g} dB"] = dict(kept=float(kept.sum() / max(n.sum(), 1)), line_bins_per_window=float(n.mean()),
                                            windows_without_lines=int(np.sum(n == 0)), by_f0=by_f0)
        print(f"Gate B at {snr:+g} dB: {kept.sum() / max(n.sum(), 1):.1%} of line bins keep gain > 0.5 "
              f"({n.mean():.1f} line bins per window); by f0: "
              + ", ".join(f"{k} {v['kept']:.0%} (n={v['windows']})" for k, v in by_f0.items()), flush=True)
    loss = np.array(bg_loss_db)
    out["gate_b"]["background_loss_db"] = dict(mean=float(loss.mean()), median=float(np.median(loss)),
                                               p10=float(np.percentile(loss, 10)), p90=float(np.percentile(loss, 90)))
    kept0 = out["gate_b"]["+0 dB"]["kept"]
    out["gate_b"]["pass"] = bool(kept0 >= 0.80 and loss.mean() >= 6.0)
    print(f"Background alone loses {loss.mean():.1f} dB in-band (median {np.median(loss):.1f}, "
          f"10-90% {np.percentile(loss, 10):.1f}-{np.percentile(loss, 90):.1f}) -> Gate B "
          f"{'PASSES' if out['gate_b']['pass'] else 'FAILS'}", flush=True)

    path = ROOT / "artifacts" / "it20_gates.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
