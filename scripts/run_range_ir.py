"""run_range.py with a measured microphone impulse response at the end of the audio pipeline (entry 272).

    python scripts/run_range_ir.py --ir data/ir/20261123_120401 -- <run_range.py arguments>

Training: every window, after mixing, augmentation and the speed change and just before the STFT, is
filtered by one of the measured IRs (IR_mic_in3 or IR_mic_in4, drawn per window). Scoring during the run
(validation, which freezes the deployment threshold; Lund, mower and the rest) filters each window by
IR_mic_in3 first, as the physical microphone would. The saved checkpoint is an ordinary network: at
deployment the real microphone applies the IR, so nothing is added at inference.

The patches live in this script only; run_range.py and dad are unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from dad import micir  # noqa: E402
from dad import rangerun as rg  # noqa: E402
from dad import realrun as rr  # noqa: E402


def install(folder: str, seed: int) -> np.ndarray:
    irs = micir.load_irs(folder)
    rng = np.random.default_rng([seed, 2311])
    print(f"mic IR: {folder}, {irs.shape[0]} responses, {irs.shape[1]} taps at 16 kHz; "
          f"training draws one per window, scoring uses the first", flush=True)

    from dad import gpufront

    def train_features(self, audio, down, eq):
        import torch
        x = self.speed(torch.from_numpy(np.asarray(audio)).to(self.device), down, self.cfg.samples_per_window)
        x = micir.convolve_torch(x, irs, rng.integers(0, len(irs), len(x)))
        L = self.logbands(x) + torch.from_numpy(eq).to(self.device)[:, :, None]
        return self.finish(L)
    gpufront.GpuFrontend.train_features = train_features

    def batch(self, index, batch_size):         # the numpy path (no CUDA), same order of steps
        audio, down, eq, y = self.raw_batch(index, batch_size)
        r = np.random.default_rng([self.seed, index, 2311])
        L = np.stack([self.fe.frames(micir.convolve_np(rg.resample_down(x, d)[: self.n_out],
                                                       irs[int(r.integers(0, len(irs)))]))
                      for x, d in zip(audio, down)])
        return rr.to_model(rr.level_normalise(L + eq[:, :, None]), self.cfg), y
    rg.RangeSampler.batch = batch

    score = rr.NetScorer.score

    def scored(self, audio, *a, **k):
        return score(self, micir.convolve_np(audio, irs[0]), *a, **k)
    rr.NetScorer.score = scored
    return irs


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--ir" not in argv or "--" not in argv:
        raise SystemExit(__doc__)
    folder = argv[argv.index("--ir") + 1]
    rest = argv[argv.index("--") + 1:]
    seed = int(rest[rest.index("--seed") + 1]) if "--seed" in rest else 0
    install(folder, seed)
    import run_range
    sys.argv = ["run_range.py"] + rest
    run_range.main()
