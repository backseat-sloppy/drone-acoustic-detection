"""All scenes of a make_scenes.py run in one figure: a grid of linear spectrograms (1024-sample
window, hop 256, as scenes_gui.py), one shared dB colour scale, each panel titled with its scene
number, cruise f0 and falloff, and a dashed line at each drone label's onset and offset.

    uv run python scripts/plot_scene_grid.py data/processed/scenes/drones1

Writes <run>/spectrograms.png. --fmax sets the top of the frequency axis (default 4000 Hz).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import stft  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--fmax", type=float, default=4000.0)
    a = ap.parse_args()

    wavs = sorted((a.run / "audio").glob("*.wav"))
    cols = math.ceil(math.sqrt(len(wavs)))
    rows = math.ceil(len(wavs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 1.9), sharex=True, sharey=True,
                             squeeze=False, constrained_layout=True)
    for ax in axes.flat[len(wavs):]:
        ax.axis("off")
    for ax, wav in zip(axes.flat, wavs):
        x, fs = sf.read(wav)
        f, t, Z = stft(x, fs, nperseg=1024, noverlap=1024 - 256)
        keep = f <= a.fmax
        im = ax.pcolormesh(t, f[keep], 20 * np.log10(np.abs(Z[keep]) + 1e-10), cmap="magma",
                           vmin=-110, vmax=-20, shading="auto", rasterized=True)
        lab = json.loads((a.run / "labels" / f"{wav.stem}.json").read_text())
        for e in lab["events"]:
            if e["role"] == "drone":
                for s in (e["onset"], e["offset"]):
                    ax.axvline(s, color="white", lw=0.6, ls="--")
        d = lab["drone"]
        ax.set_title(f"{wav.stem[-3:]}  f0 {d['f0_cruise']:.0f}  fo {d['falloff']:.1f}" if d else wav.stem[-3:],
                     fontsize=7)
        ax.tick_params(labelsize=6)
    fig.supxlabel("s")
    fig.supylabel("Hz")
    fig.suptitle(f"{a.run.name}: {len(wavs)} scenes, linear spectrogram 0-{a.fmax:.0f} Hz "
                 "(f0 = cruise fundamental Hz, fo = harmonic falloff; dashed = drone label)", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.4, label="dB")
    out = a.run / "spectrograms.png"
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
