"""War-shaped noise: steady noise with the long-term average spectrum (LTAS) of war sounds.

Like speech-shaped noise, but the target spectrum is averaged over war recordings
(artillery, small arms, vehicles, explosions ...). It is a steady, repeatable
background for testing detectors at exact SNRs; it has none of the transients of
real war audio (see docs/war_shaped_noise.md).

Build mode (default):
  1. load every WAV/MP3 under IN_DIR (recursive), mix to mono, resample to --sr
  2. LTAS of each file with Welch (Hann, --nfft, 50 % overlap), each file scaled
     to unit total power so long or loud files do not dominate
  3. average the files in the power domain, optionally smooth (--octave-frac 3
     gives 1/3-octave smoothing)
  4. write
       war_gain.txt   magnitude gain for FFT bins 0..nfft/2, one value per line,
                      for a Pd array `read` (Pure-data+Script/WarShapedNoise/)
       war_ltas.png   the LTAS, with a drone recording overlaid if --drone is given
       war_noise_ref.wav  random-phase noise with exactly the gain's magnitude in every
                      FFT bin (the most precise version), -20 dBFS RMS, loops seamlessly
     The gain is scaled so white noise keeps its RMS after shaping (unit power gain),
     and bin 0 (DC) is set to 0.

Check mode (--check REC.wav): LTAS of a recording (for example the Pd output) against
war_gain.txt, per 1/3-octave band from 50 Hz to 20 kHz, plus an overlay plot.

  uv run python scripts/war_shaped_noise.py                         # data/processed/stems/war
  uv run python scripts/war_shaped_noise.py IN_DIR --octave-frac 3 --drone drone.wav
  uv run python scripts/war_shaped_noise.py --check Pure-data+Script/WarShapedNoise/war_noise_out.wav
"""
import argparse
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, welch

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "data" / "processed" / "stems" / "war"
OUT_DIR = ROOT / "Pure-data+Script" / "WarShapedNoise"


def load(path, sr):
    """Mono float audio at sr. soundfile reads WAV and MP3 (libsndfile >= 1.1)."""
    x, fs = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if fs != sr:
        g = gcd(fs, sr)
        x = resample_poly(x, sr // g, fs // g)
    return x


def ltas(x, sr, nfft):
    """Welch power spectrum, bins 0..nfft/2, scaled to unit total power."""
    f, p = welch(x, fs=sr, window="hann", nperseg=nfft, noverlap=nfft // 2)
    return f, p / p.sum()


def smooth(f, p, frac):
    """Fractional-octave smoothing: each bin becomes the mean power over [f/2^(1/2frac), f*2^(1/2frac)]."""
    if not frac:
        return p
    c = np.concatenate([[0.0], np.cumsum(p)])
    lo = np.searchsorted(f, f * 2 ** (-0.5 / frac), side="left")
    hi = np.searchsorted(f, f * 2 ** (0.5 / frac), side="right")
    hi = np.maximum(hi, lo + 1)                       # at least the bin itself (low bins)
    out = (c[hi] - c[lo]) / (hi - lo)
    out[0] = p[0]
    return out


def unit_gain(g):
    """Scale bins 0..N/2 so a full N-point spectrum has mean power 1 (white noise keeps its RMS)."""
    full = g[0] ** 2 + g[-1] ** 2 + 2 * np.sum(g[1:-1] ** 2)
    return g / np.sqrt(full / (2 * (len(g) - 1)))


def shaped_noise(gain, sr, seconds, seed):
    """Random-phase noise: every bin of the clip's FFT has exactly the target magnitude, only the
    phase is random. Gaussian noise would add a random magnitude per bin (about +-0.5 dB per
    1/3-octave band at 20 s). The clip is periodic, so it loops without a click."""
    n = int(seconds * sr)
    f = np.fft.rfftfreq(n, 1 / sr)
    mag = np.sqrt(np.interp(f, np.linspace(0, sr / 2, len(gain)), gain ** 2))   # interpolate power
    spec = mag * np.exp(2j * np.pi * np.random.default_rng(seed).random(len(f)))
    spec[0] = 0.0                                                    # no DC
    if n % 2 == 0:
        spec[-1] = mag[-1]                                           # Nyquist bin must be real
    return np.fft.irfft(spec, n)


def db(p):
    return 10 * np.log10(np.maximum(p, 1e-20))


def plt():
    """pyplot on the Agg backend, loaded only to plot: importing this module must not change
    a GUI's backend (scripts/drone_war_grid.py uses shaped_noise)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot
    return matplotlib.pyplot


def build(a):
    files = sorted(p for p in Path(a.in_dir).rglob("*") if p.suffix.lower() in (".wav", ".mp3"))
    if not files:
        raise SystemExit(f"no WAV/MP3 files under {a.in_dir}")
    spectra, by_class = [], {}                       # class = the file's folder name
    for p in files:
        x = load(p, a.sr)
        if len(x) < a.nfft or not np.any(x):
            print(f"skip {p.name}: shorter than nfft or silent")
            continue
        f, s = ltas(x, a.sr, a.nfft)
        spectra.append(s)
        by_class.setdefault(p.parent.name, []).append(s)
        print(f"{p.relative_to(a.in_dir)}  {len(x) / a.sr:6.1f} s")
    raw = np.mean(spectra, axis=0)                   # power-domain average of normalised files
    p = smooth(f, raw, a.octave_frac)
    gain = unit_gain(np.sqrt(p))
    gain[0] = 0.0                                    # no DC in the noise

    a.out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(a.out_dir / "war_gain.txt", gain, fmt="%.8f")
    ref = shaped_noise(gain, a.sr, a.seconds, a.seed)
    ref *= 10 ** (-20 / 20) / np.sqrt(np.mean(ref ** 2))        # -20 dBFS RMS
    sf.write(a.out_dir / "war_noise_ref.wav", ref.astype(np.float32), a.sr, subtype="FLOAT")

    fig, ax = plt().subplots(figsize=(9, 5))
    ax.semilogx(f[1:], db(raw[1:]), color="0.7", lw=1, label=f"war LTAS, {len(spectra)} files")
    if a.octave_frac:
        ax.semilogx(f[1:], db(p[1:]), color="C3", lw=2, label=f"1/{a.octave_frac}-octave smoothed")
    if a.drone:
        _, d = ltas(load(a.drone, a.sr), a.sr, a.nfft)
        ax.semilogx(f[1:], db(d[1:]), color="C0", lw=1.5, label=f"drone: {Path(a.drone).name}")
    ax.set(xlabel="frequency (Hz)", ylabel="power per bin (dB re total)", xlim=(20, a.sr / 2),
           title=f"Long-term average spectrum, Welch nfft={a.nfft}, sr={a.sr}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.out_dir / "war_ltas.png", dpi=130)

    # each class's share of the average (lines add up to the black total), 1/3-octave smoothed to read
    fig, ax = plt().subplots(figsize=(9, 5))
    ax.semilogx(f[1:], db(smooth(f, raw, 3)[1:]), color="k", lw=2.5, label="all files")
    for c, v in sorted(by_class.items()):
        part = smooth(f, np.sum(v, axis=0) / len(spectra), 3)
        ax.semilogx(f[1:], db(part[1:]), lw=1.3, label=f"{c} ({len(v)})")
    ax.set(xlabel="frequency (Hz)", ylabel="power per bin (dB re total)", xlim=(20, a.sr / 2),
           ylim=(db(raw[1:].max()) - 60, db(raw[1:].max()) + 5),
           title="Contribution of each class to the war LTAS (1/3-octave smoothed)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(a.out_dir / "war_ltas_by_class.png", dpi=130)
    print(f"wrote war_gain.txt ({len(gain)} bins), war_noise_ref.wav, war_ltas.png, war_ltas_by_class.png to {a.out_dir}")


def check(a):
    gain = np.loadtxt(a.out_dir / "war_gain.txt")
    nfft = 2 * (len(gain) - 1)
    x = load(a.check, a.sr)
    print(f"{a.check}: {len(x) / a.sr:.1f} s, RMS {20 * np.log10(np.sqrt(np.mean(x ** 2))):.1f} dBFS")
    f, got = ltas(x, a.sr, nfft)
    want = gain ** 2 / np.sum(gain ** 2)
    # 1/3-octave bands from 50 Hz to 20 kHz: per-bin ripple from short noise averages out
    centres = 1000 * 2 ** (np.arange(-14, 14) / 3)
    centres = centres[(centres >= 50) & (centres <= min(20000, a.sr / 2 / 2 ** (1 / 6)))]
    err = []
    print("  band Hz   target dB   got dB   diff")
    for fc in centres:
        m = (f >= fc * 2 ** (-1 / 6)) & (f < fc * 2 ** (1 / 6))
        if m.any():
            t, g = db(want[m].sum()), db(got[m].sum())
            err.append(g - t)
            print(f"  {fc:7.0f}   {t:8.1f}   {g:7.1f}   {g - t:+5.1f}")
    err = np.array(err)
    print(f"band error: max |diff| {np.abs(err).max():.2f} dB, RMS {np.sqrt(np.mean(err ** 2)):.2f} dB")

    fig, ax = plt().subplots(figsize=(9, 5))
    ax.semilogx(f[1:], db(want[1:]), color="C3", lw=2, label="target (war_gain.txt)")
    ax.semilogx(f[1:], db(got[1:]), color="C0", lw=1, label=Path(a.check).name)
    ax.set(xlabel="frequency (Hz)", ylabel="power per bin (dB re total)", xlim=(20, a.sr / 2))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = Path(a.check).with_name(Path(a.check).stem + "_check.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return err


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("in_dir", nargs="?", type=Path, default=IN_DIR, help="folder of war WAV/MP3 files")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--nfft", type=int, default=1024, help="must match block~ in the Pd patch")
    ap.add_argument("--octave-frac", type=int, default=0, help="smoothing, 3 = 1/3 octave, 0 = off")
    ap.add_argument("--drone", type=Path, help="drone recording to overlay on the LTAS plot")
    ap.add_argument("--seconds", type=float, default=20.0, help="length of the reference WAV")
    ap.add_argument("--seed", type=int, default=0, help="seed of the reference WAV")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--check", type=Path, help="compare this recording's LTAS with war_gain.txt")
    a = ap.parse_args()
    check(a) if a.check else build(a)


if __name__ == "__main__":
    main()
