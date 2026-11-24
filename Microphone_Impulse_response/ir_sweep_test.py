#!/usr/bin/env python3
"""
Impulse response measurement with an exponential sine sweep (Farina method).

Setup (Steinberg UR44c):
    Output L  -> speaker
    Output R  -> Input 1   (loopback / timing reference)
    Input 3   -> microphone A (under test)
    Input 4   -> microphone B (under test)

Install:
    pip install numpy scipy sounddevice matplotlib

Usage:
    python ir_sweep_test.py --list                 # show audio devices
    python ir_sweep_test.py                        # run with defaults (30 s sweep)
    python ir_sweep_test.py --device "UR44" --level -12 --fs 48000

Tips:
    * Set the sample rate in the Steinberg driver panel (or Windows/macOS audio
      settings) to match --fs, otherwise the stream will fail or resample.
    * On Windows, use the ASIO or WASAPI/MME entry of the UR44c. Use --device with
      the index from --list if the name match picks the wrong host API.
    * Turn the loopback input (Input 1) gain so the sweep is well below clipping.
"""

import argparse
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import fftconvolve

# ----------------------------------------------------------------------------
# Channel routing (1-based, as used by sounddevice mappings)
# ----------------------------------------------------------------------------
OUT_SPEAKER = 1          # L output -> speaker
OUT_LOOPBACK = 2         # R output -> input 1
IN_LOOPBACK = 1          # input 1 receives R output
IN_MICS = [3, 4]         # microphones under test
IN_LABELS = ["loopback (in1)", "mic in3", "mic in4"]


# ----------------------------------------------------------------------------
# Sweep generation
# ----------------------------------------------------------------------------
def make_sweep(f1, f2, duration, fs):
    """Exponential sine sweep and its inverse filter (Farina)."""
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    rate = np.log(f2 / f1)

    sweep = np.sin(2 * np.pi * f1 * duration / rate * (np.exp(t * rate / duration) - 1))

    # Inverse filter: time-reversed sweep with -6 dB/oct amplitude envelope
    inv = sweep[::-1] * np.exp(-t * rate / duration)

    # Normalise so that sweep (*) inv has a unit peak
    ref = fftconvolve(sweep, inv)
    inv /= np.max(np.abs(ref))

    # Short fades on the played signal to avoid clicks (inverse stays untouched)
    fade = int(0.01 * fs)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(fade) / fade))
    played = sweep.copy()
    played[:fade] *= ramp
    played[-fade:] *= ramp[::-1]

    return played, inv


# ----------------------------------------------------------------------------
# Device handling
# ----------------------------------------------------------------------------
def find_device(name):
    devices = sd.query_devices()
    candidates = []
    for i, d in enumerate(devices):
        if (name.lower() in d["name"].lower()
                and d["max_input_channels"] >= max(IN_MICS)
                and d["max_output_channels"] >= 2):
            candidates.append(i)
    if not candidates:
        raise RuntimeError(
            f"No device matching '{name}' with >= {max(IN_MICS)} inputs and 2 outputs. "
            "Run with --list and pass --device <index>."
        )
    return candidates[0]


# ----------------------------------------------------------------------------
# Analysis helpers
# ----------------------------------------------------------------------------
def smooth_fractional_octave(freqs, mag, frac=6, n_points=500):
    centers = np.geomspace(20, 20000, n_points)
    out = np.zeros(n_points)
    half = 2 ** (1 / (2 * frac))
    for k, c in enumerate(centers):
        m = (freqs >= c / half) & (freqs <= c * half)
        out[k] = np.sqrt(np.mean(mag[m] ** 2)) if np.any(m) else np.nan
    return centers, out


def save_wav(path, data, fs):
    wavfile.write(path, fs, data.astype(np.float32))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Sweep-based impulse response measurement")
    p.add_argument("--list", action="store_true", help="list audio devices and exit")
    p.add_argument("--device", default="UR44", help="device name substring or index")
    p.add_argument("--fs", type=int, default=48000, help="sample rate")
    p.add_argument("--duration", type=float, default=30.0, help="sweep length in seconds")
    p.add_argument("--f1", type=float, default=20.0, help="start frequency (Hz)")
    p.add_argument("--f2", type=float, default=None, help="end frequency (Hz), default 0.45*fs")
    p.add_argument("--level", type=float, default=-12.0, help="sweep level in dBFS")
    p.add_argument("--pre", type=float, default=0.5, help="silence before sweep (s)")
    p.add_argument("--tail", type=float, default=3.0, help="silence after sweep for decay (s)")
    p.add_argument("--ir-length", type=float, default=2.0, help="IR length to keep (s)")
    p.add_argument("--pre-ir", type=float, default=0.005, help="IR pre-roll before direct sound (s)")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--outdir", default=os.path.join(script_dir, "ir_measurements"),
                   help="output folder (default: next to this script)")
    args = p.parse_args()

    if args.list:
        print(sd.query_devices())
        return

    fs = args.fs
    f2 = args.f2 if args.f2 else 0.45 * fs
    amp = 10 ** (args.level / 20)

    # Device
    dev = int(args.device) if str(args.device).isdigit() else find_device(args.device)
    info = sd.query_devices(dev)
    print(f"Using device [{dev}]: {info['name']}")
    sd.default.device = (dev, dev)
    sd.default.samplerate = fs

    # Signals
    sweep, inv = make_sweep(args.f1, f2, args.duration, fs)
    pre = np.zeros(int(args.pre * fs))
    tail = np.zeros(int(args.tail * fs))
    play_mono = np.concatenate([pre, sweep * amp, tail])

    # Same sweep on both outputs: L -> speaker, R -> loopback
    play = np.zeros((len(play_mono), 2), dtype=np.float32)
    play[:, OUT_SPEAKER - 1] = play_mono
    play[:, OUT_LOOPBACK - 1] = play_mono

    in_map = [IN_LOOPBACK] + IN_MICS
    total = len(play_mono) / fs
    print(f"Sweep {args.f1:.0f}-{f2:.0f} Hz, {args.duration:.0f} s, level {args.level} dBFS")
    print(f"Recording inputs {in_map}. Total time: {total:.1f} s ...")

    rec = sd.playrec(play, samplerate=fs, input_mapping=in_map,
                     output_mapping=[OUT_SPEAKER, OUT_LOOPBACK], dtype="float32")
    sd.wait()
    print("Done recording.")

    # Level check
    for k, label in enumerate(IN_LABELS):
        peak = np.max(np.abs(rec[:, k]))
        db = 20 * np.log10(peak + 1e-12)
        flag = "  <-- CLIPPING!" if peak >= 0.98 else ("  (very low)" if db < -50 else "")
        print(f"  {label:16s} peak {db:6.1f} dBFS{flag}")

    # Deconvolution (all channels)
    full = np.column_stack([fftconvolve(rec[:, k], inv) for k in range(rec.shape[1])])

    # Latency/time-zero from loopback: strongest peak of loopback IR
    t0 = int(np.argmax(np.abs(full[:, 0])))
    print(f"Loopback peak at sample {t0} -> system latency reference set to t = 0")

    start = max(t0 - int(args.pre_ir * fs), 0)
    stop = start + int(args.ir_length * fs)
    irs = full[start:stop, :]

    # Normalise mic IRs with a common factor so their relative levels are kept
    mic_norm = np.max(np.abs(irs[:, 1:]))
    irs_norm = irs.copy()
    irs_norm[:, 1:] /= mic_norm
    irs_norm[:, 0] /= np.max(np.abs(irs[:, 0]))

    # Save
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(args.outdir, stamp)
    os.makedirs(out, exist_ok=True)
    save_wav(os.path.join(out, "raw_recording.wav"), rec, fs)
    names = ["loopback", "mic_in3", "mic_in4"]
    for k, n in enumerate(names):
        save_wav(os.path.join(out, f"IR_{n}.wav"), irs_norm[:, k], fs)
    print(f"Saved results in: {out}")

    # Plots
    t_ms = (np.arange(len(irs)) / fs - args.pre_ir) * 1000
    fig, ax = plt.subplots(2, 1, figsize=(11, 8))

    for k in (1, 2):
        ax[0].plot(t_ms, irs_norm[:, k], label=IN_LABELS[k], alpha=0.8)
    ax[0].set_xlim(-args.pre_ir * 1000, 200)
    ax[0].set_xlabel("Time relative to loopback (ms)")
    ax[0].set_ylabel("Amplitude (normalised)")
    ax[0].set_title("Impulse responses (first 200 ms)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    # Frequency response (fade-out window on the IR tail)
    n = len(irs_norm)
    win = np.ones(n)
    fade_len = n // 5
    win[-fade_len:] = 0.5 * (1 + np.cos(np.pi * np.arange(fade_len) / fade_len))
    freqs = np.fft.rfftfreq(n, 1 / fs)
    for k in (1, 2):
        mag = np.abs(np.fft.rfft(irs_norm[:, k] * win))
        fc, ms = smooth_fractional_octave(freqs, mag, frac=6)
        ax[1].semilogx(fc, 20 * np.log10(ms + 1e-12), label=IN_LABELS[k])
    ax[1].set_xlim(20, 20000)
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Magnitude (dB, relative)")
    ax[1].set_title("Frequency response (1/6 octave smoothing)")
    ax[1].legend()
    ax[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    fig.savefig(os.path.join(out, "ir_plot.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
