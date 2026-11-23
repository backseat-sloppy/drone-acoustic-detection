"""
Measure the impulse response of a speaker -> room -> microphone chain.

Method: 10 s exponential sine sweep (Farina), played and recorded
simultaneously, then deconvolved with the inverse sweep.

Install:  pip install numpy scipy sounddevice matplotlib
Usage:    python measure_ir.py
          python measure_ir.py --list                 (show audio devices)
          python measure_ir.py --in 1 --out 3         (pick devices by index)
"""
import argparse
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import matplotlib.pyplot as plt

# ---------------- settings ----------------
FS = 48000          # sample rate
SWEEP_SEC = 10.0    # sweep length
F1, F2 = 20.0, 20000.0
TAIL_SEC = 2.0      # silence recorded after the sweep (captures room decay)
PRE_SEC = 0.5       # silence before the sweep
LEVEL = 0.3         # playback amplitude (0-1). Keep modest to avoid clipping
IR_PRE_MS = 5       # ms kept before the main peak
IR_LEN_SEC = 1.5    # IR length kept after the peak
# ------------------------------------------


def make_sweep(T, f1, f2, fs):
    t = np.arange(int(T * fs)) / fs
    R = np.log(f2 / f1)
    sweep = np.sin(2 * np.pi * f1 * T / R * (np.exp(t / T * R) - 1))

    # short fades to avoid clicks
    n = int(0.01 * fs)
    fade = np.linspace(0, 1, n)
    sweep[:n] *= fade
    sweep[-n:] *= fade[::-1]

    # inverse filter: time-reversed sweep with -6 dB/oct amplitude decay
    inverse = sweep[::-1] * np.exp(-t / T * R)
    return sweep, inverse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list audio devices")
    ap.add_argument("--in", dest="indev", type=int, default=None)
    ap.add_argument("--out", dest="outdev", type=int, default=None)
    args = ap.parse_args()

    if args.list:
        print(sd.query_devices())
        return

    sweep, inverse = make_sweep(SWEEP_SEC, F1, F2, FS)

    # pad with silence: pre + sweep + tail
    play = np.concatenate([
        np.zeros(int(PRE_SEC * FS)),
        sweep * LEVEL,
        np.zeros(int(TAIL_SEC * FS)),
    ]).astype(np.float32)

    print(f"Playing and recording {len(play) / FS:.1f} s ... keep the room quiet.")
    rec = sd.playrec(
        play.reshape(-1, 1),
        samplerate=FS,
        channels=1,
        device=(args.indev, args.outdev),
        dtype="float32",
    )
    sd.wait()
    rec = rec[:, 0]

    peak = np.max(np.abs(rec))
    print(f"Recorded peak level: {peak:.3f}")
    if peak >= 0.99:
        print("WARNING: recording clipped. Lower LEVEL or the mic gain.")
    elif peak < 0.02:
        print("WARNING: very low signal. Raise LEVEL or the mic gain.")

    # deconvolution via FFT (linear convolution with the inverse filter)
    N = len(rec) + len(inverse) - 1
    ir_full = np.fft.irfft(np.fft.rfft(rec, N) * np.fft.rfft(inverse, N), N)

    # find the main peak (includes system latency), then trim around it
    p = int(np.argmax(np.abs(ir_full)))
    start = max(0, p - int(IR_PRE_MS / 1000 * FS))
    end = p + int(IR_LEN_SEC * FS)
    ir = ir_full[start:end]
    ir /= np.max(np.abs(ir))

    latency_ms = (p - (len(sweep) - 1) - int(PRE_SEC * FS)) / FS * 1000
    print(f"Estimated round-trip latency: {latency_ms:.1f} ms")

    # save results
    wavfile.write("impulse_response.wav", FS, ir.astype(np.float32))
    wavfile.write("recording.wav", FS, rec.astype(np.float32))
    print("Saved impulse_response.wav and recording.wav")

    # plots
    t = (np.arange(len(ir)) - int(IR_PRE_MS / 1000 * FS)) / FS * 1000
    H = np.fft.rfft(ir, 1 << 16)
    f = np.fft.rfftfreq(1 << 16, 1 / FS)
    mag_db = 20 * np.log10(np.abs(H) + 1e-12)

    fig, ax = plt.subplots(2, 1, figsize=(10, 7))
    ax[0].plot(t, ir)
    ax[0].set_xlabel("Time (ms)")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_title("Impulse response")
    ax[0].set_xlim(-IR_PRE_MS, 200)

    ax[1].semilogx(f[1:], mag_db[1:])
    ax[1].set_xlim(F1, F2)
    ax[1].set_ylim(np.max(mag_db) - 70, np.max(mag_db) + 5)
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Magnitude (dB)")
    ax[1].set_title("Frequency response")
    ax[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
