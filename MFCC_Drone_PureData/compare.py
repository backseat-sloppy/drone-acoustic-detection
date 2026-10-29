import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

REAL_FILE, REAL_START = "drone_rec.wav", 5      # stable stretch of the real drone
PD_FILE,   PD_START   = "pd_drone.wav", 2   # skip the first 2 s of the patch
DURATION = 4
SR = 44100
N_FFT = 8192
HOP = 512
N_MELS = 128


def load(path, start):
    y, _ = librosa.load(path, sr=SR, offset=start, duration=DURATION, mono=True)
    return y / (np.sqrt(np.mean(y ** 2)) + 1e-12)   # equal RMS loudness


def analyse(y):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    spec_db = 20 * np.log10(S.mean(axis=1) + 1e-10)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=2048,
                                         hop_length=HOP, n_mels=N_MELS)
    mel_db = 10 * np.log10(mel.mean(axis=1) + 1e-10)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, n_fft=2048,
                                hop_length=HOP, n_mels=N_MELS)
    return spec_db, mel_db, mfcc.mean(axis=1)


def peaks_below(spec_db, fmax=3000):
    f = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    m = f <= fmax
    p, _ = find_peaks(spec_db[m], prominence=6, distance=int(20 / (SR / N_FFT)))
    return f[m][p], spec_db[m][p]


real = load(REAL_FILE, REAL_START)
pd_ = load(PD_FILE, PD_START)
r_spec, r_mel, r_mfcc = analyse(real)
p_spec, p_mel, p_mfcc = analyse(pd_)

f = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
mel_f = librosa.mel_frequencies(n_mels=N_MELS, fmax=SR / 2)

# ---- the two numbers to push down ----
mfcc_dist = np.linalg.norm(r_mfcc[1:] - p_mfcc[1:])      # skip coeff 0 (loudness)
band = mel_f <= 8000
mel_gap = np.mean(np.abs(r_mel[band] - p_mel[band]))
print(f"\nMFCC distance (coeffs 1-12): {mfcc_dist:.1f}    (lower = closer)")
print(f"Mel spectrum gap, 0-8 kHz:   {mel_gap:.1f} dB   (lower = closer)")

for name, spec in [("REAL drone", r_spec), ("PD patch", p_spec)]:
    pf, pdb = peaks_below(spec)
    print(f"\n{name} peaks (Hz, dB):")
    for a, b in zip(pf, pdb):
        print(f"{a:8.1f} Hz   {b:6.1f} dB")

# ---- plots ----
fig, ax = plt.subplots(3, 1, figsize=(12, 12))

ax[0].plot(f, r_spec, label="real drone")
ax[0].plot(f, p_spec, label="pd patch", alpha=0.8)
ax[0].set_xlim(0, 3000)
ax[0].set_xlabel("Hz"); ax[0].set_ylabel("dB")
ax[0].set_title("Average spectrum: peaks = harmonics")
ax[0].legend()

ax[1].plot(mel_f, r_mel, label="real drone")
ax[1].plot(mel_f, p_mel, label="pd patch", alpha=0.8)
ax[1].set_xlim(0, 8000)
ax[1].set_xlabel("Hz"); ax[1].set_ylabel("dB")
ax[1].set_title("Overall spectral shape (what MFCCs capture)")
ax[1].legend()

idx = np.arange(1, 13)
ax[2].bar(idx - 0.2, r_mfcc[1:], width=0.4, label="real drone")
ax[2].bar(idx + 0.2, p_mfcc[1:], width=0.4, label="pd patch")
ax[2].set_xlabel("MFCC number")
ax[2].set_title("Average MFCCs 1-12")
ax[2].legend()

plt.tight_layout()
plt.savefig("compare.png", dpi=120)
plt.show()