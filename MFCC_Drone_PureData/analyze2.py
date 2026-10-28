import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

FILE = "drone_rec.wav"     # <- your filename
SR = 44100
START = 5              # seconds: start of a stable stretch
DURATION = 4           # seconds to analyse
N_FFT = 8192           # very fine frequency resolution (~5 Hz per bin)
HOP = 512

y, sr = librosa.load(FILE, sr=SR, offset=START, duration=DURATION, mono=True)
y = y / np.max(np.abs(y))

# Average spectrum over the stretch
S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
mean_db = 20 * np.log10(S.mean(axis=1) + 1e-10)
freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

# Find peaks up to 3 kHz
mask = freqs <= 3000
f, d = freqs[mask], mean_db[mask]
bin_hz = sr / N_FFT
peaks, _ = find_peaks(d, prominence=6, distance=int(20 / bin_hz))

print("\nPEAKS FOUND (Hz, dB):")
for p in peaks:
    print(f"{f[p]:8.1f} Hz   {d[p]:6.1f} dB")

# MFCCs
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048,
                            hop_length=HOP, n_mels=128)
print("\nMEAN MFCC VECTOR (your drone's fingerprint):")
print(np.round(mfcc.mean(axis=1), 1))

# Plot
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
ax[0].plot(f, d)
ax[0].plot(f[peaks], d[peaks], 'rx')
ax[0].set_xlabel('Hz'); ax[0].set_ylabel('dB')
ax[0].set_title('Average spectrum with detected peaks')

img = librosa.display.specshow(mfcc, sr=sr, hop_length=HOP, x_axis='time', ax=ax[1])
fig.colorbar(img, ax=ax[1])
ax[1].set_title('MFCCs (13 coefficients)')

plt.tight_layout()
plt.show()