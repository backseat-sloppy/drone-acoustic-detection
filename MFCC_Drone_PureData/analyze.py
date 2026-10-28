import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

FILE = "drone_rec.wav"      # filename
SR = 44100              # fixed on purpose: Pd output and recording must match
DURATION = 15           # seconds to analyse

y, sr = librosa.load(FILE, sr=SR, duration=DURATION, mono=True)
y = y / np.max(np.abs(y))   # normalise loudness so comparisons are fair

# Linear-frequency spectrogram: this is the one that shows drone harmonics
D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=4096, hop_length=512)), ref=np.max)

# Mel spectrogram: closer to how MFCCs "see" the sound
M = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=4096, hop_length=512, n_mels=128), ref=np.max)

fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
img1 = librosa.display.specshow(D, sr=sr, hop_length=512, x_axis='time', y_axis='hz', ax=ax[0])
ax[0].set_ylim(0, 5000)
ax[0].set_title('Linear spectrogram (look for horizontal lines)')
fig.colorbar(img1, ax=ax[0], format='%+2.0f dB')

img2 = librosa.display.specshow(M, sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=ax[1])
ax[1].set_title('Mel spectrogram')
fig.colorbar(img2, ax=ax[1], format='%+2.0f dB')

plt.tight_layout()
plt.show()