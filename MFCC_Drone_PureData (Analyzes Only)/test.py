import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# librosa ships with a built-in example sound, so no file needed yet
y, sr = librosa.load(librosa.example('trumpet'))

S = librosa.feature.melspectrogram(y=y, sr=sr)
S_db = librosa.power_to_db(S, ref=np.max)

fig, ax = plt.subplots()
img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set_title('Test mel spectrogram')
plt.show()