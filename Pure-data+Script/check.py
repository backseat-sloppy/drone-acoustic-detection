import numpy as np
from render import render_dry

thr = np.full(150, 60.0)                      # steady throttle, 3 s
for ps in (0.85, 1.0, 1.25):
    x, fs = render_dry(thr, ps)
    seg = x[fs:3 * fs] * np.hanning(len(x[fs:3 * fs]))
    spec = np.abs(np.fft.rfft(seg))
    f = np.fft.rfftfreq(len(seg), 1 / fs)
    band = (f > 100) & (f < 1500)
    print(f"pscale {ps}: strongest peak {f[band][spec[band].argmax()]:.0f} Hz")