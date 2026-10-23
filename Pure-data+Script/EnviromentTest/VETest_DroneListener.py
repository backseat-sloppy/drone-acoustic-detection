import numpy as np
import soundfile as sf
from pathlib import Path
OUT = Path(__file__).parent

FS = 44100
C = 343.0                          # speed of sound, m/s
P_REF = 20e-6                      # 0 dB SPL, in pascals
L_1M = 70.0                        # dB SPL at 1 m (placeholder, we calibrate it in Step 2)

def make_trajectory(rng, speed=35.0, start_dist=400.0):
    az = rng.uniform(0, 2*np.pi)
    el = np.radians(rng.uniform(20, 60))          # above the horizon
    start = start_dist * np.array([np.cos(el)*np.cos(az),
                                   np.cos(el)*np.sin(az),
                                   np.sin(el)])
    T = start_dist / speed
    t = np.arange(int(T * FS)) / FS               # t = EMISSION time (drone's own clock)
    pos = start[None, :] * (1 - t[:, None] / T)
    return t, pos

def propagate(x, t, pos):
    r = np.maximum(np.linalg.norm(pos, axis=1), 1.0)
    # sound emitted at t arrives at t + r/C. Read it back on a regular time grid:
    y = np.interp(t, t + r / C, x / r, left=0.0, right=0.0)
    return y, r

def spl(p):                                       # RMS pressure -> dB SPL
    return 20 * np.log10(np.sqrt(np.mean(p**2)) / P_REF)

rng = np.random.default_rng(1)
t, pos = make_trajectory(rng)

# placeholder drone, scaled so its RMS at 1 m equals L_1M in pascals
x = sum(np.sin(2*np.pi*300*k*t) / k for k in range(1, 8))
x = x / np.sqrt(np.mean(x**2)) * P_REF * 10**(L_1M / 20)

y, r = propagate(x, t, pos)

print(f"flight time: {t[-1]:.1f} s, start distance: {r[0]:.0f} m")
print(f"start position: {pos[0].round(0)}  (z > 0 means above you)")

seg = y[len(y)//2 - FS : len(y)//2 + FS]
print("peak Hz:", np.abs(np.fft.rfft(seg)).argmax() / (len(seg)/FS))   # expect ~334

for d in (300, 100, 30):                          # SPL in a 0.5 s window around each distance
    i = np.argmin(np.abs(r - d))
    print(f"SPL at {d:>3} m: {spl(y[i - FS//4 : i + FS//4]):.1f} dB   (formula: {L_1M - 20*np.log10(d):.1f})")

BG_DB = 60.0                                  # background level, dB SPL (try 30 / 50 / 60)
THRESH_DB = -20.0                  # detector works when SNR > this (try -10, -20)
bg = rng.standard_normal(len(y))
bg = bg / np.sqrt(np.mean(bg**2)) * P_REF * 10**(BG_DB / 20)

SPEED = 35.0
win = FS // 2                                  # 0.5 s windows
print(f"\nBackground {BG_DB} dB SPL")
print("dist(m)  drone dB  SNR dB  warning(s)")
first = None
for i in range(0, len(y) - win, win):
    d = r[i + win // 2]                        # where the drone really is
    snr = spl(y[i:i + win]) - spl(bg[i:i + win])
    if first is None and snr > THRESH_DB:
        first = d
    if i % (win * 2) == 0:                     # print every second
        print(f"{d:7.0f}  {spl(y[i:i + win]):8.1f}  {snr:6.1f}  {d / SPEED:9.1f}")

if first is None:
    print(f"\nNever above {THRESH_DB} dB SNR -> no warning at all")
else:
    print(f"\nDetectable from {first:.0f} m -> {first / SPEED:.1f} s before impact")

mix = y + bg
sf.write(OUT / "approach_with_background.wav", (mix / np.max(np.abs(mix))).astype(np.float32), FS)
sf.write("approach_test.wav", (y / np.max(np.abs(y))).astype(np.float32), FS)