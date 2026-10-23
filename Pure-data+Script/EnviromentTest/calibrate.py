import numpy as np
from stream import MicStream
from detector import score_trace, first_alarm, sustained

NEED = 10                      # 10 hops = 1 s of continuous tonal evidence
CAL_SEEDS = range(100, 120)    # drone-free calibration scenes
TEST_SEEDS = range(1, 11)      # different seeds, never used for calibration

# 1) calibration: environment only
cal_kw = dict(bg_db=40.0, wind_db=45.0, dist_range=(800, 1000))
worst, hours = [], 0.0
for seed in CAL_SEEDS:
    mic = MicStream(seed, **cal_kw)
    t, s = score_trace(mic.mix - mic.drone, mic.fs)
    worst.append(sustained(s, NEED))
    hours += len(mic.mix) / mic.fs / 3600
print(f"calibration: {len(worst)} scenes, {hours*60:.1f} min of drone-free audio")
print(f"sustained noise score: median {np.median(worst):.1f} dB, max {max(worst):.1f} dB")
THRESH = max(worst) + 2.0
print(f"chosen threshold: {THRESH:.1f} dB\n")

# 2) test: far start, drone and no-drone versions of the same scene
test_kw = dict(bg_db=40.0, wind_db=45.0, dist_range=(1200, 1500))
print("seed  start | WITH drone: det.dist  warning | NO drone | events")
for seed in TEST_SEEDS:
    mic = MicStream(seed, **test_kw)
    T = mic.params["start_dist"] / mic.params["speed"]
    t1 = first_alarm(*score_trace(mic.mix, mic.fs), THRESH, NEED)
    t0 = first_alarm(*score_trace(mic.mix - mic.drone, mic.fs), THRESH, NEED)
    a = f"{mic.distance_at(t1):6.0f} m  {T - t1:5.1f} s" if t1 else "   missed       "
    b = f"FALSE ALARM {t0:.1f}s" if t0 else "clean"
    ev = ", ".join(f"{e['kind']}@{e['t_start']}s/{e['min_dist']:.0f}m" for e in mic.params["events"])
    print(f"{seed:4d} {mic.params['start_dist']:6.0f} |  {a}  | {b:16s} | {ev}")