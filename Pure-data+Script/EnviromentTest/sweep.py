import numpy as np
from stream import MicStream
from detector import score_trace, first_alarm, sustained

NEED = 10
CAL, TEST = range(100, 120), range(1, 11)
cal_kw = dict(bg_db=40.0, wind_db=45.0, dist_range=(800, 1000))
test_kw = dict(bg_db=40.0, wind_db=45.0, dist_range=(1200, 1500))

noise = []                                    # drone-free calibration scenes
for seed in CAL:
    m = MicStream(seed, **cal_kw)
    t, s = score_trace(m.mix - m.drone, m.fs)
    noise.append((seed, t, s, m.params["events"]))

tests = []                                    # test scenes with the drone
for seed in TEST:
    m = MicStream(seed, **test_kw)
    t, s = score_trace(m.mix, m.fs)
    T = m.params["start_dist"] / m.params["speed"]
    tests.append((m, t, s, T))

# 1) which noise scenes set the threshold?
print("TOP 5 NOISE SCENES (sustained score, events)")
ranked = sorted(noise, key=lambda n: -sustained(n[2], NEED))[:5]
for seed, t, s, ev in ranked:
    evs = ", ".join(f"{e['kind']}@{e['t_start']}s/{e['min_dist']:.0f}m" for e in ev) or "none"
    print(f"  seed {seed}: {sustained(s, NEED):5.1f} dB | {evs}")

# 2) tonal score of the drone scenes vs true distance
print("\nTONAL SCORE vs DRONE DISTANCE (all test scenes)")
edges = [1500, 1000, 500, 300, 200, 100, 50, 0]
bins = {i: [] for i in range(len(edges) - 1)}
for m, t, s, T in tests:
    for ti, si in zip(t, s):
        d = m.distance_at(ti)
        for i in range(len(edges) - 1):
            if edges[i + 1] <= d < edges[i]:
                bins[i].append(si)
for i, v in bins.items():
    if v:
        print(f"  {edges[i]:5d}-{edges[i+1]:4d} m: median {np.median(v):5.1f} dB, "
              f"90th pct {np.percentile(v, 90):5.1f} dB")

# 3) threshold sweep: false alarms vs detection range
print("\nTHRESHOLD SWEEP")
print("thresh | false-alarm scenes (of 20) | detected (of 10) | median det.dist | median warning")
for thr in range(14, 36, 2):
    fa = sum(first_alarm(t, s, thr, NEED) is not None for _, t, s, _ in noise)
    dist, warn = [], []
    for m, t, s, T in tests:
        ta = first_alarm(t, s, thr, NEED)
        if ta:
            dist.append(m.distance_at(ta)); warn.append(T - ta)
    md = f"{np.median(dist):6.0f} m" if dist else "     -  "
    mw = f"{np.median(warn):5.1f} s" if warn else "   -  "
    print(f"{thr:6d} | {fa:14d}             | {len(dist):9d}        | {md}       | {mw}")