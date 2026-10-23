import numpy as np
from stream import MicStream
from detector import TonalDetector

SEEDS = range(1, 11)
KW = dict(bg_db=40.0, wind_db=45.0, dist_range=(1200, 1500))   # start far, so the detector's limit shows

def first_detection(sig, fs):
    det, blk = TonalDetector(fs), int(fs * 0.02)
    for k in range(len(sig) // blk):
        res = det.process(sig[k * blk:(k + 1) * blk])
        if res and res[0]:
            return (k + 1) * blk / fs
    return None

print("seed  start  L_1m | WITH drone: det.dist  warning | NO drone: result    | events")
for seed in SEEDS:
    mic = MicStream(seed, **KW)
    T = mic.params["start_dist"] / mic.params["speed"]
    t1 = first_detection(mic.mix, mic.fs)                    # drone + environment
    t0 = first_detection(mic.mix - mic.drone, mic.fs)        # same environment, no drone
    a = f"{mic.distance_at(t1):6.0f} m  {T - t1:5.1f} s" if t1 else "   missed        "
    b = f"FALSE ALARM t={t0:.1f}s" if t0 else "clean"
    ev = ", ".join(f"{e['kind']}@{e['t_start']}s/{e['min_dist']:.0f}m" for e in mic.params["events"])
    print(f"{seed:4d} {mic.params['start_dist']:6.0f} {mic.params['L_1m']:5.1f} |  {a}  | {b:20s} | {ev}")