import queue
import numpy as np
import sounddevice as sd
from stream import MicStream
from detector import TonalDetector
import soundfile as sf
from pathlib import Path

SEED = 3
LIVE = False            # True: play through the speakers in real time. False: run as fast as possible
WITH_DRONE = True      # False: environment only, so any detection is a false alarm
BG_DB, WIND_DB = 40.0, 45.0
PLAY_FS_DB = 75.0       # dB SPL that maps to full scale in the speakers

mic = MicStream(SEED, bg_db=BG_DB, wind_db=WIND_DB, with_drone=WITH_DRONE)
HERE = Path(__file__).parent
scale = 20e-6 * 10 ** (110.0 / 20)                        # same 110 dB full scale as drone_sim.py
sf.write(HERE / f"live_{SEED}_calibrated.wav", (mic.mix / scale).astype(np.float32), mic.fs)
sf.write(HERE / f"live_{SEED}_listen.wav", (mic.mix / np.abs(mic.mix).max()).astype(np.float32), mic.fs)
det = TonalDetector(mic.fs)
T = mic.params["start_dist"] / mic.params["speed"]
state = dict(k=0, detected=None, best=-1e9)

def handle(block):
    state["k"] += 1
    t = state["k"] * mic.block / mic.fs
    res = det.process(block)
    if res is None:
        return
    detected, score = res
    state["best"] = max(state["best"], score)
    if detected and state["detected"] is None:
        state["detected"] = t
        print(f"DETECTED at t={t:.2f} s | true distance {mic.distance_at(t):.0f} m | "
              f"warning {T - t:.1f} s before impact")

print("params:", {k: v for k, v in mic.params.items() if k != "events"})
for e in mic.params["events"]:
    print("event:", e)

if LIVE:
    scale = 20e-6 * 10 ** (PLAY_FS_DB / 20)
    q = queue.Queue()
    def callback(outdata, frames, time_info, status):
        b = mic.next_block()
        if b is None:
            outdata[:] = 0
            raise sd.CallbackStop
        outdata[:, 0] = np.clip(b / scale, -1, 1)
        q.put(b)
    with sd.OutputStream(samplerate=mic.fs, blocksize=mic.block, channels=1,
                         dtype="float32", callback=callback) as stream:
        while stream.active or not q.empty():
            try:
                handle(q.get(timeout=0.2))
            except queue.Empty:
                pass
else:
    while (b := mic.next_block()) is not None:
        handle(b)

print(f"\nhighest tonal score: {state['best']:.1f} dB (threshold {det.thresh})")
if state["detected"] is None:
    print("no detection" + (" (correct, no drone present)" if not WITH_DRONE else " (missed the drone)"))
elif not WITH_DRONE:
    print("FALSE ALARM")