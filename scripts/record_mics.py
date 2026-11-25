"""Record the two test microphones (Fireface inputs Mic 9 and Mic 10) beside a live test, until Ctrl+C.

    uv run python scripts/record_mics.py [--channels 8,9] [--seconds S] [--out DIR]

One WAV per start, left = Mic 9, right = Mic 10, 48 kHz 24 bit, as they come in (no gain, no averaging). The name is
the start time, to pair it with the live run's folder; the exact alignment comes from the run's heard/ files.
"""
import argparse
import queue
import time
from pathlib import Path

import sounddevice as sd
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--channels", default="8,9")                 # from 0
ap.add_argument("--seconds", type=float)
ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/mic_recordings")
a = ap.parse_args()
ch = [int(c) for c in a.channels.split(",")]
dev = next((i for i, d in enumerate(sd.query_devices()) if "Fireface" in d["name"] and d["max_input_channels"] > max(ch)), None)
if dev is None:
    raise SystemExit("no Fireface with those inputs")
sr = int(sd.query_devices(dev)["default_samplerate"])
a.out.mkdir(parents=True, exist_ok=True)
f = a.out / f"{time.strftime('%Y%m%d_%H%M%S')}_mics_{'_'.join(map(str, ch))}.wav"
q: queue.Queue = queue.Queue()
lost, peak, t0, shown = 0, 0.0, time.time(), 0.0


def cb(x, n, t, status):
    global lost
    lost += bool(status.input_overflow)
    q.put(x[:, ch])                                          # a copy


with sf.SoundFile(f, "x", sr, len(ch), "PCM_24", format="RF64") as w, \
        sd.InputStream(device=dev, samplerate=sr, channels=max(ch) + 1, dtype="float32", latency="high", callback=cb):
    print(f"recording {f}  (Ctrl+C stops)")
    try:
        while a.seconds is None or time.time() - t0 < a.seconds:
            try:
                x = q.get(timeout=1.0)
            except queue.Empty:
                print("NO AUDIO is arriving from the Fireface")
                continue
            w.write(x)
            peak = abs(x).max(0) if isinstance(peak, float) else abs(x).max(0).clip(min=peak)
            if time.time() - shown > 10:
                print(f"{time.time() - t0:6.0f} s  peaks {[round(float(p), 4) for p in peak]}" + ("  CLIPPED" if peak.max() >= 0.999 else ""))
                shown = time.time()
    except KeyboardInterrupt:
        pass
print(f"saved {f}: {sf.info(f).duration:.1f} s, {lost} overflow(s)")
