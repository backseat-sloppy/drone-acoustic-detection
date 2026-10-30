import glob
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
PATCH = HERE / "DroneSound.pd"
LAUNCH = HERE / "run.pd"

_found = (glob.glob("/Applications/Pd*.app/Contents/Resources/bin/pd")      # macOS
          + glob.glob("C:/Program Files*/Pd*/bin/pd.com"))                   # Windows (console build)
if not _found:
    raise SystemExit("Pd not found in /Applications or C:/Program Files")
PD = _found[0]


def render_dry(throttle, pscale=1.0, step_ms=20):
    """throttle: array of slider values 0..127, one per step_ms."""
    lines = ["0;", f"pscale {pscale:.3f};"]
    for v in throttle:
        lines.append(f"{step_ms};")
        lines.append(f"throttle {v:.2f};")
    (HERE / "throttle.txt").write_text("\n".join(lines) + "\n")

    # tiny launcher patch: fires "autorun" after DroneSound.pd has loaded
    LAUNCH.write_text("#N canvas 0 0 300 200 12;\n"
                      "#X obj 10 10 loadbang;\n"
                      "#X msg 10 50 \\; autorun bang;\n"
                      "#X connect 0 0 1 0;\n")

    out = HERE / "out.wav"
    want = len(throttle) * step_ms * 44100 // 1000          # the patch adds 300 ms after the last step
    for attempt in range(3):
        out.unlink(missing_ok=True)
        try:
            res = subprocess.run([PD, "-nogui", "-nosound", "-batch", "-r", "44100",
                                  "-open", str(PATCH), "-open", str(LAUNCH)],
                                 timeout=40, cwd=HERE, capture_output=True, text=True)
            print(res.stdout[-800:], res.stderr[-800:])
        except subprocess.TimeoutExpired as e:
            print("Pd did not finish in 40 s. Its output:\n", e.stdout, e.stderr)

        if not out.exists():
            raise SystemExit("Pd produced no out.wav")
        x, fs = sf.read(out)
        if len(x) < (out.stat().st_size - 44) // 2:
            # Pd quits right after "wr stop", sometimes before writesf~ fixes the header (data size 0
            # or a stale size). The samples are there: 16-bit mono PCM after the 44-byte header.
            x = np.fromfile(out, dtype="<i2", offset=44) / 32768.0
        if len(x) >= want:
            return x, fs
        # "pd quit" can also beat writesf~'s disk thread, losing the last buffer (about 0.6 s)
        print(f"Pd render {len(x) / 44100:.2f} s, want {want / 44100:.2f} s: rendering again")
    raise SystemExit(f"Pd render stayed short after 3 attempts ({len(x)} of {want} samples)")


def make_throttle(rng, n, base=60, wobble=6.0, dive_start=0.7, dive_gain=80):
    """n steps of 20 ms. Steady cruise, small corrections, dive at the end."""
    slow = np.convolve(rng.standard_normal(n), np.ones(25) / 25, mode="same")
    slow = slow / slow.std() * wobble
    fast = rng.standard_normal(n) * 0.8
    ramp = np.clip((np.arange(n) / n - dive_start) / (1 - dive_start), 0, 1) * dive_gain
    return np.clip(base + slow + fast + ramp, 0, 127)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    thr = make_throttle(rng, 500)
    x, fs = render_dry(thr)
    print("sample rate:", fs, " length:", len(x) / fs, "s  peak:", np.abs(x).max())
    sf.write(HERE / "dry_test.wav", x, fs)