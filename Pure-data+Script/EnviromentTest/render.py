import glob
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
PATCH = HERE / "DroneSound.pd"
LAUNCH = HERE / "run.pd"

_found = glob.glob("/Applications/Pd*.app/Contents/Resources/bin/pd")
if not _found:
    raise SystemExit("Pd not found in /Applications. Run: ls /Applications | grep -i pd")
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
    return x, fs


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