#!/usr/bin/env python3
"""
Automate sliders of the helicopter Pure Data patch and record a 10 s wav.

How it works
------------
Python makes a temporary copy of your patch (your original is never touched),
renames the slider send names ($0-period -> hel-period, ...) so Python can set
them, then starts Pd without a window and without audio and sends it new slider
values every 20 ms. Pd records the output into a wav file.
Pd runs in real time, so a 10 s file takes about 10 s to render.

Run:   python3 automate_helicopter.py
Needs: Pure Data installed, python3 with numpy and scipy (matplotlib optional).
Put this file in the same folder as Helicipter.pd, mdel.pd and vrdel.pd.
"""
import os, re, shutil, subprocess, sys
import numpy as np
from scipy.io import wavfile

# =====================================================================
# 1) FILES AND SETTINGS
# =====================================================================
PATCH      = "Helicipter.pd"          # your patch
ABSTRACTIONS = ["mdel.pd", "vrdel.pd"]  # abstraction files next to the patch
OUTPUT     = "helicopter_automated.wav"
DURATION   = 10.0                     # seconds
PD_BIN     = "pd"                     # on Mac use e.g.
                                      # "/Applications/Pd-0.54-1.app/Contents/Resources/bin/pd"
NORMALIZE_PEAK = 0.9                  # scale the final file so its peak is 0.9 (None = leave as it is)
FIX_SHIFT  = True                     # the lower [r $0-reson] in the main rotor should be [r $0-shift]

# =====================================================================
# 2) SLIDERS THAT STAY FIXED (value 0..1)  <<< edit to taste
#    Anything you also put in CURVES below is moved over time instead.
# =====================================================================
FIXED = {
    "period":       0.209,
    "basef":        0.50,
    "deviationf":   0.21,
    "reson":        0.83,
    "shift":        0.98,
    "symmetry":     0.01,
    "engine-speed": 0.38,
    "engine-mix":   0.17,
    "tail-speed":   0.88,
    "tail-mix":     0.36,
    "gear-ratio":   0.49,
    "gear-mix":     0.47,
    "distance":     1.0,
    "rotor-mix":    0.67,
}

# =====================================================================
# 3) MIN AND MAX OF EACH AUTOMATED SLIDER   <<< change the numbers here
#    (slider values, between 0 and 1).  A curve value of 0 -> MIN, 1 -> MAX
# =====================================================================
RANGES = {
    #  name           (min,  max)
    "distance":      (0.20, 0.90),
    "deviationf":    (0.05, 0.60),
    "basef":         (0.20, 0.70),
    "reson":         (0.30, 0.90),
    "tail-mix":      (0.20, 0.50),
    "engine-mix":    (0.10, 0.30),
    "gear-mix":      (0.30, 0.60),
}

# =====================================================================
# 4) HOW EACH SLIDER MOVES OVER TIME   <<< change the shapes here
#    List of (second, position). position 0 = the MIN above, 1 = the MAX above.
#    Straight lines are drawn between the points. Delete a line (or comment
#    it out with #) to stop automating that slider; it then uses FIXED.
# =====================================================================
CURVES = {
    "distance":   [(0, 0.0), (5, 1.0), (10, 0.0)],             # up then down
    "deviationf": [(0, 0.5), (3, 1.0), (7, 0.0), (10, 0.5)],
    "basef":      [(0, 1.0), (10, 0.0)],                       # falling
    "reson":      [(0, 0.0), (2.5, 1.0), (5, 0.0), (7.5, 1.0), (10, 0.0)],
    "tail-mix":   [(0, 0.5), (10, 0.5)],
    "engine-mix": [(0, 0.0), (5, 1.0), (10, 0.0)],
    "gear-mix":   [(0, 1.0), (10, 0.0)],
}

STEP = 0.02   # seconds between slider updates

# =====================================================================
# Nothing below needs to be changed
# =====================================================================
SEND_NAMES = {   # name in this script -> send name used inside the patch
    "period": "period", "basef": "basef", "deviationf": "devf", "reson": "reson",
    "shift": "shift", "symmetry": "symmetry", "engine-speed": "engine-speed",
    "engine-mix": "engine-mix", "tail-speed": "tail-speed", "tail-mix": "tail-mix",
    "gear-ratio": "gear-ratio", "gear-mix": "gear-mix", "distance": "distance",
    "rotor-mix": "rotor-mix",
}


def build_patch(folder, workdir):
    os.makedirs(workdir, exist_ok=True)
    for a in ABSTRACTIONS:
        shutil.copy(os.path.join(folder, a), workdir)
    txt = open(os.path.join(folder, PATCH)).read()
    if FIX_SHIFT:
        txt = re.sub(r"(#X obj \d+ \d+ )r \\\$0-reson(;\n#X obj \d+ \d+ \* 300;)", r"\1r \\$0-shift\2", txt)
    for s in SEND_NAMES.values():
        txt = txt.replace("\\$0-" + s + " ", "hel-" + s + " ").replace("\\$0-" + s + ";", "hel-" + s + ";")
    lines = txt.rstrip("\n").split("\n")

    # find top-level objects, and the wired dac~ whose signal we record
    depth, idx, objs, top_conn = 0, 0, {}, []
    for n, l in enumerate(lines):
        if l.startswith("#N canvas"):
            depth += 1 if n > 0 else 0
        elif l.startswith("#X restore"):
            depth -= 1
            if depth == 0: objs[idx] = n; idx += 1
        elif depth == 0 and re.match(r"#X (obj|msg|text|floatatom|symbolatom|array)\b", l):
            objs[idx] = n; idx += 1
        elif depth == 0 and l.startswith("#X connect"):
            top_conn.append(l)
    src = None
    for i, n in objs.items():
        if lines[n].startswith("#X obj") and " dac~" in lines[n]:
            for l in top_conn:
                m = re.match(r"#X connect (\d+) (\d+) %d 0;" % i, l)
                if m: src = m.groups(); break
        if src: break
    if src is None:
        sys.exit("Could not find a [dac~] that is connected to something in the patch.")
    k = idx
    lines += ["#X obj 10 10 writesf~ 1;", "#X obj 10 40 r hel-wsf;", "#X obj 10 70 qlist;",
              "#X obj 10 100 loadbang;", "#X msg 10 130 read score.txt \\, rewind \\, bang;",
              f"#X connect {src[0]} {src[1]} {k} 0;", f"#X connect {k+1} 0 {k} 0;",
              f"#X connect {k+3} 0 {k+4} 0;", f"#X connect {k+4} 0 {k+2} 0;"]
    open(os.path.join(workdir, "render.pd"), "w").write("\n".join(lines) + "\n")


def automation():
    t = np.arange(0.0, DURATION + 1e-9, STEP)
    values = {}
    for name in SEND_NAMES:
        if name in CURVES and name in RANGES:
            lo, hi = RANGES[name]
            kt, kv = zip(*CURVES[name])
            values[name] = np.clip(lo + np.interp(t, kt, kv) * (hi - lo), 0, 1)
        else:
            values[name] = np.full_like(t, FIXED[name])
    return t, values


def write_score(workdir, t, values, wav_tmp):
    ev = [(0.0, "pd dsp 1"), (0.0, f"hel-wsf open -bytes 4 {wav_tmp}"), (0.0, "hel-wsf start")]
    for i, ti in enumerate(t):
        for name, v in values.items():
            if i == 0 or name in CURVES:      # fixed sliders only need to be sent once
                ev.append((float(ti), f"hel-{SEND_NAMES[name]} {v[i]:.5f}"))
    ev.append((DURATION, "hel-wsf stop"))
    ev.append((DURATION + 0.2, "pd quit"))
    ev.sort(key=lambda e: e[0])
    out, last = [], 0.0
    for ti, m in ev:
        out.append(f"{round((ti - last) * 1000, 3)} {m};"); last = ti
    open(os.path.join(workdir, "score.txt"), "w").write("\n".join(out) + "\n")


def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    workdir = os.path.join(folder, "pd_work")
    wav_tmp = os.path.join(workdir, "raw.wav")
    build_patch(folder, workdir)
    t, values = automation()
    write_score(workdir, t, values, wav_tmp)
    if os.path.exists(wav_tmp): os.remove(wav_tmp)
    print(f"Rendering {DURATION:.0f} s with Pd (real time)...")
    subprocess.run([PD_BIN, "-nogui", "-noaudio", "-r", "44100", "-path", workdir,
                    "-open", os.path.join(workdir, "render.pd")],
                   cwd=workdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=DURATION + 120)
    if not os.path.exists(wav_tmp):
        sys.exit("Pd did not write a file. Check PD_BIN and that Pd is installed.")
    sr, x = wavfile.read(wav_tmp)
    x = x.astype(np.float32)
    peak = float(np.abs(x).max())
    print(f"peak {peak:.2f}  rms {np.sqrt(np.mean(x**2)):.4f}")
    if NORMALIZE_PEAK and peak > 0:
        x *= NORMALIZE_PEAK / peak
    wavfile.write(os.path.join(folder, OUTPUT), sr, x)
    print("wrote", os.path.join(folder, OUTPUT))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 3.5))
        for name in CURVES:
            if name in RANGES: ax.plot(t, values[name], label=name)
        ax.set_xlabel("seconds"); ax.set_ylabel("slider value"); ax.legend(ncol=4, fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(folder, "automation_curves.png"), dpi=90)
    except Exception:
        pass


if __name__ == "__main__":
    main()
