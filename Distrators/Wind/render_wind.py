"""Render one wind .wav from wind.pd with Python-chosen speed / resonance / volume.

Usage:  python render_wind.py --speed 60 --reso 40 --vol 70 --seconds 20 --out wind1.wav
Values are the slider values: 0-127.
"""
import argparse, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "wind.pd")

def find_pd():
    pd = shutil.which("pd")
    if pd:
        return pd
    # macOS: Pd lives inside the app bundle. Adjust if yours is named differently.
    import glob
    hits = sorted(glob.glob("/Applications/Pd-*.app/Contents/Resources/bin/pd"))
    hits += sorted(glob.glob("C:/Program Files*/Pd*/bin/pd.com"))   # Windows (console build)
    if hits:
        return hits[-1]
    sys.exit("Could not find Pd. Set PD_PATH at the top of this script.")

PD_PATH = find_pd()

def render(speed, reso, vol, seconds, out_path):
    # relative names are saved next to this script (not wherever the terminal happens to be)
    final_path = out_path if os.path.isabs(out_path) else os.path.join(HERE, out_path)
    final_path = os.path.abspath(os.path.expanduser(final_path))
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="windrender")
    out_path = os.path.join(tmp, "out.wav")   # Pd breaks on paths with spaces, so record here first
    lines = open(TEMPLATE).read().strip().split("\n")

    obj_lines = [l for l in lines if l.startswith("#X obj") or l.startswith("#X msg")]
    conn_lines = [l for l in lines if l.startswith("#X connect")]
    header = [l for l in lines if l.startswith("#N canvas")]

    # find dac~ and the startup message in the template
    dac = next(i for i, l in enumerate(obj_lines) if l.endswith("dac~;"))
    start = next(i for i, l in enumerate(obj_lines) if "pd dsp 1" in l)

    # dac~ -> writesf~ (1 channel); drop the connection into dac~'s second inlet
    obj_lines[dac] = "#X obj 20 680 writesf~ 1;"
    conn_lines = [c for c in conn_lines if not c.endswith(f" {dac} 1;")]

    # startup message: turn on audio and set the three exposed controls
    x = obj_lines[start].split(" ")
    obj_lines[start] = f"#X msg {x[2]} {x[3]} \\; pd dsp 1 \\; speed {speed} \\; reso {reso} \\; vol {vol};"

    # recording chain, appended as new objects
    n = len(obj_lines)
    obj_lines += [
        "#X obj 780 120 del 300;",                                    # n    wait, then open+start
        f"#X msg 780 150 open -bytes 2 {out_path.replace(os.sep, '/')} \\, start;",   # n+1 (Pd eats backslashes)
        f"#X obj 780 190 del {int(seconds*1000)+300};",               # n+2  recording length
        "#X obj 780 220 t b b;",                                      # n+3
        "#X msg 780 250 stop;",                                       # n+4
        "#X obj 780 280 del 300;",                                    # n+5
        "#X msg 780 310 \\; pd quit;",                               # n+6
    ]
    lb = next(i for i, l in enumerate(obj_lines) if l.endswith("loadbang;"))
    conn_lines += [
        f"#X connect {lb} 0 {n} 0;",
        f"#X connect {n} 0 {n+1} 0;",
        f"#X connect {n+1} 0 {dac} 0;",
        f"#X connect {lb} 0 {n+2} 0;",
        f"#X connect {n+2} 0 {n+3} 0;",
        f"#X connect {n+3} 1 {n+4} 0;",
        f"#X connect {n+4} 0 {dac} 0;",
        f"#X connect {n+3} 0 {n+5} 0;",
        f"#X connect {n+5} 0 {n+6} 0;",
    ]

    patch = "\n".join(header + obj_lines + conn_lines) + "\n"
    p = os.path.join(tmp, "render.pd")
    open(p, "w").write(patch)
    res = subprocess.run([PD_PATH, "-nogui", "-noaudio", "-stderr", "-r", "44100", "-open", p],
                         capture_output=True, text=True, timeout=seconds * 5 + 60)
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        raise RuntimeError("Pd did not write a wav file.\nPd path: %s\nPd said:\n%s%s"
                           % (PD_PATH, res.stdout, res.stderr))
    shutil.move(out_path, final_path)
    shutil.rmtree(tmp, ignore_errors=True)
    return final_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=40)
    ap.add_argument("--reso", type=float, default=40)
    ap.add_argument("--vol", type=float, default=50)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--out", default="wind.wav")
    a = ap.parse_args()
    print("wrote", render(a.speed, a.reso, a.vol, a.seconds, a.out))
