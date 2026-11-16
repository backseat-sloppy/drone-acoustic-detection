#!/usr/bin/env python3
"""
helicopter_flyby.py  -  a helicopter flying past you, made in Pure Data, controlled from Python.

Python writes a Pure Data patch (vanilla objects only, tested with Pd 0.57.0), starts Pd without a window,
sends it the fly-by settings, and Pd records the sound into a wav file.  Pd runs in real time, so a 10 s file
takes about 10 s + 1-3 s warm-up to render.

Quick start (Pd must be installed; on a Mac pass --pd-bin, see below):
    python3 helicopter_flyby.py --speed 200 --distance 100            # 200 km/h, passes 100 m away
    python3 helicopter_flyby.py --speed 0 --distance 60               # hovering 60 m away
    python3 helicopter_flyby.py --grid --speeds 80 200 --distances 50 300   # 4 files at once
    python3 helicopter_flyby.py --write-patch                         # just write helicopter_flyby.pd to play with in Pd

Needs: python3, numpy, scipy.   Pd 0.57:  Mac  /Applications/Pd-0.57-0.app/Contents/Resources/bin/pd
                                          Linux pd      Windows  "C:/Program Files/Pd/bin/pd.com"
"""
import argparse, concurrent.futures as cf, json, math, os, shutil, subprocess, sys
import numpy as np
from scipy.io import wavfile

# =====================================================================================================
#  1) THE VARIABLES YOU WILL WANT TO CHANGE  (command-line options override these)
# =====================================================================================================
SPEED_KMH   = 200.0     # helicopter ground speed in km/h  (0 = hovering)
DISTANCE_M  = 100.0     # closest distance to you, in metres (straight-line distance at the moment it passes)
ALTITUDE_M  = None      # height of the helicopter in metres; None = half of the distance (must be <= distance)
DURATION_S  = 10.0      # length of the wav file
PASS_TIME_S = None      # second of the file when it passes closest; None = the middle
PRESET      = "A"       # "A" = 26 Hz blade rate, strong slap (like reference A); "C" = 19.5 Hz, smoother, with turbine whine
PD_BIN      = os.environ.get("PD_BIN", "pd")   # path to the Pd program
# =====================================================================================================

PRESETS = {
    "A": {
        "thump": 1.99535,
        "thump_decay": 8.13886,
        "thump_lp": 633.86141,
        "crack": 0.08513,
        "crack_decay": 46.43029,
        "crack_f": 1799.64375,
        "crack_q": 0.85656,
        "rnoise": 3.0,
        "rnoise_f": 1667.79463,
        "tail": 0.93634,
        "tail_ratio": 3.0,
        "jet": 0.04164,
        "jet_f": 1482.23058,
        "whine": 0.0,
        "rumble": 0.01426,
        "jitter": 0.0,
        "bladevar": 0.22584,
        "eq": [
            -4.119,
            13.588,
            9.397,
            18.0,
            4.993,
            2.379,
            2.723,
            -9.181,
            -18.0
        ],
        "level": 0.01011,
        "gear": 0.0,
        "brate": 26.0
    },
    "C": {
        "thump": 1.32861,
        "thump_decay": 21.3612,
        "thump_lp": 473.26046,
        "crack": 0.24121,
        "crack_decay": 102.98656,
        "crack_f": 1604.09873,
        "crack_q": 0.83314,
        "rnoise": 0.20818,
        "rnoise_f": 1622.05663,
        "tail": 0.44614,
        "tail_ratio": 5.77674,
        "jet": 0.06362,
        "jet_f": 951.69156,
        "whine": 0.00299,
        "rumble": 1.36563,
        "jitter": 8.0,
        "bladevar": 0.16521,
        "eq": [
            -14.6,
            3.799,
            5.288,
            13.92,
            7.357,
            3.636,
            2.072,
            5.054,
            -2.796
        ],
        "level": 0.08265,
        "whine_hz": 8772.0,
        "gear": 0.0,
        "brate": 19.5
    }
}

# =====================================================================================================
#  PRESETS  (found by fitting the patch to your reference recordings)
# =====================================================================================================

# ---- Pd patch generator ----
import re

def esc(t):
    t = t.replace("\\", "\\\\") if False else t
    return t.replace("$", "\\$").replace(",", " \\,").replace(";", " \\;")

class Canvas:
    def __init__(self):
        self.lines = []; self.n = 0; self.conns = []
    def _add(self, kind, x, y, text):
        self.lines.append(f"#X {kind} {int(x)} {int(y)} {text};"); self.n += 1; return self.n - 1
    def obj(self, text, x=10, y=10): return self._add("obj", x, y, esc(text))
    def msg(self, text, x=10, y=10): return self._add("msg", x, y, esc(text))
    def raw(self, line): self.lines.append(line); self.n += 1; return self.n - 1
    def text(self, text, x=10, y=10):
        self.lines.append(f"#X text {int(x)} {int(y)} {esc(text)};"); self.n += 1; return self.n - 1
    def c(self, a, b, o=0, i=0): self.conns.append(f"#X connect {a} {o} {b} {i};")
    def chain(self, texts, x, y, dy=28, first=None, kind="obj"):
        """stack objects vertically and connect them outlet0 -> inlet0; returns indices"""
        ids = []
        for k, t in enumerate(texts):
            ids.append(self.obj(t, x, y + k * dy) if not t.startswith("msg:") else self.msg(t[4:], x, y + k * dy))
        for a, b in zip(ids[:-1], ids[1:]): self.c(a, b)
        return ids
    def sub(self, name, child, x, y):
        """add a finished child canvas as a [pd name] subpatch"""
        body = [f"#N canvas 0 50 640 480 {name} 0;"] + child.lines + child.conns + [f"#X restore {int(x)} {int(y)} pd {name};"]
        self.lines.extend(body); self.n += 1; return self.n - 1
    def text_out(self, w=1100, h=720):
        return "\n".join([f"#N canvas 40 40 {w} {h} 12;"] + self.lines + self.conns) + "\n"

SR = 44100
EQ_FREQS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 10000]
EQ_Q = 1.2

DEFAULT_K = dict(
    seed=0,
    thump=1.0, thump_decay=14.0, thump_lp=700.0,
    crack=0.5, crack_decay=60.0, crack_f=1400.0, crack_q=0.8,
    rnoise=0.4, rnoise_f=2500.0,
    tail=0.3, tail_ratio=4.3, tail_decay=8.0, tail_lp=500.0,
    jet=0.2, jet_f=2500.0, jet_q=0.5,
    whine=0.02, whine_hz=8800.0, gear=0.01, gear_hz=2650.0,
    rumble=0.3,
    jitter=2.0, bladevar=0.10, slapk=0.9,
    level=1.0, amb=0.0015, brate=22.0,
    eq=[0.0] * len(EQ_FREQS),
)

def peaking(fc, gain_db, q, sr=SR):
    """RBJ peaking EQ -> Pd biquad~ coefficients (fb1 fb2 ff1 ff2 ff3)"""
    A = 10 ** (gain_db / 40); w0 = 2 * math.pi * fc / sr; al = math.sin(w0) / (2 * q); cw = math.cos(w0)
    b0, b1, b2 = 1 + al * A, -2 * cw, 1 - al * A
    a0, a1, a2 = 1 + al / A, -2 * cw, 1 - al / A
    return (-a1 / a0, -a2 / a0, b0 / a0, b1 / a0, b2 / a0)

def num(v): return f"{v:.6g}"

def build(K=None, path="helicopter_flyby.pd", score=None):
    K = dict(DEFAULT_K, **(K or {}))
    sd = int(K["seed"]) * 100
    top = Canvas()

    # ------------------------------------------------------------------ clock
    c = Canvas()
    a = c.obj("r heli-go", 10, 10); b = c.msg("0, 600 600000", 10, 40); l = c.obj("line~", 10, 70); s = c.obj("s~ heli-t", 10, 100)
    c.c(a, b); c.c(b, l); c.c(l, s)
    clock = top.sub("clock", c, 10, 300)

    # --------------------------------------------------------------- geometry (emission time)
    g = Canvas()
    te = g.obj("r~ heli-t", 10, 10)
    rv = g.obj("r heli-speed", 200, 10); rtp = g.obj("r heli-tpass", 300, 10); rd = g.obj("r heli-dist", 400, 10)
    re_ = g.obj("expr~ sqrt($f2*$f2*($v1-$f3)*($v1-$f3)+$f4*$f4)", 10, 60)
    g.c(te, re_); g.c(rv, re_, 0, 1); g.c(rtp, re_, 0, 2); g.c(rd, re_, 0, 3)
    ap = g.obj("expr~ -$f2*($v1-$f3)/sqrt($f2*$f2*($v1-$f3)*($v1-$f3)+$f4*$f4)", 400, 60)
    g.c(te, ap); g.c(rv, ap, 0, 1); g.c(rtp, ap, 0, 2); g.c(rd, ap, 0, 3)
    sa = g.obj("s~ heli-appr", 400, 100); g.c(ap, sa)
    rdr = g.obj("r heli-dref", 600, 10)
    gain = g.obj("expr~ $f4/max($v1\\,1)/pow(1-$f2*0.0029155*$v3\\,2)", 10, 110)
    g.c(re_, gain); g.c(rv, gain, 0, 1); g.c(ap, gain, 0, 2); g.c(rdr, gain, 0, 3)
    sg = g.obj("s~ heli-gain", 10, 150); g.c(gain, sg)
    fc = g.obj("expr~ min(1000*pow(681/max($v1\\,3)\\,0.7143)\\,20000)", 200, 110)
    g.c(re_, fc)
    sf = g.obj("s~ heli-fc", 200, 150); g.c(fc, sf)
    geom = top.sub("geometry", g, 10, 340)

    # ------------------------------------------------------------------ main rotor
    r = Canvas()
    rb = r.obj("r heli-brate", 10, 10); dv = r.obj("/ 4", 10, 40); rn = r.obj("r heli-nblades", 80, 10); r.c(rn, dv, 0, 1)
    sg_ = r.obj("sig~", 10, 70); r.c(rb, dv); r.c(dv, sg_)
    nz = r.obj(f"noise~ -seed {sd+11}", 160, 10); lp = r.obj("lop~ 1.2", 160, 40); jg = r.obj(f"*~ {num(K['jitter'])}", 160, 70)
    p1 = r.obj("+~ 1", 160, 100); r.c(nz, lp); r.c(lp, jg); r.c(jg, p1)
    mj = r.obj("*~", 10, 130); r.c(sg_, mj); r.c(p1, mj, 0, 1)
    ph = r.obj("phasor~", 10, 160); r.c(mj, ph)
    mn = r.obj("*~ 4", 10, 190); r.c(ph, mn); r.c(rn, mn, 0, 1)
    wr = r.obj("wrap~", 10, 220); r.c(mn, wr)                      # blade phase p (0..1)
    ab = r.obj(f"expr~ 1+{num(K['bladevar'])}*cos(6.28318*$v1+1.3)+{num(K['bladevar']*0.5)}*cos(12.56637*$v1+0.4)", 300, 190); r.c(ph, ab)
    # slap boost: 1 + k*speed*max(approach,0)
    rap = r.obj("r~ heli-appr", 500, 10); rsp = r.obj("r heli-speed", 600, 10); sk = r.obj(f"* {num(K['slapk']/70)}", 600, 40)
    bo = r.obj("expr~ 1+$f2*max($v1\\,0)", 500, 80); r.c(rap, bo); r.c(rsp, sk); r.c(sk, bo, 0, 1)
    # thump
    t1 = r.obj(f"expr~ exp(-$v1*{num(K['thump_decay'])})", 10, 260); r.c(wr, t1)
    t2 = r.obj("hip~ 10", 10, 290); t3 = r.obj(f"lop~ {num(K['thump_lp'])}", 10, 320); t4 = r.obj("*~", 10, 350)
    t5 = r.obj(f"*~ {num(K['thump'])}", 10, 380); r.c(t1, t2); r.c(t2, t3); r.c(t3, t4); r.c(ab, t4, 0, 1); r.c(t4, t5)
    # crack (blade slap)
    n2 = r.obj(f"noise~ -seed {sd+21}", 160, 260); e2 = r.obj(f"expr~ exp(-$v1*{num(K['crack_decay'])})", 260, 260); r.c(wr, e2)
    m2 = r.obj("*~", 160, 290); r.c(n2, m2); r.c(e2, m2, 0, 1)
    b2 = r.obj(f"bp~ {num(K['crack_f'])} {num(K['crack_q'])}", 160, 320); r.c(m2, b2)
    a2 = r.obj("*~", 160, 350); r.c(b2, a2); r.c(ab, a2, 0, 1)
    c2 = r.obj("*~", 160, 380); r.c(a2, c2); r.c(bo, c2, 0, 1)
    c3 = r.obj(f"*~ {num(K['crack'])}", 160, 410); r.c(c2, c3)
    # rotor broadband noise, modulated at the blade-pass rate
    n3 = r.obj(f"noise~ -seed {sd+31}", 400, 260); b3 = r.obj(f"bp~ {num(K['rnoise_f'])} 0.6", 400, 290); r.c(n3, b3)
    e3 = r.obj("expr~ 0.6+0.4*cos(6.28318*$v1)", 520, 260); r.c(wr, e3)
    m3 = r.obj("*~", 400, 320); r.c(b3, m3); r.c(e3, m3, 0, 1)
    m4 = r.obj(f"*~ {num(K['rnoise'])}", 400, 350); r.c(m3, m4)
    s1 = r.obj("+~", 10, 440); s2 = r.obj("+~", 10, 470); r.c(t5, s1); r.c(c3, s1, 0, 1); r.c(s1, s2); r.c(m4, s2, 0, 1)
    o = r.obj("outlet~", 10, 500); r.c(s2, o)
    rotor = top.sub("rotor", r, 10, 10)

    # ------------------------------------------------------------------ tail rotor
    t = Canvas()
    rb = t.obj("r heli-brate", 10, 10); mu = t.obj(f"* {num(K['tail_ratio'])}", 10, 40); sg_ = t.obj("sig~", 10, 70); t.c(rb, mu); t.c(mu, sg_)
    nz = t.obj(f"noise~ -seed {sd+41}", 150, 10); lp = t.obj("lop~ 1.5", 150, 40); jg = t.obj(f"*~ {num(K['jitter']*1.5)}", 150, 70); p1 = t.obj("+~ 1", 150, 100)
    t.c(nz, lp); t.c(lp, jg); t.c(jg, p1)
    mj = t.obj("*~", 10, 130); t.c(sg_, mj); t.c(p1, mj, 0, 1)
    ph = t.obj("phasor~", 10, 160); t.c(mj, ph)
    e1 = t.obj(f"expr~ exp(-$v1*{num(K['tail_decay'])})", 10, 200); t.c(ph, e1)
    h1 = t.obj("hip~ 20", 10, 230); l1 = t.obj(f"lop~ {num(K['tail_lp'])}", 10, 260); t.c(e1, h1); t.c(h1, l1)
    n2 = t.obj(f"noise~ -seed {sd+42}", 200, 200); e2 = t.obj("expr~ exp(-$v1*25)", 300, 200); t.c(ph, e2)
    m2 = t.obj("*~", 200, 230); t.c(n2, m2); t.c(e2, m2, 0, 1); b2 = t.obj("bp~ 900 0.8", 200, 260); t.c(m2, b2)
    m3 = t.obj("*~ 0.5", 200, 290); t.c(b2, m3)
    sm = t.obj("+~", 10, 330); t.c(l1, sm); t.c(m3, sm, 0, 1)
    gg = t.obj(f"*~ {num(K['tail'])}", 10, 360); t.c(sm, gg); o = t.obj("outlet~", 10, 390); t.c(gg, o)
    tail = top.sub("tailrotor", t, 140, 10)

    # ------------------------------------------------------------------ turbine, gearbox, rumble
    u = Canvas()
    nz = u.obj(f"noise~ -seed {sd+51}", 10, 10); lp = u.obj("lop~ 3", 10, 40); gg = u.obj("*~ 0.004", 10, 70); pp = u.obj("+~ 1", 10, 100)
    u.c(nz, lp); u.c(lp, gg); u.c(gg, pp)
    m1 = u.obj(f"*~ {num(K['whine_hz'])}", 10, 130); u.c(pp, m1); o1 = u.obj("osc~", 10, 160); u.c(m1, o1)
    w1 = u.obj(f"*~ {num(K['whine'])}", 10, 190); u.c(o1, w1)
    m2 = u.obj(f"*~ {num(K['gear_hz'])}", 120, 130); u.c(pp, m2); o2 = u.obj("osc~", 120, 160); u.c(m2, o2)
    w2 = u.obj(f"*~ {num(K['gear'])}", 120, 190); u.c(o2, w2)
    n3 = u.obj(f"noise~ -seed {sd+52}", 250, 10); b3 = u.obj(f"bp~ {num(K['jet_f'])} {num(K['jet_q'])}", 250, 40); u.c(n3, b3)
    n4 = u.obj(f"noise~ -seed {sd+53}", 380, 10); l4 = u.obj("lop~ 2", 380, 40); g4 = u.obj("*~ 6", 380, 70); p4 = u.obj("+~ 1", 380, 100); u.c(n4, l4); u.c(l4, g4); u.c(g4, p4)
    mj = u.obj("*~", 250, 130); u.c(b3, mj); u.c(p4, mj, 0, 1); jg2 = u.obj(f"*~ {num(K['jet'])}", 250, 160); u.c(mj, jg2)
    n5 = u.obj(f"noise~ -seed {sd+54}", 500, 10); l5 = u.obj("lop~ 70", 500, 40); l6 = u.obj("lop~ 70", 500, 70); g5 = u.obj(f"*~ {num(K['rumble']*8)}", 500, 100)
    u.c(n5, l5); u.c(l5, l6); u.c(l6, g5)
    a1 = u.obj("+~", 10, 240); a2 = u.obj("+~", 10, 270); a3 = u.obj("+~", 10, 300)
    u.c(w1, a1); u.c(w2, a1, 0, 1); u.c(a1, a2); u.c(jg2, a2, 0, 1); u.c(a2, a3); u.c(g5, a3, 0, 1)
    o = u.obj("outlet~", 10, 330); u.c(a3, o)
    turb = top.sub("turbine", u, 270, 10)

    # ------------------------------------------------------------------ source mix + EQ (source voicing)
    m = Canvas()
    i1 = m.obj("inlet~", 10, 10); i2 = m.obj("inlet~", 110, 10); i3 = m.obj("inlet~", 210, 10)
    s1 = m.obj("+~", 10, 50); s2 = m.obj("+~", 10, 80); m.c(i1, s1); m.c(i2, s1, 0, 1); m.c(s1, s2); m.c(i3, s2, 0, 1)
    prev = s2; ys = 120
    lb = m.obj("loadbang", 300, 10)
    for k, (f, gdb) in enumerate(zip(EQ_FREQS, K["eq"])):
        bq = m.obj("biquad~", 10, ys); m.c(prev, bq)
        co = peaking(f, gdb, EQ_Q)
        ms = m.msg(" ".join(num(v) for v in co), 300, ys + 20); m.c(lb, ms); m.c(ms, bq)
        prev = bq; ys += 40
    o = m.obj("outlet~", 10, ys); m.c(prev, o)
    mix = top.sub("sourcemix", m, 400, 10)
    top.c(rotor, mix, 0, 0); top.c(tail, mix, 0, 1); top.c(turb, mix, 0, 2)

    # ------------------------------------------------------------------ propagation
    p = Canvas()
    ii = p.obj("inlet~", 10, 10)
    gs = p.obj("r~ heli-gain", 150, 10); fs = p.obj("r~ heli-fc", 300, 10)
    mg = p.obj("*~", 10, 50); p.c(ii, mg); p.c(gs, mg, 0, 1)
    l1 = p.obj("lop~", 10, 80); l2 = p.obj("lop~", 10, 110); l3 = p.obj("lop~", 10, 140)
    p.c(mg, l1); p.c(l1, l2); p.c(l2, l3)
    for lx in (l1, l2, l3): p.c(fs, lx, 0, 1)
    dw = p.obj("delwrite~ heli-air 4000", 10, 170); p.c(l3, dw)
    rt = p.obj("r~ heli-t", 400, 10); rv = p.obj("r heli-speed", 500, 10); rtp = p.obj("r heli-tpass", 600, 10); rd = p.obj("r heli-dist", 700, 10)
    tau = p.obj("expr~ (-$f2*$f2*($v1-$f3)+sqrt(117649*$f2*$f2*($v1-$f3)*($v1-$f3)+(117649-$f2*$f2)*$f4*$f4))/(117649-$f2*$f2)*1000", 400, 50)
    p.c(rt, tau); p.c(rv, tau, 0, 1); p.c(rtp, tau, 0, 2); p.c(rd, tau, 0, 3)
    d1 = p.obj("delread4~ heli-air", 400, 90); p.c(tau, d1)
    rh = p.obj("r heli-height", 600, 90); rg = p.obj("r heli-ground", 700, 90)
    tau2 = p.obj("expr~ $v1+(sqrt(pow(0.343*$v1\\,2)+6*$f2)-0.343*$v1)/0.343", 560, 130); p.c(tau, tau2); p.c(rh, tau2, 0, 1)
    d2 = p.obj("delread4~ heli-air", 560, 170); p.c(tau2, d2)
    lg = p.obj("lop~ 2500", 560, 200); p.c(d2, lg)
    g2 = p.obj("expr~ $f2*0.343*$v1/sqrt(pow(0.343*$v1\\,2)+6*$f3)", 700, 130); p.c(tau, g2); p.c(rg, g2, 0, 1); p.c(rh, g2, 0, 2)
    gm = p.obj("*~", 560, 240); p.c(lg, gm); p.c(g2, gm, 0, 1)
    sm = p.obj("+~", 400, 280); p.c(d1, sm); p.c(gm, sm, 0, 1)
    rl = p.obj("r heli-level", 560, 280); lv = p.obj("*~", 400, 320); p.c(sm, lv); p.c(rl, lv, 0, 1)
    nzb = p.obj(f"noise~ -seed {sd+61}", 700, 280); lpb = p.obj("lop~ 600", 700, 310); gb = p.obj(f"*~ {num(K['amb'])}", 700, 340); p.c(nzb, lpb); p.c(lpb, gb)
    sb = p.obj("+~", 400, 370); p.c(lv, sb); p.c(gb, sb, 0, 1)
    so = p.obj("s~ heli-out", 400, 410); p.c(sb, so)
    prop = top.sub("propagation", p, 400, 130)
    top.c(mix, prop)

    # ------------------------------------------------------------------ output / recorder (used by the python script too)
    ro = top.obj("r~ heli-out", 600, 300)
    dac = top.obj("dac~", 600, 340); top.c(ro, dac); top.c(ro, dac, 0, 1)
    ws = top.obj("writesf~ 1", 700, 340); top.c(ro, ws)
    rw = top.obj("r heli-wsf", 700, 300); top.c(rw, ws)

    # ------------------------------------------------------------------ defaults + simple controls for use inside Pd
    d = top.msg(f"; heli-nblades 4; heli-brate {K.get('brate', 22)}; heli-speed 55; heli-dist 120; heli-height 60; heli-tpass 6; heli-dref 100; heli-level {K['level']}; heli-ground 0.6", 10, 100)
    lb = top.obj("loadbang", 10, 70); top.c(lb, d)
    top.text("Fly-by helicopter (vanilla Pd). Turn DSP on, then click [go( ; sliders send heli-speed / heli-dist / heli-brate. Closest approach happens at heli-tpass seconds.", 10, 400)
    go = top.msg("; heli-go bang", 10, 440)
    top.raw("#X obj 600 40 hsl 128 15 0 100 0 0 heli-speed empty speed_m/s -2 -8 0 10 -262144 -1 -1 0 1;")
    top.raw("#X obj 600 70 hsl 128 15 10 800 0 0 heli-dist empty distance_m -2 -8 0 10 -262144 -1 -1 0 1;")
    top.raw("#X obj 600 100 hsl 128 15 8 30 0 0 heli-brate empty blade_rate_Hz -2 -8 0 10 -262144 -1 -1 0 1;")
    top.raw("#X obj 600 130 hsl 128 15 5 400 0 0 heli-height empty altitude_m -2 -8 0 10 -262144 -1 -1 0 1;")
    if score:   # used by the python script: read a timed list of messages (qlist) when the patch loads
        lq = top.obj("loadbang", 300, 520); mq = top.msg(f"read {score} , rewind , bang", 300, 550); qq = top.obj("qlist", 300, 580)
        top.c(lq, mq); top.c(mq, qq)
    open(path, "w").write(top.text_out())
    return path


# ---- offline renderer ----
C = 343.0

def tau(u, v, d):
    """exact propagation delay (s) of the sound heard at time u after the source passed its closest point"""
    c2 = C * C; v2 = v * v
    return (-v2 * u + np.sqrt(c2 * v2 * u * u + (c2 - v2) * d * d)) / (c2 - v2)

def events(p, dur, warm):
    ev = []
    ev += [(0.10, f"heli-nblades {p['nblades']}"), (0.10, f"heli-brate {p['brate']}"), (0.10, f"heli-speed {p['speed']}"),
           (0.10, f"heli-dist {p['dist']}"), (0.10, f"heli-height {p['height']}"), (0.10, f"heli-tpass {p['tpass_abs']:.4f}"),
           (0.10, f"heli-dref {p['dref']}"), (0.10, f"heli-level {p['level']}"), (0.10, f"heli-ground {p['ground']}")]
    for name, pts in (p.get("auto") or {}).items():     # optional automation: {"brate": [(sec, value),...]}
        step = 0.02
        for t in np.arange(0, dur, step):
            ev.append((0.2 + warm + t, f"heli-{name} {np.interp(t, [a for a, b in pts], [b for a, b in pts]):.5f}"))
    return ev

def render_raw(speed=55.0, dist=120.0, height=None, brate=22.0, nblades=4, dur=10.0, pass_time=None, level=1.0,
           ground=0.6, dref=100.0, K=None, workdir="pd_work", pd_bin="pd", seed=0, out=None, extra=None):
    """speed m/s, dist m (closest approach), height m (default 0.5*dist), pass_time = second of the file when the closest
    approach is heard (default middle).  Returns (sr, samples)."""
    height = 0.5 * dist if height is None else height
    pass_time = dur / 2 if pass_time is None else pass_time
    u = np.linspace(-pass_time, dur - pass_time, 200) + dist / C
    tmax = float(np.max(tau(u, speed, dist)))
    warm = math.ceil(tmax + 0.5)
    tpass_abs = warm + pass_time - dist / C
    os.makedirs(workdir, exist_ok=True)
    wav_tmp = os.path.abspath(os.path.join(workdir, "raw.wav")).replace("\\", "/")
    p = dict(nblades=nblades, brate=brate, speed=speed, dist=dist, height=height, tpass_abs=tpass_abs,
             dref=dref, level=level, ground=ground, auto=extra)
    ev = events(p, dur, warm)
    ev += [(0.15, "heli-go bang"), (0.2, f"heli-wsf open -bytes 4 {wav_tmp}"), (0.2 + warm, "heli-wsf start"),
           (0.2 + warm + dur, "heli-wsf stop"), (0.2 + warm + dur + 0.3, "pd quit"), (0.0, "pd dsp 1")]
    ev.sort(key=lambda e: e[0])
    lines, last = [], 0.0
    for t, m in ev:
        lines.append(f"{round((t - last) * 1000, 3)} {m};"); last = t
    score = os.path.abspath(os.path.join(workdir, "score.txt")).replace("\\", "/")
    open(score, "w").write("\n".join(lines) + "\n")
    K = dict(K or {}); K["seed"] = seed
    patch = os.path.abspath(os.path.join(workdir, "render.pd"))
    build(K, patch, score=score)
    if os.path.exists(wav_tmp): os.remove(wav_tmp)
    subprocess.run([pd_bin, "-nogui", "-noaudio", "-r", "44100", "-open", patch], cwd=workdir,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=dur + warm + 60)
    if not os.path.exists(wav_tmp): raise RuntimeError("Pd produced no file")
    sr, x = wavfile.read(wav_tmp)
    return sr, x.astype(np.float64)


# =====================================================================================================
#  2) USER-LEVEL FUNCTION
# =====================================================================================================
def render_flyby(speed_kmh=SPEED_KMH, distance_m=DISTANCE_M, altitude_m=ALTITUDE_M, duration=DURATION_S,
                 pass_time=PASS_TIME_S, preset=PRESET, blade_rate=None, blades=4, ground=0.6, seed=0,
                 out="helicopter_flyby.wav", pd_bin=PD_BIN, workdir=None, normalize=None):
    """Make one wav file.  Returns the path.  blade_rate (Hz) overrides the preset's blade-pass frequency."""
    K = dict(PRESETS[preset]); v = speed_kmh / 3.6
    if altitude_m is not None and altitude_m > distance_m:
        raise ValueError("altitude_m cannot be larger than distance_m")
    workdir = workdir or os.path.join(os.path.dirname(os.path.abspath(out)), "pd_work_" + os.path.splitext(os.path.basename(out))[0])
    sr, x = render_raw(speed=v, dist=distance_m, height=altitude_m, brate=blade_rate or K["brate"], nblades=blades,
                       dur=duration, pass_time=pass_time, level=K["level"], ground=ground, K=K, workdir=workdir,
                       pd_bin=pd_bin, seed=seed)
    peak = float(np.abs(x).max())
    if normalize: x = x * (normalize / peak)
    elif peak > 0.99: x = x * (0.99 / peak); print(f"  note: {out} would clip (peak {peak:.2f}); scaled down")
    wavfile.write(out, sr, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    shutil.rmtree(workdir, ignore_errors=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="Helicopter fly-by generator (Pure Data + Python)")
    ap.add_argument("--speed", type=float, default=SPEED_KMH, help="km/h")
    ap.add_argument("--distance", type=float, default=DISTANCE_M, help="closest distance, m")
    ap.add_argument("--altitude", type=float, default=ALTITUDE_M, help="m (default: half the distance)")
    ap.add_argument("--duration", type=float, default=DURATION_S)
    ap.add_argument("--pass-time", type=float, default=PASS_TIME_S)
    ap.add_argument("--preset", choices=sorted(PRESETS), default=PRESET)
    ap.add_argument("--blade-rate", type=float, default=None, help="main-rotor blade pass frequency in Hz")
    ap.add_argument("--blades", type=int, default=4); ap.add_argument("--ground", type=float, default=0.6, help="ground reflection 0..1")
    ap.add_argument("--seed", type=int, default=0, help="different seed = different random flutter")
    ap.add_argument("--out", default=None); ap.add_argument("--pd-bin", default=PD_BIN)
    ap.add_argument("--normalize", type=float, default=None, help="scale file so its peak is this (e.g. 0.9)")
    ap.add_argument("--grid", action="store_true", help="make every combination of --speeds and --distances")
    ap.add_argument("--speeds", type=float, nargs="*", default=[80, 200]); ap.add_argument("--distances", type=float, nargs="*", default=[50, 300])
    ap.add_argument("--jobs", type=int, default=4, help="parallel Pd renders in --grid mode")
    ap.add_argument("--write-patch", action="store_true", help="only write helicopter_flyby.pd (to open in Pd)")
    a = ap.parse_args()
    if a.write_patch:
        print("wrote", build(PRESETS[a.preset], "helicopter_flyby.pd")); return
    common = dict(altitude_m=a.altitude, duration=a.duration, pass_time=a.pass_time, preset=a.preset, blade_rate=a.blade_rate,
                  blades=a.blades, ground=a.ground, seed=a.seed, pd_bin=a.pd_bin, normalize=a.normalize)
    if a.grid:
        jobs = [(s, d) for s in a.speeds for d in a.distances]
        def one(sd):
            s, d = sd; o = f"flyby_{a.preset}_{s:g}kmh_{d:g}m.wav"
            render_flyby(s, d, out=o, **common); print("wrote", o); return o
        with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex: list(ex.map(one, jobs))
    else:
        o = a.out or f"flyby_{a.preset}_{a.speed:g}kmh_{a.distance:g}m.wav"
        print("rendering", o, "(real time, please wait)...")
        render_flyby(a.speed, a.distance, out=o, **common); print("wrote", o)


if __name__ == "__main__":
    main()
