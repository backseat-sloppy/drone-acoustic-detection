"""The war soundscape as point sources: each track of WarSoundScape/WarSoundScape.RPP rendered on its own, mono,
each panned to its own direction (a voice from two speakers is not one place; Marius, 2026-11-23).

Where they are (`place`): new at random for every round (live test) or segment (REAPER projects), and moving while
they play (Marius, 2026-11-23). The whole scene is in one place (Marius, 2026-11-24: one direction, not two sides
of the array): every source at ear height, a few degrees from the scene's centre (OFFSET_DEG). It stands still
(SCENE_DEG_S = 0, 2026-11-24): no stem is recorded moving (constant level and perspective, no Doppler), and turning the
scene at 4 degrees/s took it more than three times round the listener in a 300 s segment. A rate > 0 turns it. reaper_array.moving_gains pans them (VBAP, 10 ms frames).

Renders every item of the project's unmuted tracks (position, length, source offset, fades, item and track volume;
mono = the mean of the channels) at 48 kHz over the project's 38.5 s. Not rendered: the master's kHs Reverb (an AU; the
room adds its own). Since 2026-11-24 the project's own audio is all in WarSoundScape/ (Pedro: 'shoot' = GUNSHOTS.wav,
'footsteps' = Human_Movement.wav; the old footsteps, cloth and twigs tracks muted). A track whose file is still missing
is left out, except 'shoot', which then plays WarSoundScape/Bombs' renders at its items' times (entry 268).

Writes artifacts/war_stems/: <source>.wav per point source, mix.wav (their sum: the live test's war layer for REAPER
runs, so its level and its parts match what the speakers play) and stems.csv (source, group, files, items, level).

    uv run python scripts/war_stems.py
"""
from __future__ import annotations

import csv
import functools
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
WAR = ROOT / "WarSoundScape"
RPP = WAR / "WarSoundScape.RPP"
OUT = ROOT / "artifacts/war_stems"
SR = 48000
LENGTH_S = 38.5                                              # the project's length (WarSoundScape.wav)
# Degrees from the scene's centre; a source not listed sits at the centre. Within +-12: one direction, one place.
OFFSET_DEG = {"Talk": 0.0, "cloth": 6.0, "objects": -6.0, "reload": 12.0, "footsteps": 3.0, "twigs and ground": -12.0,
              "shoot": -9.0}
SCENE_DEG_S = 0.0                                            # degrees/s at most; 0 = still (see the docstring)
BOMBS = ["bomb_far", "bomb_close", "bomb_far"]               # stand-ins for the shoot track's three items, in order


def _find(f: str) -> Path | None:
    """A project FILE path on this machine: as written (relative to the project), or by name in Media/ or beside it."""
    for p in (WAR / f, WAR / "Media" / Path(f).name, WAR / Path(f).name):
        if p.exists():
            return p
    return None


def _mono(p: Path) -> np.ndarray:
    x, sr = sf.read(p, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    return resample_poly(x, SR, sr) if sr != SR else x


def items() -> list[dict]:
    """Every item of the project: track, its volume, position, length, offset, fades, volume, file."""
    out = []
    for tr in re.split(r"\n  <TRACK", RPP.read_text())[1:]:
        name = re.search(r"\n    NAME (.*)", tr).group(1).strip('"')
        tvol = float(re.search(r"\n    VOLPAN (\S+)", tr).group(1))
        if re.search(r"\n    MUTESOLO (\S+)", tr).group(1) == "1":   # a muted track is not in the mix
            continue
        for it in re.findall(r"\n    <ITEM\n(.*?)\n    >", tr, re.S):
            g = lambda k: re.search(rf"\n?\s*{k} (.*)", it).group(1).split()  # noqa: E731
            if g("MUTE")[0] == "1":
                continue
            out.append(dict(track=name, track_vol=tvol, pos=float(g("POSITION")[0]), len=float(g("LENGTH")[0]),
                            soffs=float(g("SOFFS")[0]), fade_in=float(g("FADEIN")[1]), fade_out=float(g("FADEOUT")[1]),
                            vol=float(g("VOLPAN")[0]), file=re.search(r'FILE "([^"]*)"', it).group(1)))
    return out


def build() -> dict[str, np.ndarray]:
    """{source: mono 48 kHz signal, LENGTH_S long}; also writes OUT."""
    n = int(round(LENGTH_S * SR))
    stems, rows, cache = {}, {}, {}
    k_shoot = 0
    for it in items():
        src, p = it["track"], _find(it["file"])
        if p is None and src == "shoot":                     # the fireworks stem is missing: a bomb render instead
            src, p = BOMBS[k_shoot % len(BOMBS)], WAR / "Bombs" / f"{BOMBS[k_shoot % len(BOMBS)]}.wav"
            k_shoot += 1
            it = dict(it, soffs=0.0)
        if p is None:
            rows.setdefault(src, dict(source=src, files="MISSING: " + Path(it["file"]).name, items=0))
            continue
        x = cache.setdefault(p, _mono(p))
        a, m = int(round(it["soffs"] * SR)), int(round(it["len"] * SR))
        y = x[a:a + m].copy()                                # an item past its source's end is silent there
        fi, fo = min(int(it["fade_in"] * SR), len(y)), min(int(it["fade_out"] * SR), len(y))   # linear, as REAPER's default
        if fi:
            y[:fi] *= np.linspace(0, 1, fi)
        if fo:
            y[len(y) - fo:] *= np.linspace(1, 0, fo)
        j = int(round(it["pos"] * SR))
        s = stems.setdefault(src, np.zeros(n))
        y = y[: max(0, n - j)] * it["vol"] * it["track_vol"]
        s[j:j + len(y)] += y
        r = rows.setdefault(src, dict(source=src, files=set(), items=0))
        r["files"].add(str(p.relative_to(ROOT)))
        r["items"] += 1
    if k_shoot:                                              # the bombs at the missing fireworks' level: the mixdown
        mixdown = _mono(WAR / "WarSoundScape.wav")[:n]       # minus every rendered stem, over the shoot items
        win = np.zeros(n, bool)
        for it in (i for i in items() if i["track"] == "shoot"):
            win[int(it["pos"] * SR):int((it["pos"] + it["len"]) * SR)] = True
        bombs = [s for s in stems if s.startswith("bomb_")]
        rest = sum(v for k, v in stems.items() if k not in bombs)[:len(mixdown)]
        b = sum(stems[k] for k in bombs)[:len(mixdown)]
        g = np.sqrt(np.mean((mixdown - rest)[win[:len(mixdown)]] ** 2) / np.mean(b[win[:len(mixdown)]] ** 2))
        for k in bombs:
            stems[k] *= g
            rows[k]["note"] = f"stand-in for the missing fireworks, {20 * np.log10(g):+.1f} dB to their level in WarSoundScape.wav"
    OUT.mkdir(parents=True, exist_ok=True)
    for src, s in stems.items():
        sf.write(OUT / f"{src}.wav", s.astype(np.float32), SR, subtype="FLOAT")
    mix = np.sum(list(stems.values()), axis=0)
    sf.write(OUT / "mix.wav", mix.astype(np.float32), SR, subtype="FLOAT")
    with open(OUT / "stems.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["source", "offset_deg", "items", "rms_db_fs", "files", "note"])
        w.writeheader()
        for src, r in rows.items():
            files = r["files"] if isinstance(r["files"], str) else "; ".join(sorted(r["files"]))
            rm = f"{20 * np.log10(np.sqrt(np.mean(stems[src] ** 2)) + 1e-12):.2f}" if src in stems else ""
            w.writerow(dict(source=src, offset_deg=OFFSET_DEG.get(src, 0.0), items=r["items"], rms_db_fs=rm,
                            note=r.get("note", ""),
                            files=files))
    return stems


def load(sr: int) -> dict[str, np.ndarray]:
    """The stems at sample rate sr (built once into OUT): resampled exactly as the live test resamples mix.wav. Read
    again whenever mix.wav changes, so a long-running GUI never pairs a new mix with old stems."""
    if not (OUT / "mix.wav").exists():
        build()
    return _load(sr, (OUT / "mix.wav").stat().st_mtime_ns)


@functools.cache
def _load(sr: int, _mtime: int) -> dict[str, np.ndarray]:
    out = {}
    for row in csv.DictReader(open(OUT / "stems.csv")):
        if int(row["items"]):
            x = sf.read(OUT / f"{row['source']}.wav", dtype="float64")[0]
            out[row["source"]] = resample_poly(x, sr, SR) if sr != SR else x
    return out


def place(rng, sources) -> dict[str, dict]:
    """Where each source is for one round or segment: {source: azimuth_deg (at its start), elevation_deg, deg_s}."""
    az, rate = float(rng.uniform(0, 360)), float(rng.uniform(-SCENE_DEG_S, SCENE_DEG_S))
    return {s: dict(azimuth_deg=round((az + OFFSET_DEG.get(s, 0.0)) % 360, 3), elevation_deg=0.0, deg_s=round(rate, 4))
            for s in sorted(sources)}


if __name__ == "__main__":
    s = build()
    mix = sf.read(OUT / "mix.wav", dtype="float64")[0]
    assert np.allclose(sum(s.values()), mix, atol=1e-6), "mix.wav is not the sum of the stems"
    print(open(OUT / "stems.csv").read())
