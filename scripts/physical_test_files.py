"""The audio files of a Live demo test plan, for a physical test: each distinct sound once, as its own WAV.

Used by the "Download for physical test" button on the Live page of scripts/Complete_Test_Gui.py:

    export(plan, folder, progress) -> the folder written

A plan's rounds repeat sounds: 4 drone files x 5 speeds x 2 winds is 40 rounds, but 20 drone sounds and 2 wind
sounds. Each distinct sound is written once:

    drones/       one file per drone round setting that changes the drone's sound: file x speed x pitch x
                  level at 1 m. Exactly the signal the live test mixes in (complete_test_engine.Drones.get):
                  10 s, flown straight at the microphone, its last sample the impact.
    background/   one file per background layer setting: layer x level x its Pure Data settings (the wind's
                  speed and resonance, the cars' speed and distance). The layer's whole source at the round's
                  level, rotated to start where the live test's soundscape starts (Test.start_s), so looping it
                  from its first sample plays what the live test plays without a break. A layer that plays once
                  per round (car_once) is the one pass-by, not rotated.
    rounds.csv    every round in the plan's order, naming its drone file, where the drone starts in the round
                  and its background files: the playlist of the physical test.
    files.csv     every file: what it is, length, level (dB SPL, full scale = 110 dB SPL), peak (dBFS).

All files are 16 kHz mono 32-bit float, the live engine's own rate and units (full scale = 110 dB SPL). A peak
above 0 dBFS is kept as it is in float; files.csv shows it, so a playback chain that converts to PCM can be
turned down first. Speaker and room are not in the files: what the microphone hears in the physical test
is not the plan's dB SPL until the playback level is calibrated.
"""
from __future__ import annotations

import csv
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import complete_test_engine as E

SR = E.SR


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def _num(x: float) -> str:
    return f"{float(x):g}".replace("-", "m")


def drone_name(key: str, level_db) -> str:
    p, v, pitch = E.split_modified(key)
    name = f"drone_{_safe(Path(p).stem)}_{_num(v)}kmh"
    if pitch != 1.0:
        name += f"_pitch{_num(pitch)}"
    return name + (f"_{_num(level_db)}dB" if level_db is not None else "") + ".wav"


def layer_settings(name: str, pd: dict) -> dict:
    """The Pure Data settings of round r that belong to layer `name` (the wind's for the wind, ...)."""
    return {k: v for k, v in pd.items() if E.PD_ROUND_KEYS[k][0] == name}


def background_name(name: str, level_db: float, settings: dict) -> str:
    extra = "".join(f"_{k}{_num(v)}" for k, v in sorted(settings.items()))
    return f"bg_{_safe(name)}_{_num(level_db)}dB{extra}.wav"


def _write(path: Path, x: np.ndarray) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, np.float32)
    sf.write(str(path), x, SR, subtype="FLOAT")
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return dict(seconds=round(len(x) / SR, 4), level_db_spl=round(E.db_spl(x), 2),
                peak_dbfs=round(20 * np.log10(peak), 2) if peak > 0 else None)


def export(plan: E.Plan, folder: Path, progress=print) -> Path:
    """Every distinct drone and background sound of `plan` as its own WAV in a new subfolder of `folder`."""
    out = Path(folder) / f"physical_test_{time.strftime('%Y%m%d_%H%M%S')}"
    t = E.Test(plan, out_dir=out)
    rounds = t.rounds
    for r in rounds:                                          # the Pure Data renders the rounds need
        t.layers_for(r)
    for i, (key, layer) in enumerate(t._pd_layers.items(), 1):
        progress(f"Pure Data: {key[0]}, combination {i} of {len(t._pd_layers)}")
        layer.signal()
    if t.start_s is None:                                     # as Test.prepare: one soundscape start per batch
        first = t.layers_for(rounds[0]) if rounds else t.layers
        longest = max((first[n].signal()[0].size for n in plan.layers), default=0) / SR
        span = max(longest - plan.length_s, 0)
        t.start_s = float(np.random.default_rng(E.dwg.WSN_SEED).integers(0, int(span) + 1)) if span else 0.0
    progress("making the drones")
    drones = E.Drones(plan, progress)

    files, drone_file, bg_file, playlist = [], {}, {}, []
    for r in rounds:
        row = dict(round=r["round"], drone_file="", drone_start_s="", drone_level_db="", speed_kmh="",
                   round_length_s=plan.length_s)
        if r["drone"] is not None:
            k = (r["drone"], r["drone_level_db"])
            if k not in drone_file:
                name = drone_name(*k)
                progress(f"writing {name}")
                info = _write(out / "drones" / name, drones.get(*k))
                p, v, pitch = E.split_modified(r["drone"])
                files.append(dict(file=f"drones/{name}", kind="drone", source=Path(p).name, speed_kmh=v, pitch=pitch,
                                  level_setting_db=r["drone_level_db"], settings="", **info))
                drone_file[k] = f"drones/{name}"
            p, v, _ = E.split_modified(r["drone"])
            row.update(drone_file=drone_file[k], drone_start_s=r["start_s"], drone_level_db=r["drone_level_db"],
                       speed_kmh=v)
        layers = t.layers_for(r)
        bgs = []
        for name, lv in r["background"].items():
            st = layer_settings(name, r["pd"])
            k = (name, lv, tuple(sorted(st.items())))
            if k not in bg_file:
                fn = background_name(name, lv, st)
                progress(f"writing {fn}")
                layer = layers[name]
                x, _ = layer.signal()
                start = 0.0 if layer.once else t.start_s
                info = _write(out / "background" / fn, layer.window(len(x), start, lv))
                files.append(dict(file=f"background/{fn}", kind="background", source=name, speed_kmh="", pitch="",
                                  level_setting_db=lv, settings=" ".join(f"{a}={b:g}" for a, b in sorted(st.items())),
                                  **info))
                bg_file[k] = f"background/{fn}"
            bgs.append(bg_file[k])
        row["background_files"] = " + ".join(bgs)
        playlist.append(row)

    with open(out / "rounds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, playlist[0].keys() if playlist else ["round"])
        w.writeheader()
        w.writerows(playlist)
    with open(out / "files.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, files[0].keys() if files else ["file"])
        w.writeheader()
        w.writerows(files)
    progress(f"{len(files)} files for {len(rounds)} rounds ({len(drone_file)} drone, {len(bg_file)} background) in {out}")
    return out
