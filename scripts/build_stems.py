"""Clean, level-normalised mono stems by class (supervisor notes S3 and S4).

Four groups, each a set of classes:

    drone       Glasgow drone-authentication set: each drone hovering at 1 m,
                one recording per drone (24 drones), indoors.
    background  steady ambience: rain, stream, insects, traffic, crowd
    distracter  events that can be mistaken for, or mask, a drone: gunfire,
                explosion, car, truck, motorcycle, aircraft, dog, shout, birds,
                siren, idling engine, power tool, thunder (EXCLUDE drops clips
                whose label does not fit their content)
    wind        wind
    war         sounds of a war zone, for the iteration-7 war track: small arms,
                machine guns, mortar and artillery, far explosions, tanks, jets,
                air-raid siren, shouting (PICKED)

PICKED adds hand-picked clips from FSD50K dev or eval, chosen by title and
labels (report entry 060) for classes FSD50K has no label for (helicopter,
scooter, mower, chainsaw, ...) and for the war group. --picked builds only
those.

Non-drone classes come from FSD50K eval (CC0 or CC BY only; labels limited to
the class and its ancestors in the ontology; nothing whose title, tags or
description mention a drone), read by range request with
`scripts/fetch_env_clips.py`'s reader. Helicopters are not in FSD50K and are
not here yet.

Every stem is mono at 44.1 kHz, at most --max-seconds long (the most energetic
stretch), denoised per its group (below), and scaled so its active RMS (frames
within 20 dB of the loudest frame) is --target-dbfs, never above -1 dBFS peak.

Denoising is spectral gating (`gate`): bins that do not rise --thresh-db above
a noise profile are attenuated, at most by --floor-db, with a smoothed mask.

    drone       not denoised by default (report entry 052). The drone runs
                through every recording, so the room floor under it cannot be
                measured (a profile from the clip itself would take the steady
                drone for noise), and the set's 60 min "noise" recording is no
                stand-in: it is louder than the drones (21 dB in 50-200 Hz).
                Gating with it lowered the floor by about 1 dB and cost 1-2 dB
                of the drone's harmonics. --drone-denoise gates with it anyway.
    distracter  profile = 20th percentile of each bin over the clip: the
                clip's own steady floor, under its events. Kept only if the
                loudest frames lose at most --max-loss-db; a steady engine
                (aircraft, truck, power tool) is its own floor and fails that,
                and is then kept undenoised ("percentile->none").
    background, wind
                not denoised: the noise is the content.

Output: data/processed/stems/<group>/<class>/<source>_<id>.wav and
data/processed/stems/manifest.csv (source, licence, levels before and after,
how far the gate lowered the noise floor (10th percentile of
frame levels) and how much content it kept: a drone's harmonic peaks, or a
distracter's loudest frames).

    uv run python scripts/build_stems.py
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter
from scipy.signal import istft, resample_poly, stft

sys.path.insert(0, str(Path(__file__).parent))
import fetch_env_clips as fz  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GLASGOW = ROOT / "data" / "extracted" / "Glasgow"
OUT = ROOT / "data" / "processed" / "stems"
FS = 44100
NPERSEG, HOP = 2048, 512
DRONE_WORDS = re.compile(r"drone|uav|quadcopter|multicopter|multirotor|quadrotor", re.I)

# group -> class -> (FSD50K label, labels allowed beside it and its ancestors)
ENGINE = {"Engine", "Idling", "Accelerating_and_revving_and_vroom", "Engine_starting"}
FSD_CLASSES = {
    "background": {"rain": ("Rain", set()), "stream": ("Stream", set()), "insects": ("Insect", {"Cricket"}),
                   "traffic": ("Traffic_noise_and_roadway_noise", set()), "crowd": ("Crowd", set())},
    "distracter": {"gunfire": ("Gunshot_and_gunfire", set()), "explosion": ("Explosion", {"Boom"}),
                   "car": ("Car_passing_by", set()), "truck": ("Truck", ENGINE),
                   "motorcycle": ("Motorcycle", ENGINE), "aircraft": ("Fixed-wing_aircraft_and_airplane", ENGINE),
                   "dog": ("Bark", set()), "shout": ("Shout", set()),
                   "birds": ("Bird_vocalization_and_bird_call_and_bird_song", {"Chirp_and_tweet"}),
                   "siren": ("Siren", set()), "idling": ("Idling", {"Engine"}),
                   "power_tool": ("Power_tool", {"Drill"}), "thunder": ("Thunder", {"Thunderstorm"})},
    "wind": {"wind": ("Wind", set())},
}
# group -> class -> FSD50K (split, id), picked by title and labels (entry 060): CC0 or CC BY, no speech or
# music label, no drone words. Not yet listened to.
PICKED = {
    "war": {
        "small_arms": [("eval", "204949"), ("eval", "259758"), ("eval", "174290")],
        "machine_gun": [("dev", "417688"), ("dev", "101961"), ("dev", "160773")],
        "mortar_artillery": [("dev", "426161"), ("dev", "426165"), ("dev", "171475"), ("dev", "184275"),
                             ("dev", "187767")],
        "explosion_far": [("dev", "156500"), ("dev", "156499"), ("dev", "130974")],
        "tank": [("dev", "161886"), ("dev", "187676")],
        "jet": [("eval", "397165"), ("eval", "420560"), ("dev", "431473")],
        "air_raid_siren": [("dev", "262278")],
        "shout": [("dev", "362665"), ("dev", "187294"), ("dev", "387829")],
    },
    "distracter": {
        "helicopter": [("eval", "322179"), ("dev", "181071"), ("eval", "263458")],
        "scooter": [("dev", "3185"), ("dev", "55035"), ("dev", "166717")],
        "mower": [("dev", "97595"), ("dev", "267973"), ("dev", "156223")],
        "chainsaw": [("dev", "157611"), ("dev", "345994"), ("dev", "431739")],
        "generator": [("eval", "406110"), ("dev", "164306"), ("dev", "367175")],
        "mosquito": [("dev", "188708"), ("eval", "320723"), ("dev", "98648")],
        "buzzing_insect": [("dev", "273214"), ("dev", "233783"), ("eval", "395593")],
        "prop_plane": [("dev", "403316"), ("dev", "369482"), ("eval", "431206")],
    },
}
# Clips whose label does not fit their content, found by reading the titles (2026-10-26)
EXCLUDE = {"214675": "'Bubble wrap', labelled Explosion", "238926": "'Click-Click.wav', labelled traffic",
           "25287": "'Wails_ses2.wav', labelled Crowd",
           "72560": "'Lapwing.wav': Mads, 2026-10-29, on listening: 'That bird is so bad'"}


# ------------------------------------------------------------------ denoise

def spec(x):
    return stft(x, FS, nperseg=NPERSEG, noverlap=NPERSEG - HOP)


def gate(x, profile, thresh_db=12.0, floor_db=-20.0):
    """Spectral gate. profile: noise magnitude per frequency bin, same STFT as spec().
    12 dB above a 20th-percentile profile: noise magnitudes are Rayleigh-distributed, and at 6 dB about
    41% of pure-noise bins would pass the gate, at 12 dB about 3%."""
    _, _, Z = spec(x)
    M = np.abs(Z)
    # digital silence in a bin, or in most of the clip (profile all 0): no division by 0
    profile = np.maximum(profile, max(profile.max() * 1e-6, 1e-12))
    above = 20 * np.log10(M / (profile[:, None] * 10 ** (thresh_db / 20)) + 1e-12)
    mask = np.clip(above / 6 + 0.5, 0, 1)                 # 0 at 3 dB below the threshold, 1 at 3 dB above
    mask = uniform_filter(mask, size=(3, 5))              # 3 bins x 5 frames: no musical noise
    floor = 10 ** (floor_db / 20)
    _, y = istft(Z * (floor + (1 - floor) * mask), FS, nperseg=NPERSEG, noverlap=NPERSEG - HOP)
    return y[: len(x)], mask


def frame_db(x, n=HOP * 4):
    m = len(x) // n
    return 10 * np.log10(np.mean(x[: m * n].reshape(m, n) ** 2, axis=1) + 1e-20)


def active_rms_db(x):
    e = frame_db(x)
    return 10 * np.log10(np.mean(10 ** (e[e > e.max() - 20] / 10)))


def loudest(x, seconds):
    n = int(seconds * FS)
    if len(x) <= n:
        return x
    c = np.concatenate([[0.0], np.cumsum(x ** 2)])
    s = int(np.argmax(c[n:] - c[:-n]))
    return x[s:s + n]


def events_kept_db(before, after):
    """Change in energy of the frames within 10 dB of the loudest one (a distracter's events), in dB."""
    e0, e1 = frame_db(before), frame_db(after)
    k = min(len(e0), len(e1))
    loud = e0[:k] > e0[:k].max() - 10
    return 10 * np.log10(np.mean(10 ** (e1[:k][loud] / 10)) / np.mean(10 ** (e0[:k][loud] / 10)))


def harmonic_kept_db(before, after):
    """Change in energy in the strongest spectral peaks (the drone's harmonics), in dB."""
    P0 = np.abs(np.fft.rfft(before * np.hanning(len(before)))) ** 2
    P1 = np.abs(np.fft.rfft(after * np.hanning(len(after)))) ** 2
    peaks = P0 > np.percentile(P0, 99.5)
    return 10 * np.log10(P1[peaks].sum() / P0[peaks].sum())


def finish(x, target_dbfs):
    g = 10 ** ((target_dbfs - active_rms_db(x)) / 20)
    g = min(g, 10 ** (-1 / 20) / np.abs(x).max())        # never above -1 dBFS peak
    return (x * g).astype(np.float32), 20 * np.log10(g)


# ------------------------------------------------------------------ sources

def fsd_candidates(per_class, min_s):
    """Up to per_class clips per class: licence-clean, labels within the class's branch, no drone words."""
    gt = {r["fname"]: r["labels"].split(",") for r in csv.DictReader(fz.open_meta("FSD50K.ground_truth/eval.csv"))}
    info = json.loads(fz.open_meta("FSD50K.metadata/eval_clips_info_FSD50K.json").read())
    cd = fz.central_directory()
    sizes = {Path(k).stem: (k, v) for k, v in cd.items() if k.endswith(".wav")}
    out = []
    for group, classes in FSD_CLASSES.items():
        for cls, (label, extra) in classes.items():
            with_label = [set(l) for l in gt.values() if label in l]
            ancestors = set.intersection(*with_label) - {label}  # FSD50K always adds a label's ancestors
            rows = []
            for fname, labels in gt.items():
                meta = info[fname]
                text = " ".join([meta["title"], meta.get("description", ""), " ".join(meta.get("tags", []))])
                secs = sizes[fname][1][4] / fz.BYTES_PER_S
                if (label in labels and set(labels) <= ancestors | extra | {label} and secs >= min_s
                        and fname not in EXCLUDE
                        and any(a in meta["license"] for a in fz.ALLOWED) and not DRONE_WORDS.search(text)):
                    rows.append(dict(group=group, cls=cls, source="fsd50k_eval", id=fname, seconds=round(secs, 1),
                                     labels=",".join(labels), license=meta["license"], uploader=meta["uploader"],
                                     title=meta["title"], member=sizes[fname]))
            rows.sort(key=lambda r: -r["seconds"])
            out += rows[:per_class]
            print(f"  {group}/{cls}: {len(rows)} candidates, taking {min(per_class, len(rows))}")
    return out


def picked_clips():
    """PICKED as rows like fsd_candidates', each with the archive parts to read it from."""
    rows = []
    for split, parts in (("eval", fz.PARTS), ("dev", fz.DEV_PARTS)):
        wanted = [(g, c, i) for g, cs in PICKED.items() for c, ids in cs.items() for s, i in ids if s == split]
        gt = {r["fname"]: r["labels"] for r in csv.DictReader(fz.open_meta(f"FSD50K.ground_truth/{split}.csv"))}
        info = json.loads(fz.open_meta(f"FSD50K.metadata/{split}_clips_info_FSD50K.json").read())
        cd = {Path(k).stem: v for k, v in fz.central_directory(parts).items() if k.endswith(".wav")}
        for group, cls, fname in wanted:
            meta = info[fname]
            if not any(a in meta["license"] for a in fz.ALLOWED):
                raise SystemExit(f"{fname}: licence {meta['license']} is not CC0 or CC BY")
            rows.append(dict(group=group, cls=cls, source=f"fsd50k_{split}", id=fname, labels=gt[fname],
                             license=meta["license"], uploader=meta["uploader"], title=meta["title"],
                             member=(None, cd[fname]), parts=parts))
    return rows


def glasgow_files():
    """One hover recording at 1 m per drone (the first index), and the noise recording(s)."""
    wavs = sorted(GLASGOW.rglob("*.[wW][aA][vV]"))
    if not wavs:
        raise SystemExit(f"no WAVs under {GLASGOW}; extract ACSAC_2022_Drone_Authentication.7z there first")
    drones, noise = {}, []
    for p in wavs:
        m = re.search(r"_d(\d+)_hover_1m_", p.name)
        if m:
            drones.setdefault(int(m.group(1)), p)
        elif "noise" in p.name.lower() or "noise" in str(p.parent).lower():
            noise.append(p)
    return drones, noise


def read_mono(path_or_bytes):
    x, fs = sf.read(io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes,
                    always_2d=True)
    x = x[:, 0]
    return x if fs == FS else resample_poly(x, FS, fs)


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=3, help="FSD50K clips per class")
    ap.add_argument("--min-seconds", type=float, default=3.0)
    ap.add_argument("--max-seconds", type=float, default=10.0)
    ap.add_argument("--target-dbfs", type=float, default=-26.0, help="active RMS of every stem")
    ap.add_argument("--thresh-db", type=float, default=12.0)
    ap.add_argument("--floor-db", type=float, default=-20.0)
    ap.add_argument("--max-loss-db", type=float, default=1.0,
                    help="keep a gated stem only if it lost at most this much content")
    ap.add_argument("--skip-fsd", action="store_true")
    ap.add_argument("--skip-drones", action="store_true")
    ap.add_argument("--picked", action="store_true", help="build only the PICKED clips")
    ap.add_argument("--drone-denoise", action="store_true", help="gate drones with the Glasgow noise recording")
    a = ap.parse_args()
    rows = []

    def save(group, cls, source, sid, x, policy, profile, extra):
        x = loudest(x, a.max_seconds)
        before_db = active_rms_db(x)
        if policy == "none":
            y, removed, kept = x, 0.0, None
        else:
            y, _ = gate(x, profile, a.thresh_db, a.floor_db)
            removed = float(np.percentile(frame_db(x), 10) - np.percentile(frame_db(y), 10))  # floor lowered by
            kept = harmonic_kept_db(x, y) if group == "drone" else events_kept_db(x, y)
            if kept < -a.max_loss_db:                     # the gate ate the content (a steady engine is its
                policy, y, removed = f"{policy}->none", x, 0.0   # own percentile floor): keep the clip as it
                                                                 # was; content_kept_db then says why
        y, gain = finish(y, a.target_dbfs)
        path = OUT / group / cls / f"{source}_{sid}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, y, FS, subtype="PCM_24")
        rows.append(dict(group=group, cls=cls, source=source, id=sid, file=str(path.relative_to(ROOT)),
                         seconds=round(len(y) / FS, 2), denoise=policy,
                         noise_removed_db=round(removed, 1), content_kept_db=None if kept is None else round(kept, 2),
                         active_rms_in_dbfs=round(before_db, 1), gain_db=round(gain, 1),
                         active_rms_out_dbfs=round(active_rms_db(y), 1),
                         peak_dbfs=round(20 * np.log10(np.abs(y).max()), 1), **extra))
        print(f"  {group}/{cls}/{source}_{sid}: {policy:10} removed {removed:4.1f} dB"
              + ("" if kept is None else f", content kept {kept:+.2f} dB"))

    if a.picked:
        a.skip_drones = a.skip_fsd = True
        print("FSD50K picked:")
        for r in picked_clips():
            disk, off, csize, method, _ = r.pop("member")[1]
            x = read_mono(fz.read_member(disk, off, csize, method, r.pop("parts")))
            policy = "percentile" if r["group"] in ("distracter", "war") else "none"
            profile = np.percentile(np.abs(spec(x)[2]), 20, axis=1) if policy != "none" else None
            save(r.pop("group"), r.pop("cls"), r.pop("source"), r.pop("id"), x, policy, profile,
                 {k: r[k] for k in ("license", "uploader", "title", "labels")})

    if not a.skip_drones:
        drones, noise = glasgow_files()
        profile, policy = None, "none"
        if a.drone_denoise:
            if not noise:
                raise SystemExit("no Glasgow noise recording found, so no profile for the drone gate")
            total, frames = 0.0, 0                        # mean spectrum in 60 s chunks: one STFT of
            for p in noise:                               # the 60 min recording needs about 5 GB
                nz = read_mono(p)
                for i in range(0, len(nz), 60 * FS):
                    M = np.abs(spec(nz[i:i + 60 * FS])[2])
                    total, frames = total + M.sum(axis=1), frames + M.shape[1]
            profile, policy = total / frames, "profile"
        print(f"Glasgow: {len(drones)} drones at 1 m, denoise: {policy}")
        for d, p in sorted(drones.items()):
            save("drone", "glasgow_1m", "glasgow", f"d{d:02d}", read_mono(p), policy, profile,
                 dict(license="CC BY 4.0", uploader="Diao, Zhang, Zhao, Khamis (U. Glasgow)",
                      title=p.name, labels="drone,hover,1m"))

    if not a.skip_fsd:
        print("FSD50K eval:")
        for r in fsd_candidates(a.per_class, a.min_seconds):
            disk, off, csize, method, _ = r.pop("member")[1]
            x = read_mono(fz.read_member(disk, off, csize, method))
            policy = "percentile" if r["group"] == "distracter" else "none"
            profile = np.percentile(np.abs(spec(x)[2]), 20, axis=1) if policy != "none" else None
            save(r.pop("group"), r.pop("cls"), r.pop("source"), r.pop("id"), x, policy, profile,
                 {k: r[k] for k in ("license", "uploader", "title", "labels")})

    OUT.mkdir(parents=True, exist_ok=True)
    built = {(r["group"], r["cls"]) for r in rows}        # keep the rows of classes this run did not build
    if (OUT / "manifest.csv").exists():
        rows = [r for r in csv.DictReader(open(OUT / "manifest.csv", encoding="utf-8"))
                if (r["group"], r["cls"]) not in built] + rows
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} stems -> {OUT}")


if __name__ == "__main__":
    main()
