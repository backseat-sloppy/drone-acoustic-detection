"""Report entry 242: checks behind the review of the three tests. No model is run and no ITU ARIS audio is read.

    uv run python scripts/review_tests.py

1. The war soundscape layer at the test level (90 dB SPL): window levels, peaks, silent gaps, and the
   per-window SNR each drone level of test 1 gives against the 20 s excerpt every logged run used.
2. dev_test.py's war stretches, which were picked on the 600 s war track, taken on today's 38.5 s file.
3. What the 1 s buffer after the drone's end labels as drone, for 1 s and 2 s windows.
4. approach() (test 2) against spherical spreading + ISO 9613-1, and its Doppler factor (asserted).
5. The soundscape start of every run folder on this machine.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import complete_test_engine as E  # noqa: E402
import dev_test  # noqa: E402

from dad import propagation as prop  # noqa: E402
from dad.wavio import Decisions, Truth, window_labels  # noqa: E402

SR, HOP = E.SR, 0.125
EXCERPT = (8.0, 20.0)                            # soundscape start and round length of the logged runs


def window_db(x: np.ndarray, win: float = 1.0) -> np.ndarray:
    """dB SPL of every `win` s window on the detectors' 0.125 s hop."""
    c = np.concatenate([[0.0], np.cumsum(x ** 2)])
    w, s = int(win * SR), np.arange(0, len(x) - int(win * SR) + 1, int(HOP * SR))
    with np.errstate(divide="ignore"):
        return E.FULL_SCALE_DB + 10 * np.log10((c[s + w] - c[s]) / w)


def war() -> None:
    layer = E.make_layers()["war soundscape"]
    x = layer.window(len(layer.signal()[0]), 0.0, 90.0)
    ex = layer.window(int(EXCERPT[1] * SR), EXCERPT[0], 90.0)
    q = lambda v: " / ".join(f"{np.percentile(v, p):.1f}" for p in (0, 10, 50, 90, 100))   # noqa: E731
    e = np.sort(x ** 2)[::-1]
    print(f"1. {E.WAR_SOUNDSCAPE.name}: {len(x) / SR:.1f} s, set to {E.db_spl(x):.1f} dB SPL; peak "
          f"{20 * np.log10(np.abs(x).max()):+.1f} dBFS, crest {20 * np.log10(np.abs(x).max() / np.sqrt(np.mean(x ** 2))):.1f} dB; "
          f"loudest 1 % / 5 % of samples hold {e[: len(e) // 100].sum() / e.sum():.0%} / {e[: len(e) // 20].sum() / e.sum():.0%} of the energy")
    w = window_db(ex)
    print(f"   excerpt {EXCERPT[0]:g}-{sum(EXCERPT):g} s: {E.db_spl(ex):.1f} dB SPL, samples over 0 dBFS {int((np.abs(ex) > 1).sum())}; "
          f"1 s windows min/p10/median/p90/max {q(w)} dB SPL")
    z = np.abs(layer.signal()[0]) < 1e-7
    d = np.diff(np.concatenate([[0], z.astype(int), [0]]))
    gaps = [(a / SR, b / SR) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a > 0.25 * SR]
    print("   digital silence over 0.25 s: " + ", ".join(f"{a:.2f}-{b:.2f} s" for a, b in gaps))
    for lv in (100, 90, 80, 70, 60):
        snr = lv - w
        print(f"   drone {lv} dB SPL (set SNR {lv - 90:+d} dB): per-window SNR min/median/max "
              f"{snr.min():+.1f} / {np.median(snr):+.1f} / {snr.max():+.1f} dB")


def dev_stretches() -> None:
    n = len(E.make_layers()["war soundscape"].signal()[0])
    at = [round(s * SR % n / SR, 2) for s in dev_test.WAR_STARTS]
    print(f"2. dev_test.WAR_STARTS {dev_test.WAR_STARTS} on the {n / SR:.1f} s file start at {at} s (10 s each); "
          f"the logged runs use {EXCERPT[0]:g}-{sum(EXCERPT):g} s")


def buffer() -> None:
    for win in (1.0, 2.0):
        t_end = np.arange(win, 20 + 1e-9, HOP)
        d, tr = Decisions(t_end, np.zeros(len(t_end)), None), Truth([dict(start_s=2.0, end_s=12.0)])
        before, after = window_labels(d, tr, win), window_labels(d, E.with_buffer(tr, 1.0), win)
        added = (after == 1) & (before != 1)
        share = np.clip((12.0 - (t_end - win)) / win, 0, 1)[added]
        print(f"3. {win:g} s window, 10 s drone, 1 s buffer: drone windows {int((before == 1).sum())} -> {int((after == 1).sum())}; "
              f"the added ones hold {100 * share.max():.1f} % down to {100 * share.min():.1f} % drone audio")


def approach() -> None:
    t = np.arange(int(E.DRONE_S * SR)) / SR
    worst = 0.0
    for f0 in (200.0, 1000.0, 4000.0):
        x = np.sqrt(2) * np.sin(2 * np.pi * f0 * t)                    # RMS 1 = the level at 1 m
        for v in (50.0, 100.0, 200.0):
            y, _ = E.approach(x, v, 1.0, doppler=False)
            r = E.approach_distance(len(x), v, 1.0)
            for before in (7.0, 5.0, 3.0):
                i, n = int((E.DRONE_S - before) * SR), int(0.25 * SR)
                got = 20 * np.log10(np.sqrt(np.mean(y[i:i + n] ** 2)))
                rr = r[i:i + n]
                want = 10 * np.log10(np.mean(10 ** (-(20 * np.log10(rr) + prop.air_absorption_db_per_m(f0) * (rr - 1)) / 10)))
                worst = max(worst, abs(got - want))
    print(f"4. approach(), no Doppler: largest level error against spreading + ISO 9613-1 at 3-7 s before impact, "
          f"200 Hz-4 kHz, 50-200 km/h: {worst:.3f} dB")
    assert worst < 0.05
    x = np.sqrt(2) * np.sin(2 * np.pi * 1000.0 * t)
    for v in (50.0, 100.0, 200.0):
        y, heard = E.approach(x, v, 1.0, doppler=True)
        seg = y[int(4 * SR):int(6 * SR)]
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 8 * len(seg)))
        f = np.fft.rfftfreq(8 * len(seg), 1 / SR)[spec.argmax()]
        theory = 1000.0 / (1 - v / 3.6 / prop.speed_of_sound())
        print(f"   Doppler {v:g} km/h: a 1000 Hz tone is heard at {f:.1f} Hz (c / (c - v): {theory:.1f}); first sound {heard:.2f} s")
        assert abs(f - theory) < 1.0


def runs() -> None:
    for c in sorted(E.OUT_ROOT.glob("*/config.json")):
        d = json.loads(c.read_text(encoding="utf-8"))
        p = d["plan"]
        print(f"5. {c.parent.name}: {p['kind']}, soundscape start {d.get('soundscape_start_s')} s, length {p['length_s']:g} s, "
              f"buffer {p.get('buffer_after_s')}, Doppler {p.get('approach', {}).get('doppler')}, levels {p.get('drone_levels')}")


if __name__ == "__main__":
    war()
    dev_stretches()
    buffer()
    approach()
    runs()
