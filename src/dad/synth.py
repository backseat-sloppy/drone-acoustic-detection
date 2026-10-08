"""Synthetic audio so the pipeline runs end-to-end on day one.

This is scaffolding, NOT a result. Any accuracy you measure here is fiction:
the generator and the model share assumptions, so the task is far easier than
reality. Its jobs are (a) to let you time the latency budget honestly, which
is hardware-dependent and not data-dependent, and (b) to give the shape of the
real dataset, including session structure.

Session structure is deliberate. The council was unanimous that you must split
train/test by RECORDING SESSION, never by clip. Clips from one session share
mic, placement, wind and background, so a clip-level split leaks and inflates
accuracy, which tricks you into thinking you need a bigger, slower model.

Swap this for real audio before you report anything. See README for datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT, FrontendConfig


@dataclass
class Clip:
    audio: np.ndarray
    label: int        # 1 = drone present, 0 = background/confuser
    session: str      # group key for leakage-free splitting
    kind: str
    snr_db: float


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise via spectral shaping. Stands in for wind and city rumble."""
    spec = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    f = np.arange(len(spec))
    spec /= np.sqrt(np.maximum(f, 1.0))
    x = np.fft.irfft(spec, n=n)
    return (x / (np.std(x) + 1e-12)).astype(np.float32)


def _harmonic_stack(
    n: int,
    sr: int,
    f0: float,
    n_harm: int,
    rng: np.random.Generator,
    wobble_hz: float = 0.8,
    wobble_depth: float = 0.02,
    rolloff: float = 0.75,
) -> np.ndarray:
    """Blade-pass comb with slow RPM wobble and a harmonic roll-off."""
    t = np.arange(n, dtype=np.float64) / sr
    # Slow frequency modulation: real rotors never hold constant RPM.
    inst = f0 * (1.0 + wobble_depth * np.sin(2 * np.pi * wobble_hz * t + rng.uniform(0, 6.28)))
    phase_base = 2 * np.pi * np.cumsum(inst) / sr
    out = np.zeros(n, dtype=np.float64)
    for k in range(1, n_harm + 1):
        amp = rolloff ** (k - 1)
        out += amp * np.sin(k * phase_base + rng.uniform(0, 6.28))
    return (out / (np.std(out) + 1e-12)).astype(np.float32)


def _esc_whine(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Narrowband 4-7 kHz component. This is exactly the content you throw away
    if you sample at 8 kHz instead of 16 kHz."""
    t = np.arange(n, dtype=np.float64) / sr
    f = rng.uniform(4200.0, 6800.0)
    x = np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.3, 1.5) * t)
    return ((x * env) / (np.std(x * env) + 1e-12)).astype(np.float32)


def _mix_at_snr(sig: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    ps, pn = float(np.mean(sig ** 2)), float(np.mean(noise ** 2)) + 1e-12
    gain = np.sqrt(ps / (pn * (10.0 ** (snr_db / 10.0)))) if ps > 0 else 1.0
    out = sig + gain * noise
    return (out / (np.max(np.abs(out)) + 1e-9) * 0.9).astype(np.float32)


def make_session(
    session: str,
    n_clips: int,
    cfg: FrontendConfig = DEFAULT,
    seconds: float = 2.0,
    seed: int = 0,
) -> list[Clip]:
    """One recording session: a fixed mic character and background, varied
    clips. Positives and negatives both present."""
    rng = np.random.default_rng(seed)
    n = int(seconds * cfg.sample_rate)
    clips: list[Clip] = []

    # Session-level nuisance factors, constant within a session. This is what
    # makes a clip-level split leak.
    session_tilt = rng.uniform(0.5, 1.5)
    session_f0 = rng.uniform(110.0, 280.0)

    for i in range(n_clips):
        bg = _pink(n, rng) * session_tilt
        # Deliberately hard SNR range. A generous range makes every model look
        # perfect and the comparison stops discriminating, which is worse than
        # useless because it hides the thing you are trying to measure.
        snr = float(rng.uniform(-12.0, 4.0))

        if i % 2 == 0:
            sig = _harmonic_stack(n, cfg.sample_rate, session_f0 * rng.uniform(0.9, 1.1),
                                  n_harm=8, rng=rng)
            # ESC whine is present but weak, and not on every clip. Making it
            # a reliable giveaway would hand the classifier a free shortcut.
            if rng.random() < 0.7:
                sig = sig + rng.uniform(0.08, 0.25) * _esc_whine(n, cfg.sample_rate, rng)
            clips.append(Clip(_mix_at_snr(sig, bg, snr), 1, session, "drone", snr))
        else:
            # Confusers, not just noise. A gate or classifier that only ever
            # sees pink noise as the negative class will look great here and
            # fail outdoors on lawn equipment and helicopters.
            kind = str(rng.choice(["wind", "mower", "pink", "machine"]))
            if kind == "mower":
                # Harmonic like a drone, but lower fundamental and much
                # steadier. This is what defeats a naive comb detector via the
                # subharmonic/octave error. See gate.py.
                sig = _harmonic_stack(n, cfg.sample_rate, rng.uniform(45.0, 95.0),
                                      n_harm=10, rng=rng, wobble_depth=0.004)
            elif kind == "machine":
                # The genuinely hard one: harmonic content INSIDE the drone
                # band with mild wobble, plus a little high-frequency whine.
                # Think an air-conditioning unit or a pump. The tier-0 gate
                # cannot reject this on f0 alone, which is exactly why a
                # second tier exists.
                sig = _harmonic_stack(n, cfg.sample_rate, rng.uniform(105.0, 290.0),
                                      n_harm=5, rng=rng, wobble_depth=0.006,
                                      rolloff=0.55)
                if rng.random() < 0.4:
                    sig = sig + 0.1 * _esc_whine(n, cfg.sample_rate, rng)
            elif kind == "wind":
                sig = _pink(n, rng)
                sig = np.convolve(sig, np.ones(64) / 64, mode="same").astype(np.float32)
            else:
                sig = _pink(n, rng)
            clips.append(Clip(_mix_at_snr(sig, bg, snr), 0, session, kind, snr))

    return clips


def make_dataset(
    n_sessions: int = 8,
    clips_per_session: int = 24,
    cfg: FrontendConfig = DEFAULT,
    seed: int = 0,
) -> list[Clip]:
    out: list[Clip] = []
    for s in range(n_sessions):
        out.extend(make_session(f"sess{s:02d}", clips_per_session, cfg, seed=seed + 1000 * s))
    return out


def session_split(
    clips: list[Clip], test_sessions: int = 2, seed: int = 0
) -> tuple[list[Clip], list[Clip]]:
    """Leakage-free split: whole sessions go to test, never individual clips."""
    sessions = sorted({c.session for c in clips})
    rng = np.random.default_rng(seed)
    held = set(rng.choice(sessions, size=min(test_sessions, len(sessions)), replace=False))
    train = [c for c in clips if c.session not in held]
    test = [c for c in clips if c.session in held]
    return train, test
