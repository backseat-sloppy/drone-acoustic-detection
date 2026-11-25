"""The speaker array's room as a filter per speaker, estimated from a finished array run, so that a virtual test can
hear what the microphones would have heard.

    uv run python scripts/array_room.py RUN_FOLDER [RUN_FOLDER ...]   # fit; writes artifacts/array_room/room.npz

An array run keeps what each of the 39 speakers played (reaper/round_*.wav) and what the two microphones averaged
heard (heard/round_*.wav). Per frequency the 39 transfer functions are the weighted least squares solution over many
rounds; their impulse responses are cut to 20 ms before to 120 ms after. Least squares loses what does not line up
to a fraction of a period (4-8 kHz: 4 dB), so the power per frequency is put back, and what is left at
every level is the noise floor (`power_fit`).

Rounds with an ITU ARIS drone are never used in the fit (CLAUDE.md section 5). Every fourth of the others is held out
of it, and the errors printed are on those.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "artifacts/array_room/room.npz"
SR, NFFT, HOP = 16000, 16384, 8192
PRE, POST = int(0.020 * SR), int(0.120 * SR)
RIDGE = (1e-4, 1e-3, 1e-2, 1e-1)
BANDS = ((125, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000))


def own(r: dict) -> bool:
    """A round that may be used in the fit: no drone, or one of our own recordings."""
    return r["drone"] is None or "Audio_2_raw" in str(r["drone"])


def frames(x: np.ndarray) -> np.ndarray:
    """(channels, samples) -> (bins, channels, frames), Hann."""
    w = np.hanning(NFFT).astype(np.float32)
    v = np.lib.stride_tricks.sliding_window_view(x, NFFT, axis=-1)[..., ::HOP, :] * w
    return np.moveaxis(np.fft.rfft(v, axis=-1), -1, 0).astype(np.complex64)


def to_ir(H: np.ndarray) -> np.ndarray:
    """(bins, 39) -> (39, PRE + POST) taps, the first PRE of them before time 0; faded at both ends."""
    h = np.fft.irfft(H.T, NFFT, axis=-1)
    h = np.concatenate([h[:, -PRE:], h[:, :POST]], axis=1)
    f = PRE // 2
    win = np.ones(h.shape[1])
    win[:f], win[-f:] = np.linspace(0, 1, f), np.linspace(1, 0, f)
    return (h * win).astype(np.float32)


def convolve(y: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """What the microphones hear of speaker signals y (n x 39), without noise."""
    return fftconvolve(y, ir.T, axes=0)[PRE:PRE + len(y)].sum(axis=1)


def hear(y: np.ndarray, room: dict, seed: int) -> np.ndarray:
    """What the microphones hear of speaker signals y (n x 39), with the room's noise floor."""
    n = len(y)
    f = np.fft.rfftfreq(n, 1 / SR)
    psd = np.interp(f, np.fft.rfftfreq(NFFT, 1 / SR), room["noise_psd"])        # power per Hz, one-sided
    rng = np.random.default_rng(seed)
    z = (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))) * np.sqrt(psd * SR * n / 4)
    return convolve(y, room["ir"]) + np.fft.irfft(z, n)


def as_played(y: np.ndarray) -> np.ndarray:
    """Speaker signals as reaper_live writes them for REAPER: 16 bit (the wind lies near the last bit on each speaker)."""
    b = io.BytesIO()
    sf.write(b, np.clip(y, -1, 1), SR, format="WAV", subtype="PCM_16")
    b.seek(0)
    return sf.read(b, dtype="float32")[0]


def load(path: Path = FILE) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def shift(x: np.ndarray, d: float) -> np.ndarray:
    """x delayed by d samples (a fraction too)."""
    X = np.fft.rfft(x)
    return np.fft.irfft(X * np.exp(-2j * np.pi * np.fft.rfftfreq(len(x)) * d), len(x))


def delay(ref: np.ndarray, x: np.ndarray, most: int = 160) -> float:
    """How many samples x is later than ref, to a fraction (parabola at the correlation's peak)."""
    n = 2 * len(x)
    c = np.fft.irfft(np.fft.rfft(x, n) * np.conj(np.fft.rfft(ref, n)), n)
    c = np.concatenate([c[-most:], c[:most + 1]])
    i = int(np.clip(np.argmax(c), 1, len(c) - 2))
    a, b, e = c[i - 1:i + 2]
    return i - most + 0.5 * (a - e) / (a - 2 * b + e)


def rounds_of(folders: list[Path]) -> list[tuple[Path, dict]]:
    out = []
    for d in folders:
        live = json.loads((d / "live.json").read_text(encoding="utf-8"))["rounds"]
        for k, v in live.items():
            f = d / "reaper" / f"round_{int(k):04d}.wav"
            if own(v["round"]) and f.exists():
                out.append((d, v["round"]))
    return out


def read(d: Path, r: dict, lag: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    h = sf.read(d / "heard" / f"round_{r['round']:04d}.wav", dtype="float32")[0]
    y = sf.read(d / "reaper" / f"round_{r['round']:04d}.wav", dtype="float32")[0][:len(h)]
    return y, (shift(h, -lag) if lag else h).astype(np.float32)


def solve(XX: np.ndarray, XY: np.ndarray, ridge: float) -> np.ndarray:
    tr = np.trace(XX, axis1=1, axis2=2).real[:, None, None] / XX.shape[1]
    return np.linalg.solve(XX + ridge * tr * np.eye(XX.shape[1]), XY[..., None])[..., 0]


def accumulate(rounds, lags):
    XX = np.zeros((NFFT // 2 + 1, 39, 39), np.complex128)
    XY = np.zeros((NFFT // 2 + 1, 39), np.complex128)
    for (d, r), lag in zip(rounds, lags):
        y, h = read(d, r, lag)
        X, Y = frames(y.T), frames(h[None])[:, 0]            # (bins, 39, t), (bins, t)
        p = np.abs(Y) ** 2
        k = np.ones(33) / 33                                 # the recording's level around each bin: every moment counts alike
        w = 1 / (np.apply_along_axis(np.convolve, 0, p, k, "same") + 1e-9)
        Xw = X.conj() * w[:, None]
        XX += np.matmul(Xw, X.transpose(0, 2, 1))            # sum over frames of conj(x_s) x_q
        XY += np.einsum("bst,bt->bs", Xw, Y)
    return XX, XY


def power_fit(rounds, ir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The recordings' power in 128 ms frames as gain x the filters' prediction + noise, per frequency (every moment
    counting alike): (the filters with the gain in them, the noise's power per Hz on NFFT's bins)."""
    n = 2048
    w = np.hanning(n)
    H, P = [], []
    k = np.ones(5) / 5                                       # 5 bins, 39 Hz
    for (d, r), lag in rounds:
        y, h = read(d, r, lag)
        for out, x in ((H, h), (P, convolve(y, ir))):
            v = np.lib.stride_tricks.sliding_window_view(x, n)[::n] * w
            q = np.pad(np.abs(np.fft.rfft(v, axis=1)) ** 2, ((0, 0), (2, 2)), "edge")
            out.append(np.apply_along_axis(np.convolve, 1, q, k, "valid"))
    H, P = np.concatenate(H), np.concatenate(P)
    g, N = np.ones(H.shape[1]), np.quantile(H, 0.01, axis=0)
    for _ in range(30):                                      # weights from the model, not the data: no bias to quiet frames
        W = 1 / (g * P + N) ** 2
        s0, s1, s2, t0, t1 = W.sum(0), (W * P).sum(0), (W * P * P).sum(0), (W * H).sum(0), (W * P * H).sum(0)
        det = s0 * s2 - s1 ** 2
        g, N = np.clip((s0 * t1 - s1 * t0) / det, 0.5, 16), np.maximum((s2 * t0 - s1 * t1) / det, 1e-3 * np.quantile(H, 0.01, axis=0))
    f, f0 = np.fft.rfftfreq(NFFT, 1 / SR), np.fft.rfftfreq(n, 1 / SR)
    print("power put back (dB) per band", [round(float(10 * np.log10(g[(f0 >= lo) & (f0 < hi)].mean())), 1) for lo, hi in BANDS])
    psd = np.interp(f, f0, 2 * N / (SR * np.sum(w ** 2)))
    m = ir.shape[1] * 2
    return np.fft.irfft(np.fft.rfft(ir, m) * np.interp(np.fft.rfftfreq(m, 1 / SR), f0, np.sqrt(g)), m)[:, :ir.shape[1]].astype(np.float32), psd


def band_power(x: np.ndarray) -> np.ndarray:
    """(bands, frames of 0.25 s) in dB."""
    n = len(x) // 4000 * 4000
    P = np.abs(np.fft.rfft(x[:n].reshape(-1, 4000) * np.hanning(4000), axis=1)) ** 2
    f = np.fft.rfftfreq(4000, 1 / SR)
    return np.stack([10 * np.log10(P[:, (f >= lo) & (f < hi)].sum(1) + 1e-9) for lo, hi in BANDS])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    a = ap.parse_args()
    rounds = rounds_of(a.runs)
    train, val = [x for i, x in enumerate(rounds) if i % 4], rounds[::4]
    print(f"{len(train)} rounds to fit, {len(val)} held out")
    lags = [0.0] * len(train)
    vlags = [0.0] * len(val)
    for turn in range(3):                                    # fit, realign each round to what the fit predicts, again
        XX, XY = accumulate(train, lags)
        ir = to_ir(solve(XX, XY, 1e-2))
        new = []
        for (d, r), lag in zip(train, lags):
            y, h = read(d, r, lag)
            new.append(lag + delay(convolve(y, ir), h))
        new = np.array(new) - np.median(new)                 # the room's own delay stays in the filters
        print(f"turn {turn + 1}: realigned by {np.percentile(np.abs(new - lags), [50, 95]).round(2)} samples (median, 95 %)")
        lags = list(new)
    XX, XY = accumulate(train, lags)
    best = None
    for ridge in RIDGE:
        ir = to_ir(solve(XX, XY, ridge))
        e = []
        for j, (d, r) in enumerate(val):
            y, h = read(d, r)
            p = convolve(y, ir)
            vlags[j] = delay(p, h)
            h = shift(h, -vlags[j])
            e.append(np.mean(np.abs(band_power(p) - band_power(h)), axis=1))
        e = np.mean(e, 0)
        print(f"ridge {ridge:g}: mean |level error| per band (dB, no noise added)", e.round(2))
        if best is None or e.mean() < best[0]:
            best = (e.mean(), ridge, ir)
    _, ridge, ir = best
    ir, psd = power_fit(list(zip(train, lags))[::3], ir)
    FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FILE, ir=ir, noise_psd=psd.astype(np.float32), ridge=ridge, runs=[d.name for d in a.runs],
                        held_out=[r["round"] for _, r in val])
    room = load()
    e, g, raw = [], [], []
    for (d, r), lag in zip(val, vlags):
        y, h = read(d, r, lag)
        p, H = band_power(hear(y, room, r["round"])), band_power(h)
        e.append(np.mean(np.abs(p - H), 1)), g.append(np.mean(p - H, 1)), raw.append(np.mean(np.abs(band_power(y.sum(1)) - H), 1))
    print("held out, with noise: |error|", np.mean(e, 0).round(2), "bias", np.mean(g, 0).round(2),
          "  without any room: |error|", np.mean(raw, 0).round(2), f"dB   ridge {ridge:g}   -> {FILE}")


if __name__ == "__main__":
    main()
