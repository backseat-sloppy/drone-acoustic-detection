"""Tier-0 gate: a harmonic-comb detector that runs every hop and costs well
under a millisecond.

The Architect's cascade: tune this for recall above 0.99 and let it reject the
overwhelming majority of windows, so the CNN only runs on candidates. That
buys you the average power and average latency of a classical detector while
keeping the CNN's accuracy on the windows that matter.

Free bonus the Skeptic spotted: the comb's argmax IS the blade-pass frequency,
so a passing gate hands you an RPM estimate at no extra cost.

Three things had to be got right, and the first two versions of this file got
them wrong in instructive ways. Keeping the record here because the failures
are more informative than the fix:

  1. RESOLUTION. The classifier frontend uses 25 ms frames, which give 40 Hz
     bins. You cannot resolve a 100-300 Hz comb with 40 Hz bins, and the first
     attempt duly estimated 52 Hz for a clean 180 Hz stack. The gate therefore
     runs its OWN longer transform. This is the one place where sharing the
     frontend spectrum is a false economy.

  2. SUBHARMONIC / OCTAVE ERROR. A steady low-fundamental source (lawn mower,
     generator, pump) produces a stronger comb than a wobbling drone, and a
     plain comb-sum locks onto its harmonic inside the drone band. Version one
     scored 6% recall for exactly this reason. The fix resolves the true
     fundamental before deciding, and a candidate only counts as the
     fundamental if there is real energy AT that frequency, which is what
     distinguishes a genuine 90 Hz source from the second harmonic of 45 Hz.

  3. WOBBLE. Real rotors never hold RPM, so a drone's upper harmonics smear
     across bins while a steady machine's stay sharp. Harmonic energy is taken
     as a local max over a tolerance that GROWS with harmonic number, because
     the smear does too. Without this the detector is biased toward the
     steadier source, which is the confuser.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT, FrontendConfig


@dataclass
class GateResult:
    passed: bool
    score: float          # raw comb strength at the resolved fundamental
    f0_hz: float
    rpm: float
    in_band: bool = True


class HarmonicCombGate:
    """Comb-sum over candidate fundamentals, with subharmonic disambiguation.

    Frequency resolution of the GATE is set by `gate_n_fft`, independent of the
    classifier frontend. Integration time against noise remains the binding
    constraint on detection, which is why window length is a detection-range
    question rather than a resolution question.
    """

    def __init__(
        self,
        cfg: FrontendConfig = DEFAULT,
        drone_band: tuple[float, float] = (100.0, 300.0),
        search_floor_hz: float = 35.0,
        n_harmonics: int = 6,
        n_candidates: int = 320,
        threshold: float = 4.0,
        n_blades: int = 2,
        gate_n_fft: int = 2048,
        subharmonic_tolerance: float = 0.75,
        fundamental_presence: float = 0.15,
        min_fundamental_snr: float = 3.0,
        wobble_frac: float = 0.02,
        whiten_hz: float = 400.0,
    ) -> None:
        self.cfg = cfg
        self.threshold = threshold
        self.n_blades = n_blades
        self.n_harmonics = n_harmonics
        self.drone_band = drone_band
        self.subharmonic_tolerance = subharmonic_tolerance
        # Fraction of its own mean harmonic energy a candidate's fundamental
        # must carry to be believed.
        self.fundamental_presence = fundamental_presence
        # Absolute guard against pure noise, measured against the spectral floor.
        self.min_fundamental_snr = min_fundamental_snr

        self.gate_n_fft = gate_n_fft
        self._gate_hop = gate_n_fft // 2
        self._bin_hz = cfg.sample_rate / gate_n_fft
        self._n_bins = gate_n_fft // 2 + 1
        self._win = np.hanning(gate_n_fft).astype(np.float32)

        self._f0_grid = np.linspace(search_floor_hz, drone_band[1] + 30.0, n_candidates)

        # Per-harmonic search windows. The wobble tolerance is PROPORTIONAL to
        # each candidate's own harmonic frequency, not to the band maximum.
        # Scaling it to the band maximum is a trap: it gives low-f0 candidates
        # a tolerance wider than their own harmonic spacing, so they vacuum up
        # whatever real energy is nearby and win the comb. That is how an
        # earlier version reported 37 Hz for a clean 180 Hz stack.
        self._harm_idx: list[np.ndarray] = []
        self._harm_mask: list[np.ndarray] = []
        for k in range(1, n_harmonics + 1):
            bins = np.rint(self._f0_grid * k / self._bin_hz).astype(np.int64)
            tol = np.maximum(1, np.ceil(wobble_frac * bins)).astype(np.int64)
            span = int(tol.max())
            offsets = np.arange(-span, span + 1)
            idx = np.clip(bins[:, None] + offsets[None, :], 0, self._n_bins - 1)
            # Power is non-negative, so masking with zero is safe for a max and
            # the centre bin is always inside the tolerance.
            mask = (np.abs(offsets)[None, :] <= tol[:, None]).astype(np.float64)
            self._harm_idx.append(idx)
            self._harm_mask.append(mask)

        self._floor_band = slice(int(round(40.0 / self._bin_hz)), self._n_bins)
        # Local-background span for whitening. Wide enough that harmonic peaks
        # cannot dominate the median, narrow enough to track the 1/f tilt.
        self._whiten_bins = max(9, int(round(whiten_hz / self._bin_hz)) | 1)

    # ---- spectrum -------------------------------------------------------

    def mean_power(self, audio: np.ndarray) -> np.ndarray:
        """Mean power spectrum over the window at GATE resolution."""
        x = np.asarray(audio, dtype=np.float32)
        if len(x) < self.gate_n_fft:
            x = np.pad(x, (0, self.gate_n_fft - len(x)))
        n = 1 + (len(x) - self.gate_n_fft) // self._gate_hop
        idx = np.lib.stride_tricks.as_strided(
            x,
            shape=(n, self.gate_n_fft),
            strides=(x.strides[0] * self._gate_hop, x.strides[0]),
        )
        power = np.abs(np.fft.rfft(idx * self._win, axis=1)) ** 2
        return power.mean(axis=0).astype(np.float64)

    # ---- scoring --------------------------------------------------------

    def _harmonic_energy(self, spec: np.ndarray, k: int) -> np.ndarray:
        """Max power near the k-th harmonic of every candidate f0."""
        return (spec[self._harm_idx[k]] * self._harm_mask[k]).max(axis=1)

    def whiten(self, spec: np.ndarray) -> np.ndarray:
        """Divide out the local spectral background.

        Without this the gate measures POWER, not tonality, and outdoor
        background is steeply 1/f-tilted. A low-frequency candidate then wins
        the comb purely on the tilt, which is how the previous version
        estimated a 35-56 Hz fundamental for drones sitting in pink noise and
        forwarded 100% of background. Whitening makes the score
        peak-above-local-neighbourhood, which is the quantity that actually
        distinguishes a harmonic source from coloured noise.
        """
        from scipy.ndimage import median_filter

        background = median_filter(spec, size=self._whiten_bins, mode="nearest")
        return spec / (background + 1e-30)

    def score_spectrum(self, mean_power: np.ndarray) -> GateResult:
        spec = self.whiten(np.asarray(mean_power, dtype=np.float64))
        floor = 1.0  # whitened background is unity by construction

        harm = np.stack([self._harmonic_energy(spec, k) for k in range(self.n_harmonics)])
        comb = harm.sum(axis=0) / (self.n_harmonics * floor)
        fundamental = harm[0]

        # A candidate only counts if there is real energy AT its own
        # fundamental, judged RELATIVE TO ITS OWN HARMONICS rather than to the
        # spectral median. The median is useless here: for a clean tonal
        # signal it sits near zero, so even spectral leakage clears an
        # absolute threshold and every subharmonic looks valid. Measured
        # against its own harmonics, a true fundamental is comparable to them
        # while a subharmonic's "fundamental" is leakage, orders of magnitude
        # down. That is what separates a genuine 90 Hz source from the second
        # harmonic of a 45 Hz mower.
        valid = (fundamental >= self.fundamental_presence * harm.mean(axis=0)) & (
            fundamental >= self.min_fundamental_snr * floor
        )
        if not valid.any():
            return GateResult(passed=False, score=0.0, f0_hz=0.0, rpm=0.0, in_band=False)

        masked = np.where(valid, comb, -np.inf)
        peak = int(np.argmax(masked))
        peak_f0 = float(self._f0_grid[peak])

        # Candidates derive from the ORIGINAL peak. Deriving them from a
        # running estimate cascades the division and walks f0 down to the grid
        # floor, which is the bug that gave version one 6% recall.
        candidates = [peak]
        for divisor in (2, 3):
            cand_hz = peak_f0 / divisor
            if cand_hz >= self._f0_grid[0]:
                j = int(np.argmin(np.abs(self._f0_grid - cand_hz)))
                if valid[j]:
                    candidates.append(j)

        best_score = max(comb[j] for j in candidates)
        viable = [j for j in candidates
                  if comb[j] >= self.subharmonic_tolerance * best_score]
        chosen = min(viable, key=lambda j: self._f0_grid[j]) if viable else peak

        f0 = float(self._f0_grid[chosen])
        score = float(comb[chosen])
        lo, hi = self.drone_band
        in_band = bool(lo <= f0 <= hi)

        return GateResult(
            passed=bool(in_band and score >= self.threshold),
            score=score,
            f0_hz=f0,
            rpm=60.0 * f0 / self.n_blades,
            in_band=in_band,
        )

    def __call__(self, audio: np.ndarray) -> GateResult:
        return self.score_spectrum(self.mean_power(audio))


def decision_score(res: GateResult) -> float:
    """The score the DECISION is made on.

    Out-of-band sources score zero no matter how strong their comb is, so a
    lawn mower with an enormous harmonic stack cannot drag the operating
    threshold upward during calibration. `GateResult.score` stays raw because
    you want to see that strength when diagnosing a confuser.
    """
    return res.score if res.in_band else 0.0


def calibrate_threshold(
    gate: HarmonicCombGate,
    positives: list[np.ndarray],
    negatives: list[np.ndarray],
    target_recall: float = 0.99,
) -> dict:
    """Calibrate RECALL-FIRST, the only correct way for a tier-0 gate.

    A gate exists to cheaply discard background. A missed detection here can
    never be recovered downstream, so recall is the CONSTRAINT and the
    forwarded-background fraction is what you then measure, not what you
    target. Calibrating on the negative distribution instead, which is the
    intuitive thing to do, silently trades away detections.

    If the forwarded fraction comes out high, the gate is not earning its
    place: you pay the CNN on most windows anyway, so drop the tier and
    simplify rather than keeping a cascade that only looks clever.
    """
    pos = np.array([decision_score(gate(a)) for a in positives])
    neg = np.array([decision_score(gate(a)) for a in negatives])

    thr = float(np.quantile(pos, 1.0 - target_recall))
    recall = float((pos >= thr).mean())
    forwarded = float((neg >= thr).mean())

    return {
        "threshold": thr,
        "recall": recall,
        "target_recall": target_recall,
        # Two independent verdicts, deliberately not merged. A cascade can be
        # worth keeping (it rejects most background) while still having a
        # recall ceiling below target, and the fixes for those are different.
        "negative_forward_rate": forwarded,
        "useful": bool(forwarded < 0.5),
        # Shortfall means some positives score zero no matter the threshold,
        # i.e. no in-band fundamental was found at all. Lowering the threshold
        # cannot recover those; only a better frontend or a longer window can.
        "recall_shortfall": max(0.0, target_recall - recall),
        "pos_median": float(np.median(pos)),
        "neg_median": float(np.median(neg)),
        "unreachable_frac": float((pos <= 0.0).mean()),
        "n_positives": int(len(pos)),
        "n_negatives": int(len(neg)),
    }
