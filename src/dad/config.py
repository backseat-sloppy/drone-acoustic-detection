"""The frozen frontend contract.

The council's central finding: with a 1 s analysis window the classifier is
1-5% of end-to-end latency. Window fill and hop dominate. So the frontend is
pinned here ONCE, every model is benchmarked against the identical frontend,
and the only knobs you sweep for the report are `window_seconds` and
`hop_seconds`.

Decoupling matters: window length sets accuracy and detection range, hop sets
responsiveness. They are independent.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrontendConfig:
    # 16 kHz, not 44.1/48. Blade-pass fundamentals and the useful harmonic
    # stack live below 8 kHz, so a higher rate is a pure 3x tax on FFT and mel
    # cost. Do NOT drop to 8 kHz: that discards the 4-8 kHz band holding
    # electronic-speed-controller whine and the upper harmonics.
    sample_rate: int = 16_000

    n_fft: int = 400          # 25 ms frame
    hop_length: int = 160     # 10 ms stride
    n_mels: int = 64
    fmin: float = 50.0
    fmax: float = 8_000.0

    window_seconds: float = 1.0   # accuracy / detection-range knob
    hop_seconds: float = 0.125    # responsiveness knob

    # What the network sees (iteration 9, entry 090): "mfe" log-mel filterbank energies (the
    # default), "mfcc" the first n_mfcc cepstral coefficients of that log-mel, "linear" the log
    # power of every STFT bin between fmin and fmax.
    kind: str = "mfe"
    n_mfcc: int = 20

    # Iteration 20 (entry 172), both off by default. `bandpass` (lo, hi) Hz: a Butterworth of order
    # `bandpass_order`, as |H(f)|^2 on the power spectrum. `wiener`: each STFT bin's power times
    # max(1 - N/P, wiener_floor), N the median power over `wiener_hz` of bins in the same frame.
    bandpass: tuple[float, float] | None = None
    bandpass_order: int = 4
    wiener: bool = False
    wiener_hz: float = 400.0
    wiener_floor: float = 0.1

    # Iteration 22, off by default. `pcen`: per-channel energy normalisation (Wang et al. 2017) on the mel
    # energies after level normalisation, in place of the log: (E / M^alpha + delta)^r - delta^r, M a
    # one-pole smoother of E over frames with time constant `pcen_t` s, started at each window's first frame.
    pcen: bool = False
    pcen_t: float = 0.4
    pcen_alpha: float = 0.98
    pcen_delta: float = 2.0
    pcen_r: float = 0.5

    @property
    def frames_per_window(self) -> int:
        return int(round(self.window_seconds * self.sample_rate / self.hop_length))

    @property
    def samples_per_window(self) -> int:
        return int(round(self.window_seconds * self.sample_rate))

    @property
    def samples_per_hop(self) -> int:
        return int(round(self.hop_seconds * self.sample_rate))

    @property
    def n_freq_bins(self) -> int:
        return self.n_fft // 2 + 1

    def linear_bins(self) -> range:
        """STFT bins with centre frequency in [fmin, fmax]."""
        hz = self.sample_rate / self.n_fft
        return range(int(-(-self.fmin // hz)), min(int(self.fmax // hz), self.n_freq_bins - 1) + 1)

    @property
    def wiener_bins(self) -> int:
        """Odd width of the Wiener noise median, in STFT bins (11 at the default 40 Hz bins)."""
        return 2 * int(round(self.wiener_hz / 2 / (self.sample_rate / self.n_fft))) + 1

    @property
    def pcen_s(self) -> float:
        """PCEN smoother weight per frame."""
        import math
        return 1.0 - math.exp(-self.hop_length / self.sample_rate / self.pcen_t)

    @property
    def n_bands(self) -> int:
        """Rows of the feature image the network gets."""
        return {"mfe": self.n_mels, "mfcc": self.n_mfcc, "linear": len(self.linear_bins())}[self.kind]

    def describe(self) -> str:
        feat = {"mfe": f"{self.n_mels} log-mel", "mfcc": f"{self.n_mfcc} MFCC of {self.n_mels} log-mel",
                "linear": f"{self.n_bands} linear log-power bins"}[self.kind]
        if self.bandpass:
            feat += f" | bandpass {self.bandpass[0]:g}-{self.bandpass[1]:g} Hz order {self.bandpass_order}"
        if self.pcen:
            feat += f" | PCEN (T {self.pcen_t:g} s, alpha {self.pcen_alpha:g}, delta {self.pcen_delta:g}, r {self.pcen_r:g})"
        if self.wiener:
            feat += f" | Wiener (median over {self.wiener_hz:g} Hz, floor {self.wiener_floor:g})"
        return (
            f"{self.sample_rate} Hz mono | {self.window_seconds:g} s window "
            f"({self.frames_per_window} frames) | {self.hop_seconds:g} s hop | "
            f"{feat} {self.fmin:g}-{self.fmax:g} Hz | "
            f"{1000 * self.n_fft / self.sample_rate:g}/{1000 * self.hop_length / self.sample_rate:g} ms frame/stride"
        )


DEFAULT = FrontendConfig()
