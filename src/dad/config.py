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

    def describe(self) -> str:
        return (
            f"{self.sample_rate} Hz mono | {self.window_seconds:g} s window "
            f"({self.frames_per_window} frames) | {self.hop_seconds:g} s hop | "
            f"{self.n_mels} log-mel {self.fmin:g}-{self.fmax:g} Hz | "
            f"{1000 * self.n_fft / self.sample_rate:g}/{1000 * self.hop_length / self.sample_rate:g} ms frame/stride"
        )


DEFAULT = FrontendConfig()
