"""Per-stage latency benchmark. This produces the headline figure.

Report p50/p95/p99 wall clock, never the mean. A mean hides the tail, and an
always-on detector is judged on its tail. Measure on the ACTUAL target
hardware: numbers from your laptop say nothing about a Raspberry Pi, and the
council's whole point is that implementation and target, not architecture,
decide latency.

What this demonstrates:
  1. The frontend costs more than the classifier.
  2. Incremental (hop-sized) feature extraction is several times cheaper than
     recomputing the full window.
  3. Both are a rounding error next to window fill.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import DEFAULT, FrontendConfig
from .features import LogMel, StreamingLogMel, summary_stats
from .gate import HarmonicCombGate


@dataclass
class Timing:
    name: str
    samples_ms: list[float] = field(default_factory=list)

    def add(self, dt_s: float) -> None:
        self.samples_ms.append(dt_s * 1000.0)

    def pct(self, q: float) -> float:
        return float(np.percentile(self.samples_ms, q)) if self.samples_ms else float("nan")

    @property
    def p50(self) -> float:
        return self.pct(50)

    @property
    def p95(self) -> float:
        return self.pct(95)

    @property
    def p99(self) -> float:
        return self.pct(99)

    def row(self) -> str:
        return f"{self.name:<34} {self.p50:>9.3f} {self.p95:>9.3f} {self.p99:>9.3f}"


def _time_it(fn, n_iter: int, warmup: int = 10) -> Timing:
    t = Timing(getattr(fn, "_label", "stage"))
    for _ in range(warmup):
        fn()
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        t.add(time.perf_counter() - t0)
    return t


def _label(fn, text: str):
    fn._label = text
    return fn


def benchmark(cfg: FrontendConfig = DEFAULT, n_iter: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    window_audio = rng.standard_normal(cfg.samples_per_window).astype(np.float32)
    hop_audio = rng.standard_normal(cfg.samples_per_hop).astype(np.float32)

    full = LogMel(cfg)
    stream = StreamingLogMel(cfg)
    gate = HarmonicCombGate(cfg)

    # Prime the streaming buffer so we time steady state, not startup.
    stream.push(window_audio)
    logmel_window = full.frames(window_audio)

    timings: list[Timing] = [
        _time_it(_label(lambda: full.frames(window_audio),
                        "log-mel, full window (naive)"), n_iter),
        _time_it(_label(lambda: stream.push(hop_audio),
                        "log-mel, incremental (hop only)"), n_iter),
        _time_it(_label(lambda: gate(window_audio),
                        "tier-0 harmonic gate"), n_iter),
        _time_it(_label(lambda: summary_stats(logmel_window),
                        "MFCC summary stats"), n_iter),
    ]

    results: dict = {"config": cfg, "timings": timings, "model_timings": []}

    # Classical head, trained on a small synthetic set purely so the timing
    # path is real rather than a stub.
    try:
        from .models.baseline import MfccBaseline
        from .synth import make_dataset

        clips = make_dataset(n_sessions=2, clips_per_session=8, cfg=cfg)
        X = np.stack([summary_stats(full.frames(c.audio)) for c in clips])
        y = np.array([c.label for c in clips])
        base = MfccBaseline().fit(X, y)
        xv = X[0]
        results["model_timings"].append(
            _time_it(_label(lambda: base.predict_one(xv), f"classifier: {base.name}"), n_iter)
        )
    except Exception as exc:  # pragma: no cover
        results["baseline_error"] = repr(exc)

    # DS-CNN, if torch is installed.
    try:
        from .models.dscnn import TORCH_AVAILABLE, DsCnn

        if TORCH_AVAILABLE:
            import torch

            torch.set_num_threads(1)  # single core, like the embedded target
            net = DsCnn(n_mels=cfg.n_mels).eval()
            xt = torch.from_numpy(logmel_window)[None, None]
            with torch.inference_mode():
                label = f"classifier: DS-CNN fp32 ({net.n_params() / 1e3:.1f}k)"
                results["model_timings"].append(
                    _time_it(_label(lambda: net(xt), label), n_iter)
                )
            results["dscnn_params"] = net.n_params()
    except Exception as exc:  # pragma: no cover
        results["dscnn_error"] = repr(exc)

    return results


def latency_budget(
    cfg: FrontendConfig, compute_ms: float, n_of_m: tuple[int, int] | None = None
) -> dict:
    """End-to-end detection latency, decomposed.

      L = window_fill + hop_wait + compute + smoothing

    Two different quantities get conflated constantly, so both are reported:

    cold_onset_ms   a drone that was silent starts emitting now. You must wait
                    for the window to actually fill with signal.
    steady_state_ms the drone is already audible and the sliding window is
                    full. Median extra wait is hop/2 plus compute.

    Quote the cold-onset figure as your latency. The steady-state figure is the
    optimistic one, and it is what makes a 1 s window look harmless when it is
    not.
    """
    window_ms = cfg.window_seconds * 1000.0
    hop_ms = cfg.hop_seconds * 1000.0
    smoothing_ms = 0.0
    if n_of_m:
        _, m = n_of_m
        smoothing_ms = (m - 1) * hop_ms  # N-of-M voting multiplies hop by M-1

    return {
        "window_ms": window_ms,
        "hop_ms": hop_ms,
        "compute_ms": compute_ms,
        "smoothing_ms": smoothing_ms,
        "cold_onset_ms": window_ms + compute_ms + smoothing_ms,
        "steady_state_ms": hop_ms / 2.0 + compute_ms + smoothing_ms,
        "compute_share_pct": 100.0 * compute_ms / (window_ms + compute_ms + smoothing_ms),
    }


def range_penalty(
    window_s: float,
    reference_s: float = 1.0,
    db_per_doubling: float = 1.5,
    spreading_db_per_doubling: float = 6.0,
) -> dict:
    """The Skeptic's reframe, as code.

    Incoherent integration buys about 1.5 dB per doubling of window length,
    while source SNR falls about 6 dB per doubling of range. So shortening the
    window spends SNR, and SNR is range.

    Shortening 1 s to 125 ms costs about 4.5 dB, which is 0.75 of a range
    doubling, so usable range falls to roughly 0.59x. Not "halves".
    Atmospheric absorption makes the real penalty slightly worse than this
    geometric estimate.
    """
    doublings = np.log2(window_s / reference_s)
    db = doublings * db_per_doubling
    range_factor = 2.0 ** (db / spreading_db_per_doubling)
    return {
        "window_s": window_s,
        "snr_delta_db": float(db),
        "range_factor": float(range_factor),
        "range_change_pct": float(100.0 * (range_factor - 1.0)),
    }


def format_report(results: dict) -> str:
    cfg: FrontendConfig = results["config"]
    bar = "=" * 72
    dash = "-" * 72
    head = "stage".ljust(34) + "p50".rjust(9) + "p95".rjust(9) + "p99".rjust(9)

    lines = [
        bar,
        "PER-STAGE LATENCY  (milliseconds, single-threaded)",
        f"frontend: {cfg.describe()}",
        bar,
        head,
        dash,
    ]
    lines += [t.row() for t in results["timings"]]
    if results["model_timings"]:
        lines.append(dash)
        lines += [t.row() for t in results["model_timings"]]
    lines.append(bar)

    # Amortisation win: full-window versus hop-only feature extraction.
    full_ms = results["timings"][0].p50
    inc_ms = results["timings"][1].p50
    if inc_ms > 0:
        lines.append(
            f"Incremental frontend speedup: {full_ms / inc_ms:.1f}x  "
            f"[{full_ms:.3f} ms full window vs {inc_ms:.3f} ms per decision]"
        )

    # The headline: compute versus the window.
    gate_ms = results["timings"][2].p50
    clf_ms = max((t.p50 for t in results["model_timings"]), default=0.0)
    compute_ms = inc_ms + gate_ms + clf_ms
    budget = latency_budget(cfg, compute_ms)
    lines += [
        "",
        "END-TO-END LATENCY BUDGET",
        dash,
        f"  window fill       {budget['window_ms']:>9.1f} ms",
        f"  hop               {budget['hop_ms']:>9.1f} ms",
        f"  compute total     {budget['compute_ms']:>9.3f} ms",
        f"  cold-onset        {budget['cold_onset_ms']:>9.1f} ms   <-- quote this one",
        f"  steady-state      {budget['steady_state_ms']:>9.1f} ms",
        "",
        f"  COMPUTE IS {budget['compute_share_pct']:.2f}% OF COLD-ONSET LATENCY.",
        "  Architecture search optimises that term. The window is the other 99%.",
        "",
        "WINDOW LENGTH AS A DETECTION-RANGE KNOB",
        dash,
        "window".rjust(10) + "SNR delta".rjust(14) + "range factor".rjust(15),
    ]
    for w in (0.125, 0.25, 0.5, 1.0, 2.0):
        r = range_penalty(w)
        lines.append(
            f"{w:>9.3f}s {r['snr_delta_db']:>+12.2f} dB {r['range_factor']:>13.2f}x"
        )
    lines.append(bar)
    return "\n".join(lines)
