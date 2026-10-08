"""Acoustic drone detection: a latency-first pipeline.

Design follows one finding: the classifier is 1-5% of end-to-end latency, so
window length and hop are the real knobs. See README.md.
"""
from .config import DEFAULT, FrontendConfig  # noqa: F401

__all__ = ["DEFAULT", "FrontendConfig"]
