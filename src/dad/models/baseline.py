"""Classical baseline: MFCC summary statistics into a gradient-boosted tree.

Genuinely the fastest CLASSIFIER, by one to two orders of magnitude. Build it
first, report it always. Two reasons the council agreed on:

1. It is your honest floor. If the CNN cannot beat this on a session-level
   split, the CNN is not earning its milliseconds.
2. It shares the identical mel frontend, so the comparison isolates the
   classifier head, which is the only thing that actually differs.

What it will NOT do is survive the field. Published tests show MFCC+SVM
false-alarming on lawn equipment and helicopters where a small CNN holds up.
Expect it to tie on clean data and lose off-distribution.
"""

from __future__ import annotations

import numpy as np

try:  # LightGBM if present, else sklearn's equivalent histogram booster.
    from lightgbm import LGBMClassifier as _Booster
    _BACKEND = "lightgbm"
except ImportError:  # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingClassifier as _Booster
    _BACKEND = "sklearn-hist"


class MfccBaseline:
    name = f"mfcc-stats + {_BACKEND}"

    def __init__(self, **kwargs) -> None:
        defaults = dict(n_estimators=200, num_leaves=31, learning_rate=0.1, verbose=-1)
        if _BACKEND != "lightgbm":
            defaults = dict(max_iter=200, max_leaf_nodes=31, learning_rate=0.1)
        self.clf = _Booster(**{**defaults, **kwargs})

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MfccBaseline":
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

    def predict_one(self, x: np.ndarray) -> float:
        """Single-window path. This is what the latency benchmark times, since
        a real detector classifies one window at a time, not a batch."""
        return float(self.clf.predict_proba(x[None, :])[0, 1])
