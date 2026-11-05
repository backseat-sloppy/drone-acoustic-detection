import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import steady_suite as ss  # noqa: E402

from dad.wavio import Decisions  # noqa: E402


def test_matched_auc_and_recall():
    neg = np.arange(100) / 100.0                       # 0.00 .. 0.99
    line = ss.matched({"m": {"pos": [np.array([0.995, 0.955, 0.5, 0.5])], "neg": [neg]}}).splitlines()[1]
    auc, r10, r5 = map(float, line.split()[1:4])
    # 0.995 beats all 100, 0.955 beats 96, 0.5 ties one and beats 50 (50.5 each)
    assert abs(auc - (100 + 96 + 50.5 + 50.5) / 400) < 1e-3
    assert r10 == 0.5 and r5 == 0.5                   # 0.955 clears the 10% and 5% quantiles


def test_smooth_is_a_causal_running_mean():
    s = np.array([1.0, 0.0, 1.0, 0.0])
    d = ss.smooth(Decisions(np.arange(4.0), s, 0.5), 2)
    assert np.allclose(d.score, [1.0, 0.5, 0.5, 0.5])
    assert ss.smooth(Decisions(np.arange(4.0), s, 0.5), 1).score is s
