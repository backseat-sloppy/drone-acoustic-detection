"""train_clean.load_nodrone never returns a test source (report entry 132)."""
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_clean as tc  # noqa: E402


def test_load_nodrone_skips_test_sources(tmp_path):
    rows = [("a.wav", "d3set"), ("b.wav", "halmstad"), ("c.wav", "fpv"), ("d.wav", "nightjar"), ("e.wav", "dronenoise")]
    for f, _ in rows:
        sf.write(tmp_path / f, np.full(32_000, 0.1, np.float32), 16_000)
    with open(tmp_path / "recordings.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["wav", "source"]); w.writerows(rows)
    (tmp_path / "stems.csv").write_text("file,group,source\n")
    train, val = tc.load_nodrone(tmp_path, "nightjar", None)
    groups = {g for _, g in train + val}
    assert groups == {"d3set", "nightjar"}
    assert not groups & tc.TEST_SOURCES
