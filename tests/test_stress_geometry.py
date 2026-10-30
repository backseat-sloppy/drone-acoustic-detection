"""The stress test's drone placement: onset, end distance, impact, Doppler."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("stress_test", Path(__file__).parents[1] / "scripts" / "stress_test.py")
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)


def test_drone_at_mic():
    fs, L, f0, v, c = st.FS, 5.0, 1000.0, 50.0, st.C_SOUND
    v_rx = v / (1 - v / c)                               # metres closed per second of listening
    dry = np.sin(2 * np.pi * f0 * np.arange(int(7 * fs)) / fs)
    y, t0, impact, r1 = st.drone_at_mic(dry, v_ms=v, r0=400.0, d=3.0, L=L, l1m=80.0)
    assert t0 == 2.0 and impact is None and abs(r1 - (400 - 3 * v_rx)) < 1e-9
    assert not y[: int(1.99 * fs)].any() and y[int(2.1 * fs):].any()
    seg = y[int(3 * fs):int(4 * fs)]                     # approaching: pitch up by c / (c - v)
    f = np.argmax(np.abs(np.fft.rfft(seg))) * fs / len(seg)
    assert abs(f - f0 * c / (c - v)) < 2

    y, t0, impact, r1 = st.drone_at_mic(dry, v_ms=v, r0=100.0, d=4.0, L=L, l1m=80.0)
    assert abs(impact - (1 + 100 / v_rx)) < 1e-9 and r1 == 0.0 and not y[int((impact + 0.01) * fs):].any()
