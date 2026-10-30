"""The numpy port of DroneSound.pd: pitch follows throttle, level is the patch's, Pd's filters."""
import numpy as np
from scipy.signal import freqz

from dad import pdsynth as ps

FS = ps.FS


def test_pitch_and_level(monkeypatch):
    monkeypatch.setattr(ps, "PATCH_DOUBLE_DIVIDE", True)   # the mapping Pd itself renders
    for thr, rms in ((40, 0.028), (100, 0.047)):           # Pd 0.57 renders: RMS 0.0280 and 0.0467
        x = ps.render(np.full(200, float(thr)), seed=1)
        seg = x[FS:3 * FS] * np.hanning(2 * FS)
        f = np.fft.rfftfreq(2 * FS, 1 / FS)
        band = (f > 100) & (f < 1500)
        f0 = f[band][np.abs(np.fft.rfft(seg))[band].argmax()]
        motors = (150 + 180 * thr / 127) + np.mean(ps.DETUNE)
        assert abs(f0 - motors) < 6, (thr, f0, motors)
        assert abs(np.sqrt(np.mean(x[FS:] ** 2)) / rms - 1) < 0.05


def test_filters_match_pd_formulas():
    """Constant-cutoff responses at a few frequencies, measured from Pd 0.57 on white noise."""
    imp = np.zeros(1 << 15); imp[0] = 1
    n, steps = 882, len(imp) // 882
    imp = imp[:steps * n]
    for out, want in ((ps._lop(imp, np.full(steps, 1445.0), FS, n), {969: -1.3, 3004: -6.4, 8000: -13.6}),
                      (ps._hip(imp, np.full(steps, 218.0), FS, n), {54: -12.7, 97: -7.9, 215: -3.1}),
                      (ps._bp(imp, np.full(steps, 970.0), 0.8, FS, n), {497: 6.5, 3004: -4.6, 8000: -19.9})):
        for hz, db in want.items():
            _, h = freqz(out, worN=[hz], fs=FS)
            assert abs(20 * np.log10(abs(h[0])) - db) < 0.2, (hz, db)
