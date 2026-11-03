import numpy as np

from dad.dronesynth import drone


def test_drone_is_finite_unit_rms_and_never_silent_at_the_start():
    rng = np.random.default_rng(0)
    for _ in range(10):
        x, p = drone(rng, 16_000)
        assert x.shape == (16_000,) and np.isfinite(x).all()
        assert abs(x.std() - 1.0) < 1e-6
        assert 0.4 <= p["pscale"] <= 3.0 and p["motion"] in ("approach", "flyby")
        # The window is cut after the first emitted sample has arrived: no silent lead-in.
        assert np.abs(x[:800]).max() > 0.05
