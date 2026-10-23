import numpy as np
from scipy.ndimage import median_filter

class TonalDetector:
    """Looks for narrow spectral peaks (motor harmonics) sticking out of the local noise."""
    def __init__(self, fs, win_s=0.5, hop_s=0.1, thresh_db=14.0, need=3):
        self.n, self.hop = int(win_s * fs), int(hop_s * fs)
        self.buf, self.count, self.hits = np.zeros(self.n), 0, 0
        f = np.fft.rfftfreq(self.n, 1 / fs)
        self.band = (f > 100) & (f < 1500)
        self.win = np.hanning(self.n)
        self.thresh, self.need = thresh_db, need

    def process(self, block):
        """Feed one block. Returns None, or (detected, score_db) after each analysis hop."""
        self.buf = np.concatenate([self.buf[len(block):], block])
        self.count += len(block)
        if self.count < self.hop:
            return None
        self.count = 0
        P = np.abs(np.fft.rfft(self.buf * self.win))**2
        Pb = P[self.band] + 1e-30
        local = median_filter(Pb, size=101, mode="nearest")   # removes wind and slow spectral tilt
        score = 10 * np.log10((Pb / local).max())
        self.hits = self.hits + 1 if score > self.thresh else 0
        return self.hits >= self.need, score

def score_trace(sig, fs):
    det = TonalDetector(fs, thresh_db=1e9)
    blk, t, s = int(fs * 0.02), [], []
    for k in range(len(sig) // blk):
        res = det.process(sig[k * blk:(k + 1) * blk])
        if res:
            t.append((k + 1) * blk / fs)
            s.append(res[1])
    return np.array(t), np.array(s)

def first_alarm(t, s, thresh, need):
    run = 0
    for ti, si in zip(t, s):
        run = run + 1 if si > thresh else 0
        if run >= need:
            return ti
    return None

def sustained(s, need):
    """Highest score that was held for `need` consecutive hops."""
    if len(s) < need:
        return -1e9
    return np.lib.stride_tricks.sliding_window_view(s, need).min(axis=1).max()

    