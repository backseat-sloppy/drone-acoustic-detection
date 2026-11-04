import numpy as np
from scene import make_full_scene

C = 343.0

class MicStream:
    """Serves the simulated microphone signal in small blocks."""
    def __init__(self, seed, block_ms=20, **scene_kw):
        self.mix, self.drone, self.fs, self.params = make_full_scene(seed, **scene_kw)
        self.block = int(self.fs * block_ms / 1000)
        self.pos = 0

    def next_block(self):
        if self.pos + self.block > len(self.mix):
            return None
        b = self.mix[self.pos:self.pos + self.block]
        self.pos += self.block
        return b

    def distance_at(self, t):
        """True drone distance when the sound heard at time t was emitted."""
        D, v = self.params["start_dist"], self.params["speed"]
        te = max((t - D / C) / (1 - v / C), 0.0)
        return max(D - v * te, 1.0)

if __name__ == "__main__":
    s = MicStream(1)
    out = []
    while (b := s.next_block()) is not None:
        out.append(b)
    joined = np.concatenate(out)
    print("blocks:", len(out), " identical to offline:", np.array_equal(joined, s.mix[:len(joined)]))