import random

DROP_MIN = 1
DROP_MAX = 7

PRESETS = {
    "uniform": {v: 1 for v in range(1, 8)},
    "low": {v: 8 - v for v in range(1, 8)},
    "high": {v: v for v in range(1, 8)},
    "mid": {v: 4 - abs(v - 4) for v in range(1, 8)},
}


class DropDist:
    def __init__(self, weights):
        self.weights = dict(weights)
        self.pool = []
        for v, w in self.weights.items():
            self.pool.extend([v] * w)

    @classmethod
    def preset(cls, name):
        return cls(PRESETS[name])

    def sample(self, cap):
        c = min(cap, DROP_MAX)
        if c < DROP_MIN:
            return DROP_MIN
        while True:
            v = random.choice(self.pool)
            if v <= c:
                return v
