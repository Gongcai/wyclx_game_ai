import json
import os

DROP_CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "drop_dists.json")


def load_weights(name):
    with open(DROP_CFG) as f:
        cfg = json.load(f)
    if name not in cfg:
        raise KeyError(f"未知分布: {name}, 可选 {sorted(cfg)}")
    d = cfg[name]
    capped = d.get("capped", True)
    weights = {int(k): int(v) for k, v in d.items() if k != "capped"}
    return weights, capped


def make_sampler(weights, capped=True):
    vals = sorted(weights)
    pool = [v for v in vals for _ in range(weights[v])]
    hi = vals[-1]

    def sampler(rng, max_merged=None):
        limit = hi
        if capped and max_merged is not None:
            limit = min(limit, max_merged)
        while True:
            v = rng.choice(pool)
            if v <= limit:
                return v

    return sampler
