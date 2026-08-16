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


# ---- H5 源码真实掉落分布（兔王争霸赛 index_0e3a250e.js 的 _ 表）----
# 按当前合成最高等级（compositeLevel，即 max_merged）查表；合成 9 后等级=9 用表 8
# （表 8 无 1 级，6/7 级占 65%）。每次采样独立随机，无时间相关 regime。
# 详见 docs/h5_analysis.md
REAL_TABLE = {
    1: {1: 50, 2: 50},
    2: {1: 50, 2: 50},
    3: {1: 30, 2: 40, 3: 30},
    4: {1: 20, 2: 30, 3: 30, 4: 20},
    5: {1: 10, 2: 10, 3: 20, 4: 30, 5: 30},
    6: {1: 5, 2: 10, 3: 10, 4: 25, 5: 30, 6: 20},
    7: {1: 3, 2: 4, 3: 9, 4: 14, 5: 25, 6: 25, 7: 20},
    8: {2: 2, 3: 6, 4: 10, 5: 17, 6: 30, 7: 35},  # compositeLevel=9 时也用此表
}

REAL_POOLS = {lv: [v for v, w in tbl.items() for _ in range(w)]
              for lv, tbl in REAL_TABLE.items()}


def make_real_sampler():
    """真实分布采样器：按 max_merged 查表。

    与假设分布不同：表 1 就有 50% 概率掉 2（模型 cap 语义下开局只掉 1，
    真实游戏开局允许掉 2）。忽略 capped 逻辑，完全按真实表。
    """

    def sampler(rng, max_merged=None):
        lv = max_merged if max_merged is not None else 1
        lv = max(1, min(int(lv), 8))
        return rng.choice(REAL_POOLS[lv])

    return sampler
