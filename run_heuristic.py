#!/usr/bin/env python
"""启发式策略长局运行器（真实分布 + H5 官方节奏）。

用法：
    .venv/bin/python run_heuristic.py --seed 42 --games 1          # 单局长局
    .venv/bin/python run_heuristic.py --games 10 --max-steps 1000  # 多局统计
"""

import argparse
import random
import time

from agents.dist import make_real_sampler
from agents.heuristic import heuristic_policy_beam
from game import Game


def play_one(seed, max_steps):
    g = Game(rng=random.Random(seed), drop_sampler=make_real_sampler())
    n9 = 0
    n9_steps = []
    while not g.dead and g.moves < max_steps:
        a = heuristic_policy_beam(g, width=8, depth=4, seed=seed)
        if a is None or not g.move(*a):
            break
        if any(e == 9 for e in g.events):
            n9 += 1
            n9_steps.append(g.moves)
    return g, n9, n9_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=5000, help="步数上限（0=不限，仅受死亡约束）")
    args = ap.parse_args()
    max_steps = args.max_steps or 10 ** 9
    t0 = time.time()
    for i in range(args.games):
        seed = args.seed + i * 1000
        g, n9, n9_steps = play_one(seed, max_steps)
        rate = n9 / (g.moves / 1000) if g.moves else 0
        print(f"局{i} seed={seed}: {g.moves} 步 | 9数 {n9} (每千步 {rate:.1f}) | "
              f"得分 {g.score} | 最高级 {g.max_merged} | 死={g.dead}")
        if n9_steps:
            print(f"    9 的步数: {n9_steps[:30]}{'...' if len(n9_steps) > 30 else ''}")
    print(f"总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
