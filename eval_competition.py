"""按比赛目标评估：跨死亡重开，估算 30 分钟累计合成 9 数量。"""

import argparse
import random
import time

import torch

from agents.baselines import greedy_policy, random_policy
from agents.dist import load_weights, make_sampler
from agents.dqn import DQN, encode, index_action, legal_mask
from agents.search import beam_policy
from game import Game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=("agent", "beam", "greedy", "random"), default="agent")
    ap.add_argument("--model", default=None)
    ap.add_argument("--arch", choices=("mlp", "equivariant"), default="mlp")
    ap.add_argument("--dist", default="uniform")
    ap.add_argument("--moves", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--action-seconds", type=float, default=0.2, help="实机每次移动的点击/动画耗时")
    ap.add_argument("--search-depth", type=int, default=4)
    ap.add_argument("--beam-width", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.policy == "agent" and not args.model:
        raise ValueError("agent 策略必须提供 --model")
    weights, capped = load_weights(args.dist)
    game = Game(rng=random.Random(args.seed), drop_sampler=make_sampler(weights, capped))
    agent = None
    if args.policy == "agent":
        agent = DQN(device=args.device, arch=args.arch)
        agent.load(args.model)

    n9 = deaths = decision_count = 0
    decision_seconds = 0.0
    for _ in range(args.moves):
        if game.dead:
            deaths += 1
            game.reset()
        started = time.perf_counter()
        if args.policy == "agent":
            action_id = agent.act(encode(game, args.device), legal_mask(game, args.device), 0.0)
            action = index_action(action_id)
        elif args.policy == "beam":
            action = beam_policy(game, args.search_depth, args.beam_width)
        elif args.policy == "greedy":
            action = greedy_policy(game)
        else:
            action = random_policy(game)
        decision_seconds += time.perf_counter() - started
        decision_count += 1
        if action is None or not game.move(*action):
            break
        n9 += sum(event >= 9 for event in game.events)

    avg_think = decision_seconds / max(1, decision_count)
    n9_per_1000 = 1000.0 * n9 / max(1, decision_count)
    move_seconds = args.action_seconds + avg_think
    estimate_30m = n9_per_1000 / 1000.0 * 1800.0 / move_seconds
    print(
        f"policy={args.policy} arch={args.arch} dist={args.dist} moves={decision_count}\n"
        f"合成9={n9}  每千步9={n9_per_1000:.2f}  死亡重开={deaths}\n"
        f"平均决策={avg_think * 1000:.2f}ms  实机动作假设={args.action_seconds:.3f}s\n"
        f"预计30分钟={estimate_30m:.1f}个9  人类纪录≈160  第一名≈750"
    )


if __name__ == "__main__":
    main()
