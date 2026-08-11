"""按比赛目标评估：跨死亡重开，估算 30 分钟累计合成 9 数量。"""

import argparse
import json
import os
import random
import time

import torch

from agents.baselines import greedy_policy, random_policy
from agents.dist import load_weights, make_sampler
from agents.dqn import DQN, encode, index_action, legal_mask
from agents.mcts import mcts_policy
from agents.policy_value import PolicyValueNet
from agents.puct import puct_search
from agents.search import beam_policy
from game import Game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=("agent", "beam", "mcts", "puct", "greedy", "random"), default="agent")
    ap.add_argument("--model", default=None)
    ap.add_argument("--arch", choices=("mlp", "equivariant"), default="mlp")
    ap.add_argument("--dist", default="uniform")
    ap.add_argument("--moves", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--action-seconds", type=float, default=0.4, help="实机每次移动的点击/动画耗时")
    ap.add_argument("--search-depth", type=int, default=4)
    ap.add_argument("--beam-width", type=int, default=8)
    ap.add_argument("--mcts-simulations", type=int, default=96)
    ap.add_argument("--mcts-depth", type=int, default=16)
    ap.add_argument("--pv-model", default=None)
    ap.add_argument("--puct-simulations", type=int, default=64)
    ap.add_argument("--puct-depth", type=int, default=24)
    ap.add_argument("--puct-death-penalty", type=float, default=0.0)
    ap.add_argument("--puct-chance-samples", type=int, default=0, help="大于0时启用显式 afterstate chance node")
    ap.add_argument("--puct-chance-widening", type=float, default=0.0)
    ap.add_argument("--puct-root-min-visits", type=int, default=0)
    ap.add_argument("--puct-gamma", type=float, default=None, help="默认读取 Policy+Value checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json-out", default=None, help="同时保存机器可读结果")
    args = ap.parse_args()

    if args.policy == "agent" and not args.model:
        raise ValueError("agent 策略必须提供 --model")
    if args.policy == "puct" and not args.pv_model:
        raise ValueError("puct 策略必须提供 --pv-model")
    weights, capped = load_weights(args.dist)
    game = Game(rng=random.Random(args.seed), drop_sampler=make_sampler(weights, capped))
    agent = None
    pv_net = None
    puct_gamma = 0.99
    if args.policy == "agent":
        agent = DQN(device=args.device, arch=args.arch)
        agent.load(args.model)
    elif args.policy == "puct":
        checkpoint = torch.load(args.pv_model, weights_only=True, map_location=args.device)
        pv_net = PolicyValueNet(
            value_outputs=checkpoint.get("value_outputs", 2),
            afterstate_q=checkpoint.get("afterstate_q", False),
        ).to(args.device)
        pv_net.load_state_dict(checkpoint["model"])
        pv_net.eval()
        puct_gamma = args.puct_gamma if args.puct_gamma is not None else checkpoint.get("gamma", 0.99)

    n9 = deaths = decision_count = 0
    episode_moves = 0
    first9_moves = []
    post9_gaps = []
    last9_move = None
    decision_seconds = 0.0
    for _ in range(args.moves):
        if game.dead:
            deaths += 1
            game.reset()
            episode_moves = 0
            last9_move = None
        started = time.perf_counter()
        if args.policy == "agent":
            action_id = agent.act(encode(game, args.device), legal_mask(game, args.device), 0.0)
            action = index_action(action_id)
        elif args.policy == "beam":
            action = beam_policy(game, args.search_depth, args.beam_width)
        elif args.policy == "mcts":
            action = mcts_policy(game, args.mcts_simulations, args.mcts_depth)
        elif args.policy == "puct":
            action = puct_search(
                game, pv_net, args.device,
                args.puct_simulations, args.puct_depth,
                gamma=puct_gamma,
                death_penalty=args.puct_death_penalty,
                chance_samples=args.puct_chance_samples,
                chance_widening=args.puct_chance_widening,
                root_min_visits=args.puct_root_min_visits,
            )
        elif args.policy == "greedy":
            action = greedy_policy(game)
        else:
            action = random_policy(game)
        decision_seconds += time.perf_counter() - started
        decision_count += 1
        if action is None or not game.move(*action):
            break
        episode_moves += 1
        for event in game.events:
            if event >= 9:
                n9 += 1
                if last9_move is None:
                    first9_moves.append(episode_moves)
                else:
                    post9_gaps.append(episode_moves - last9_move)
                last9_move = episode_moves

    avg_think = decision_seconds / max(1, decision_count)
    n9_per_1000 = 1000.0 * n9 / max(1, decision_count)
    move_seconds = args.action_seconds + avg_think
    estimate_30m = n9_per_1000 / 1000.0 * 1800.0 / move_seconds
    first9_avg = sum(first9_moves) / len(first9_moves) if first9_moves else None
    post9_avg = sum(post9_gaps) / len(post9_gaps) if post9_gaps else None
    first9_text = f"{first9_avg:.1f}" if first9_avg is not None else "无"
    post9_text = f"{post9_avg:.1f}" if post9_avg is not None else "无"
    result = {
        "policy": args.policy,
        "arch": args.arch,
        "dist": args.dist,
        "moves": decision_count,
        "n9": n9,
        "n9_per_1000": n9_per_1000,
        "deaths": deaths,
        "first9_avg_moves": first9_avg,
        "post9_avg_gap": post9_avg,
        "post9_samples": len(post9_gaps),
        "avg_decision_seconds": avg_think,
        "action_seconds": args.action_seconds,
        "estimate_30m": estimate_30m,
    }
    print(
        f"policy={args.policy} arch={args.arch} dist={args.dist} moves={decision_count}\n"
        f"合成9={n9}  每千步9={n9_per_1000:.2f}  死亡重开={deaths}\n"
        f"首个9步数={first9_text}  后续9间隔={post9_text} "
        f"(后续样本 {len(post9_gaps)})\n"
        f"平均决策={avg_think * 1000:.2f}ms  实机动作假设={args.action_seconds:.3f}s\n"
        f"预计30分钟={estimate_30m:.1f}个9  人类纪录≈160  第一名≈750"
    )
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
