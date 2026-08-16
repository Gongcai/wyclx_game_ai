"""用当前推荐配置跑一局，逐步骤记录盘面/动作/预告/合并事件，到首个 9 或步数上限。

    .venv/bin/python record_game.py --model runs/demos/high-halving8-gumbel48-pv.pt \
        --seed 5 --max-moves 120
    输出每步的文本渲染 + JSONL 轨迹（--json-out）。
"""

import argparse
import json
import random

import torch

from agents.dist import load_weights, make_sampler
from agents.policy_value import PolicyValueNet
from agents.puct import PuctTree, advance_tree, puct_search
from game import Game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="puct 策略需要的 PV 模型")
    ap.add_argument("--policy", choices=("puct", "heuristic"), default="puct")
    ap.add_argument("--dist", default="high")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-moves", type=int, default=120, help="到首个9或此步数止")
    ap.add_argument("--simulations", type=int, default=128)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--death-penalty", type=float, default=0.5)
    ap.add_argument("--chance-samples", type=int, default=8)
    ap.add_argument("--chance-widening", type=float, default=0.5)
    ap.add_argument("--root-min-visits", type=int, default=2)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    net = None
    gamma = 0.99
    if args.policy == "puct":
        if not args.model:
            raise ValueError("puct 策略需要 --model")
        checkpoint = torch.load(args.model, weights_only=True, map_location=args.device)
        net = PolicyValueNet(
            value_outputs=checkpoint.get("value_outputs", 2),
            afterstate_q=checkpoint.get("afterstate_q", False),
            history_features=checkpoint.get("history_features", False),
        ).to(args.device)
        net.load_state_dict(checkpoint["model"])
        net.eval()
        gamma = checkpoint.get("gamma", 0.99)
    weights, capped = load_weights(args.dist)
    game = Game(rng=random.Random(args.seed), drop_sampler=make_sampler(weights, capped))
    tree = PuctTree()
    trace = []
    print(f"===== 开局 seed={args.seed} =====")
    print(game.render())
    print()
    while not game.dead and game.moves < args.max_moves and game.n9_count < 1:
        pre = [list(st) for st in game.stacks]
        preview = list(game.preview) if game.preview is not None else None
        if args.policy == "heuristic":
            from agents.heuristic import heuristic_policy_beam
            action = heuristic_policy_beam(game, width=16, depth=4)
        else:
            action = puct_search(
                game, net, args.device, args.simulations, args.depth,
                c_puct=args.c_puct, gamma=gamma,
                death_penalty=args.death_penalty,
                chance_samples=args.chance_samples,
                chance_widening=args.chance_widening,
                root_min_visits=args.root_min_visits,
                tree=tree,
            )
        if action is None or not game.move(*action):
            break
        if args.policy == "puct":
            advance_tree(tree, action, game)
        events = list(game.events)
        first9 = game.n9_count >= 1
        trace.append({
            "step": game.moves, "action": list(action),
            "pre": pre, "preview": preview,
            "events": events, "score": game.score,
            "post_stacks": [list(st) for st in game.stacks],
        })
        print(f"===== 步 {game.moves}  动作 {action[0]+1}→{action[1]+1} =====")
        if preview:
            print(f"预告: [{ ' '.join(map(str, preview)) }]")
        if events:
            print(f"合并事件: {events}")
        print(f"得分: {game.score}{'  ← 达成首个9!' if first9 else ''}")
        print(game.render())
        print()
        if first9:
            break

    print(f"===== 结束: 第 {game.moves} 步, 得分 {game.score}, "
          f"首9达成={game.n9_count >= 1} =====")
    if args.json_out:
        json.dump({"seed": args.seed, "trace": trace}, open(args.json_out, "w"),
                  ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
