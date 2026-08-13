"""用启发式 beam 生成自我对弈示范（奖励感知：主动合成低牌、控制列高）。

    .venv/bin/python generate_heuristic_demos.py --episodes 40 --width 8 --depth 4
    产物兼容 train_policy_value.py：含 states/actions/n9/dead（policy 目标取 one-hot 动作）。
"""

import argparse
import os
import random
import time

import torch

from agents.dist import load_weights, make_sampler
from agents.dqn import encode
from agents.heuristic import HeuristicWeights, heuristic_policy_beam
from game import Game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default="high")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--max-moves", type=int, default=500)
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--buried", type=float, default=1.5)
    ap.add_argument("--height6", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=400000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    weights, capped = load_weights(args.dist)
    hw = HeuristicWeights(buried=args.buried, height6=args.height6)
    episodes = []
    started = time.time()
    for ep in range(args.episodes):
        seed = args.seed + ep
        game = Game(rng=random.Random(seed), drop_sampler=make_sampler(weights, capped))
        states, actions, n9_events = [], [], []
        while not game.dead and game.moves < args.max_moves:
            states.append(encode(game, history=False))
            action = heuristic_policy_beam(game, w=hw, width=args.width, depth=args.depth)
            if action is None:
                break
            actions.append(action[0] * 5 + action[1] - (1 if action[1] > action[0] else 0))
            if not game.move(*action):
                break
            n9_events.append(sum(e >= 9 for e in game.events))
        episodes.append({
            "seed": seed, "score": game.score, "moves": game.moves, "dead": game.dead,
            "states": torch.stack(states),
            "actions": torch.tensor(actions, dtype=torch.long),
            "n9": torch.tensor(n9_events, dtype=torch.float32),
            "n9_steps": [i + 1 for i, c in enumerate(n9_events) for _ in range(c)],
        })
        print(f"[{ep + 1}/{args.episodes}] 本局9={game.score // 9} 步数={game.moves}",
              flush=True)

    out = args.out or os.path.join("runs", "demos", f"{args.dist}-heuristic-beam.pt")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({
        "version": 1, "kind": "heuristic-selfplay", "dist": args.dist,
        "width": args.width, "depth": args.depth, "episodes": episodes,
    }, out)
    total_n9 = sum(len(e["n9_steps"]) for e in episodes)
    print(f"已保存 {out}: {len(episodes)} 局, {sum(len(e['actions']) for e in episodes)} 步, "
          f"{total_n9} 个9, 用时 {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
