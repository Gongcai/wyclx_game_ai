"""用启发式 beam 生成自我对弈示范（奖励感知：主动合成低牌、控制列高）。

    .venv/bin/python generate_heuristic_demos.py --episodes 40 --width 8 --depth 4
    产物兼容 train_policy_value.py：含 states/actions/n9/dead（policy 目标取 one-hot 动作）。
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import random
import time

import torch

from agents.dist import load_weights, make_real_sampler, make_sampler
from agents.dqn import encode
from agents.heuristic import HeuristicWeights, heuristic_policy_beam
from game import Game


def make_drop_sampler(dist):
    if dist == "real":
        return make_real_sampler()
    weights, capped = load_weights(dist)
    return make_sampler(weights, capped)


def generate_episode(config):
    episode_id, dist, max_moves, width, depth, buried, height6, seed = config
    game = Game(rng=random.Random(seed), drop_sampler=make_drop_sampler(dist))
    hw = HeuristicWeights(buried=buried, height6=height6)
    states, actions, n9_events = [], [], []
    while not game.dead and game.moves < max_moves:
        states.append(encode(game, history=False))
        action = heuristic_policy_beam(game, w=hw, width=width, depth=depth)
        if action is None:
            break
        actions.append(action[0] * 5 + action[1] - (1 if action[1] > action[0] else 0))
        if not game.move(*action):
            break
        n9_events.append(sum(e >= 9 for e in game.events))
    return episode_id, {
        "seed": seed, "score": game.score, "n9_count": game.n9_count,
        "moves": game.moves, "dead": game.dead,
        "states": torch.stack(states),
        "actions": torch.tensor(actions, dtype=torch.long),
        "n9": torch.tensor(n9_events, dtype=torch.float32),
        "n9_steps": [i + 1 for i, c in enumerate(n9_events) for _ in range(c)],
    }


def save_demos(out, args, episodes):
    ordered = [episodes[index] for index in sorted(episodes)]
    temporary = out + ".tmp"
    torch.save({
        "version": 1, "kind": "heuristic-selfplay", "dist": args.dist,
        "width": args.width, "depth": args.depth, "episodes": ordered,
    }, temporary)
    os.replace(temporary, out)


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
    ap.add_argument("--workers", type=int, default=1, help="并行生成局数的进程数")
    ap.add_argument("--checkpoint-every", type=int, default=5, help="每 N 局原子保存一次")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.workers < 1:
        raise ValueError("--workers 必须至少为 1")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every 必须至少为 1")
    out = args.out or os.path.join("runs", "demos", f"{args.dist}-heuristic-beam.pt")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    episodes = {}
    started = time.time()
    configs = [
        (episode_id, args.dist, args.max_moves, args.width, args.depth,
         args.buried, args.height6, args.seed + episode_id)
        for episode_id in range(args.episodes)
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, args.episodes)) as pool:
        futures = [pool.submit(generate_episode, config) for config in configs]
        for complete, future in enumerate(as_completed(futures), start=1):
            episode_id, episode = future.result()
            episodes[episode_id] = episode
            print(f"[{complete}/{args.episodes}] seed={episode['seed']} "
                  f"本局9={episode['n9_count']} 步数={episode['moves']}", flush=True)
            if complete % args.checkpoint_every == 0 or complete == args.episodes:
                save_demos(out, args, episodes)

    ordered = [episodes[index] for index in sorted(episodes)]
    total_n9 = sum(len(e["n9_steps"]) for e in ordered)
    print(f"已保存 {out}: {len(ordered)} 局, {sum(len(e['actions']) for e in ordered)} 步, "
          f"{total_n9} 个9, 用时 {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
