"""用 beam search 生成达到目标分数的示范轨迹。"""

import argparse
import multiprocessing as mp
import os
import random
import time

import torch

from agents.dist import load_weights, make_sampler
from agents.dqn import N_ACTIONS, encode, legal_mask
from agents.search import beam_policy
from game import Game


def play_one(args):
    seed, weights, capped, depth, width, min_score, max_moves = args
    game = Game(rng=random.Random(seed), drop_sampler=make_sampler(weights, capped))
    states = []
    actions = []
    rewards = []
    next_states = []
    next_masks = []
    dones = []
    while not game.dead and game.moves < max_moves and game.score < min_score:
        action = beam_policy(game, depth=depth, width=width)
        if action is None:
            break
        states.append(encode(game))
        actions.append(action[0] * 5 + action[1] - (1 if action[1] > action[0] else 0))
        if not game.move(*action):
            break
        n9 = sum(event >= 9 for event in game.events)
        interm = sum(event for event in game.events if event < 9)
        reward = -0.05 + 10.0 * n9 + interm - 5.0 * game.dead
        reached_goal = game.score >= min_score
        rewards.append(reward)
        next_states.append(encode(game))
        dones.append(float(game.dead or reached_goal))
        next_masks.append(torch.zeros(N_ACTIONS) if game.dead or reached_goal else legal_mask(game))
    if game.score < min_score:
        return None
    return {
        "seed": seed,
        "score": game.score,
        "moves": game.moves,
        "states": torch.stack(states),
        "actions": torch.tensor(actions, dtype=torch.long),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "next_states": torch.stack(next_states),
        "next_masks": torch.stack(next_masks),
        "dones": torch.tensor(dones, dtype=torch.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default="uniform")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--search-depth", type=int, default=4)
    ap.add_argument("--beam-width", type=int, default=8)
    ap.add_argument("--min-score", type=int, default=9)
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.episodes <= 0 or args.workers <= 0:
        raise ValueError("episodes 和 workers 必须为正数")
    weights, capped = load_weights(args.dist)
    out = args.out or os.path.join("runs", "demos", f"{args.dist}-beam.pt")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    work = [
        (
            args.seed + i,
            weights,
            capped,
            args.search_depth,
            args.beam_width,
            args.min_score,
            args.max_moves,
        )
        for i in range(args.episodes)
    ]

    started = time.time()
    demos = []
    with mp.Pool(args.workers) as pool:
        for i, demo in enumerate(pool.imap_unordered(play_one, work), start=1):
            if demo is not None:
                demos.append(demo)
            if i % max(1, args.episodes // 20) == 0 or i == args.episodes:
                print(f"[{i:>6}/{args.episodes}] 成功 {len(demos):>5}", flush=True)

    transitions = sum(len(demo["actions"]) for demo in demos)
    torch.save(
        {
            "version": 2,
            "dist": args.dist,
            "search_depth": args.search_depth,
            "beam_width": args.beam_width,
            "min_score": args.min_score,
            "max_moves": args.max_moves,
            "episodes": demos,
        },
        out,
    )
    elapsed = time.time() - started
    print(
        f"已保存 {out}: 成功 {len(demos)}/{args.episodes}, "
        f"转移 {transitions}, 用时 {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
