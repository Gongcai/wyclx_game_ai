"""N-Tuple Network + afterstate TD(lambda) 训练入口。"""

import argparse
import json
import os
import random
import signal
import sys
import time

from agents.baselines import play
from agents.dist import load_weights, make_real_sampler, make_sampler
from agents.ntuple import (
    EligibilityTrace,
    NtupleNetwork,
    NtuplePolicy,
    deterministic_reward,
)
from agents.ntuple_search import NtupleBeamPolicy
from game import Game


def make_env(weights, capped, seed):
    return Game(
        rng=random.Random(seed),
        drop_sampler=make_real_sampler() if weights == "real" else make_sampler(weights, capped),
    )


def run_ntuple_episode(network, game, rng, max_moves, gamma, trace_lambda, alpha,
                       eps, merge_w, n9_w, death_w, policy=None):
    trace = EligibilityTrace()
    if policy is None:
        policy = NtuplePolicy(
            network, gamma=gamma, merge_w=merge_w, n9_w=n9_w, death_w=death_w,
        )
    pending_counts = None
    pending_drop_reward = 0.0
    moves = 0
    while not game.dead and moves < max_moves:
        action = policy.choose(game, eps, rng)
        if action is None:
            break
        if not game.move_afterstate(*action):
            raise RuntimeError(f"N-Tuple 选出非法动作: {action}")

        # 必须在 resolve 前保存 afterstate；此时不能读隐藏预告。
        current_counts = network.feature_counts(game)
        move_event_count = len(game.events)
        move_dead = game.dead
        move_reward = deterministic_reward(
            game.events, move_dead, merge_w, include_step=True,
            n9_w=n9_w, death_w=death_w,
        )

        if pending_counts is not None:
            trace.update(
                network, pending_counts,
                pending_drop_reward + move_reward,
                0.0 if move_dead else network.value(counts=current_counts),
                gamma, trace_lambda, alpha, terminal=move_dead,
            )

        moves += 1
        if move_dead:
            trace.clear()
            break

        game.resolve_afterstate()
        drop_events = game.events[move_event_count:]
        drop_reward = deterministic_reward(
            drop_events, game.dead, merge_w, include_step=False,
            n9_w=n9_w, death_w=death_w,
        )
        if game.dead:
            trace.update(
                network, current_counts, drop_reward, 0.0,
                gamma, trace_lambda, alpha, terminal=True,
            )
            break

        pending_counts = current_counts
        pending_drop_reward = drop_reward

    # 时间截断时，把已发生的掉落作为 pending afterstate 的终止目标。
    if pending_counts is not None:
        trace.update(
            network, pending_counts, pending_drop_reward, 0.0,
            gamma, trace_lambda, alpha, terminal=True,
        )
    return moves


def evaluate(network, weights, capped, n, seed, max_moves, merge_w, n9_w=10.0,
             gamma=0.99, death_w=5.0, policy=None):
    base = random.Random(seed)
    result = {}
    for name in ("ntuple", "greedy", "random"):
        n9 = score = moves = 0.0
        for _ in range(n):
            game = make_env(weights, capped, base.randint(0, 2**31 - 1))
            if name == "ntuple":
                if policy is None:
                    policy = NtuplePolicy(
                        network, merge_w=merge_w, n9_w=n9_w, gamma=gamma,
                        death_w=death_w,
                    )
                while not game.dead and game.moves < max_moves:
                    action = policy.choose(game, 0.0, base)
                    if action is None or not game.move(*action):
                        break
            else:
                play(game, name)
            n9 += game.n9_count
            score += game.score
            moves += game.moves
        result[name] = {
            "n9": n9 / n,
            "score": score / n,
            "moves": moves / n,
            "n9_per_1000_moves": 1000.0 * n9 / max(1.0, moves),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default="real")
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--trace-lambda", type=float, default=0.6)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--eps-start", type=float, default=0.2)
    ap.add_argument("--eps-end", type=float, default=0.01)
    ap.add_argument("--eps-decay", type=int, default=200_000)
    ap.add_argument("--merge-w", type=float, default=1.0)
    ap.add_argument("--n9-w", type=float, default=10.0)
    ap.add_argument("--death-w", type=float, default=5.0)
    ap.add_argument("--behavior", choices=("greedy", "beam"), default="greedy",
                    help="行为策略：1-ply 贪心（默认）或 beam 搜索")
    ap.add_argument("--behavior-depth", type=int, default=4)
    ap.add_argument("--behavior-width", type=int, default=8)
    ap.add_argument("--behavior-root-width", type=int, default=None)
    ap.add_argument("--max-episode-moves", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=20_000)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    if args.dist == "real":
        weights, capped = "real", None
    else:
        weights, capped = load_weights(args.dist)
    run_id = args.resume or args.run_id or f"ntuple-{args.dist}-{int(time.time())}"
    out = os.path.join("runs", run_id)
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, "metrics.jsonl")
    network = NtupleNetwork()
    steps = 0
    best = float("-inf")
    next_eval = args.eval_every
    if args.resume:
        data = network.load(os.path.join(out, "checkpoint.pt"))
        steps = int(data.get("steps", 0))
        best = float(data.get("best", best))
        next_eval = int(data.get("next_eval", next_eval))

    rng = random.Random(args.seed)
    envs = [make_env(weights, capped, args.seed * 1000 + i) for i in range(args.n_envs)]
    if args.behavior == "beam":
        # FastNtupleValue 的 numpy 镜像是零拷贝视图，训练中原地更新自动生效。
        behavior_policy = NtupleBeamPolicy(
            network,
            depth_moves=args.behavior_depth,
            width=args.behavior_width,
            root_width=args.behavior_root_width,
            gamma=args.gamma,
            merge_w=args.merge_w,
            n9_w=args.n9_w,
            death_w=args.death_w,
        )
        eval_policy = behavior_policy
    else:
        behavior_policy = None
        eval_policy = None
    if args.resume:
        if "rng_state" in data:
            rng.setstate(data["rng_state"])
        for env, state in zip(envs, data.get("env_rng_states", [])):
            env.rng.setstate(state)
    print(
        f"run_id={run_id} dist={args.dist} steps={steps} "
        f"lambda={args.trace_lambda} alpha={args.alpha}"
    )

    def save_checkpoint():
        network.save(
            os.path.join(out, "checkpoint.pt"),
            steps=steps, best=best, next_eval=next_eval,
            gamma=args.gamma, trace_lambda=args.trace_lambda, alpha=args.alpha,
            merge_w=args.merge_w, n9_w=args.n9_w,
            rng_state=rng.getstate(),
            env_rng_states=[env.rng.getstate() for env in envs],
        )

    def on_signal(signum, _frame):
        save_checkpoint()
        print(f"收到信号 {signum}，断点已保存至 {out}/checkpoint.pt，退出")
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    while steps < args.steps:
        for game in envs:
            if game.dead:
                game.reset()
            epsilon = (
                args.eps_start - min(steps, args.eps_decay)
                * (args.eps_start - args.eps_end) / max(1, args.eps_decay)
            )
            episode_moves = run_ntuple_episode(
                network, game, rng,
                min(args.max_episode_moves, max(1, args.steps - steps)),
                args.gamma,
                args.trace_lambda, args.alpha, epsilon, args.merge_w,
                args.n9_w,
                args.death_w,
                policy=behavior_policy,
            )
            steps += episode_moves
            if steps >= args.steps:
                break
            # 每个调用拥有独立 trace；截断局也应从干净的初始盘面继续。
            if not game.dead:
                game.reset()

        if steps >= next_eval:
            stats = evaluate(
                network, weights, capped, args.eval_n, args.seed,
                args.max_episode_moves, args.merge_w,
                args.n9_w, args.gamma,
                args.death_w, policy=eval_policy,
            )
            line = {"step": steps, **stats}
            print(
                f"[{steps:>9}] "
                + "  ".join(
                    f"{name}: n9={value['n9']:.2f} "
                    f"steps={value['moves']:.0f} "
                    f"per1000={value['n9_per_1000_moves']:.2f}"
                    for name, value in stats.items()
                ),
                flush=True,
            )
            with open(log_path, "a") as log:
                log.write(json.dumps(line) + "\n")
            score = stats["ntuple"]["n9_per_1000_moves"]
            if score > best:
                best = score
                network.save(
                    os.path.join(out, "best.pt"), best=best,
                    gamma=args.gamma, merge_w=args.merge_w, n9_w=args.n9_w,
                    death_w=args.death_w,
                )
            next_eval += args.eval_every
            save_checkpoint()

    stats = evaluate(
        network, weights, capped, args.eval_n, args.seed,
        args.max_episode_moves, args.merge_w,
        args.n9_w, args.gamma,
        args.death_w,
    )
    print("final:", stats)
    final_score = stats["ntuple"]["n9_per_1000_moves"]
    if final_score > best:
        best = final_score
        network.save(
            os.path.join(out, "best.pt"), best=best,
            gamma=args.gamma, merge_w=args.merge_w, n9_w=args.n9_w,
            death_w=args.death_w,
        )
    network.save(
        os.path.join(out, "final.pt"), best=best,
        gamma=args.gamma, merge_w=args.merge_w, n9_w=args.n9_w,
        death_w=args.death_w,
    )
    save_checkpoint()
    print("saved to", out)


if __name__ == "__main__":
    main()
