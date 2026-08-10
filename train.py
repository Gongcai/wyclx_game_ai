import argparse
import os
import random
import time

import torch

from agents.baselines import play
from agents.dist import load_weights, make_sampler
from agents.dqn import DQN, encode, index_action, legal_mask
from game import Game


def make_env(weights, capped, seed=None):
    return Game(
        rng=random.Random(seed),
        drop_sampler=make_sampler(weights, capped),
    )


def step_reward(g):
    n9 = sum(1 for v in g.events if v >= 9)
    interm = sum(v for v in g.events if v < 9)
    r = -0.05 + 10.0 * n9 + 0.5 * interm
    if g.dead:
        r -= 5.0
    return r


def evaluate(agent, weights, capped, n, seed, device):
    base = random.Random(seed)
    stats = {}
    for policy in ("agent", "greedy", "random"):
        tot_score = tot_steps = tot_n9 = 0
        for i in range(n):
            g = make_env(weights, capped, base.randint(0, 2**31))
            if policy == "agent":
                while not g.dead:
                    mask = legal_mask(g, device)
                    a = agent.act(encode(g, device), mask, 0.0)
                    src, dst = index_action(a)
                    if not g.move(src, dst):
                        break
                score, steps, n9 = g.score, g.moves, g.score // 9
            else:
                score, steps, n9 = play(g, policy)
            tot_score += score
            tot_steps += steps
            tot_n9 += n9
        stats[policy] = (tot_score / n, tot_steps / n, tot_n9 / n)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default="uniform")
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--replay", type=int, default=100_000)
    ap.add_argument("--target-sync", type=int, default=1000)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-decay", type=int, default=200_000)
    ap.add_argument("--eval-every", type=int, default=20_000)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    weights, capped = load_weights(args.dist)
    envs = [make_env(weights, capped, args.seed * 1000 + i) for i in range(args.n_envs)]
    agent = DQN(lr=args.lr, replay=args.replay, device=device)

    run_id = args.run_id or f"{args.dist}-{int(time.time())}"
    out = os.path.join("runs", run_id)
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, "metrics.jsonl")

    eps = args.eps_start
    steps = 0
    best = -1e9
    next_eval = args.eval_every
    print(f"run_id={run_id} dist={args.dist} device={device}")

    while steps < args.steps:
        for g in envs:
            if g.dead:
                g.reset()
            s = encode(g, device)
            mask = legal_mask(g, device)
            a = agent.act(s, mask, eps)
            src, dst = index_action(a)
            g.move(src, dst)
            r = step_reward(g)
            ns = encode(g, device)
            nmask = legal_mask(g, device) if not g.dead else torch.zeros_like(mask)
            agent.remember(s, a, r, ns, nmask, float(g.dead))
            steps += 1
            if steps <= args.eps_decay:
                eps = args.eps_start - steps * (args.eps_start - args.eps_end) / args.eps_decay
            else:
                eps = args.eps_end
            agent.learn(args.batch, args.gamma, args.target_sync)

        if steps >= next_eval:
            stats = evaluate(agent, weights, capped, args.eval_n, args.seed, device)
            line = {
                "step": steps,
                "eps": round(eps, 4),
                "agent": stats["agent"],
                "greedy": stats["greedy"],
                "random": stats["random"],
            }
            print(f"[{steps:>9}] " + "  ".join(f"{k}: score={v[0]:.1f} steps={v[1]:.0f} n9={v[2]:.2f}" for k, v in stats.items()))
            with open(log_path, "a") as f:
                f.write(str(line) + "\n")
            if stats["agent"][0] > best:
                best = stats["agent"][0]
                agent.save(os.path.join(out, "best.pt"))
            next_eval += args.eval_every

    stats = evaluate(agent, weights, capped, args.eval_n, args.seed, device)
    print("final:", {k: tuple(round(x, 2) for x in v) for k, v in stats.items()})
    agent.save(os.path.join(out, "final.pt"))
    print("saved to", out)


if __name__ == "__main__":
    main()
