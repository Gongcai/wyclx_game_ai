import argparse
import json
import os
import random
import signal
import sys
import time

import torch

from agents.baselines import play
from agents.dist import load_weights, make_sampler, make_real_sampler
from agents.dqn import DQN, encode, index_action, legal_mask
from game import Game


def make_env(weights, capped, seed=None):
    return Game(
        rng=random.Random(seed),
        drop_sampler=make_real_sampler() if weights == "real" else make_sampler(weights, capped),
    )


def step_reward(g, merge_w=1.0):
    n9 = sum(1 for v in g.events if v >= 9)
    interm = sum(v for v in g.events if v < 9)
    r = -0.05 + 10.0 * n9 + merge_w * interm
    if g.dead:
        r -= 5.0
    return r


def top_pair_count(g):
    n = 0
    for st in g.stacks:
        if len(st) >= 2 and st[0] == st[1]:
            n += 1
    return n


def empty_col_count(g):
    return sum(1 for st in g.stacks if not st)


def phi(g, pair_bonus, empty_bonus):
    return pair_bonus * top_pair_count(g) + empty_bonus * empty_col_count(g)


def evaluate(agent, weights, capped, n, seed, device):
    base = random.Random(seed)
    stats = {}
    for policy in ("agent", "greedy", "random"):
        tot = [0.0] * 5  # score, steps, n9, max_merged, fullness
        for i in range(n):
            g = make_env(weights, capped, base.randint(0, 2**31))
            if policy == "agent":
                while not g.dead:
                    mask = legal_mask(g, device)
                    a = agent.act(encode(g, device), mask, 0.0)
                    src, dst = index_action(a)
                    if not g.move(src, dst):
                        break
            else:
                play(g, policy)
            tot[0] += g.score
            tot[1] += g.moves
            tot[2] += g.n9_count
            tot[3] += g.max_merged
            tot[4] += sum(len(st) for st in g.stacks)
        stats[policy] = tuple(v / n for v in tot)
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
    ap.add_argument("--merge-w", type=float, default=1.0, help="中间合成奖励系数（默认 1.0，原 0.5）")
    ap.add_argument("--pair-bonus", type=float, default=0.3, help="栈顶成对势函数系数（0 关闭）")
    ap.add_argument("--empty-bonus", type=float, default=0.2, help="空列势函数系数（0 关闭）")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--resume", default=None, help="runs/ 下已有 run 目录，从 checkpoint.pt 断点续训（需与中断时同超参同 seed）")
    ap.add_argument("--prefill-demos", default=None, help="beam 示范 .pt；仅新训练时预填充 replay")
    ap.add_argument("--expert-ratio", type=float, default=0.25, help="每个 batch 的固定示范比例")
    ap.add_argument("--expert-bc-w", type=float, default=1.0, help="示范动作监督损失权重")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--arch", choices=("mlp", "equivariant"), default="mlp")
    args = ap.parse_args()

    device = args.device
    if args.dist == "real":
        weights, capped = "real", None
    else:
        weights, capped = load_weights(args.dist)
    envs = [make_env(weights, capped, args.seed * 1000 + i) for i in range(args.n_envs)]
    agent = DQN(lr=args.lr, replay=args.replay, device=device, arch=args.arch)

    if args.prefill_demos and args.resume:
        raise ValueError("prefill-demos 不能与 resume 同时使用")

    run_id = args.resume or args.run_id or f"{args.dist}-{int(time.time())}"
    out = os.path.join("runs", run_id)
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, "metrics.jsonl")

    steps = 0
    best = -1e9
    next_eval = args.eval_every
    eps = args.eps_start
    if args.resume:
        ckpt = agent.load_full(os.path.join(out, "checkpoint.pt"))
        steps, eps, best = ckpt["steps"], ckpt["eps"], ckpt["best"]
        next_eval = ckpt["next_eval"]
        if next_eval <= steps:
            next_eval = ((steps // args.eval_every) + 1) * args.eval_every
        print(f"已断点续训: run_id={run_id} steps={steps} best={best} device={device}")
    else:
        print(f"run_id={run_id} dist={args.dist} device={device}")
        if args.prefill_demos:
            demos = torch.load(args.prefill_demos, weights_only=True, map_location="cpu")
            episodes = demos.get("episodes", [])
            required = {"states", "actions", "rewards", "next_states", "next_masks", "dones"}
            if not episodes or not required <= set(episodes[0]):
                raise ValueError("示范文件不含完整 transition，请用新版 generate_demos.py 重新生成")
            count = 0
            for episode in episodes:
                for i in range(len(episode["actions"])):
                    agent.remember_expert(
                        episode["states"][i],
                        int(episode["actions"][i]),
                        float(episode["rewards"][i]),
                        episode["next_states"][i],
                        episode["next_masks"][i],
                        float(episode["dones"][i]),
                    )
                    count += 1
            print(f"已加载固定 beam 示范池: {count} 条 transition")

    def save_ckpt():
        agent.save_full(
            os.path.join(out, "checkpoint.pt"),
            steps=steps, eps=eps, best=best, next_eval=next_eval,
        )

    def on_signal(signum, frame):
        save_ckpt()
        print(f"收到信号 {signum}，断点已保存至 {out}/checkpoint.pt，退出")
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    while steps < args.steps:
        for g in envs:
            if g.dead:
                g.reset()
            s = encode(g, device)
            mask = legal_mask(g, device)
            a = agent.act(s, mask, eps)
            src, dst = index_action(a)
            phi0 = phi(g, args.pair_bonus, args.empty_bonus)
            g.move(src, dst)
            r = step_reward(g, args.merge_w)
            phi1 = phi(g, args.pair_bonus, args.empty_bonus)
            r += (args.gamma * phi1 if not g.dead else 0.0) - phi0
            ns = encode(g, device)
            nmask = legal_mask(g, device) if not g.dead else torch.zeros_like(mask)
            agent.remember(s, a, r, ns, nmask, float(g.dead))
            steps += 1
            if steps <= args.eps_decay:
                eps = args.eps_start - steps * (args.eps_start - args.eps_end) / args.eps_decay
            else:
                eps = args.eps_end
            agent.learn(
                args.batch,
                args.gamma,
                args.target_sync,
                args.expert_ratio,
                args.expert_bc_w,
            )

        if steps >= next_eval:
            stats = evaluate(agent, weights, capped, args.eval_n, args.seed, device)
            line = {
                "step": steps,
                "eps": round(eps, 4),
                "agent": stats["agent"],
                "greedy": stats["greedy"],
                "random": stats["random"],
            }
            print(f"[{steps:>9}] " + "  ".join(
                f"{k}: score={v[0]:.1f} steps={v[1]:.0f} n9={v[2]:.2f} max={v[3]:.2f} fill={v[4]:.1f}"
                for k, v in stats.items()
            ))
            with open(log_path, "a") as f:
                f.write(json.dumps(line) + "\n")
            if stats["agent"][0] > best:
                best = stats["agent"][0]
                agent.save(os.path.join(out, "best.pt"))
            next_eval += args.eval_every
            save_ckpt()

    stats = evaluate(agent, weights, capped, args.eval_n, args.seed, device)
    print("final:", {k: tuple(round(x, 2) for x in v) for k, v in stats.items()})
    agent.save(os.path.join(out, "final.pt"))
    save_ckpt()
    print("saved to", out)


if __name__ == "__main__":
    main()
