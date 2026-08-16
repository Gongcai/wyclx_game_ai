import argparse
import random

from agents.baselines import play
from agents.dist import load_weights, make_sampler, make_real_sampler
from agents.dqn import DQN, encode, index_action, legal_mask
from agents.search import beam_policy
from game import Game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="runs/<id>/best.pt, 不传则只评测基线")
    ap.add_argument("--dist", default="uniform")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--arch", choices=("mlp", "equivariant"), default="mlp")
    ap.add_argument("--search-depth", type=int, default=4)
    ap.add_argument("--beam-width", type=int, default=8)
    ap.add_argument(
        "--policies",
        default="agent,beam,greedy,random",
        help="逗号分隔：agent,beam,greedy,random",
    )
    args = ap.parse_args()

    policies = tuple(item.strip() for item in args.policies.split(",") if item.strip())
    allowed = {"agent", "beam", "greedy", "random"}
    if not policies or not set(policies) <= allowed:
        raise ValueError(f"policies 必须属于 {sorted(allowed)}")

    if args.dist == "real":
        drop_sampler = make_real_sampler()
    else:
        weights, capped = load_weights(args.dist)
        drop_sampler = make_sampler(weights, capped)
    base = random.Random(args.seed)
    agent = None
    if args.model:
        agent = DQN(device=args.device, arch=args.arch)
        agent.load(args.model)

    print(f"dist={args.dist}  n={args.n}  model={args.model or '无'}")
    for policy in policies:
        if policy == "agent" and agent is None:
            continue
        tot = [0.0] * 6  # score, steps, n9, max_merged, fullness, success
        for i in range(args.n):
            g = Game(
                rng=random.Random(base.randint(0, 2**31)),
                drop_sampler=drop_sampler,
            )
            if policy == "agent":
                while not g.dead:
                    mask = legal_mask(g, args.device)
                    a = agent.act(encode(g, args.device), mask, 0.0)
                    src, dst = index_action(a)
                    if not g.move(src, dst):
                        break
            elif policy == "beam":
                while not g.dead:
                    action = beam_policy(g, args.search_depth, args.beam_width)
                    if action is None or not g.move(*action):
                        break
            else:
                play(g, policy)
            tot[0] += g.score
            tot[1] += g.moves
            tot[2] += g.n9_count
            tot[3] += g.max_merged
            tot[4] += sum(len(st) for st in g.stacks)
            tot[5] += g.n9_count > 0
        print(
            f"{policy:6s}  平均得分 {tot[0] / args.n:7.2f}  平均步数 {tot[1] / args.n:7.1f}"
            f"  平均合成9 {tot[2] / args.n:6.2f}  成功率 {tot[5] / args.n:5.1%}"
            f"  最大合成 {tot[3] / args.n:4.2f}  满度 {tot[4] / args.n:5.1f}"
        )


if __name__ == "__main__":
    main()
