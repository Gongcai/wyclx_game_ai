import argparse
import random

from agents.baselines import play
from agents.dist import load_weights, make_sampler
from agents.dqn import DQN, encode, index_action, legal_mask
from game import Game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="runs/<id>/best.pt, 不传则只评测基线")
    ap.add_argument("--dist", default="uniform")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = ap.parse_args()

    weights, capped = load_weights(args.dist)
    base = random.Random(args.seed)
    agent = None
    if args.model:
        agent = DQN(device=args.device)
        agent.load(args.model)

    print(f"dist={args.dist}  n={args.n}  model={args.model or '无'}")
    for policy in ("agent", "greedy", "random"):
        if policy == "agent" and agent is None:
            continue
        tot_score = tot_steps = tot_n9 = 0
        for i in range(args.n):
            g = Game(
                rng=random.Random(base.randint(0, 2**31)),
                drop_sampler=make_sampler(weights, capped),
            )
            if policy == "agent":
                while not g.dead:
                    mask = legal_mask(g, args.device)
                    a = agent.act(encode(g, args.device), mask, 0.0)
                    src, dst = index_action(a)
                    if not g.move(src, dst):
                        break
                score, steps, n9 = g.score, g.moves, g.score // 9
            else:
                score, steps, n9 = play(g, policy)
            tot_score += score
            tot_steps += steps
            tot_n9 += n9
        print(f"{policy:6s}  平均得分 {tot_score / args.n:7.2f}  平均步数 {tot_steps / args.n:7.1f}  平均合成9 {tot_n9 / args.n:6.2f}")


if __name__ == "__main__":
    main()
