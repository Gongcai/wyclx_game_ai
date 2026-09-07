"""评测 N-Tuple TD(lambda) 模型。"""

import argparse
import json

from agents.dist import load_weights
from agents.ntuple import NtupleNetwork
from train_ntuple import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dist", default="real")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-moves", type=int, default=1000)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--merge-w", type=float, default=None)
    ap.add_argument("--n9-w", type=float, default=None)
    ap.add_argument("--death-w", type=float, default=None)
    args = ap.parse_args()

    network = NtupleNetwork()
    metadata = network.load(args.model)
    gamma = args.gamma if args.gamma is not None else metadata.get("gamma", 0.99)
    merge_w = args.merge_w if args.merge_w is not None else metadata.get("merge_w", 1.0)
    n9_w = args.n9_w if args.n9_w is not None else metadata.get("n9_w", 10.0)
    death_w = args.death_w if args.death_w is not None else metadata.get("death_w", 5.0)
    if args.dist == "real":
        weights, capped = "real", None
    else:
        weights, capped = load_weights(args.dist)
    stats = evaluate(
        network, weights, capped, args.n, args.seed,
        args.max_moves, merge_w, n9_w, gamma, death_w,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
