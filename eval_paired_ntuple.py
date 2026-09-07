"""配对评测：N-Tuple(+beam 搜索) 挑战者 vs PUCT 基线（同种子同掉落序列）。

协议与 eval_paired_puct.py 一致：逐局原子写入 JSON、断点续跑、bootstrap 95% CI。
主指标为每局 n9 均值（配对差），每千步吞吐仅作参考。基线固定为 PUCT 部署配置
（sims64-d16-reuse-halving8-q2，见 docs/progress.md）。
"""

import argparse
import json
import os
import random
import time

import torch

from agents.dist import load_weights, make_real_sampler, make_sampler
from agents.ntuple import NtupleNetwork
from agents.ntuple_search import NtupleBeamPolicy
from agents.puct import PuctTree, advance_tree, puct_search
from eval_paired_puct import (
    bootstrap_ci,
    load_model,
    percentile,
    print_summary,
    save_result,
)
from game import Game


def load_ntuple(path):
    network = NtupleNetwork()
    metadata = network.load(path)
    return network, metadata


def ntuple_label_default(path):
    return "ntuple-" + os.path.splitext(os.path.basename(path))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntuple-models", nargs="+", required=True,
                    help="LABEL=path/to/best.pt（N-Tuple 挑战者）")
    ap.add_argument("--ntuple-overrides", nargs="*", default=[],
                    help="LABEL:depth:width 覆盖单个挑战者的搜索参数")
    ap.add_argument("--search-depth", type=int, default=8)
    ap.add_argument("--search-width", type=int, default=8)
    ap.add_argument("--search-root-width", type=int, default=None)
    ap.add_argument("--puct-model", required=True, help="PUCT 基线 checkpoint")
    ap.add_argument("--puct-label", default="puct-r2")
    ap.add_argument("--dist", default="real")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=860_000)
    ap.add_argument("--max-moves", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--max-new", type=int, default=0, help="本次最多新增局数，0 不限")
    # PUCT 基线部署配置（real-h5-puct-r2 推荐参数，docs/progress.md）
    ap.add_argument("--simulations", type=int, default=64)
    ap.add_argument("--puct-depth", type=int, default=16)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--death-penalty", type=float, default=0.5)
    ap.add_argument("--chance-samples", type=int, default=8)
    ap.add_argument("--chance-widening", type=float, default=0.5)
    ap.add_argument("--root-min-visits", type=int, default=2)
    ap.add_argument("--root-sequential-halving", type=int, default=8)
    ap.add_argument("--root-q-scale", type=float, default=2.0)
    args = ap.parse_args()

    ntuple_specs = []
    for item in args.ntuple_models:
        label, path = item.split("=", 1)
        ntuple_specs.append((label, path))
    overrides = {}
    for item in args.ntuple_overrides:
        label, depth, width = item.split(":", 2)
        overrides[label] = (int(depth), int(width))

    config = {
        "ntuple_models": dict(ntuple_specs),
        "ntuple_overrides": {k: list(v) for k, v in overrides.items()},
        "search_depth": args.search_depth,
        "search_width": args.search_width,
        "search_root_width": args.search_root_width,
        "puct_model": args.puct_model,
        "puct_label": args.puct_label,
        "dist": args.dist,
        "episodes": args.episodes,
        "seed": args.seed,
        "max_moves": args.max_moves,
        "simulations": args.simulations,
        "puct_depth": args.puct_depth,
        "c_puct": args.c_puct,
        "death_penalty": args.death_penalty,
        "chance_samples": args.chance_samples,
        "chance_widening": args.chance_widening,
        "root_min_visits": args.root_min_visits,
        "root_sequential_halving": args.root_sequential_halving,
        "root_q_scale": args.root_q_scale,
    }
    rows = []
    if os.path.exists(args.json_out):
        with open(args.json_out) as f:
            saved = json.load(f)
        if saved.get("config") != config:
            raise ValueError("已有评测文件的配置与当前参数不同")
        rows = saved.get("rows", [])

    ntuple_loaded = {}
    for label, path in ntuple_specs:
        network, metadata = load_ntuple(path)
        depth, width = overrides.get(label, (args.search_depth, args.search_width))
        ntuple_loaded[label] = NtupleBeamPolicy(
            network,
            depth_moves=depth,
            width=width,
            root_width=args.search_root_width,
            gamma=metadata.get("gamma", 0.99),
            merge_w=metadata.get("merge_w", 1.0),
            n9_w=metadata.get("n9_w", 30.0),
            death_w=metadata.get("death_w", 15.0),
        )
    puct_net, puct_gamma = load_model(args.puct_model, args.device)

    if args.dist == "real":
        drop_sampler = make_real_sampler()
    else:
        weights, capped = load_weights(args.dist)
        drop_sampler = make_sampler(weights, capped)

    all_labels = [args.puct_label] + [label for label, _path in ntuple_specs]
    total = args.episodes * len(all_labels)
    new_count = 0
    for episode_id in range(args.episodes):
        seed = args.seed + episode_id
        for label in all_labels:
            if any(row["model"] == label and row["seed"] == seed for row in rows):
                continue
            game = Game(rng=random.Random(seed), drop_sampler=drop_sampler)
            started = time.time()
            n9_steps = []
            tree = PuctTree() if label == args.puct_label else None
            while not game.dead and game.moves < args.max_moves:
                if label == args.puct_label:
                    action = puct_search(
                        game, puct_net, args.device, args.simulations,
                        args.puct_depth, c_puct=args.c_puct, gamma=puct_gamma,
                        death_penalty=args.death_penalty,
                        chance_samples=args.chance_samples,
                        chance_widening=args.chance_widening,
                        root_min_visits=args.root_min_visits,
                        root_sequential_halving=args.root_sequential_halving,
                        root_q_scale=args.root_q_scale,
                        tree=tree,
                    )
                else:
                    action = ntuple_loaded[label].choose(game, 0.0, random.Random(seed * 31 + game.moves))
                if action is None or not game.move(*action):
                    break
                if tree is not None:
                    advance_tree(tree, action, game)
                n9_steps.extend(
                    [game.moves] * sum(event >= 9 for event in game.events)
                )
            rows.append({
                "model": label, "seed": seed, "n9": len(n9_steps),
                "moves": game.moves, "dead": game.dead,
                "n9_steps": n9_steps, "seconds": time.time() - started,
            })
            new_count += 1
            save_result(args.json_out, config, rows)
            print(
                f"[{len(rows):>4}/{total}] {label} seed={seed} "
                f"9={len(n9_steps)} moves={game.moves}",
                flush=True,
            )
            if args.max_new > 0 and new_count >= args.max_new:
                print_summary(all_labels, rows)
                return
    print_summary(all_labels, rows)


if __name__ == "__main__":
    main()
