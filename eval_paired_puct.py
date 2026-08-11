"""在相同 seed 上配对评测多个 PUCT 模型，并持续保存置信区间数据。"""

import argparse
import json
import os
import random
import statistics
import time

import torch

from agents.dist import load_weights, make_sampler
from agents.policy_value import PolicyValueNet
from agents.puct import puct_search
from game import Game


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[round((len(ordered) - 1) * q)]


def bootstrap_ci(values, samples=10_000, seed=0):
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(
        statistics.mean(rng.choices(values, k=len(values)))
        for _ in range(samples)
    )
    return means[round(0.025 * (samples - 1))], means[round(0.975 * (samples - 1))]


def parse_models(items):
    models = []
    for item in items:
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            path = item
            label = os.path.splitext(os.path.basename(path))[0]
        if not label or any(label == old_label for old_label, _path in models):
            raise ValueError(f"模型标签为空或重复: {label!r}")
        models.append((label, path))
    return models


def load_model(path, device):
    checkpoint = torch.load(path, weights_only=True, map_location=device)
    net = PolicyValueNet(
        value_outputs=checkpoint.get("value_outputs", 2),
        afterstate_q=checkpoint.get("afterstate_q", False),
        history_features=checkpoint.get("history_features", False),
    ).to(device)
    net.load_state_dict(checkpoint["model"])
    net.eval()
    return net, checkpoint.get("gamma", 0.99)


def save_result(path, config, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as f:
        json.dump({"config": config, "rows": rows}, f, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def print_summary(labels, rows):
    print("\n模型汇总")
    by_label = {label: [] for label in labels}
    for row in rows:
        by_label[row["model"]].append(row)
    for label in labels:
        selected = by_label[label]
        scores = [row["n9"] for row in selected]
        moves = sum(row["moves"] for row in selected)
        if not scores:
            continue
        print(
            f"{label}: n={len(scores)} mean={statistics.mean(scores):.2f} "
            f"P10/P50/P90={percentile(scores, 0.1)}/"
            f"{percentile(scores, 0.5)}/{percentile(scores, 0.9)} "
            f"max={max(scores)} 每千步={sum(scores) * 1000 / max(1, moves):.2f}"
        )

    baseline = labels[0]
    baseline_rows = {row["seed"]: row for row in by_label[baseline]}
    print(f"\n相对基线: {baseline}")
    for label in labels[1:]:
        candidate_rows = {row["seed"]: row for row in by_label[label]}
        seeds = sorted(set(baseline_rows) & set(candidate_rows))
        differences = [
            candidate_rows[seed]["n9"] - baseline_rows[seed]["n9"]
            for seed in seeds
        ]
        low, high = bootstrap_ci(differences)
        wins = sum(value > 0 for value in differences)
        ties = sum(value == 0 for value in differences)
        losses = sum(value < 0 for value in differences)
        print(
            f"{label}: paired_n={len(seeds)} mean_diff="
            f"{statistics.mean(differences) if differences else 0.0:+.2f} "
            f"95%CI=[{low:+.2f}, {high:+.2f}] W/T/L={wins}/{ties}/{losses}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="LABEL=checkpoint.pt")
    ap.add_argument("--dist", default="high")
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument("--seed", type=int, default=100_000)
    ap.add_argument("--max-moves", type=int, default=500)
    ap.add_argument("--simulations", type=int, default=128)
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--death-penalty", type=float, default=0.5)
    ap.add_argument("--chance-samples", type=int, default=8)
    ap.add_argument("--chance-widening", type=float, default=0.5)
    ap.add_argument("--root-min-visits", type=int, default=2)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--max-new", type=int, default=0, help="本次最多新增的模型局数，0 表示不限")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model_specs = parse_models(args.models)
    config = {
        "models": dict(model_specs), "dist": args.dist, "episodes": args.episodes,
        "seed": args.seed, "max_moves": args.max_moves,
        "simulations": args.simulations, "depth": args.depth,
        "c_puct": args.c_puct, "death_penalty": args.death_penalty,
        "chance_samples": args.chance_samples,
        "chance_widening": args.chance_widening,
        "root_min_visits": args.root_min_visits,
    }
    rows = []
    if os.path.exists(args.json_out):
        with open(args.json_out) as f:
            saved = json.load(f)
        if saved.get("config") != config:
            raise ValueError("已有评测文件的配置与当前参数不同")
        rows = saved.get("rows", [])

    completed = {(row["model"], row["seed"]) for row in rows}
    loaded = {
        label: (*load_model(path, args.device), path)
        for label, path in model_specs
    }
    weights, capped = load_weights(args.dist)
    total = args.episodes * len(model_specs)
    new_count = 0
    for episode_id in range(args.episodes):
        seed = args.seed + episode_id
        for label, _path in model_specs:
            if (label, seed) in completed:
                continue
            net, gamma, _model_path = loaded[label]
            game = Game(
                rng=random.Random(seed),
                drop_sampler=make_sampler(weights, capped),
            )
            started = time.time()
            n9_steps = []
            while not game.dead and game.moves < args.max_moves:
                action = puct_search(
                    game, net, args.device, args.simulations, args.depth,
                    c_puct=args.c_puct, gamma=gamma,
                    death_penalty=args.death_penalty,
                    chance_samples=args.chance_samples,
                    chance_widening=args.chance_widening,
                    root_min_visits=args.root_min_visits,
                )
                if action is None or not game.move(*action):
                    break
                n9_steps.extend([game.moves] * sum(event >= 9 for event in game.events))
            rows.append({
                "model": label, "seed": seed, "n9": len(n9_steps),
                "moves": game.moves, "dead": game.dead,
                "n9_steps": n9_steps, "seconds": time.time() - started,
            })
            completed.add((label, seed))
            new_count += 1
            save_result(args.json_out, config, rows)
            print(
                f"[{len(rows):>4}/{total}] {label} seed={seed} "
                f"9={len(n9_steps)} moves={game.moves}",
                flush=True,
            )
            if args.max_new > 0 and new_count >= args.max_new:
                print_summary([name for name, _path in model_specs], rows)
                return
    print_summary([label for label, _path in model_specs], rows)


if __name__ == "__main__":
    main()
