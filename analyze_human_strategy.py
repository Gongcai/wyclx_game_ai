"""按人类长局结构指标分析 PUCT/beam 示范轨迹。"""

import argparse
import math
import statistics

import torch

from agents.dqn import COLS, GRID_CLASSES, H
from agents.human_strategy import human_structure_from_stacks


def decode_stacks(state):
    grid = state[:COLS * H * GRID_CLASSES].reshape(COLS, H, GRID_CLASSES)
    return [
        [int(row.argmax()) + 1 for row in grid[col] if row.sum() > 0]
        for col in range(COLS)
    ]


def mean_metrics(states):
    metrics = [human_structure_from_stacks(decode_stacks(state)) for state in states]
    if not metrics:
        return None
    names = (
        "score", "empty_cols", "inversions", "ordered_ratio",
        "top_pairs", "low_bottom_cols", "low_value_cols",
    )
    return {
        name: statistics.mean(getattr(metric, name) for metric in metrics)
        for name in names
    }


def correlation(xs, ys):
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs)
        * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", nargs="+", required=True)
    ap.add_argument("--low-max", type=int, default=3)
    ap.add_argument("--elite-min", type=int, default=6)
    args = ap.parse_args()

    rows = []
    for path in args.demos:
        data = torch.load(path, weights_only=True, map_location="cpu")
        for episode in data.get("episodes", []):
            n9 = len(episode.get("n9_steps", []))
            first9 = episode["n9_steps"][0] if n9 else len(episode["states"])
            metrics = mean_metrics(episode["states"][first9:])
            if metrics is not None:
                rows.append((n9, metrics))
    if not rows:
        raise ValueError("轨迹中没有首个 9 之后的成熟状态")

    groups = (
        ("高产", lambda n9: n9 >= args.elite_min),
        ("中等", lambda n9: args.low_max < n9 < args.elite_min),
        ("低产", lambda n9: n9 <= args.low_max),
    )
    print("分组  局数  结构分  空列  逆序数  有序率  顶部对子  低值垫底列")
    for label, predicate in groups:
        selected = [metrics for n9, metrics in rows if predicate(n9)]
        if not selected:
            continue
        avg = {
            name: statistics.mean(item[name] for item in selected)
            for name in selected[0]
        }
        print(
            f"{label:<4} {len(selected):>4}  {avg['score']:.3f}  "
            f"{avg['empty_cols']:.2f}  {avg['inversions']:.2f}  "
            f"{avg['ordered_ratio']:.3f}  {avg['top_pairs']:.2f}  "
            f"{avg['low_bottom_cols']:.2f}"
        )
    print(
        "整局9数量与成熟盘面平均结构分相关系数: "
        f"{correlation([n9 for n9, _ in rows], [m['score'] for _, m in rows]):.3f}"
    )


if __name__ == "__main__":
    main()
