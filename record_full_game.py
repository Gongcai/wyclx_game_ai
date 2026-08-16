"""录制一整局（到死亡或步数上限），输出逐步骤文本渲染 + JSON 轨迹。

与 record_game.py（只到首个 9）不同，本工具跑到整局结束，并区分死亡原因
（移动溢出 vs 掉落溢出），供人类复盘对比 AI 与人类策略差距。

用法：
  .venv/bin/python record_full_game.py --policy puct --model runs/demos/high-halving8-gumbel48-pv.pt --seed 5
  .venv/bin/python record_full_game.py --policy heuristic --seed 5 --width 8

输出：
  --json-out  完整轨迹（每步动作/预告/合并/盘面/空列/最高列/栈顶对子/掉落）
  --txt-out   文本渲染（每步盘面 + 死亡前 3 步 + 死亡盘面 + 整局摘要）
"""

import argparse
import json
import random

import torch

from agents.dist import load_weights, make_sampler
from agents.policy_value import PolicyValueNet
from agents.puct import PuctTree, advance_tree, puct_search
from game import Game, MAX_HEIGHT


def top_pair_count(game):
    return sum(
        len(st) >= 2 and st[0] == st[1] for st in game.stacks
    )


def buried_low(game):
    """被埋低牌数：压在 >=2 个更高牌之下的 1/2/3 数量（人类规则：低牌不垫底）。"""
    n = 0
    for st in game.stacks:
        for i, v in enumerate(st):
            if v <= 3 and any(w > v for w in st[i + 1:]):
                n += 1
    return n


def run(policy, model, seed, dist, max_moves, sims, depth, c_puct, death_penalty,
        chance_samples, chance_widening, root_min_visits, width, device):
    net = None
    gamma = 0.99
    if policy == "puct":
        checkpoint = torch.load(model, weights_only=True, map_location=device)
        net = PolicyValueNet(
            value_outputs=checkpoint.get("value_outputs", 2),
            afterstate_q=checkpoint.get("afterstate_q", False),
            history_features=checkpoint.get("history_features", False),
        ).to(device)
        net.load_state_dict(checkpoint["model"])
        net.eval()
        gamma = checkpoint.get("gamma", 0.99)

    weights, capped = load_weights(dist)
    game = Game(rng=random.Random(seed), drop_sampler=make_sampler(weights, capped))
    tree = PuctTree()
    trace = []
    lines = []
    lines.append(f"===== 开局 seed={seed} policy={policy} =====")
    lines.append(game.render())
    lines.append("")

    death = None
    while not game.dead and game.moves < max_moves:
        step = game.moves
        pre = [list(st) for st in game.stacks]
        preview = list(game.preview) if game.preview is not None else None
        if policy == "heuristic":
            from agents.heuristic import heuristic_policy_beam
            action = heuristic_policy_beam(game, width=width, depth=4, seed=seed)
        else:
            action = puct_search(
                game, net, device, sims, depth,
                c_puct=c_puct, gamma=gamma,
                death_penalty=death_penalty,
                chance_samples=chance_samples,
                chance_widening=chance_widening,
                root_min_visits=root_min_visits,
                tree=tree,
            )
        if action is None:
            break
        # 分阶段执行以归因死亡：移动合并溢出 vs 掉落溢出
        ok = game.move_afterstate(*action)
        if not ok:
            break
        cause = None
        if game.dead:
            cause = "移动溢出（合并后列高>7）"
        else:
            game.resolve_afterstate()
            if game.dead:
                cause = "掉落溢出（预告掉落压死）"
        if policy == "puct":
            advance_tree(tree, action, game)
        events = list(game.events)
        entry = {
            "step": game.moves,
            "action": [action[0] + 1, action[1] + 1],  # 1-based 显示
            "phase": game.moves % 4,
            "preview_before": preview,
            "events": events,
            "score": game.score,
            "stacks": [list(st) for st in game.stacks],
            "empty_cols": sum(not st for st in game.stacks),
            "max_height": max(len(st) for st in game.stacks),
            "top_pairs": top_pair_count(game),
            "buried_low": buried_low(game),
            "last_drop": game.last_drop,
        }
        trace.append(entry)
        a = entry["action"]
        lines.append(f"===== 步 {game.moves}  动作 {a[0]}→{a[1]}  空列{entry['empty_cols']} "
                     f"最高{entry['max_height']} 顶对{entry['top_pairs']} 埋低{entry['buried_low']} =====")
        if preview:
            lines.append(f"预告: [{' '.join(map(str, preview))}]")
        if events:
            lines.append(f"合并事件: {events}")
        lines.append(f"得分: {game.score}")
        lines.append(game.render())
        lines.append("")
        if cause:
            death = {
                "cause": cause,
                "step": game.moves,
                "last_drop": game.last_drop,
                "stacks": [list(st) for st in game.stacks],
            }
            lines.append(f"===== 死亡: {cause} =====  (第 {game.moves} 步)")
            lines.append(game.render())
            lines.append("")
            # 死亡前 3 步盘面（来自 trace）
            lines.append("----- 死亡前 3 步 -----")
            for e in trace[-4:-1]:
                lines.append(f"步 {e['step']} 动作 {e['action'][0]}→{e['action'][1]} "
                             f"空列{e['empty_cols']} 最高{e['max_height']} 顶对{e['top_pairs']} 埋低{e['buried_low']}")
            lines.append("")
            break

    # 整局摘要
    n9 = [e for e in trace if 9 in e["events"]]
    lines.append("===== 整局摘要 =====")
    lines.append(f"步数: {game.moves}  得分: {game.score}  合成9数: {len(n9)}  "
                 f"最大合成: {game.max_merged}  死亡: {death['cause'] if death else '未死'}")
    if n9:
        lines.append("9 的步数: " + ", ".join(str(e["step"]) for e in n9))
    cycles = {}
    for e in trace:
        cycles.setdefault(e["step"] // 4, []).append(e["empty_cols"])
    peak = {c: max(v) for c, v in cycles.items()}
    lines.append("每周期空列峰值: " + " ".join(f"c{i}:{p}" for i, p in sorted(peak.items())))
    lines.append(f"空列峰值=0 的周期数: {sum(1 for p in peak.values() if p == 0)} / {len(peak)}")
    seven = [e["step"] for e in trace if e["max_height"] >= 7]
    lines.append(f"出现 7 高列的步数: {len(seven)} / {len(trace)} 步（首次 {seven[0] if seven else '-'}）")
    print("\n".join(lines))
    return {"seed": seed, "policy": policy, "trace": trace, "death": death,
            "summary": {
                "steps": game.moves, "score": game.score, "n9": len(n9),
                "max_merged": game.max_merged, "death_cause": death["cause"] if death else None,
                "n9_steps": [e["step"] for e in n9],
                "cycles_empty_zero": sum(1 for p in peak.values() if p == 0),
                "cycles_total": len(peak),
                "steps_with_7high": len(seven),
            }}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=("puct", "heuristic"), required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dist", default="high")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--max-moves", type=int, default=1000)
    ap.add_argument("--simulations", type=int, default=128)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--death-penalty", type=float, default=0.5)
    ap.add_argument("--chance-samples", type=int, default=8)
    ap.add_argument("--chance-widening", type=float, default=0.5)
    ap.add_argument("--root-min-visits", type=int, default=2)
    ap.add_argument("--width", type=int, default=8, help="启发式 beam 宽度")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--txt-out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import io
    import sys
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    result = run(args.policy, args.model, args.seed, args.dist, args.max_moves,
                 args.simulations, args.depth, args.c_puct, args.death_penalty,
                 args.chance_samples, args.chance_widening, args.root_min_visits,
                 args.width, args.device)
    sys.stdout = old
    text = buf.getvalue()
    if args.txt_out:
        open(args.txt_out, "w").write(text)
    if args.json_out:
        json.dump(result, open(args.json_out, "w"), ensure_ascii=False, indent=1)
    print(text)


if __name__ == "__main__":
    main()
