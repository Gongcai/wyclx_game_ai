"""运行当前推荐 PUCT 配置，收集死亡前的完整轨迹并归因死亡原因，渲染成可读报告。

用法示例（与 eval_competition 推荐配置一致）：
    .venv/bin/python analyze_deaths.py \
        --model runs/demos/high-halving8-gumbel48-pv.pt \
        --dist high --episodes 40 --tail 12 \
        --json-out runs/death-analysis.jsonl

每次死亡记录：致命动作、死亡阶段(move/drop)、溢出列与高度、死亡前若干步盘面渲染。
JSONL 供程序化汇总，文本报告（--render-dir，默认 stdout）供人工阅读。
"""

import argparse
import json
import os
import random

import torch

from agents.dist import load_weights, make_sampler
from agents.policy_value import PolicyValueNet
from agents.puct import PuctTree, advance_tree, puct_search
from game import Game, MAX_HEIGHT


def classify_death(pre_move_stacks, post_move_stacks, phase, action):
    """根据 move_afterstate / resolve_afterstate 分阶段判断死因。"""
    if phase == "move":
        for c in range(len(pre_move_stacks)):
            if len(post_move_stacks[c]) > MAX_HEIGHT:
                return {
                    "cause": "move_overflow",
                    "col": c,
                    "pre_height": len(pre_move_stacks[c]),
                    "post_height": len(post_move_stacks[c]),
                }
        return {"cause": "move_unknown", "col": -1}
    for c in range(len(pre_move_stacks)):
        if len(post_move_stacks[c]) > MAX_HEIGHT:
            return {
                "cause": "drop_overflow",
                "col": c,
                "pre_height": len(pre_move_stacks[c]),
                "post_height": len(post_move_stacks[c]),
            }
    return {"cause": "drop_unknown", "col": -1}


def render_board(game, title):
    lines = [title, game.render()]
    for c in range(game.cols):
        heights = [str(len(st)) for st in game.stacks]
    lines.append("列高: " + " ".join(f"c{i}={h}" for i, h in enumerate(heights)))
    empty = sum(not st for st in game.stacks)
    lines.append(f"空列数={empty}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="puct 策略需要的 PV 模型")
    ap.add_argument("--policy", choices=("puct", "heuristic"), default="puct")
    ap.add_argument("--heur-width", type=int, default=8)
    ap.add_argument("--heur-depth", type=int, default=4)
    ap.add_argument("--heur-buried", type=float, default=1.5)
    ap.add_argument("--heur-height6", type=float, default=3.0)
    ap.add_argument("--dist", default="high")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--max-moves", type=int, default=500)
    ap.add_argument("--simulations", type=int, default=128)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--death-penalty", type=float, default=0.5)
    ap.add_argument("--chance-samples", type=int, default=8)
    ap.add_argument("--chance-widening", type=float, default=0.5)
    ap.add_argument("--root-min-visits", type=int, default=2)
    ap.add_argument("--safe-veto", action="store_true", help="第4步预告已知时硬性排除必然溢出动作（若存在安全动作）")
    ap.add_argument("--shape-over-w", type=float, default=0.0, help="叶价值塑形：列高超过6每格惩罚权重")
    ap.add_argument("--shape-low-w", type=float, default=0.0, help="叶价值塑形：低牌(<=3)每张惩罚权重")
    ap.add_argument("--tail", type=int, default=12, help="渲染死亡前多少步")
    ap.add_argument("--seed", type=int, default=310000)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--render-dir", default=None, help="文本报告目录；缺省打 stdout")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    net = None
    gamma = 0.99
    if args.policy == "puct":
        if not args.model:
            raise ValueError("puct 策略需要 --model")
        checkpoint = torch.load(args.model, weights_only=True, map_location=args.device)
        net = PolicyValueNet(
            value_outputs=checkpoint.get("value_outputs", 2),
            afterstate_q=checkpoint.get("afterstate_q", False),
            history_features=checkpoint.get("history_features", False),
        ).to(args.device)
        net.load_state_dict(checkpoint["model"])
        net.eval()
        gamma = checkpoint.get("gamma", 0.99)
    weights, capped = load_weights(args.dist)
    os.makedirs(os.path.dirname(args.render_dir) or ".", exist_ok=True) if args.render_dir else None

    causes = {}
    n9_total = 0
    moves_total = 0
    jsonl = open(args.json_out, "w") if args.json_out else None
    reports = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        game = Game(rng=random.Random(seed), drop_sampler=make_sampler(weights, capped))
        tree = PuctTree()
        history = []  # 每步: (step, action, pre_stacks, preview, events, death_phase)
        while not game.dead and game.moves < args.max_moves:
            pre_stacks = [list(st) for st in game.stacks]
            pre_preview = list(game.preview) if game.preview is not None else None
            if args.policy == "heuristic":
                from agents.heuristic import HeuristicWeights, heuristic_policy_beam
                hw = HeuristicWeights(buried=args.heur_buried, height6=args.heur_height6)
                action = heuristic_policy_beam(
                    game, w=hw, width=args.heur_width, depth=args.heur_depth,
                )
            else:
                action = puct_search(
                    game, net, args.device, args.simulations, args.depth,
                    c_puct=args.c_puct, gamma=gamma,
                    death_penalty=args.death_penalty,
                    chance_samples=args.chance_samples,
                    chance_widening=args.chance_widening,
                    root_min_visits=args.root_min_visits,
                    tree=tree,
                    shape_over_w=args.shape_over_w,
                    shape_low_w=args.shape_low_w,
                    safe_veto=args.safe_veto,
                )
            if action is None:
                break
            # 分阶段执行以归因死亡来源（与 game.move 等价）
            phase = None
            if not game.move_afterstate(*action):
                break
            if game.dead:
                phase = "move"
            else:
                game.resolve_afterstate()
                if game.dead:
                    phase = "drop"
            if phase is not None:
                death = classify_death(pre_stacks, game.stacks, phase, action)
                death.update({
                    "episode": ep, "seed": seed, "step": game.moves,
                    "action": action, "n9": game.score // 9,
                    "death": True,
                })
                causes.setdefault(death["cause"], []).append(death)
                history.append({
                    "step": game.moves, "action": list(action),
                    "pre_stacks": pre_stacks, "preview": pre_preview,
                    "events": list(game.events), "phase": phase,
                })
                break
            n9_events = sum(e >= 9 for e in game.events)
            history.append({
                "step": game.moves, "action": list(action),
                "pre_stacks": pre_stacks, "preview": pre_preview,
                "events": list(game.events), "phase": phase,
            })
            if args.policy == "puct":
                advance_tree(tree, action, game)
            n9_total += n9_events
        moves_total += game.moves

        # 渲染死亡报告
        if game.dead:
            report = [f"===== 局 {ep} seed={seed}: 死亡于第 {game.moves} 步, "
                      f"本局 {game.score // 9} 个9, 死因={history[-1]['phase']} ====="]
            for item in history[-args.tail:]:
                sim = Game(rng=random.Random(seed), drop_sampler=make_sampler(weights, capped))
                sim.stacks = [list(st) for st in item["pre_stacks"]]
                sim.moves = item["step"] - 1
                sim.max_merged = game.max_merged
                sim.preview = list(item["preview"]) if item["preview"] else None
                title = (f"步 {item['step']}  动作 {item['action'][0]+1}→{item['action'][1]+1}  "
                         f"预告={item['preview']}  事件={item['events']}  阶段={item['phase']}")
                report.append(render_board(sim, title))
            if history[-1]["phase"] == "drop":
                sim = Game(rng=random.Random(seed), drop_sampler=make_sampler(weights, capped))
                sim.stacks = [list(st) for st in game.stacks]
                sim.moves = game.moves
                sim.max_merged = game.max_merged
                report.append(render_board(sim, "死亡后盘面（掉落已应用）"))
            reports.append("\n".join(report))
            if jsonl:
                jsonl.write(json.dumps({
                    "episode": ep, "seed": seed, "n9": game.score // 9,
                    "moves": game.moves, "cause": history[-1]["phase"],
                    "overflow_col": None,
                    "tail": history[-args.tail:],
                }) + "\n")
                jsonl.flush()
        elif history:
            reports.append(f"===== 局 {ep} seed={seed}: 达到 {args.max_moves} 步上限未死亡, "
                           f"{game.score // 9} 个9 =====")

    if jsonl:
        jsonl.close()
    output = "\n\n".join(reports)
    if args.render_dir:
        path = os.path.join(args.render_dir, f"deaths-{args.seed}.txt")
        os.makedirs(args.render_dir, exist_ok=True)
        with open(path, "w") as f:
            f.write(output + "\n")
        print(f"报告已写入 {path}")
    else:
        print(output)
    print(f"\n===== 汇总: {args.episodes} 局, 平均步数={moves_total / args.episodes:.0f}, "
          f"总9={n9_total}, 平均每局9={n9_total / args.episodes:.1f} =====")
    print("死亡原因分布:")
    for cause, deaths in sorted(causes.items(), key=lambda kv: -len(kv[1])):
        cols = [d["col"] for d in deaths]
        pre = [d["pre_height"] for d in deaths]
        print(f"  {cause}: {len(deaths)} 次 | 溢出列死亡前高度分布={sorted(pre)}")


if __name__ == "__main__":
    main()
