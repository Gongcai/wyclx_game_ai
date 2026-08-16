"""用当前 Policy+Value PUCT 生成自举长局数据。"""

import argparse
import os
import random
import time

import torch

from agents.dist import load_weights, make_real_sampler, make_sampler
from agents.dqn import encode, index_action
from agents.policy_value import PolicyValueNet, hidden_from_checkpoint
from agents.puct import PuctTree, advance_tree, puct_search
from game import Game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dist", default="high")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--max-moves", type=int, default=300)
    ap.add_argument("--simulations", type=int, default=64)
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--c-puct", type=float, default=1.5, help="PUCT 先验探索强度")
    ap.add_argument("--temperature-moves", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--death-penalty", type=float, default=0.0)
    ap.add_argument("--chance-samples", type=int, default=0, help="大于0时启用显式 afterstate chance node")
    ap.add_argument("--chance-widening", type=float, default=0.0, help="chance node 渐进扩展指数，建议 0.5")
    ap.add_argument("--tree-reuse", action="store_true", help="跨真实动作复用 chance 搜索树")
    ap.add_argument("--safe-veto", action="store_true", help="预告已知时排除必然溢出动作")
    ap.add_argument("--shape-over-w", type=float, default=0.0)
    ap.add_argument("--shape-low-w", type=float, default=0.0)
    ap.add_argument("--save-q-targets", action="store_true", help="保存根搜索的全动作 Q reanalyse 目标")
    ap.add_argument("--root-min-visits", type=int, default=0, help="reanalyse 时强制根动作最少访问次数")
    ap.add_argument("--root-sequential-halving", type=int, default=0, help="Gumbel 根候选数，0 表示关闭")
    ap.add_argument("--root-q-scale", type=float, default=2.0)
    ap.add_argument("--root-gumbel-noise", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=None, help="默认读取 Policy+Value checkpoint")
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    checkpoint = torch.load(args.model, weights_only=True, map_location=args.device)
    net = PolicyValueNet(hidden=hidden_from_checkpoint(checkpoint),
        value_outputs=checkpoint.get("value_outputs", 2),
        afterstate_q=checkpoint.get("afterstate_q", False),
        history_features=checkpoint.get("history_features", False),
    ).to(args.device)
    net.load_state_dict(checkpoint["model"])
    net.eval()
    gamma = args.gamma if args.gamma is not None else checkpoint.get("gamma", 0.99)
    if args.dist == "real":
        drop_sampler = make_real_sampler()
    else:
        weights, capped = load_weights(args.dist)
        drop_sampler = make_sampler(weights, capped)
    if args.tree_reuse and args.chance_samples <= 0:
        raise ValueError("--tree-reuse 需要 --chance-samples > 0")
    sample_generator = torch.Generator().manual_seed(args.seed)
    episodes = []
    started = time.time()
    for episode_id in range(args.episodes):
        seed = args.seed + episode_id
        game = Game(rng=random.Random(seed), drop_sampler=drop_sampler)
        states = []
        actions = []
        policy_targets = []
        q_targets = []
        q_masks = []
        n9_events = []
        tree = PuctTree() if args.tree_reuse else None
        while not game.dead and game.moves < args.max_moves:
            states.append(encode(game, history=net.history_features))
            search_result = puct_search(
                game, net, args.device, args.simulations, args.depth,
                c_puct=args.c_puct,
                gamma=gamma,
                death_penalty=args.death_penalty,
                chance_samples=args.chance_samples,
                chance_widening=args.chance_widening,
                root_min_visits=args.root_min_visits,
                root_sequential_halving=args.root_sequential_halving,
                root_q_scale=args.root_q_scale,
                root_gumbel_noise=args.root_gumbel_noise,
                tree=tree,
                shape_over_w=args.shape_over_w,
                shape_low_w=args.shape_low_w,
                safe_veto=args.safe_veto,
                return_policy=True,
                return_q=args.save_q_targets,
            )
            if args.save_q_targets:
                action, policy, root_q, root_q_mask = search_result
                q_targets.append(root_q)
                q_masks.append(root_q_mask)
            else:
                action, policy = search_result
            if game.moves < args.temperature_moves and args.temperature > 0:
                probs = policy.pow(1.0 / args.temperature)
                if probs.sum() > 0:
                    action_id = int(torch.multinomial(
                        probs / probs.sum(), 1, generator=sample_generator
                    ))
                    action = index_action(action_id)
            actions.append(action[0] * 5 + action[1] - (1 if action[1] > action[0] else 0))
            policy_targets.append(policy)
            if not game.move(*action):
                break
            if tree is not None:
                advance_tree(tree, action, game)
            n9_events.append(sum(event >= 9 for event in game.events))
        episode = {
            "seed": seed,
            "score": game.score,
            "n9_count": game.n9_count,
            "moves": game.moves,
            "dead": game.dead,
            "states": torch.stack(states),
            "actions": torch.tensor(actions, dtype=torch.long),
            "policy_targets": torch.stack(policy_targets),
            "n9": torch.tensor(n9_events, dtype=torch.float32),
            "n9_steps": [
                i + 1 for i, count in enumerate(n9_events) for _ in range(count)
            ],
        }
        if args.save_q_targets:
            episode["afterstate_q_targets"] = torch.stack(q_targets)
            episode["afterstate_q_masks"] = torch.stack(q_masks)
        episodes.append(episode)
        total_n9 = sum(len(episode["n9_steps"]) for episode in episodes)
        print(
            f"[{episode_id + 1:>4}/{args.episodes}] 本局9={game.n9_count} "
            f"累计9={total_n9}",
            flush=True,
        )

    out = args.out or os.path.join("runs", "demos", f"{args.dist}-puct.pt")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(
        {
            "version": 1,
            "kind": "puct-selfplay",
            "dist": args.dist,
            "source_model": args.model,
            "simulations": args.simulations,
            "depth": args.depth,
            "c_puct": args.c_puct,
            "death_penalty": args.death_penalty,
            "chance_samples": args.chance_samples,
            "chance_widening": args.chance_widening,
            "tree_reuse": args.tree_reuse,
            "safe_veto": args.safe_veto,
            "shape_over_w": args.shape_over_w,
            "shape_low_w": args.shape_low_w,
            "has_q_targets": args.save_q_targets,
            "root_min_visits": args.root_min_visits,
            "root_sequential_halving": args.root_sequential_halving,
            "root_q_scale": args.root_q_scale,
            "root_gumbel_noise": args.root_gumbel_noise,
            "gamma": gamma,
            "episodes": episodes,
        },
        out,
    )
    game_n9 = sorted(len(episode["n9_steps"]) for episode in episodes)
    p90 = game_n9[max(0, (len(game_n9) * 9 + 9) // 10 - 1)]
    print(
        f"已保存 {out}: {len(episodes)} 局, "
        f"{sum(len(e['actions']) for e in episodes)} 步, "
        f"{sum(len(e['n9_steps']) for e in episodes)} 个9, "
        f"单局均值={sum(game_n9) / len(game_n9):.2f} P90={p90} "
        f"最高={max(game_n9)} 最长={max(len(e['actions']) for e in episodes)}步, "
        f"用时 {time.time() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
