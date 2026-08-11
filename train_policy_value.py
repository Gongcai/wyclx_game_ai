"""从长局 beam 示范联合训练等变策略与价值网络。"""

import argparse
import itertools
import json
import os
import random

import torch
import torch.nn.functional as F

from agents.policy_value import PolicyValueNet
from agents.dqn import COLS, GRID_CLASSES, H, N_ACTIONS, action_index, index_action


PERMUTATIONS = torch.tensor(list(itertools.permutations(range(COLS))), dtype=torch.long)
ACTION_MAPS = []
for permutation in PERMUTATIONS.tolist():
    inverse = [permutation.index(old) for old in range(COLS)]
    ACTION_MAPS.append([
        action_index(inverse[src], inverse[dst])
        for src, dst in (index_action(i) for i in range(N_ACTIONS))
    ])
ACTION_MAPS = torch.tensor(ACTION_MAPS, dtype=torch.long)


def augment_batch(states, policy_targets, actions, generator):
    batch = len(states)
    ids = torch.randint(len(PERMUTATIONS), (batch,), generator=generator)
    permutations = PERMUTATIONS[ids]
    grid_size = COLS * H * GRID_CLASSES
    grid = states[:, :grid_size].reshape(batch, COLS, H, GRID_CLASSES)
    gather_grid = permutations[:, :, None, None].expand(-1, -1, H, GRID_CLASSES)
    out = states.clone()
    out[:, :grid_size] = torch.gather(grid, 1, gather_grid).reshape(batch, -1)
    preview = states[:, grid_size + 4:grid_size + 4 + COLS]
    out[:, grid_size + 4:grid_size + 4 + COLS] = torch.gather(preview, 1, permutations)
    out_policy = torch.zeros_like(policy_targets)
    out_policy.scatter_(1, ACTION_MAPS[ids], policy_targets)
    out_actions = torch.gather(
        ACTION_MAPS[ids], 1, actions.long().unsqueeze(1),
    ).squeeze(1)
    return out, out_policy, out_actions


def ended_by_death(episode):
    if "dead" in episode:
        return bool(episode["dead"])
    if "dones" in episode and len(episode["dones"]):
        return bool(episode["dones"][-1])
    # 旧版 PUCT 长局未保存 dead；这些轨迹均运行至死亡。
    return True


def episode_targets(n9, horizon, gamma, death_horizon, terminal_death=True):
    n = len(n9)
    future = torch.zeros(n)
    distance = torch.ones(n)
    for i in range(n):
        discount = 1.0
        for j in range(i, min(n, i + horizon)):
            future[i] += discount * n9[j]
            if n9[j] > 0 and distance[i] == 1:
                distance[i] = (j - i + 1) / horizon
            discount *= gamma
    death = torch.zeros(n)
    if terminal_death:
        death[max(0, n - death_horizon):] = 1.0
    return future, distance, death


@torch.no_grad()
def metrics(
    net, states, policy_targets, actions, future, distance, death,
    device, batch, afterstate_q,
):
    correct = 0
    policy_loss = 0.0
    value_error = distance_error = death_error = q_error = 0.0
    for start in range(0, len(policy_targets), batch):
        end = start + batch
        policy, pred_value, pred_distance, pred_death, pred_q = net(
            states[start:end].to(device)
        )
        targets = policy_targets[start:end].to(device)
        correct += int((policy.argmax(1).cpu() == policy_targets[start:end].argmax(1)).sum())
        policy_loss += float((-(targets * F.log_softmax(policy, dim=1)).sum(1)).sum())
        value_error += float((pred_value.cpu() - future[start:end]).abs().sum())
        distance_error += float((pred_distance.cpu() - distance[start:end]).abs().sum())
        death_error += float((pred_death.cpu() - death[start:end]).abs().sum())
        if afterstate_q:
            chosen_q = pred_q.gather(
                1, actions[start:end].to(device).long().unsqueeze(1),
            ).squeeze(1)
            q_error += float((chosen_q.cpu() - future[start:end]).abs().sum())
    n = len(policy_targets)
    return (
        correct / n, policy_loss / n, value_error / n,
        distance_error / n, death_error / n, q_error / n,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--value-w", type=float, default=1.0)
    ap.add_argument("--distance-w", type=float, default=1.0)
    ap.add_argument("--death-w", type=float, default=0.5)
    ap.add_argument("--afterstate-q-w", type=float, default=0.0)
    ap.add_argument("--init-model", default=None)
    ap.add_argument("--freeze-base-for-q", action="store_true")
    ap.add_argument(
        "--policy-head-only", action="store_true",
        help="冻结编码器和价值头，只蒸馏 PUCT 策略头",
    )
    ap.add_argument("--death-horizon", type=int, default=16)
    ap.add_argument("--mature-policy-weight", type=float, default=1.0, help="首个9之后状态的额外策略权重")
    ap.add_argument("--high-tile-policy-weight", type=float, default=0.0, help="棋盘含7/8状态的额外策略权重")
    ap.add_argument("--elite-score-threshold", type=int, default=0, help="整局9数量达到该值视为精英轨迹")
    ap.add_argument("--elite-policy-weight", type=float, default=0.0, help="精英轨迹的额外策略权重")
    ap.add_argument("--puct-policy-weight", type=float, default=1.0, help="PUCT 软策略目标的权重倍率")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    datasets = [torch.load(path, weights_only=True, map_location="cpu") for path in args.demos]
    episodes = [episode for data in datasets for episode in data.get("episodes", [])]
    if not episodes or "n9" not in episodes[0]:
        raise ValueError("需要新版长局示范（每局必须包含 n9）")
    ids = list(range(len(episodes)))
    random.shuffle(ids)
    n_val = max(1, round(len(ids) * args.val_ratio))
    val_ids = set(ids[:n_val])

    def flatten(selected):
        states = torch.cat([episode["states"] for episode in selected]).float()
        policy_targets = torch.cat([
            episode.get(
                "policy_targets",
                F.one_hot(episode["actions"].long(), N_ACTIONS).float(),
            )
            for episode in selected
        ]).float()
        actions = torch.cat([episode["actions"] for episode in selected]).long()
        targets = [
            episode_targets(
                episode["n9"], args.horizon, args.gamma, args.death_horizon,
                ended_by_death(episode),
            )
            for episode in selected
        ]
        future = torch.cat([target[0] for target in targets])
        distance = torch.cat([target[1] for target in targets])
        death = torch.cat([target[2] for target in targets])
        mature = torch.cat([
            (torch.cumsum(episode["n9"], dim=0) - episode["n9"] > 0).float()
            for episode in selected
        ])
        source_weight = torch.cat([
            torch.full(
                (len(episode["n9"]),),
                args.puct_policy_weight if "policy_targets" in episode else 1.0,
            )
            for episode in selected
        ])
        grid = states[:, :COLS * H * GRID_CLASSES].reshape(
            -1, COLS, H, GRID_CLASSES,
        )
        high_tile = (grid[..., 6:8].sum(dim=(1, 2, 3)) > 0).float()
        elite = torch.cat([
            torch.full(
                (len(episode["n9"]),),
                float(int(episode["n9"].sum()) >= args.elite_score_threshold),
            )
            for episode in selected
        ])
        policy_weight = source_weight * (
            1.0
            + args.mature_policy_weight * mature
            + args.high_tile_policy_weight * high_tile
            + args.elite_policy_weight * elite
        )
        return (
            states, policy_targets, actions, future, distance, death,
            policy_weight,
        )

    train = flatten([episode for i, episode in enumerate(episodes) if i not in val_ids])
    val = flatten([episode for i, episode in enumerate(episodes) if i in val_ids])
    use_afterstate_q = args.afterstate_q_w > 0
    net = PolicyValueNet(
        value_outputs=3, afterstate_q=use_afterstate_q,
    ).to(args.device)
    if args.init_model:
        checkpoint = torch.load(
            args.init_model, weights_only=True, map_location=args.device,
        )
        missing, unexpected = net.load_state_dict(
            checkpoint["model"], strict=False,
        )
        allowed_missing = {
            key for key in net.state_dict() if key.startswith("afterstate_q_head.")
        }
        if set(missing) != allowed_missing or unexpected:
            raise ValueError(
                f"初始化模型结构不兼容: missing={missing}, unexpected={unexpected}"
            )
    if args.freeze_base_for_q:
        if not use_afterstate_q or not args.init_model:
            raise ValueError("--freeze-base-for-q 需要 Q 头和 --init-model")
        for name, parameter in net.named_parameters():
            parameter.requires_grad = name.startswith("afterstate_q_head.")
    if args.policy_head_only:
        if args.freeze_base_for_q or use_afterstate_q or not args.init_model:
            raise ValueError("--policy-head-only 需要 --init-model，且不能与 Q 头训练同时使用")
        for name, parameter in net.named_parameters():
            parameter.requires_grad = name.startswith("policy_head.")
    opt = torch.optim.Adam(
        [parameter for parameter in net.parameters() if parameter.requires_grad],
        lr=args.lr,
    )
    generator = torch.Generator().manual_seed(args.seed)
    best = None
    best_loss = float("inf")
    print(f"demos={args.demos} train={len(train[1])} val={len(val[1])} device={args.device}")
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = torch.randperm(len(train[1]), generator=generator)
        for start in range(0, len(order), args.batch):
            idx = order[start:start + args.batch]
            states, policy_targets, actions = augment_batch(
                train[0][idx], train[1][idx], train[2][idx], generator,
            )
            policy, pred_value, pred_distance, pred_death, pred_q = net(
                states.to(args.device)
            )
            targets = policy_targets.to(args.device)
            per_sample_policy = -(targets * F.log_softmax(policy, dim=1)).sum(1)
            policy_weight = train[6][idx].to(args.device)
            policy_loss = (per_sample_policy * policy_weight).sum() / policy_weight.sum()
            value_loss = F.smooth_l1_loss(pred_value, train[3][idx].to(args.device))
            distance_loss = F.smooth_l1_loss(pred_distance, train[4][idx].to(args.device))
            death_loss = F.binary_cross_entropy(
                pred_death, train[5][idx].to(args.device),
            )
            if use_afterstate_q:
                chosen_q = pred_q.gather(
                    1, actions.to(args.device).unsqueeze(1),
                ).squeeze(1)
                q_loss = F.smooth_l1_loss(
                    chosen_q, train[3][idx].to(args.device),
                )
            else:
                q_loss = torch.zeros((), device=args.device)
            loss = (
                policy_loss
                + args.value_w * value_loss
                + args.distance_w * distance_loss
                + args.death_w * death_loss
                + args.afterstate_q_w * q_loss
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        stats = metrics(
            net, val[0], val[1], val[2], val[3], val[4], val[5],
            args.device, args.batch, use_afterstate_q,
        )
        if args.policy_head_only:
            val_loss = stats[1]
        else:
            val_loss = (
                0.2 * stats[1] + stats[2] + stats[3]
                + args.death_w * stats[4]
                + args.afterstate_q_w * stats[5]
            )
        if val_loss < best_loss:
            best_loss = val_loss
            best = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        print(
            f"[{epoch:>3}/{args.epochs}] val_policy={stats[0]:.2%} "
            f"policy_ce={stats[1]:.3f} value_mae={stats[2]:.3f} "
            f"next9_mae={stats[3]:.3f} death_mae={stats[4]:.3f} "
            f"after_q_mae={stats[5]:.3f}",
            flush=True,
        )

    out = args.out or os.path.join(os.path.dirname(args.demos[0]), "policy-value.pt")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(
        {
            "model": best,
            "horizon": args.horizon,
            "gamma": args.gamma,
            "value_outputs": 3,
            "death_horizon": args.death_horizon,
            "afterstate_q": use_afterstate_q,
            "dists": [data.get("dist") for data in datasets],
        },
        out,
    )
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(
            {
                "best_value_distance_mae": best_loss,
                "demos": args.demos,
                "mature_policy_weight": args.mature_policy_weight,
                "puct_policy_weight": args.puct_policy_weight,
                "high_tile_policy_weight": args.high_tile_policy_weight,
                "elite_score_threshold": args.elite_score_threshold,
                "elite_policy_weight": args.elite_policy_weight,
                "death_weight": args.death_w,
                "death_horizon": args.death_horizon,
                "afterstate_q_weight": args.afterstate_q_w,
                "init_model": args.init_model,
                "freeze_base_for_q": args.freeze_base_for_q,
                "policy_head_only": args.policy_head_only,
            },
            f,
            indent=2,
        )
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()
