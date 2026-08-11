"""从长局 beam 示范联合训练等变策略与价值网络。"""

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F

from agents.policy_value import PolicyValueNet
from train_bc import augment_columns


def episode_targets(n9, horizon, gamma):
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
    return future, distance


@torch.no_grad()
def metrics(net, states, actions, future, distance, device, batch):
    correct = 0
    value_error = distance_error = 0.0
    for start in range(0, len(actions), batch):
        end = start + batch
        policy, pred_value, pred_distance = net(states[start:end].to(device))
        correct += int((policy.argmax(1).cpu() == actions[start:end]).sum())
        value_error += float((pred_value.cpu() - future[start:end]).abs().sum())
        distance_error += float((pred_distance.cpu() - distance[start:end]).abs().sum())
    n = len(actions)
    return correct / n, value_error / n, distance_error / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--value-w", type=float, default=1.0)
    ap.add_argument("--distance-w", type=float, default=1.0)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = torch.load(args.demos, weights_only=True, map_location="cpu")
    episodes = data.get("episodes", [])
    if not episodes or "n9" not in episodes[0]:
        raise ValueError("需要新版长局示范（每局必须包含 n9）")
    ids = list(range(len(episodes)))
    random.shuffle(ids)
    n_val = max(1, round(len(ids) * args.val_ratio))
    val_ids = set(ids[:n_val])

    def flatten(selected):
        states = torch.cat([episode["states"] for episode in selected]).float()
        actions = torch.cat([episode["actions"] for episode in selected]).long()
        targets = [episode_targets(episode["n9"], args.horizon, args.gamma) for episode in selected]
        future = torch.cat([target[0] for target in targets])
        distance = torch.cat([target[1] for target in targets])
        return states, actions, future, distance

    train = flatten([episode for i, episode in enumerate(episodes) if i not in val_ids])
    val = flatten([episode for i, episode in enumerate(episodes) if i in val_ids])
    net = PolicyValueNet().to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    best = None
    best_loss = float("inf")
    print(f"demos={args.demos} train={len(train[1])} val={len(val[1])} device={args.device}")
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = torch.randperm(len(train[1]), generator=generator)
        for start in range(0, len(order), args.batch):
            idx = order[start:start + args.batch]
            states, actions = augment_columns(train[0][idx], train[1][idx], generator)
            policy, pred_value, pred_distance = net(states.to(args.device))
            policy_loss = F.cross_entropy(policy, actions.to(args.device))
            value_loss = F.smooth_l1_loss(pred_value, train[2][idx].to(args.device))
            distance_loss = F.smooth_l1_loss(pred_distance, train[3][idx].to(args.device))
            loss = policy_loss + args.value_w * value_loss + args.distance_w * distance_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        stats = metrics(net, *val, args.device, args.batch)
        val_loss = stats[1] + stats[2]
        if val_loss < best_loss:
            best_loss = val_loss
            best = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        print(
            f"[{epoch:>3}/{args.epochs}] val_policy={stats[0]:.2%} "
            f"value_mae={stats[1]:.3f} next9_mae={stats[2]:.3f}",
            flush=True,
        )

    out = args.out or os.path.join(os.path.dirname(args.demos), "policy-value.pt")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(
        {
            "model": best,
            "horizon": args.horizon,
            "gamma": args.gamma,
            "dist": data.get("dist"),
        },
        out,
    )
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump({"best_value_distance_mae": best_loss, "demos": args.demos}, f, indent=2)
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()
