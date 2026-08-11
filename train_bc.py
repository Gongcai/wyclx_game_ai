"""从 beam search 示范训练与 DQN 兼容的行为克隆策略。"""

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F

from agents.dqn import COLS, GRID_CLASSES, H, META_DIM, Net


def load_episodes(path):
    data = torch.load(path, weights_only=True, map_location="cpu")
    episodes = data.get("episodes", [])
    if not episodes:
        raise ValueError(f"示范文件没有成功轨迹: {path}")
    for episode in episodes:
        states = episode["states"]
        if states.ndim != 2 or states.shape[1] != 6 * 7 * 9 + META_DIM:
            raise ValueError("示范状态维度与当前网络不兼容")
    return data, episodes


def batches(states, actions, batch_size, generator):
    order = torch.randperm(len(actions), generator=generator)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        yield states[idx], actions[idx]


def augment_columns(states, actions, generator):
    """对每个样本随机重排列，保持游戏动力学和动作语义不变。"""
    grid_size = COLS * H * GRID_CLASSES
    out = states.clone()
    grid = states[:, :grid_size].reshape(-1, COLS, H, GRID_CLASSES)
    preview = states[:, grid_size + 4:grid_size + 4 + COLS]
    new_actions = actions.clone()
    for i in range(len(states)):
        # perm[new_col] = old_col；inverse 将旧动作映射到新列。
        perm = torch.randperm(COLS, generator=generator)
        inverse = torch.empty(COLS, dtype=torch.long)
        inverse[perm] = torch.arange(COLS)
        out[i, :grid_size] = grid[i, perm].reshape(-1)
        out[i, grid_size + 4:grid_size + 4 + COLS] = preview[i, perm]
        src = int(actions[i]) // (COLS - 1)
        dst = int(actions[i]) % (COLS - 1)
        if dst >= src:
            dst += 1
        src = int(inverse[src])
        dst = int(inverse[dst])
        new_actions[i] = src * (COLS - 1) + dst - (1 if dst > src else 0)
    return out, new_actions


@torch.no_grad()
def accuracy(net, states, actions, device, batch_size):
    correct = 0
    for start in range(0, len(actions), batch_size):
        logits = net(states[start:start + batch_size].to(device))
        correct += int((logits.argmax(1).cpu() == actions[start:start + batch_size]).sum())
    return correct / len(actions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True, help="generate_demos.py 生成的 .pt 文件")
    ap.add_argument("--out", default=None, help="输出 DQN 兼容权重路径")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-augment-columns", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.epochs <= 0 or args.batch <= 0 or not 0 < args.val_ratio < 1:
        raise ValueError("epochs、batch 必须为正数，val-ratio 必须在 (0, 1) 内")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data, episodes = load_episodes(args.demos)
    shuffled = list(range(len(episodes)))
    random.shuffle(shuffled)
    n_val = max(1, round(len(episodes) * args.val_ratio))
    val_ids = set(shuffled[:n_val])
    train_eps = [episode for i, episode in enumerate(episodes) if i not in val_ids]
    val_eps = [episode for i, episode in enumerate(episodes) if i in val_ids]
    train_s = torch.cat([episode["states"] for episode in train_eps]).float()
    train_a = torch.cat([episode["actions"] for episode in train_eps]).long()
    val_s = torch.cat([episode["states"] for episode in val_eps]).float()
    val_a = torch.cat([episode["actions"] for episode in val_eps]).long()

    net = Net().to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    best_acc = -1.0
    best_state = None
    print(
        f"demos={args.demos} dist={data.get('dist')} train={len(train_a)} "
        f"val={len(val_a)} device={args.device}"
    )
    for epoch in range(1, args.epochs + 1):
        net.train()
        loss_sum = 0.0
        count = 0
        for states, actions in batches(train_s, train_a, args.batch, generator):
            if not args.no_augment_columns:
                states, actions = augment_columns(states, actions, generator)
            logits = net(states.to(args.device))
            loss = F.cross_entropy(logits, actions.to(args.device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(actions)
            count += len(actions)
        net.eval()
        train_acc = accuracy(net, train_s, train_a, args.device, args.batch)
        val_acc = accuracy(net, val_s, val_a, args.device, args.batch)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        print(
            f"[{epoch:>3}/{args.epochs}] loss={loss_sum / count:.4f} "
            f"train_acc={train_acc:.3%} val_acc={val_acc:.3%}",
            flush=True,
        )

    out = args.out or os.path.join(os.path.dirname(args.demos), "bc.pt")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(best_state, out)
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(
            {
                "demos": args.demos,
                "dist": data.get("dist"),
                "epochs": args.epochs,
                "train_transitions": len(train_a),
                "val_transitions": len(val_a),
                "best_val_accuracy": best_acc,
                "column_augmentation": not args.no_augment_columns,
            },
            f,
            indent=2,
        )
    print(f"已保存 {out}, 最佳验证动作准确率 {best_acc:.3%}")


if __name__ == "__main__":
    main()
