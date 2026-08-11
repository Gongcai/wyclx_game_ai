"""用 chance PUCT reanalyse 的全动作搜索值训练独立 afterstate Q 头。"""

import argparse
import itertools
import os
import random

import torch
import torch.nn.functional as F

from agents.dqn import COLS, GRID_CLASSES, H, N_ACTIONS, action_index, index_action
from agents.policy_value import PolicyValueNet


PERMUTATIONS = torch.tensor(list(itertools.permutations(range(COLS))), dtype=torch.long)
ACTION_MAPS = []
for permutation in PERMUTATIONS.tolist():
    inverse = [permutation.index(old) for old in range(COLS)]
    ACTION_MAPS.append([
        action_index(inverse[src], inverse[dst])
        for src, dst in (index_action(i) for i in range(N_ACTIONS))
    ])
ACTION_MAPS = torch.tensor(ACTION_MAPS, dtype=torch.long)


def augment_batch(states, q_targets, q_masks, generator):
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
    out_q = torch.zeros_like(q_targets)
    out_mask = torch.zeros_like(q_masks)
    out_q.scatter_(1, ACTION_MAPS[ids], q_targets)
    out_mask.scatter_(1, ACTION_MAPS[ids], q_masks)
    return out, out_q, out_mask


@torch.no_grad()
def q_mae(net, states, targets, masks, device, batch):
    error = count = 0.0
    for start in range(0, len(states), batch):
        end = start + batch
        _policy, _value, _distance, _death, q = net(states[start:end].to(device))
        mask = masks[start:end].to(device)
        error += float(((q - targets[start:end].to(device)).abs() * mask).sum())
        count += float(mask.sum())
    return error / max(1.0, count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", nargs="+", required=True)
    ap.add_argument("--init-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    datasets = [torch.load(path, weights_only=True, map_location="cpu") for path in args.demos]
    episodes = [episode for data in datasets for episode in data.get("episodes", [])]
    if not episodes or any("afterstate_q_targets" not in episode for episode in episodes):
        raise ValueError("所有轨迹都必须包含 afterstate_q_targets")
    ids = list(range(len(episodes)))
    random.shuffle(ids)
    n_val = max(1, round(len(ids) * args.val_ratio))
    val_ids = set(ids[:n_val])

    def flatten(selected):
        return (
            torch.cat([episode["states"] for episode in selected]).float(),
            torch.cat([episode["afterstate_q_targets"] for episode in selected]).float(),
            torch.cat([episode["afterstate_q_masks"] for episode in selected]).float(),
        )

    train = flatten([episode for i, episode in enumerate(episodes) if i not in val_ids])
    val = flatten([episode for i, episode in enumerate(episodes) if i in val_ids])
    checkpoint = torch.load(args.init_model, weights_only=True, map_location=args.device)
    net = PolicyValueNet(
        value_outputs=checkpoint.get("value_outputs", 2), afterstate_q=True,
        history_features=checkpoint.get("history_features", False),
    ).to(args.device)
    missing, unexpected = net.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = {
        key for key in net.state_dict() if key.startswith("afterstate_q_head.")
    }
    if set(missing) != allowed_missing or unexpected:
        raise ValueError(f"初始化模型结构不兼容: missing={missing}, unexpected={unexpected}")
    for name, parameter in net.named_parameters():
        parameter.requires_grad = name.startswith("afterstate_q_head.")
    opt = torch.optim.Adam(
        [parameter for parameter in net.parameters() if parameter.requires_grad], lr=args.lr,
    )
    generator = torch.Generator().manual_seed(args.seed)
    best = None
    best_mae = float("inf")
    print(f"demos={args.demos} train={len(train[0])} val={len(val[0])} device={args.device}")
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = torch.randperm(len(train[0]), generator=generator)
        for start in range(0, len(order), args.batch):
            idx = order[start:start + args.batch]
            states, targets, masks = augment_batch(
                train[0][idx], train[1][idx], train[2][idx], generator,
            )
            _policy, _value, _distance, _death, q = net(states.to(args.device))
            targets = targets.to(args.device)
            masks = masks.to(args.device)
            loss = (F.smooth_l1_loss(q, targets, reduction="none") * masks).sum()
            loss = loss / masks.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        mae = q_mae(net, val[0], val[1], val[2], args.device, args.batch)
        if mae < best_mae:
            best_mae = mae
            best = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        print(f"[{epoch:>3}/{args.epochs}] after_q_mae={mae:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    output = dict(checkpoint)
    output.update({
        "model": best,
        "afterstate_q": True,
        "afterstate_q_source": args.demos,
        "afterstate_q_mae": best_mae,
    })
    torch.save(output, args.out)
    print(f"已保存 {args.out}")


if __name__ == "__main__":
    main()
