"""DAgger：在模型自身访问的状态上用 beam search 标注纠正动作。"""

import argparse
import os
import random

import torch
import torch.nn.functional as F

from agents.dist import load_weights, make_sampler
from agents.dqn import Net, action_index, encode, index_action, legal_mask
from agents.search import beam_policy
from game import Game


def train_epochs(net, opt, states, actions, epochs, batch_size, device, generator):
    net.train()
    for _ in range(epochs):
        order = torch.randperm(len(actions), generator=generator)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            logits = net(states[idx].to(device))
            loss = F.cross_entropy(logits, actions[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.no_grad()
def student_action(net, game, device):
    logits = net(encode(game, device).unsqueeze(0))[0]
    logits = logits.masked_fill(~legal_mask(game, device).bool(), float("-inf"))
    return int(logits.argmax())


def collect(net, weights, capped, seeds, depth, width, beta, max_moves, device):
    states = []
    actions = []
    results = []
    selector = random.Random(seeds[0] + 104729)
    for seed in seeds:
        game = Game(rng=random.Random(seed), drop_sampler=make_sampler(weights, capped))
        while not game.dead and game.moves < max_moves:
            state = encode(game)
            teacher = beam_policy(game, depth=depth, width=width)
            if teacher is None:
                break
            label = action_index(*teacher)
            student = student_action(net, game, device)
            states.append(state)
            actions.append(label)
            chosen = label if selector.random() < beta else student
            if not game.move(*index_action(chosen)):
                break
        results.append((game.score, game.moves, game.max_merged))
    return torch.stack(states), torch.tensor(actions, dtype=torch.long), results


@torch.no_grad()
def evaluate(net, weights, capped, n, seed, max_moves, device):
    total_score = total_moves = total_max = successes = 0
    for i in range(n):
        game = Game(rng=random.Random(seed + i), drop_sampler=make_sampler(weights, capped))
        while not game.dead and game.moves < max_moves:
            action = student_action(net, game, device)
            if not game.move(*index_action(action)):
                break
        total_score += game.score
        total_moves += game.moves
        total_max += game.max_merged
        successes += game.score > 0
    return total_score / n, total_moves / n, total_max / n, successes / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True)
    ap.add_argument("--init-model", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dist", default="uniform")
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--rollout-episodes", type=int, default=20)
    ap.add_argument("--train-epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=0.5, help="首轮执行教师动作的概率，每轮减半")
    ap.add_argument("--search-depth", type=int, default=4)
    ap.add_argument("--beam-width", type=int, default=8)
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--eval-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    data = torch.load(args.demos, weights_only=True, map_location="cpu")
    episodes = data.get("episodes", [])
    if not episodes:
        raise ValueError("示范文件为空")
    states = torch.cat([episode["states"] for episode in episodes]).float()
    actions = torch.cat([episode["actions"] for episode in episodes]).long()
    weights, capped = load_weights(args.dist)
    net = Net().to(args.device)
    if args.init_model:
        net.load_state_dict(torch.load(args.init_model, weights_only=True, map_location=args.device))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    out = args.out or os.path.join(os.path.dirname(args.demos), f"{args.dist}-dagger.pt")
    best_success = -1.0
    best_state = None

    print(f"初始聚合数据 {len(actions)} 条，device={args.device}")
    for iteration in range(1, args.iterations + 1):
        beta = args.beta * 0.5 ** (iteration - 1)
        seeds = [args.seed + iteration * 100000 + i for i in range(args.rollout_episodes)]
        new_s, new_a, rollout = collect(
            net, weights, capped, seeds, args.search_depth, args.beam_width,
            beta, args.max_moves, args.device,
        )
        states = torch.cat([states, new_s])
        actions = torch.cat([actions, new_a])
        train_epochs(
            net, opt, states, actions, args.train_epochs, args.batch,
            args.device, generator,
        )
        stats = evaluate(
            net, weights, capped, args.eval_n, args.seed + 900000,
            args.max_moves, args.device,
        )
        teacher_success = sum(score > 0 for score, _, _ in rollout) / len(rollout)
        print(
            f"[{iteration}/{args.iterations}] beta={beta:.3f} 新增={len(new_a)} "
            f"聚合={len(actions)} rollout成功={teacher_success:.1%} "
            f"agent得分={stats[0]:.2f} 成功率={stats[3]:.1%} "
            f"步数={stats[1]:.1f} max={stats[2]:.2f}",
            flush=True,
        )
        if stats[3] > best_success:
            best_success = stats[3]
            best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(best_state, out)
    print(f"已保存 {out}，最佳成功率 {best_success:.1%}")


if __name__ == "__main__":
    main()
