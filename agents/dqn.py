import random

import torch
import torch.nn as nn

COLS = 6
H = 7
N_ACTIONS = COLS * (COLS - 1)
GRID_CLASSES = 9
META_DIM = 4 + COLS + 2
HISTORY_DIM = 4


def action_index(src, dst):
    return src * (COLS - 1) + (dst - (1 if dst > src else 0))


def index_action(i):
    src = i // (COLS - 1)
    dst = i % (COLS - 1)
    if dst >= src:
        dst += 1
    return src, dst


class Net(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        in_dim = COLS * H * GRID_CLASSES + META_DIM
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(self, x):
        return self.fc(x)


class EquivariantNet(nn.Module):
    """共享列编码器 + 有序源/目标成对打分，严格保持列置换等变。"""

    def __init__(self, hidden=128):
        super().__init__()
        col_dim = H * GRID_CLASSES + 1  # 单列网格 + 该列预告
        global_dim = 4 + 2  # 周期相位 + 预告存在标志 + 掉落上限
        self.col_encoder = nn.Sequential(
            nn.Linear(col_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pair_head = nn.Sequential(
            nn.Linear(hidden * 3 + global_dim, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        pairs = [index_action(i) for i in range(N_ACTIONS)]
        self.register_buffer("src_idx", torch.tensor([s for s, _ in pairs]))
        self.register_buffer("dst_idx", torch.tensor([d for _, d in pairs]))

    def forward(self, x):
        grid_dim = COLS * H * GRID_CLASSES
        grid = x[:, :grid_dim].reshape(-1, COLS, H * GRID_CLASSES)
        meta = x[:, grid_dim:]
        preview = meta[:, 4:4 + COLS].unsqueeze(-1)
        col = self.col_encoder(torch.cat([grid, preview], dim=-1))
        pooled = col.mean(dim=1)
        global_meta = torch.cat([meta[:, :4], meta[:, 4 + COLS:]], dim=1)
        pair_global = torch.cat([pooled, global_meta], dim=1)
        pair_global = pair_global.unsqueeze(1).expand(-1, N_ACTIONS, -1)
        pair = torch.cat(
            [col[:, self.src_idx], col[:, self.dst_idx], pair_global], dim=-1
        )
        return self.pair_head(pair).squeeze(-1)


def make_net(arch="mlp"):
    if arch == "mlp":
        return Net()
    if arch == "equivariant":
        return EquivariantNet()
    raise ValueError(f"未知网络结构: {arch}")


def encode(game, device="cpu", history=False):
    g = torch.zeros(COLS * H * GRID_CLASSES, device=device)
    for c in range(COLS):
        st = game.stacks[c]
        for i, v in enumerate(st):
            if i < H:
                assert 1 <= v <= GRID_CLASSES, f"值 {v} 超出编码类别"
                g[(c * H + i) * GRID_CLASSES + v - 1] = 1
    meta = torch.zeros(META_DIM, device=device)
    meta[game.moves % 4] = 1
    if game.preview is not None:
        meta[4 + COLS] = 1.0
        for c in range(COLS):
            meta[4 + c] = game.preview[c] / 7.0
    meta[5 + COLS] = min(game.max_merged, 7) / 7.0
    parts = [g, meta]
    if history:
        empty_history = [
            game.current_cycle_empty_peak,
            *game.recent_cycle_empty_peaks,
        ]
        parts.append(torch.tensor(
            [min(2, count) / 2 for count in empty_history],
            device=device,
        ))
    return torch.cat(parts)


def legal_mask(game, device="cpu"):
    m = torch.zeros(N_ACTIONS, device=device)
    for s, d in game.legal_moves():
        m[action_index(s, d)] = 1
    return m


class DQN:
    def __init__(self, lr=3e-4, replay=100_000, device="cpu", arch="mlp"):
        self.device = device
        self.arch = arch
        self.online = make_net(arch).to(device)
        self.target = make_net(arch).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.replay = []
        self.expert = []
        self.replay_max = replay
        self.pos = 0
        self.train_steps = 0

    def remember(self, s, a, r, ns, nmask, done):
        if len(self.replay) < self.replay_max:
            self.replay.append(None)
        self.replay[self.pos] = (s.cpu(), a, r, ns.cpu(), nmask.cpu(), done)
        self.pos = (self.pos + 1) % self.replay_max

    def remember_expert(self, s, a, r, ns, nmask, done):
        self.expert.append((s.cpu(), a, r, ns.cpu(), nmask.cpu(), done))

    @torch.no_grad()
    def act(self, s, mask, eps):
        if random.random() < eps:
            legal = mask.nonzero().flatten().tolist()
            return random.choice(legal)
        q = self.online(s.unsqueeze(0))[0]
        q = q.masked_fill(~mask.bool(), float("-inf"))
        return int(q.argmax())

    def learn(self, batch_size, gamma, target_sync, expert_ratio=0.0, expert_bc_w=0.0):
        n_expert = min(len(self.expert), round(batch_size * expert_ratio))
        n_replay = batch_size - n_expert
        if len(self.replay) < n_replay:
            return None
        samples = random.sample(self.expert, n_expert) if n_expert else []
        samples += random.sample(self.replay, n_replay)
        s = torch.stack([item[0] for item in samples]).to(self.device)
        a = torch.tensor([item[1] for item in samples], device=self.device)
        r = torch.tensor([item[2] for item in samples], device=self.device)
        ns = torch.stack([item[3] for item in samples]).to(self.device)
        nmask = torch.stack([item[4] for item in samples]).to(self.device)
        done = torch.tensor([item[5] for item in samples], device=self.device)

        q_all = self.online(s)
        q = q_all.gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            pick = self.online(ns).masked_fill(~nmask.bool(), float("-inf")).argmax(1)
            nq = self.target(ns).gather(1, pick.unsqueeze(1)).squeeze(1)
            nq = nq.masked_fill(~nmask.bool()[torch.arange(batch_size, device=self.device), pick], 0.0)
        target = r + gamma * nq * (1 - done)

        loss = torch.nn.functional.mse_loss(q, target)
        if n_expert and expert_bc_w:
            loss = loss + expert_bc_w * torch.nn.functional.cross_entropy(
                q_all[:n_expert], a[:n_expert]
            )
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.train_steps += 1
        if self.train_steps % target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path):
        torch.save(self.online.state_dict(), path)

    def load(self, path):
        self.online.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))
        self.target.load_state_dict(self.online.state_dict())

    def save_full(self, path, **extra):
        def _cpu(x):
            if isinstance(x, dict):
                return {k: _cpu(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return type(x)(_cpu(v) for v in x)
            if isinstance(x, torch.Tensor):
                return x.cpu()
            return x

        torch.save(
            {
                "online": {k: v.cpu() for k, v in self.online.state_dict().items()},
                "target": {k: v.cpu() for k, v in self.target.state_dict().items()},
                "opt": _cpu(self.opt.state_dict()),
                "replay": self.replay,
                "expert": self.expert,
                "pos": self.pos,
                "replay_max": self.replay_max,
                "train_steps": self.train_steps,
                "arch": self.arch,
                **extra,
            },
            path,
        )

    def load_full(self, path):
        d = torch.load(path, weights_only=True, map_location="cpu")
        saved_arch = d.get("arch", "mlp")
        if saved_arch != self.arch:
            raise ValueError(f"checkpoint arch={saved_arch}，当前 arch={self.arch}")
        self.online.load_state_dict(d["online"])
        self.target.load_state_dict(d["target"])
        self.opt.load_state_dict(d["opt"])
        for state in self.opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.replay = d["replay"]
        self.expert = d.get("expert", [])
        self.pos = d["pos"]
        self.replay_max = d["replay_max"]
        self.train_steps = d["train_steps"]
        return d
