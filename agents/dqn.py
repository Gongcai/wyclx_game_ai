import random

import torch
import torch.nn as nn

COLS = 6
H = 7
N_ACTIONS = COLS * (COLS - 1)
GRID_CLASSES = 9
META_DIM = 4 + COLS + 2


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


def encode(game, device="cpu"):
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
    return torch.cat([g, meta])


def legal_mask(game, device="cpu"):
    m = torch.zeros(N_ACTIONS, device=device)
    for s, d in game.legal_moves():
        m[action_index(s, d)] = 1
    return m


class DQN:
    def __init__(self, lr=3e-4, replay=100_000, device="cpu"):
        self.device = device
        self.online = Net().to(device)
        self.target = Net().to(device)
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
                **extra,
            },
            path,
        )

    def load_full(self, path):
        d = torch.load(path, weights_only=True, map_location="cpu")
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
