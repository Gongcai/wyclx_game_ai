import random

import torch
import torch.nn as nn

COLS = 6
H = 7
N_ACTIONS = COLS * (COLS - 1)
GRID_CLASSES = 9
META_DIM = 4 + COLS + 1


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
                g[(c * H + i) * GRID_CLASSES + v] = 1
    meta = torch.zeros(META_DIM, device=device)
    meta[game.moves % 4] = 1
    if game.preview:
        for c in range(COLS):
            meta[4 + c] = game.preview[c] / 7.0
    meta[4 + COLS] = game.max_merged / 7.0
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
        self.replay_max = replay
        self.pos = 0
        self.train_steps = 0

    def remember(self, s, a, r, ns, nmask, done):
        if len(self.replay) < self.replay_max:
            self.replay.append(None)
        self.replay[self.pos] = (s, a, r, ns, nmask, done)
        self.pos = (self.pos + 1) % self.replay_max

    @torch.no_grad()
    def act(self, s, mask, eps):
        if random.random() < eps:
            legal = mask.nonzero().flatten().tolist()
            return random.choice(legal)
        q = self.online(s.unsqueeze(0))[0]
        q = q.masked_fill(~mask.bool(), float("-inf"))
        return int(q.argmax())

    def learn(self, batch_size, gamma, target_sync):
        if len(self.replay) < batch_size:
            return None
        idx = random.sample(range(len(self.replay)), batch_size)
        s = torch.stack([self.replay[i][0] for i in idx])
        a = torch.tensor([self.replay[i][1] for i in idx], device=self.device)
        r = torch.tensor([self.replay[i][2] for i in idx], device=self.device)
        ns = torch.stack([self.replay[i][3] for i in idx])
        nmask = torch.stack([self.replay[i][4] for i in idx])
        done = torch.tensor([self.replay[i][5] for i in idx], device=self.device)

        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            pick = self.online(ns).masked_fill(~nmask.bool(), float("-inf")).argmax(1)
            nq = self.target(ns).gather(1, pick.unsqueeze(1)).squeeze(1)
            nq = nq.masked_fill(~nmask.bool()[torch.arange(batch_size, device=self.device), pick], 0.0)
        target = r + gamma * nq * (1 - done)

        loss = torch.nn.functional.mse_loss(q, target)
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
