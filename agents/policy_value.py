"""用于 PUCT/MCTS 的列置换等变 Policy+Value 网络。"""

import torch
import torch.nn as nn

from agents.dqn import COLS, GRID_CLASSES, H, HISTORY_DIM, N_ACTIONS, index_action


def hidden_from_checkpoint(checkpoint):
    """从 checkpoint 的 col_encoder 首层权重推断 hidden 大小（加载大网络用）。"""
    weight = checkpoint["model"]["col_encoder.0.weight"]
    return weight.shape[0]


class PolicyValueNet(nn.Module):
    def __init__(
        self, hidden=128, value_outputs=2, afterstate_q=False,
        history_features=False,
    ):
        super().__init__()
        if value_outputs not in (2, 3):
            raise ValueError("value_outputs 只能是 2 或 3")
        self.value_outputs = value_outputs
        self.afterstate_q = afterstate_q
        self.history_features = history_features
        col_dim = H * GRID_CLASSES + 1
        global_dim = 6 + (HISTORY_DIM if history_features else 0)
        self.col_encoder = nn.Sequential(
            nn.Linear(col_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden * 3 + global_dim, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        if afterstate_q:
            self.afterstate_q_head = nn.Sequential(
                nn.Linear(hidden * 3 + global_dim, hidden * 2),
                nn.ReLU(),
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        self.value_head = nn.Sequential(
            nn.Linear(hidden + global_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, value_outputs),
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
        global_features = torch.cat([pooled, global_meta], dim=1)
        expanded = global_features.unsqueeze(1).expand(-1, N_ACTIONS, -1)
        pair = torch.cat([col[:, self.src_idx], col[:, self.dst_idx], expanded], dim=-1)
        policy = self.policy_head(pair).squeeze(-1)
        if self.afterstate_q:
            afterstate_q = self.afterstate_q_head(pair.detach()).squeeze(-1)
        else:
            afterstate_q = torch.zeros_like(policy)
        values = self.value_head(global_features)
        future_n9 = values[:, 0]
        next9_fraction = torch.sigmoid(values[:, 1])
        if self.value_outputs == 3:
            death_risk = torch.sigmoid(values[:, 2])
        else:
            death_risk = torch.zeros_like(future_n9)
        return policy, future_n9, next9_fraction, death_risk, afterstate_q
