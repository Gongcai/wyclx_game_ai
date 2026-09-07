"""N-Tuple Network 与 afterstate TD(lambda) 控制器。

网络只保存若干共享的查表权重。特征从 afterstate 提取，因此不会读取
Game._hidden_preview；随机掉落由 TD 目标中的 chance 部分承担。
"""

import copy
import random
from collections import Counter

import torch

from agents.dqn import COLS, H, index_action


_BASE = 9
_TOP3_SIZE = _BASE ** 3


TABLE_SIZES = {
    "col4_preview": _BASE ** 4 * 8,
    "col6": _BASE ** 6,
    "pair3": _BASE ** 6,
    "heights": 8 ** COLS,
    "global": 4 * 2 * 9,
    "health": 7 * 7 * 7 * 43,
}


def _cell_code(stack, length):
    """按栈顶到栈底编码，0 表示该位置为空。"""
    code = 0
    factor = 1
    for value in stack[:length]:
        value = max(0, min(8, int(value)))
        code += value * factor
        factor *= _BASE
    return code


def _column_key(game, col):
    stack = game.stacks[col]
    preview = 0 if game.preview is None else int(game.preview[col])
    return (_cell_code(stack, 3), preview, min(len(stack), H))


class NtupleNetwork:
    """共享权重的 N-Tuple value function。"""

    format_version = 2

    def __init__(self):
        self.tables = {
            name: torch.zeros(size, dtype=torch.float32)
            for name, size in TABLE_SIZES.items()
        }

    def feature_counts(self, game):
        """返回 ``(表名, 索引)`` 的计数；列置换不会改变结果。"""
        features = []
        for col in range(COLS):
            stack = game.stacks[col]
            preview = 0 if game.preview is None else int(game.preview[col])
            code = _cell_code(stack, 4) + preview * (_BASE ** 4)
            features.append(("col4_preview", code))
            features.append(("col6", _cell_code(stack, 6)))

        # 同一张 pair 表覆盖所有无序列对。先按带 preview 的列签名排序，
        # 保持 preview 与所属列绑定，同时消除列编号带来的伪差异。
        columns = sorted(range(COLS), key=lambda col: _column_key(game, col))
        top3 = [_cell_code(game.stacks[col], 3) for col in columns]
        for left in range(COLS):
            for right in range(left + 1, COLS):
                features.append(("pair3", top3[left] * _TOP3_SIZE + top3[right]))

        heights = sorted(min(len(game.stacks[col]), H) for col in range(COLS))
        code = 0
        factor = 1
        for height in heights:
            code += height * factor
            factor *= 8
        features.append(("heights", code))

        phase = int(game.moves % 4)
        has_preview = int(game.preview is not None)
        max_merged = max(1, min(int(game.max_merged), 8))
        global_code = (phase * 2 + has_preview) * 9 + max_merged
        features.append(("global", global_code))

        empty = sum(not stack for stack in game.stacks)
        top_pairs = sum(
            len(stack) >= 2 and stack[0] == stack[1]
            for stack in game.stacks
        )
        high_columns = sum(len(stack) >= 6 for stack in game.stacks)
        total_height = sum(min(len(stack), H) for stack in game.stacks)
        health_code = (((empty * 7 + top_pairs) * 7 + high_columns) * 43
                       + min(total_height, 42))
        features.append(("health", health_code))
        return Counter(features)

    def value(self, game=None, counts=None):
        if counts is None:
            counts = self.feature_counts(game)
        return sum(
            float(self.tables[name][index]) * count
            for (name, index), count in counts.items()
        )

    def update(self, counts, amount):
        for (name, index), count in counts.items():
            self.tables[name][index] += float(amount) * count

    def state_dict(self):
        return {name: table.clone() for name, table in self.tables.items()}

    def load_state_dict(self, state):
        expected = set(TABLE_SIZES)
        missing = expected - set(state)
        if set(state) - expected or missing - {"health"}:
            raise ValueError(f"N-Tuple 表结构不兼容: {sorted(state)}")
        for name, size in TABLE_SIZES.items():
            if name not in state:
                continue
            table = state[name]
            if tuple(table.shape) != (size,):
                raise ValueError(f"N-Tuple 表 {name} 大小错误: {tuple(table.shape)}")
            self.tables[name].copy_(table.float())

    def save(self, path, **extra):
        torch.save({
            "format": "ntuple-td-lambda",
            "version": self.format_version,
            "tables": self.state_dict(),
            **extra,
        }, path)

    def load(self, path):
        data = torch.load(path, weights_only=True, map_location="cpu")
        if data.get("format") not in (None, "ntuple-td-lambda"):
            raise ValueError(f"不是 N-Tuple 模型: {path}")
        self.load_state_dict(data["tables"] if "tables" in data else data)
        return data


class EligibilityTrace:
    """稀疏 replacing trace；每个并行环境应单独持有一个实例。"""

    def __init__(self, prune=1e-5):
        self.values = {}
        self.prune = prune

    def clear(self):
        self.values.clear()

    def update(
        self, network, counts, reward, next_value, gamma, trace_lambda, alpha,
        terminal=False,
    ):
        current = network.value(counts=counts)
        delta = float(reward) + (0.0 if terminal else gamma * float(next_value)) - current

        decay = gamma * trace_lambda
        for key in list(self.values):
            value = self.values[key] * decay
            if abs(value) < self.prune:
                del self.values[key]
            else:
                self.values[key] = value
        # replacing trace 对重复出现的共享 tuple 保留出现次数。
        for key, count in counts.items():
            self.values[key] = float(count)

        step_alpha = float(alpha) / max(1, sum(counts.values()))
        for (name, index), trace in self.values.items():
            network.tables[name][index] += step_alpha * delta * trace
        if terminal:
            self.clear()
        return delta


def deterministic_reward(events, dead=False, merge_w=1.0, include_step=True,
                        n9_w=10.0, death_w=5.0):
    """计算一段 events 的奖励；掉落和移动阶段可分别调用。"""
    n9 = sum(1 for value in events if value >= 9)
    intermediate = sum(value for value in events if value < 9)
    reward = (-0.05 if include_step else 0.0) + n9_w * n9 + merge_w * intermediate
    if dead:
        reward -= death_w
    return reward


class NtuplePolicy:
    """使用 afterstate value 选动作。"""

    def __init__(self, network, gamma=0.99, merge_w=1.0, n9_w=10.0, death_w=5.0):
        self.network = network
        self.gamma = gamma
        self.merge_w = merge_w
        self.n9_w = n9_w
        self.death_w = death_w

    def choose(self, game, epsilon=0.0, rng=None, return_counts=False):
        rng = rng or random
        legal = game.legal_moves()
        if not legal:
            return None
        if rng.random() < epsilon:
            action = rng.choice(legal)
            return action

        best = None
        best_score = float("-inf")
        tied = []
        for action in legal:
            sim = copy.deepcopy(game)
            if not sim.move_afterstate(*action):
                continue
            move_reward = deterministic_reward(
                sim.events, sim.dead, self.merge_w, include_step=True,
                n9_w=self.n9_w, death_w=self.death_w,
            )
            score = move_reward + self.gamma * self.network.value(game=sim)
            if score > best_score + 1e-8:
                best_score = score
                tied = [action]
            elif abs(score - best_score) <= 1e-8:
                tied.append(action)
        if tied:
            best = rng.choice(tied)
        return best if best is not None else legal[0]


def action_values(network, game, merge_w=1.0, n9_w=10.0, gamma=0.99, death_w=5.0):
    """返回合法动作的 afterstate 分数，便于诊断和离线评测。"""
    policy = NtuplePolicy(
        network, merge_w=merge_w, n9_w=n9_w, gamma=gamma, death_w=death_w,
    )
    values = {}
    for action in game.legal_moves():
        sim = copy.deepcopy(game)
        sim.move_afterstate(*action)
        values[index_action(action[0], action[1])] = (
            deterministic_reward(
                sim.events, sim.dead, merge_w, n9_w=n9_w, death_w=death_w,
            )
            + gamma * network.value(game=sim)
        )
    return values
