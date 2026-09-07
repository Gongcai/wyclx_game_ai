"""N-Tuple 价值函数之上的 beam 搜索策略。

结构沿用启发式 beam（agents/heuristic.py，仓库已验证"推演深度是最大杠杆"）
的两条关键语义，叶子估值换成训练出的 N-Tuple afterstate 价值：

1. 折扣记账与 TD 训练语义一致：Q(s,a) = move_r(a) + γ·F(A)，F(A,1)=V(A)，
   F(A,d>1) = drop_r(A) + max_{a'} Q(s',a',d-1)。V 在 pending afterstate
   （move 后、resolve 前）上评估，与 train_ntuple 的训练分布完全一致。
2. chance 诚实采样：揭示点（moves%4==2 的 afterstate）resolve 时用
   sample_chance 按预抽等级分布采样，绝不读取 _hidden_preview；预告已知时
   （moves%4==3）resolve 原样应用已知掉落，确定性精确推演。

beam 为全局裁剪（跨根动作混合 chance 实现），与启发式 beam 一致；每个根动作
以其 1-ply 分数为兜底终端分，深线只在更好时覆盖。
"""

import random

from agents.ntuple import NtupleNetwork, deterministic_reward


class FastNtupleValue:
    """numpy 镜像查表，避免 torch 标量索引开销（~24us → ~2us）。

    依赖 feature_counts 的单一实现，不重复特征逻辑；表更新后需重建实例。
    """

    def __init__(self, network: NtupleNetwork):
        self.network = network
        self.arrays = {name: table.numpy() for name, table in network.tables.items()}

    def value(self, game):
        arrays = self.arrays
        total = 0.0
        for (name, index), count in self.network.feature_counts(game).items():
            total += float(arrays[name][index]) * count
        return total


class NtupleBeamPolicy:
    """beam 搜索 + N-Tuple afterstate 价值的策略，接口与 NtuplePolicy 兼容。"""

    def __init__(
        self,
        network,
        depth_moves=8,
        width=8,
        root_width=None,
        num_det=1,
        gamma=0.99,
        merge_w=1.0,
        n9_w=30.0,
        death_w=15.0,
    ):
        self.network = network
        self.value = FastNtupleValue(network)
        self.depth_moves = max(1, int(depth_moves))
        self.width = max(1, int(width))
        self.root_width = max(self.width, int(root_width or width))
        self.num_det = max(1, int(num_det))
        self.gamma = gamma
        self.merge_w = merge_w
        self.n9_w = n9_w
        self.death_w = death_w

    def _move_reward(self, events, dead):
        return deterministic_reward(
            events, dead, self.merge_w, include_step=True,
            n9_w=self.n9_w, death_w=self.death_w,
        )

    def _drop_reward(self, events, dead):
        return deterministic_reward(
            events, dead, self.merge_w, include_step=False,
            n9_w=self.n9_w, death_w=self.death_w,
        )

    def choose(self, game, epsilon=0.0, rng=None):
        rng = rng or random
        legal = game.legal_moves()
        if not legal:
            return None
        if epsilon > 0 and rng.random() < epsilon:
            return rng.choice(legal)
        return self._search(game, rng)

    def _search(self, game, rng):
        """返回 beam 搜索的最优首步动作。

        num_det > 1 时用不同共享采样各跑一次 beam，按根动作平均终端分
        （多确定性化平均，降低单次采样的决策方差）。
        """
        totals = {}
        for _det in range(self.num_det):
            for first, value in self._search_once(game, rng).items():
                totals[first] = totals.get(first, 0.0) + value
        best_action = None
        best_value = float("-inf")
        for first, value in totals.items():
            if value > best_value:
                best_value = value
                best_action = first
        if best_action is None:
            legal = game.legal_moves()
            best_action = legal[0] if legal else None
        return best_action

    def _search_once(self, game, rng):
        """单次确定性化的 beam；返回 {首步动作: 终端分}。"""
        root = game.clone()
        root.rng = random.Random(rng.randrange(2**63))

        # 条目: (pending_afterstate, 首步动作, 累计折扣奖励 G, 折扣系数 disc, 排序分)
        # score = G + disc·V(A)（A.dead 时 score = G，即 V(dead)=0）。
        terminal_best = {}
        current = []
        for action in root.legal_moves():
            sim = root.clone()
            if not sim.move_afterstate(*action):
                continue
            g = self._move_reward(sim.events, sim.dead)
            sim.events.clear()
            first = action
            if sim.dead:
                if g > terminal_best.get(first, float("-inf")):
                    terminal_best[first] = g
                continue
            score = g + self.gamma * self.value.value(sim)
            if score > terminal_best.get(first, float("-inf")):
                terminal_best[first] = score
            if self.depth_moves > 1:
                current.append((sim, first, g, self.gamma, score))

        for ply in range(self.depth_moves - 1):
            if not current:
                break
            current.sort(key=lambda entry: entry[4], reverse=True)
            keep = self.root_width if ply == 0 else self.width
            last_ply = ply == self.depth_moves - 2
            nxt = []
            # 同层所有条目按隐藏等级共享同一次预告采样：消除"幸运预告线"
            # 的跨线运气差（逐条目独立采样会让乐观偏差随深度复合，实测
            # depth 8 反而差于 depth 4）。
            shared_samples = {}
            for sim, first, g, disc, _score in current[:keep]:
                if sim.moves % 4 == 2:
                    level = sim._hidden_preview_level
                    sample = shared_samples.get(level)
                    if sample is None:
                        sample = sim.sample_chance(rng)
                        shared_samples[level] = sample
                    sim.resolve_afterstate(sample)
                else:
                    sim.resolve_afterstate()
                drop_r = self._drop_reward(sim.events, sim.dead)
                sim.events.clear()
                if sim.dead:
                    terminal = g + disc * drop_r
                    if terminal > terminal_best.get(first, float("-inf")):
                        terminal_best[first] = terminal
                    continue
                base_g = g + disc * drop_r
                child_disc = disc * self.gamma
                for action in sim.legal_moves():
                    child = sim.clone()
                    if not child.move_afterstate(*action):
                        continue
                    move_r = self._move_reward(child.events, child.dead)
                    child.events.clear()
                    child_g = base_g + disc * move_r
                    if child.dead:
                        if child_g > terminal_best.get(first, float("-inf")):
                            terminal_best[first] = child_g
                        continue
                    score = child_g + child_disc * self.value.value(child)
                    if last_ply:
                        if score > terminal_best.get(first, float("-inf")):
                            terminal_best[first] = score
                    else:
                        nxt.append((child, first, child_g, child_disc, score))
            current = nxt

        return terminal_best
