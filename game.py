import random

from agents.dist import make_real_sampler

MAX_HEIGHT = 7
MAX_VALUE = 8
DROP_MAX = 7


class Game:
    """H5《兔王争霸赛》的无界面状态机。

    ``preview`` 只保存已经展示给玩家的预告。H5 会提前抽好下一轮，故未
    展示的值存在 ``_hidden_preview``，绝不可被策略编码读取。
    """

    def __init__(self, cols=6, rng=None, drop_sampler=None, future_seed=None, h5_style=True):
        self.cols = cols
        self.rng = rng or random.Random()
        self.h5_style = h5_style
        self.drop_sampler = drop_sampler or make_real_sampler()
        self.future_seed = future_seed
        self.reset()

    def reset(self):
        # H5 initRabbits：每列底部两个 1；本地栈顶在索引 0。
        self.stacks = [[1, 1] for _ in range(self.cols)] if self.h5_style else [[1] for _ in range(self.cols)]
        # 原版 score 只累计 2~8 级合成的 2**level；9 单独计数。
        self.score = 0
        self.n9_count = 0
        self.moves = 0
        self.max_merged = 1
        self.dead = False
        self.preview = None
        self._hidden_preview = None
        self._hidden_preview_level = self.max_merged
        self.last_drop = None
        self.events = []
        self._afterstate_pending = False
        self._drop_count = 0
        self.current_cycle_empty_peak = 0
        self.recent_cycle_empty_peaks = [0, 0, 0]
        # initRabbits() 的 refreshTopPreviewItem()：开局已抽样，但界面隐藏。
        self._prepare_hidden_preview()
        return self.observe()

    def _future_drop(self, i, level):
        """预知模式第 i 轮的掉落；level 必须是 H5 实际抽样时的等级。"""
        rng = random.Random((self.future_seed * 1000003 + i) & ((1 << 63) - 1))
        return tuple(self.drop_sampler(rng, level) for _ in range(self.cols))

    def _prepare_hidden_preview(self):
        """H5 refreshTopPreviewItem：抽好六列但暂不让玩家看见。"""
        level = self.max_merged
        if self.future_seed is not None:
            values = self._future_drop(self._drop_count, level)
        else:
            values = tuple(self.drop_sampler(self.rng, level) for _ in range(self.cols))
        self._hidden_preview = list(values)
        self._hidden_preview_level = level
        self._drop_count += 1

    def future_drops(self, ahead):
        """搜索专用的预知接口，不代表真实玩家可见信息。"""
        if self.future_seed is None:
            return None
        if ahead <= 0:
            return []
        values = [tuple(self._hidden_preview)] if self._hidden_preview is not None else []
        # 后续轮的实际等级尚取决于未来合成；保留旧接口的短视近似语义。
        for i in range(len(values), ahead):
            values.append(self._future_drop(self._drop_count + i - len(values), self.max_merged))
        return values

    def observe(self):
        return [list(st) for st in self.stacks]

    def legal_moves(self):
        if self._afterstate_pending:
            return []
        return [
            (src, dst)
            for src in range(self.cols) if self.stacks[src]
            for dst in range(self.cols) if src != dst
        ]

    def move(self, src, dst):
        if not self.move_afterstate(src, dst):
            return False
        self.resolve_afterstate()
        return True

    def move_afterstate(self, src, dst):
        """执行玩家移动，但推迟预告揭示/掉落，以供 PUCT 建 chance node。"""
        if self.dead or self._afterstate_pending or src == dst or not self.stacks[src]:
            return False
        self.events.clear()
        st = self.stacks[src]
        value = st[0]
        # H5 tryLiftRabbit 只拿顶部一张及其下方同级的一张（最多两张）。
        count = 1
        while count < 2 and count < len(st) and st[count] == value:
            count += 1
        segment = st[:count]
        del st[:count]
        self.stacks[dst][:0] = segment
        self._merge_col(dst)
        self.current_cycle_empty_peak = max(
            self.current_cycle_empty_peak, sum(not stack for stack in self.stacks),
        )
        self.moves += 1
        self.last_drop = None
        self._afterstate_pending = not self.dead
        return True

    def chance_required(self):
        """第 2、6、10…步后会揭示已预抽好的预告，供 PUCT 显式建模。"""
        return self._afterstate_pending and not self.dead and self.moves % 4 == 2

    def sample_chance(self, rng=None):
        """从预抽时的等级分布采样，供搜索分支使用，不偷看真实隐藏预告。"""
        rng = rng or self.rng
        return [self.drop_sampler(rng, self._hidden_preview_level) for _ in range(self.cols)]

    def resolve_afterstate(self, chance=None):
        """完成 afterstate。

        H5 时序：第 2 步揭示开局/上轮掉落时预抽的预告；第 3 步先抽好下
        一轮隐藏预告，再应用当前预告。后一个顺序至关重要：掉落造成的合成不
        能影响紧随其后的预告分布。
        """
        if not self._afterstate_pending:
            return False
        self._afterstate_pending = False
        if self.dead:
            return True
        if self.moves % 4 == 2:
            if chance is not None:
                values = list(chance)
                if len(values) != self.cols:
                    raise ValueError(f"随机掉落应有 {self.cols} 列，实际 {len(values)}")
            else:
                values = list(self._hidden_preview)
            self.preview = values
            self._hidden_preview = None
        elif self.moves % 4 == 3 and self.preview is not None:
            # doDelayDropTopItems：启动六个下落后立刻 refreshTopPreviewItem。
            self._prepare_hidden_preview()
            self._apply_drops(self.preview)
            self.preview = None
            self.recent_cycle_empty_peaks = [
                self.current_cycle_empty_peak, *self.recent_cycle_empty_peaks[:2],
            ]
            self.current_cycle_empty_peak = sum(not stack for stack in self.stacks)
        return True

    def _sample(self):
        return self.drop_sampler(self.rng, self.max_merged)

    def _apply_drops(self, values):
        values = list(values)
        if len(values) != self.cols:
            raise ValueError("掉落必须使用已知的六列预告")
        for col, value in enumerate(values):
            self.stacks[col].insert(0, value)
            self._merge_col(col)
            if self.dead:
                break
        self.last_drop = values

    def _merge_col(self, col):
        st = self.stacks[col]
        while st:
            top = st[0]
            count = 1
            while count < len(st) and st[count] == top:
                count += 1
            if count < 3:
                break
            del st[:count]
            new_value = top + 1
            self.events.append(new_value)
            if new_value > MAX_VALUE:
                self.n9_count += 1
                self.max_merged = max(self.max_merged, 9)
            else:
                st.insert(0, new_value)
                self.score += 2 ** new_value
                self.max_merged = max(self.max_merged, new_value)
        if len(st) > MAX_HEIGHT:
            self.dead = True

    def render(self):
        n = (3 - self.moves % 4) if self.moves % 4 != 3 else 4
        lines = [
            f"步数:{self.moves}  得分:{self.score}  合成9:{self.n9_count}  已合成最大:{self.max_merged}",
        ]
        if self.preview is not None:
            lines.append(f"下次掉落预告: [{' '.join(map(str, self.preview))}]  ({n} 步后掉落)")
        else:
            lines.append(f"下次掉落: ????  ({n} 步后掉落)")
        lines.append("")
        lines.append("      " + "  ".join(str(i + 1) for i in range(self.cols)))
        for row in range(1, MAX_HEIGHT + 1):
            cells = []
            for col in range(self.cols):
                st = self.stacks[col]
                idx = row - MAX_HEIGHT + len(st) - 1
                cells.append(f"{st[idx] if 0 <= idx < len(st) else '.':>2} ")
            lines.append(f" r{row}: " + "".join(cells))
        return "\n".join(lines)
