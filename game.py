import random

MAX_HEIGHT = 7
MAX_VALUE = 8
DROP_MAX = 7


class Game:
    def __init__(self, cols=6, rng=None, drop_sampler=None):
        self.cols = cols
        self.rng = rng or random.Random()
        self.drop_sampler = drop_sampler or (
            lambda rng, max_merged=None: rng.randint(1, min(DROP_MAX, max_merged or DROP_MAX))
        )
        self.reset()

    def reset(self):
        self.stacks = [[1] for _ in range(self.cols)]
        self.score = 0
        self.moves = 0
        self.max_merged = 1
        self.dead = False
        self.preview = None
        self.last_drop = None
        self.events = []
        self._afterstate_pending = False
        self.current_cycle_empty_peak = 0
        self.recent_cycle_empty_peaks = [0, 0, 0]
        return self.observe()

    def observe(self):
        return [list(st) for st in self.stacks]

    def legal_moves(self):
        if self._afterstate_pending:
            return []
        out = []
        for s in range(self.cols):
            if not self.stacks[s]:
                continue
            for d in range(self.cols):
                if s != d:
                    out.append((s, d))
        return out

    def move(self, src, dst):
        if not self.move_afterstate(src, dst):
            return False
        self.resolve_afterstate()
        return True

    def move_afterstate(self, src, dst):
        """执行玩家动作，但暂不生成预告或应用周期掉落。"""
        if self.dead or self._afterstate_pending or src == dst or not self.stacks[src]:
            return False
        self.events.clear()
        st = self.stacks[src]
        v = st[0]
        k = 1
        while k < len(st) and st[k] == v:
            k += 1
        seg = st[:k]
        del st[:k]
        self.stacks[dst][:0] = seg
        self._merge_col(dst)
        self.current_cycle_empty_peak = max(
            self.current_cycle_empty_peak,
            sum(not stack for stack in self.stacks),
        )
        self.moves += 1
        self.last_drop = None
        self._afterstate_pending = not self.dead
        return True

    def chance_required(self):
        """当前 afterstate 是否需要采样未知随机事件。"""
        if not self._afterstate_pending or self.dead:
            return False
        return self.moves % 4 == 3 or (
            self.moves % 4 == 0 and self.preview is None
        )

    def sample_chance(self, rng=None):
        """按当前 afterstate 的掉落上限采样完整六列随机结果。"""
        rng = rng or self.rng
        return [self.drop_sampler(rng, self.max_merged) for _ in range(self.cols)]

    def resolve_afterstate(self, chance=None):
        """将随机结果应用到 pending afterstate，恢复为正常决策状态。"""
        if not self._afterstate_pending:
            return False
        self._afterstate_pending = False
        if self.dead:
            return True
        if self.moves % 4 == 3:
            values = self.sample_chance() if chance is None else list(chance)
            if len(values) != self.cols:
                raise ValueError(f"随机掉落应有 {self.cols} 列，实际 {len(values)}")
            self.preview = values
        elif self.moves % 4 == 0:
            if self.preview is not None:
                if chance is not None and list(chance) != self.preview:
                    raise ValueError("已有 preview 时不得提供不同的随机结果")
                self._apply_drops()
            else:
                values = self.sample_chance() if chance is None else list(chance)
                self._apply_drops(values)
            self.recent_cycle_empty_peaks = [
                self.current_cycle_empty_peak,
                *self.recent_cycle_empty_peaks[:2],
            ]
            self.current_cycle_empty_peak = sum(not stack for stack in self.stacks)
        return True

    def _sample(self):
        return self.drop_sampler(self.rng, self.max_merged)

    def _apply_drops(self, values=None):
        values = self.preview if self.preview is not None else (
            self.sample_chance() if values is None else list(values)
        )
        self.preview = None
        for c, v in enumerate(values):
            self.stacks[c].insert(0, v)
            self._merge_col(c)
            if self.dead:
                break
        self.last_drop = values

    def _merge_col(self, c):
        st = self.stacks[c]
        while st:
            top = st[0]
            k = 1
            while k < len(st) and st[k] == top:
                k += 1
            if k < 3:
                break
            del st[:k]
            nv = top + 1
            self.events.append(nv)
            if nv > MAX_VALUE:
                self.score += 9
                self.max_merged = max(self.max_merged, 9)
            else:
                st.insert(0, nv)
                self.max_merged = max(self.max_merged, nv)
        if len(st) > MAX_HEIGHT:
            self.dead = True

    def render(self):
        n = 4 - (self.moves % 4)
        lines = [f"步数:{self.moves}  得分:{self.score}  已合成最大:{self.max_merged}"]
        if self.preview is not None:
            lines.append(f"下次掉落预告: [{ ' '.join(map(str, self.preview)) }]  ({n} 步后掉落)")
        else:
            lines.append(f"下次掉落: ????  ({n} 步后掉落)")
        lines.append("")
        lines.append("      " + "  ".join(str(i + 1) for i in range(self.cols)))
        for row in range(1, MAX_HEIGHT + 1):
            cells = []
            for c in range(self.cols):
                st = self.stacks[c]
                idx = row - MAX_HEIGHT + len(st) - 1
                cells.append(f"{st[idx] if 0 <= idx < len(st) else '.':>2} ")
            lines.append(f" r{row}: " + "".join(cells))
        return "\n".join(lines)
