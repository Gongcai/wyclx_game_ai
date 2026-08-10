import random

MAX_HEIGHT = 7
MAX_VALUE = 8
DROP_MAX = 7


class Game:
    def __init__(self, cols=6, rng=None, drop_sampler=None):
        self.cols = cols
        self.rng = rng or random.Random()
        self.drop_sampler = drop_sampler or (lambda rng: rng.randint(1, DROP_MAX))
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
        return self.observe()

    def observe(self):
        return [list(st) for st in self.stacks]

    def legal_moves(self):
        out = []
        for s in range(self.cols):
            if not self.stacks[s]:
                continue
            for d in range(self.cols):
                if s != d:
                    out.append((s, d))
        return out

    def move(self, src, dst):
        if self.dead or src == dst or not self.stacks[src]:
            return False
        self.events.clear()
        v = self.stacks[src].pop(0)
        self.stacks[dst].insert(0, v)
        self._merge_col(dst)
        self.moves += 1
        self.last_drop = None
        if not self.dead:
            self._cycle()
        return True

    def _cycle(self):
        if self.moves % 4 == 3:
            self.preview = [self._sample() for _ in range(self.cols)]
        elif self.moves % 4 == 0:
            self._apply_drops()

    def _sample(self):
        return self.drop_sampler(self.rng)

    def _apply_drops(self):
        values = []
        for c in range(self.cols):
            v = self._sample()
            values.append(v)
            self.stacks[c].insert(0, v)
            self._merge_col(c)
            if self.dead:
                break
        self.last_drop = values
        self.preview = None

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
