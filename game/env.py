from .drop import DropDist

NUM_COLS = 6
HEIGHT = 7
SCORE_PER_9 = 100
MERGE_MIN = 3


class Game:
    def __init__(self, drop_dist=None):
        self.drop_dist = drop_dist or DropDist.preset("uniform")
        self.reset()

    def reset(self):
        self.stacks = [[1] for _ in range(NUM_COLS)]
        self.score = 0
        self.step = 0
        self.max_created = 1
        self.preview = None
        self.dead = False
        self.messages = []
        return self

    def _merge(self, col):
        st = self.stacks[col]
        while len(st) >= MERGE_MIN and st[-1] == st[-2] == st[-3]:
            v = st[-1]
            k = 0
            while k < len(st) and st[-(k + 1)] == v:
                k += 1
            del st[len(st) - k:]
            nv = v + 1
            self.max_created = max(self.max_created, nv)
            if nv >= 9:
                self.score += SCORE_PER_9
                self.messages.append(f"合成 9！+{SCORE_PER_9} 分")
            else:
                st.append(nv)
                self.messages.append(f"{v}x{k} -> {nv}")

    def _push(self, col, value):
        self.stacks[col].append(value)
        self._merge(col)
        if len(self.stacks[col]) > HEIGHT:
            self.dead = True

    def move(self, src, dst):
        self.messages.clear()
        if not (0 <= src < NUM_COLS and 0 <= dst < NUM_COLS):
            raise ValueError("列号必须在 0..5")
        if src == dst:
            raise ValueError("不能移动到同一列")
        if not self.stacks[src]:
            raise ValueError("源列为空")
        value = self.stacks[src][-1]
        k = 0
        while k < len(self.stacks[src]) and self.stacks[src][-(k + 1)] == value:
            k += 1
        del self.stacks[src][len(self.stacks[src]) - k:]
        self.stacks[dst].extend([value] * k)
        self._merge(dst)
        if len(self.stacks[dst]) > HEIGHT:
            self.dead = True
        self.step += 1
        if self.dead:
            return
        if self.step % 4 == 3:
            self.preview = [self.drop_dist.sample(self.max_created) for _ in range(NUM_COLS)]
            self.messages.append("预告下次掉落: " + " ".join(map(str, self.preview)))
        elif self.step % 4 == 0:
            self.messages.append("掉落: " + " ".join(map(str, self.preview)))
            for c in range(NUM_COLS):
                self._push(c, self.preview[c])
            self.preview = None
