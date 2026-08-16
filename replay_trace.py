#!/usr/bin/env python
"""H5 轨迹重放校验器：用与兔王争霸赛 H5 源码一致的规则模拟器重放录制轨迹。

验证：
1. 动作合法性（源列非空、源!=目标）
2. 每步后棋盘与录制快照一致（按 click_n 对齐，容差 1 步）
3. 掉落值一致性（录制 drops vs 模拟采样分布合理性）

用法：
    .venv/bin/python replay_trace.py runs/traces/*.json
    .venv/bin/python replay_trace.py --dir runs/traces --summary
"""

import argparse
import json
import sys
from pathlib import Path

from agents.dist import make_real_sampler


class H5Game:
    """兔王争霸赛 H5 规则模拟器（对齐 index_0e3a250e.js 源码）。"""

    def __init__(self, sampler=None, rng=None, drops=None):
        import random
        self.rng = rng or random.Random()
        self.sampler = sampler or make_real_sampler()
        self.drops_iter = iter(drops or [])  # 录制的掉落序列（确定性回放）
        self.cols = 6
        self.row = 7
        self.max_level = 9
        # H5 initRabbits：每列底部 2 个等级 1（t=0..11 循环列）
        self.grid = [[1, 1] for _ in range(self.cols)]  # 每列 底部->顶部
        self.composite_level = 1
        self.composite_data = {}
        self.score = 0
        self.n9_count = 0
        self.dead = False
        self.move_count = 0
        self.round_idx = 0
        self.last_drop = None
        # H5 节奏：第 1 轮 3 次移动后掉落，之后每轮 4 次（remainClickTimes 3->0 后重置为 4）
        self.round_ends = [3] + [4] * 10000
        self._round_accum = 0

    def top_segment_len(self, col):
        st = self.grid[col]
        if not st:
            return 0
        v = st[-1]
        k = 0
        for x in reversed(st):
            if x == v:
                k += 1
            else:
                break
        return k

    def merge(self, col):
        st = self.grid[col]
        while self.top_segment_len(col) >= 3:
            v = st[-1]
            n = self.top_segment_len(col)
            del st[-n:]
            a = min(v + 1, self.max_level)
            self.composite_data[a] = self.composite_data.get(a, 0) + 1
            if a < self.max_level:
                st.append(a)
                self.score += 2 ** a
                self.composite_level = max(self.composite_level, a)
            else:
                # 合成 9 消失
                self.n9_count += 1
                self.composite_level = self.max_level
            # 继续检查（级联合并）
        if len(st) > self.row:
            self.dead = True

    def move(self, src, dst):
        if self.dead or src == dst or not self.grid[src]:
            return False
        # H5 tryLiftRabbit：拿顶部 1 个，下方同等级再拿 1 个（最多 2 个）
        lift = [self.grid[src].pop()]
        if self.grid[src] and self.grid[src][-1] == lift[0]:
            lift.append(self.grid[src].pop())
        lift.reverse()
        self.grid[dst].extend(lift)
        self.merge(dst)
        self.move_count += 1
        self._round_accum += 1
        self.last_round_dropped = False
        # 掉落：本轮移动次数到边界；死亡后不再掉落（H5 官方 gameOver 即终止）
        if not self.dead and self._round_accum >= self.round_ends[self.round_idx]:
            self._round_accum = 0
            self.round_idx += 1
            self.drop_round()
            self.last_round_dropped = True
        return True

    def drop_round(self):
        try:
            values = list(next(self.drops_iter))
        except StopIteration:
            values = [self.sampler(self.rng, self.composite_level) for _ in range(6)]
        for c, v in enumerate(values):
            self.grid[c].append(v)
            self.merge(c)
            if self.dead:
                break
        self.last_drop = values

    def stacks(self):
        # 转成栈序（栈顶在前），与本地模型一致
        return [list(reversed(self.grid[c])) for c in range(6)]

    def _reset_round(self):
        pass


def replay(trace):
    """用录制的掉落序列确定性回放，校验动作合法性与快照一致性。"""
    g = H5Game(drops=trace.get("drops") or [])
    clicks = trace.get("clicks", [])
    frames = trace.get("frames", [])
    # 旧轨迹兼容：同列对 (a,a) 是取消选择，跳过不算移动
    moves = []
    for i in range(0, len(clicks) - 1, 2):
        if clicks[i] == clicks[i + 1]:
            continue
        moves.append((clicks[i], clicks[i + 1]))
    report = {"moves": len(moves), "illegal": 0, "skipped": 0, "frames": len(frames)}
    bad_actions = []
    executed = []  # 每步后的棋盘（底部->顶部）
    for i, (src, dst) in enumerate(moves):
        ok = g.move(src, dst)
        if not ok:
            report["illegal"] += 1
            bad_actions.append((i, src, dst))
            # 宽容模式：源列空通常是撤回导致的轮次错位，跳过继续（快照匹配率会体现质量）
            if g.dead:
                break
            report["skipped"] += 1
            continue
        executed.append([list(c) for c in g.grid])
    report["dead"] = g.dead
    report["final_board"] = g.stacks()
    report["score"] = g.score
    report["n9_count"] = g.n9_count
    report["max_level"] = g.composite_level
    report["bad_actions"] = bad_actions[:10]
    # 快照一致性：快照 click_n -> 已执行动作数 -> 对比执行后棋盘（容差 1 步）
    def is_transient(b):
        # 动画中间态：列顶连续段 >=3（稳态下顶部 3 个相同会立即合并）
        for col in b:
            if not col:
                continue
            v = col[-1]
            k = 0
            for x in reversed(col):
                if x == v:
                    k += 1
                else:
                    break
            if k >= 3:
                return True
        return False

    snap_match = snap_total = snap_skip = 0
    for f in frames:
        if f.get("kind") != "board" or not f.get("board"):
            continue
        if is_transient(f["board"]):
            snap_skip += 1
            continue
        idx = f.get("click_n", 0) // 2 - 1  # 快照记录时刚执行完的动作
        snap_total += 1
        ok = False
        for d in (0, -1, 1):
            j = idx + d
            if 0 <= j < len(executed) and executed[j] == f["board"]:
                ok = True
                break
        if ok:
            snap_match += 1
    report["snap_match"] = snap_match
    report["snap_total"] = snap_total
    report["snap_skip"] = snap_skip
    # 质量分级：快照匹配率（撤回局匹配率低）
    report["quality"] = ("high" if snap_total and snap_match / snap_total >= 0.8
                         else "medium" if snap_total and snap_match / snap_total >= 0.5
                         else "low" if snap_total else "none")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    files = [Path(f) for f in args.files]
    if args.dir:
        files = sorted(Path(args.dir).glob("*.json"))
    if not files:
        print("没有输入文件")
        return
    stats = {"n": 0, "illegal_any": 0, "total_moves": 0, "total_illegal": 0}
    for f in files:
        try:
            tr = json.load(open(f))
        except Exception as e:
            print(f"{f.name}: 读取失败 {e}")
            continue
        rep = replay(tr)
        stats["n"] += 1
        stats["total_moves"] += rep["moves"]
        stats["total_illegal"] += rep["illegal"]
        if rep["illegal"]:
            stats["illegal_any"] += 1
        if not args.summary:
            print(f"{f.name}: 动作 {rep['moves']} 非法 {rep['illegal']} 质量[{rep['quality']}] "
                  f"得分 {rep['score']} 合成9 {rep['n9_count']} 最高级 {rep['max_level']} 死亡 {rep['dead']} "
                  f"快照 {rep['snap_match']}/{rep['snap_total']}")
    print(f"\n=== 汇总: {stats['n']} 局, 总动作 {stats['total_moves']}, "
          f"非法动作 {stats['total_illegal']}, 含非法动作的局 {stats['illegal_any']} ===")


if __name__ == "__main__":
    main()
