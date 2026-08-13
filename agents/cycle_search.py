"""周期感知搜索：确定性 1-3 步精确推演 + 周期边界分布加权 expectimax + 第 4 步已知掉落延伸。

设计依据（用户与 human_policy）：
- 游戏 4 步一循环：第 1-3 步无掉落（完全确定），第 3 步操作后揭示预告，第 4 步应用掉落。
- 第 1/2/3 步决策时第 4 步掉落未知 → 在周期边界做分布加权 expectimax：
  value = (1/N) Σ_preview max_{第4步} [reward + 终态价值]，即平均期望而非取最大。
- 第 4 步决策时掉落已知 → 完全确定性推演（应用已知预告），延伸到下一周期。
- 奖励 R3：合成 9 + 中间合成(1→2, 2→3) + 顶对子前提（凑出 [x,x] 只差一个）。
- 边界价值可插拔：V1 启发式 heuristic_health（分布无关）/ V2 学习网络。
"""

import copy
import random

from agents.heuristic import heuristic_health


def top_pairs(game):
    """顶对子数：列顶连续 ≥2 个相同元素（只差一个同值即可合成）。"""
    return sum(
        1 for st in game.stacks if len(st) >= 2 and st[0] == st[1]
    )


def cycle_search(
    game, value_fn=heuristic_health, width=16, depth_moves=4, n_samples=8,
    w_merge=1.0, w_pair=0.5, w_value=1.0, seed=0, rng=None,
):
    """从当前游戏状态执行一个周期感知搜索，返回首选动作。

    判定规则：state.moves%4==3 且 preview is None → 搜索生成的"第3步结束"边界
    （预告未知，expectimax）；state.moves%4==3 且 preview 已知 → 真实第4步
    （确定性，扩展）。
    """
    rng = rng or random.Random(seed)

    def _reward(sim, pairs_before):
        n9 = sum(1 for e in sim.events if e >= 9)
        interm = sum(e for e in sim.events if e < 9)
        pairs_after = top_pairs(sim)
        return 9.0 * n9 + w_merge * interm + w_pair * (pairs_after - pairs_before)

    def _boundary(state):
        """state 处于第3步结束、预告未知。正确 expectimax：对采样的每个 preview，
        求最优第4步 + 终态价值，再取平均。"""
        total = 0.0
        for _ in range(max(1, n_samples)):
            sim = copy.deepcopy(state)
            sim.rng = random.Random(rng.randrange(2**63))
            sim.resolve_afterstate(None)          # 采样 preview → 第4步选择点
            pb = top_pairs(sim)
            best4 = float("-inf")
            for a4 in sim.legal_moves():
                s4 = copy.deepcopy(sim)
                if not s4.move_afterstate(*a4):
                    continue
                s4.resolve_afterstate()           # 应用 preview（掉落确定）
                v = _reward(s4, pb) + w_value * value_fn(s4)
                if v > best4:
                    best4 = v
            if best4 == float("-inf"):
                best4 = 0.0
            total += best4
        return total / max(1, n_samples)

    def _is_boundary(st):
        return st.moves % 4 == 3 and st.preview is None

    def _terminal(st, cum):
        # 边界状态的值已含 expectimax；否则加终态价值
        return cum if _is_boundary(st) else cum + w_value * value_fn(st)

    beam = [(copy.deepcopy(game), [], 0.0)]        # (state, action_seq, cum_value)
    for _ in range(depth_moves):
        expanded = []
        for state, seq, cum in beam:
            if state.dead:
                continue
            if _is_boundary(state):
                expanded.append((state, seq, cum))     # 终态，不再扩展
                continue
            pairs_before = top_pairs(state)
            for src, dst in state.legal_moves():
                sim = copy.deepcopy(state)
                if not sim.move_afterstate(src, dst):
                    continue
                if _is_boundary(sim):
                    # 边界：先放占位（cum 未含 expectimax），剪枝后再补算
                    expanded.append((sim, seq + [(src, dst)], cum))
                else:
                    sim.resolve_afterstate()
                    r = _reward(sim, pairs_before)
                    expanded.append((sim, seq + [(src, dst)], cum + r))
        if not expanded:
            break
        # 用廉价代理分剪枝：边界状态用启发式健康度近似，非边界用累计值
        expanded.sort(
            key=lambda c: (
                c[2] + w_value * value_fn(c[0]) if not _is_boundary(c[0]) else c[2]
            ),
            reverse=True,
        )
        beam = expanded[:width]
        # 只对存活的边界状态补算 expectimax（降 n_samples×30 重复）
        beam = [
            (st, seq, cum + (_boundary(st) if _is_boundary(st) else 0.0))
            for st, seq, cum in beam
        ]
    best = max(beam, key=lambda b: _terminal(b[0], b[2]))
    return best[1][0] if best[1] else None
