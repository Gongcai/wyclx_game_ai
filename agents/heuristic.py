"""基于人类策略的启发式策略：可解释评估函数 + 浅层贪心（掉落步用已知预告推演）。

依据 human_policy.md 与死亡分析：
- 列高 ≤6 最佳，7 高列离死亡一步（100% 死亡都有一列到 7）→ 惩罚 max(0, 高-6)
- 低牌(≤3)要尽早消耗，被埋在高值下面就永远清不掉 → 惩罚被埋的低牌
- 保持空列（长局储备）→ 奖励空列
- 第 3、7、11…步预告已知 → 应用预告后评估健康，避免必然溢出
- 即时合并 → 奖励（合成 9 的前进）

human_policy.md 规则映射：
(1) 列内最小逆序、大牌垫底 → order 项
(2) 保持空列 → empty 项；六级以下集中一列尽快合成 → buried 项
(3) 首轮三次、随后每轮四次移动；预告公开后的下一步会应用掉落
(4) 简单形/复杂形 → 由高度/埋雷惩罚自动保守处理
"""

import copy
from dataclasses import dataclass


@dataclass
class HeuristicWeights:
    gain: float = 1.0        # 即时合并事件值权重
    empty: float = 2.0       # 每个空列
    height6: float = 3.0     # 每格超过 6 高（即 7+）的惩罚
    height4: float = 0.4     # 每格超过 4 高（5,6 高的轻微压力）
    buried: float = 1.5      # 每个被埋低牌(≤3,不在顶run)的惩罚
    pair: float = 0.6        # 每个顶对子（可立即合并）
    order: float = 0.3       # 列内有序度（逆序惩罚）
    death: float = 500.0     # 死亡状态惩罚（必须远低于任何存活盘面的健康下限，否则会"选死")


def heuristic_health(game, w=HeuristicWeights()):
    """评估盘面健康度，越高越好。"""
    if game.dead:
        return -w.death
    score = w.empty * sum(not st for st in game.stacks)
    for st in game.stacks:
        score -= w.height6 * max(0, len(st) - 6)
        score -= w.height4 * max(0, len(st) - 4)
    for st in game.stacks:
        # 顶 run 长度（可立即合并的部分）
        k = 1
        while k < len(st) and st[k] == st[0]:
            k += 1
        # 被埋低牌：值<=3 且不在顶 run 内（上面压着别的东西，难以清除）
        for v in st[k:]:
            if v <= 3:
                score -= w.buried
        # 顶对子：可合并
        if k >= 2:
            score += w.pair
        # 列内逆序（大牌在下更好，小牌垫底是雷）
        for i in range(len(st)):
            for j in range(i + 1, len(st)):
                if st[i] > st[j]:
                    score -= w.order
    return score


def heuristic_policy(game, w=HeuristicWeights(), rng=None):
    """贪心：对每个合法动作，计算即时合并增益 + 结果状态健康度，选最优。

    预告应用步（H5 节奏 moves%4==3）：move 后应用已知预告再评估健康，
    避免必然溢出死亡。其余步：评估 move 后 afterstate 健康（惩罚 7 高列）。
    """
    best_action = None
    best_score = float("-inf")
    for src, dst in game.legal_moves():
        sim = copy.deepcopy(game)
        if not sim.move_afterstate(src, dst):
            continue
        gain = float(sum(e for e in sim.events))
        if sim.moves % 4 == 3 and sim.preview is not None:
            sim.resolve_afterstate()          # 应用已知预告（确定性）
        score = w.gain * gain + heuristic_health(sim, w)
        if score > best_score:
            best_score = score
            best_action = (src, dst)
    return best_action


def heuristic_policy_beam(game, w=HeuristicWeights(), width=16, depth=4, seed=0):
    """浅层 beam（人类推演深度，默认一个完整周期 4 步）。

    预告已公开时精确推演应用；尚未公开时只按其分布采样，绝不读取本局隐藏值。
    累积即时合并增益 + 终态健康作为序列评分，返回首步动作。

    预知模式（game.future_seed 非 None）：边界处用真实未来掉落（future_drops）
    而非采样分布，搜索可精确推演多个周期（建议加大 depth 到 8-12）。
    """
    import random
    foresee = game.future_seed is not None
    beam = [(copy.deepcopy(game), [], 0.0)]
    rng = random.Random(seed)
    for _step in range(depth):
        candidates = []
        for state, seq, cum in beam:
            if state.dead:
                continue
            for src, dst in state.legal_moves():
                sim = copy.deepcopy(state)
                if not sim.move_afterstate(src, dst):
                    continue
                gain = float(sum(e for e in sim.events))
                if sim.moves % 4 == 2:
                    # H5 节奏：揭示预抽预告。非预知搜索必须重新采样，不能偷看。
                    fd = sim.future_drops(1)
                    if fd is not None:
                        sim.resolve_afterstate(list(fd[0]))   # 预知：真实预告
                    else:
                        sim.resolve_afterstate(sim.sample_chance(rng))
                else:
                    sim.resolve_afterstate()   # %4==3 应用已知预告（确定性）；其余无操作
                h = heuristic_health(sim, w)
                total = w.gain * (cum + gain) + h
                candidates.append((sim, seq + [(src, dst)], cum + gain, total))
        if not candidates:
            break
        candidates.sort(key=lambda c: c[3], reverse=True)
        beam = [(c[0], c[1], c[2]) for c in candidates[:width]]
    best = max(beam, key=lambda b: w.gain * b[2] + heuristic_health(b[0], w))
    if best[1]:
        return best[1][0]
    # 全部路径都死亡等异常：兜底返回一个合法动作（尽量晚死）
    fallback = None
    fallback_score = float("-inf")
    for src, dst in game.legal_moves():
        sim = copy.deepcopy(game)
        if sim.move_afterstate(src, dst):
            if sim.dead:
                score = -100.0
            else:
                score = heuristic_health(sim, w)
            if score > fallback_score:
                fallback_score = score
                fallback = (src, dst)
    return fallback
