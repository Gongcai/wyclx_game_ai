"""基于已知游戏模型的短视 beam search baseline。"""

import copy
import random


def _top_run(st):
    if not st:
        return 0
    value = st[0]
    n = 1
    while n < len(st) and st[n] == value:
        n += 1
    return n


def _state_value(game):
    """估计非终止状态价值；合成 9 使用极大值确保优先完成目标。"""
    if game.dead:
        return -10000.0
    value = 0.0
    for event in game.events:
        value += 10000.0 if event >= 9 else 12.0 * event
    for st in game.stacks:
        run = _top_run(st)
        if run >= 2:
            value += 18.0 * run
        elif run == 1:
            value += 2.0
        value -= 1.5 * len(st) ** 2
    value += 5.0 * sum(not st for st in game.stacks)
    if game.preview is not None:
        value += 1.0
    return value - 0.05 * game.moves


def _observable_seed(game):
    """只从智能体可观察的信息构造稳定种子，避免搜索偷看真实 RNG。"""
    values = [game.moves % 4, game.max_merged]
    for st in game.stacks:
        values.extend(st)
        values.append(0)
    values.extend(game.preview or [])
    seed = 1469598103934665603
    for value in values:
        seed ^= value + 1
        seed = (seed * 1099511628211) & ((1 << 64) - 1)
    return seed


def beam_search(game, depth=8, width=48):
    """返回当前状态下的首个动作，找不到时返回任一合法动作。"""
    legal = game.legal_moves()
    if not legal:
        return None
    root = copy.deepcopy(game)
    root.rng = random.Random(_observable_seed(game))
    beam = [(root, None, 0.0)]
    best = None
    for _ in range(max(1, depth)):
        candidates = []
        for state, first, path_value in beam:
            for action in state.legal_moves():
                nxt = copy.deepcopy(state)
                if not nxt.move(*action):
                    continue
                root_action = action if first is None else first
                score = path_value + _state_value(nxt)
                candidates.append((score, nxt, root_action))
                if best is None or score > best[0]:
                    best = (score, root_action)
        if not candidates:
            break
        candidates.sort(key=lambda item: item[0], reverse=True)
        beam = [(state, first, score) for score, state, first in candidates[: max(1, width)]]
    return best[1] if best is not None else legal[0]


def beam_policy(game, depth=8, width=48):
    return beam_search(game, depth=depth, width=width)
