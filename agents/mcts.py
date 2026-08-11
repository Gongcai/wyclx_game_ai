"""带随机掉落机会的 Monte Carlo Tree Search baseline。"""

import copy
import math
import random

from agents.search import _state_value


class _Node:
    def __init__(self):
        self.visits = 0
        self.value = 0.0
        self.untried = None
        self.children = {}


def _event_reward(game):
    return sum(1000.0 if event >= 9 else 2.0 * event for event in game.events)


def _rollout_action(state, rng, width=4):
    actions = state.legal_moves()
    if not actions:
        return None
    candidates = actions if len(actions) <= width else rng.sample(actions, width)
    best = None
    for action in candidates:
        sim = copy.deepcopy(state)
        if not sim.move(*action):
            continue
        value = _event_reward(sim) + 0.05 * _state_value(sim)
        if best is None or value > best[0]:
            best = (value, action)
    return best[1] if best is not None else candidates[0]


def mcts_search(game, simulations=96, depth=16, exploration=1.4, seed=None):
    legal = game.legal_moves()
    if not legal:
        return None
    rng = random.Random(seed if seed is not None else 0xC0FFEE)
    root = _Node()
    for _ in range(max(1, simulations)):
        state = copy.deepcopy(game)
        # 真实 RNG 状态不可观察；每条模拟使用独立的未来掉落样本。
        state.rng = random.Random(rng.randrange(2**63))
        node = root
        path = [node]
        total = 0.0
        discount = 1.0
        expanded = False
        for _step in range(max(1, depth)):
            if state.dead:
                total -= 100.0
                break
            actions = state.legal_moves()
            if not actions:
                break
            if node.untried is None:
                node.untried = list(actions)
                rng.shuffle(node.untried)
            if node.untried:
                action = node.untried.pop()
                child = _Node()
                node.children[action] = child
                expanded = True
            else:
                log_n = math.log(max(1, node.visits))
                action, child = max(
                    node.children.items(),
                    key=lambda item: item[1].value / item[1].visits
                    + exploration * math.sqrt(log_n / item[1].visits),
                )
            if not state.move(*action):
                break
            total += discount * _event_reward(state)
            discount *= 0.97
            node = child
            path.append(node)
            if expanded:
                # 新节点之后用随机 rollout 估计随机掉落下的长期回报。
                for _ in range(_step + 1, max(1, depth)):
                    if state.dead:
                        total -= 100.0
                        break
                    rollout_action = _rollout_action(state, rng)
                    if rollout_action is None:
                        break
                    if not state.move(*rollout_action):
                        break
                    total += discount * _event_reward(state)
                    discount *= 0.97
                if not state.dead:
                    total += 0.1 * _state_value(state)
                break
        if not expanded and not state.dead:
            total += 0.1 * _state_value(state)
        for visited in path:
            visited.visits += 1
            visited.value += total
    if not root.children:
        return legal[0]
    return max(root.children.items(), key=lambda item: item[1].visits)[0]


def mcts_policy(game, simulations=96, depth=16, exploration=1.4):
    seed = game.moves * 1009 + game.max_merged * 9176
    for st in game.stacks:
        for value in st:
            seed = (seed * 31 + value) & ((1 << 63) - 1)
    return mcts_search(
        game, simulations=simulations, depth=depth,
        exploration=exploration, seed=seed,
    )
