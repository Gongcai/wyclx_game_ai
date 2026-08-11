"""使用 Policy+Value 网络的随机 root-sampling PUCT。"""

import copy
import math
import random

import torch

from agents.dqn import N_ACTIONS, action_index, encode, index_action, legal_mask


class _Edge:
    def __init__(self, prior):
        self.prior = prior
        self.visits = 0
        self.value = 0.0
        self.child = _Node()


class _Node:
    def __init__(self):
        self.visits = 0
        self.edges = None


@torch.no_grad()
def _evaluate(net, game, device, cache, death_penalty):
    state = encode(game)
    key = state.numpy().tobytes()
    cached = cache.get(key)
    if cached is not None:
        return cached
    policy, future_n9, _distance, death_risk = net(state.to(device).unsqueeze(0))
    mask = legal_mask(game, device).bool()
    policy = policy[0].masked_fill(~mask, float("-inf"))
    priors = torch.softmax(policy, dim=0).cpu()
    result = (
        priors,
        max(0.0, float(future_n9[0])) - death_penalty * float(death_risk[0]),
    )
    cache[key] = result
    return result


def puct_search(
    game, net, device="cpu", simulations=64, depth=24,
    c_puct=1.5, gamma=0.99, death_penalty=0.0, return_policy=False,
):
    legal = game.legal_moves()
    if not legal:
        return None
    root = _Node()
    cache = {}
    seed = game.moves * 1009 + game.max_merged * 9176
    for st in game.stacks:
        for value in st:
            seed = (seed * 31 + value) & ((1 << 63) - 1)
    rng = random.Random(seed)

    for _ in range(max(1, simulations)):
        state = copy.deepcopy(game)
        state.rng = random.Random(rng.randrange(2**63))
        node = root
        path = []
        leaf_value = 0.0
        for _step in range(max(1, depth)):
            if state.dead:
                leaf_value = -0.5
                break
            if node.edges is None:
                priors, leaf_value = _evaluate(
                    net, state, device, cache, death_penalty,
                )
                node.edges = {
                    action_index(s, d): _Edge(float(priors[action_index(s, d)]))
                    for s, d in state.legal_moves()
                }
                break
            sqrt_n = math.sqrt(max(1, node.visits))
            action_id, edge = max(
                node.edges.items(),
                key=lambda item: (
                    item[1].value / item[1].visits if item[1].visits else 0.0
                ) + c_puct * item[1].prior * sqrt_n / (1 + item[1].visits),
            )
            if not state.move(*index_action(action_id)):
                leaf_value = -0.5
                break
            reward = float(sum(event >= 9 for event in state.events))
            path.append((node, edge, reward))
            node = edge.child
        value = leaf_value
        for visited, edge, reward in reversed(path):
            value = reward + gamma * value
            edge.visits += 1
            edge.value += value
            visited.visits += 1
        if not path:
            root.visits += 1

    if root.edges is None:
        action = legal[0]
        if not return_policy:
            return action
        policy = torch.zeros(N_ACTIONS)
        policy[action_index(*action)] = 1.0
        return action, policy
    action_id = max(root.edges.items(), key=lambda item: item[1].visits)[0]
    action = index_action(action_id)
    if not return_policy:
        return action
    visits = torch.zeros(N_ACTIONS)
    for child_action, edge in root.edges.items():
        visits[child_action] = edge.visits
    if visits.sum() == 0:
        visits[action_id] = 1.0
    return action, visits / visits.sum()
