"""使用 Policy+Value 网络和显式 chance node 的随机环境 PUCT。"""

import copy
import math
import random

import torch

from agents.dqn import N_ACTIONS, action_index, encode, index_action, legal_mask


class _Edge:
    def __init__(self, prior, afterstate_value=0.0):
        self.prior = prior
        self.afterstate_value = afterstate_value
        self.visits = 0
        self.value = 0.0
        self.child = _Node()
        self.chance = None


class _Node:
    def __init__(self):
        self.visits = 0
        self.edges = None


class _ChanceOutcome:
    def __init__(self, values):
        self.values = values
        self.child = _Node()


class _ChanceNode:
    """用固定蒙特卡洛粒子近似一个 afterstate 的随机转移分布。"""

    def __init__(self):
        self.particles = None
        self.outcomes = None
        self.visits = 0

    def _add_particle(self, state, rng):
        values = tuple(state.sample_chance(rng))
        outcome = self.outcomes.get(values)
        if outcome is None:
            outcome = _ChanceOutcome(values)
            self.outcomes[values] = outcome
        self.particles.append(outcome)

    def sample(self, state, rng, n_samples, widening):
        if self.particles is None:
            self.particles = []
            self.outcomes = {}
            if widening <= 0:
                for _ in range(max(1, n_samples)):
                    self._add_particle(state, rng)
        if widening > 0:
            target = min(
                max(1, n_samples),
                max(1, math.ceil((self.visits + 1) ** widening)),
            )
            if len(self.particles) < target:
                self._add_particle(state, rng)
        self.visits += 1
        return rng.choice(self.particles)


@torch.no_grad()
def _evaluate(net, game, device, cache, death_penalty):
    state = encode(game)
    key = state.numpy().tobytes()
    cached = cache.get(key)
    if cached is not None:
        return cached
    policy, future_n9, _distance, death_risk, afterstate_q = net(
        state.to(device).unsqueeze(0)
    )
    mask = legal_mask(game, device).bool()
    policy = policy[0].masked_fill(~mask, float("-inf"))
    priors = torch.softmax(policy, dim=0).cpu()
    result = (
        priors,
        max(0.0, float(future_n9[0])) - death_penalty * float(death_risk[0]),
        afterstate_q[0].cpu(),
    )
    cache[key] = result
    return result


def puct_search(
    game, net, device="cpu", simulations=64, depth=24,
    c_puct=1.5, gamma=0.99, death_penalty=0.0, chance_samples=0,
    chance_widening=0.0, root_min_visits=0, return_policy=False,
    return_q=False,
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
        updated_leaf_edge = False
        for _step in range(max(1, depth)):
            if state.dead:
                leaf_value = -0.5
                break
            if node.edges is None:
                priors, leaf_value, afterstate_q = _evaluate(
                    net, state, device, cache, death_penalty,
                )
                node.edges = {
                    action_index(s, d): _Edge(
                        float(priors[action_index(s, d)]),
                        float(afterstate_q[action_index(s, d)]),
                    )
                    for s, d in state.legal_moves()
                }
                break
            underexplored = (
                [item for item in node.edges.items() if item[1].visits < root_min_visits]
                if node is root and root_min_visits > 0 else []
            )
            if underexplored:
                action_id, edge = max(
                    underexplored,
                    key=lambda item: (-item[1].visits, item[1].prior),
                )
            else:
                sqrt_n = math.sqrt(max(1, node.visits))
                action_id, edge = max(
                    node.edges.items(),
                    key=lambda item: (
                        item[1].value / item[1].visits if item[1].visits else 0.0
                    ) + c_puct * item[1].prior * sqrt_n / (1 + item[1].visits),
                )
            action = index_action(action_id)
            if chance_samples > 0:
                if not state.move_afterstate(*action):
                    leaf_value = -0.5
                    break
                if state.chance_required():
                    if edge.chance is None:
                        edge.chance = _ChanceNode()
                        if net.afterstate_q:
                            leaf_value = max(0.0, edge.afterstate_value)
                            edge.visits += 1
                            edge.value += leaf_value
                            node.visits += 1
                            updated_leaf_edge = True
                            break
                    outcome = edge.chance.sample(
                        state, rng, chance_samples, chance_widening,
                    )
                    state.resolve_afterstate(outcome.values)
                    child = outcome.child
                else:
                    state.resolve_afterstate()
                    child = edge.child
            else:
                if not state.move(*action):
                    leaf_value = -0.5
                    break
                child = edge.child
            reward = float(sum(event >= 9 for event in state.events))
            path.append((node, edge, reward))
            node = child
        value = leaf_value
        for visited, edge, reward in reversed(path):
            value = reward + gamma * value
            edge.visits += 1
            edge.value += value
            visited.visits += 1
        if not path and not updated_leaf_edge:
            root.visits += 1

    if root.edges is None:
        action = legal[0]
        if not return_policy:
            return action
        policy = torch.zeros(N_ACTIONS)
        policy[action_index(*action)] = 1.0
        if return_q:
            return action, policy, torch.zeros(N_ACTIONS), policy.clone()
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
    if not return_q:
        return action, visits / visits.sum()
    q_targets = torch.zeros(N_ACTIONS)
    q_mask = torch.zeros(N_ACTIONS)
    for child_action, edge in root.edges.items():
        if edge.visits > 0:
            q_targets[child_action] = edge.value / edge.visits
            q_mask[child_action] = 1.0
    return action, visits / visits.sum(), q_targets, q_mask
