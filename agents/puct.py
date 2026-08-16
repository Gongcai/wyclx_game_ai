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


class PuctTree:
    """跨真实动作复用搜索树：保存本次搜索的根，执行动作后推进到对应子树。"""

    def __init__(self):
        self.node = None


def advance_tree(tree, action, game):
    """在真实 game.move(*action) 之后调用，把根推进到已执行动作的子树。

    chance 边用实际揭示的 preview 匹配 outcome；未命中（该结果没被搜索采样过、
    动作未展开、或树为空）则清空，下一次搜索从头建树。
    """
    root = tree.node
    tree.node = None
    if root is None or root.edges is None:
        return
    edge = root.edges.get(action_index(*action))
    if edge is None:
        return
    if edge.chance is not None:
        if game.preview is None or edge.chance.outcomes is None:
            return
        outcome = edge.chance.outcomes.get(tuple(game.preview))
        if outcome is not None:
            tree.node = outcome.child
        return
    tree.node = edge.child


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


def _unsafe_preview_actions(game):
    """预告已知且下一次移动后即应用（决策状态 moves%4==2）时，
    动作+已知预告掉落必然溢出一列则为不安全动作。

    若所有动作都不安全（盘面已注定）则返回空集，不拦截，避免无路可走。
    """
    if game.moves % 4 != 2 or game.preview is None:
        return set()
    unsafe = set()
    safe_exists = False
    for s, d in game.legal_moves():
        sim = copy.deepcopy(game)
        if not sim.move_afterstate(s, d):
            unsafe.add(action_index(s, d))
            continue
        sim.resolve_afterstate()
        if sim.dead:
            unsafe.add(action_index(s, d))
        else:
            safe_exists = True
    if not safe_exists:
        return set()
    return unsafe


def _shape_bonus(game, shape_over_w, shape_low_w, height_ok=6, low_max=3):
    """人类规则塑形：列高超过 height_ok 的每格惩罚，低牌(<=low_max)每张惩罚。

    依据 human_policy.md 与死亡分析：列高 7 是死亡临界（100% 死亡都有一列到 7），
    1/2/3 低牌不消耗会在后期拥堵棋盘。
    """
    if shape_over_w == 0 and shape_low_w == 0:
        return 0.0
    bonus = 0.0
    if shape_over_w:
        bonus -= shape_over_w * sum(
            max(0, len(st) - height_ok) for st in game.stacks
        )
    if shape_low_w:
        bonus -= shape_low_w * sum(
            1 for st in game.stacks for v in st if v <= low_max
        )
    return bonus


@torch.no_grad()
def _evaluate(net, game, device, cache, death_penalty,
              shape_over_w=0.0, shape_low_w=0.0):
    state = encode(game, history=net.history_features)
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
        max(0.0, float(future_n9[0])) - death_penalty * float(death_risk[0])
        + _shape_bonus(game, shape_over_w, shape_low_w),
        afterstate_q[0].cpu(),
    )
    cache[key] = result
    return result


def puct_search(
    game, net, device="cpu", simulations=64, depth=24,
    c_puct=1.5, gamma=0.99, death_penalty=0.0, chance_samples=0,
    chance_widening=0.0, root_min_visits=0, return_policy=False,
    return_q=False, root_sequential_halving=0, root_q_scale=2.0,
    root_gumbel_noise=0.0, tree=None,
    shape_over_w=0.0, shape_low_w=0.0, safe_veto=False,
):
    """tree: 可选 PuctTree。传入时复用其中的暖子树作为根（须配合
    chance_samples>0 的显式 chance node，root-sampling 模式下子树统计混合了
    不同随机实现，复用不成立），搜索后把根写回 tree 供 advance_tree 推进。"""
    legal = game.legal_moves()
    if not legal:
        return None
    root = _Node()
    if tree is not None:
        if tree.node is not None:
            root = tree.node
        tree.node = root
    root_unsafe = _unsafe_preview_actions(game) if safe_veto else set()
    if root_unsafe and root.edges is not None:
        for aid in root_unsafe:
            root.edges.pop(aid, None)
    cache = {}
    seed = game.moves * 1009 + game.max_merged * 9176
    for st in game.stacks:
        for value in st:
            seed = (seed * 31 + value) & ((1 << 63) - 1)
    rng = random.Random(seed)
    sequential_candidates = None
    forced_root_queue = []
    root_gumbels = {}

    def root_scores(candidates):
        q_values = [
            root.edges[action].value / max(1, root.edges[action].visits)
            for action in candidates
        ]
        low = min(q_values)
        span = max(q_values) - low
        normalized = [
            (value - low) / span if span > 1e-8 else 0.0
            for value in q_values
        ]
        return {
            action: (
                root_gumbel_noise * root_gumbels[action]
                + math.log(max(1e-12, root.edges[action].prior))
                + root_q_scale * normalized[index]
            )
            for index, action in enumerate(candidates)
        }

    total_simulations = max(1, simulations)
    for simulation_id in range(total_simulations):
        forced_root_action = None
        if root_sequential_halving > 0 and root.edges is not None:
            remaining = total_simulations - simulation_id
            if not forced_root_queue:
                if sequential_candidates is None:
                    for action in root.edges:
                        uniform = min(1 - 1e-12, max(1e-12, rng.random()))
                        root_gumbels[action] = -math.log(-math.log(uniform))
                    initial_scores = root_scores(list(root.edges))
                    sequential_candidates = sorted(
                        root.edges, key=initial_scores.get, reverse=True,
                    )[:min(root_sequential_halving, len(root.edges), remaining)]
                elif len(sequential_candidates) > 1:
                    scores = root_scores(sequential_candidates)
                    keep = max(1, math.ceil(len(sequential_candidates) / 2))
                    sequential_candidates = sorted(
                        sequential_candidates, key=scores.get, reverse=True,
                    )[:keep]
                if len(sequential_candidates) == 1:
                    forced_root_queue = [sequential_candidates[0]] * remaining
                else:
                    rounds_left = max(1, math.ceil(math.log2(len(sequential_candidates))))
                    visits_each = max(
                        1, remaining // (len(sequential_candidates) * rounds_left),
                    )
                    forced_root_queue = [
                        action
                        for _visit in range(visits_each)
                        for action in sequential_candidates
                    ][:remaining]
            if forced_root_queue:
                forced_root_action = forced_root_queue.pop(0)
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
                    shape_over_w, shape_low_w,
                )
                legal_moves = state.legal_moves()
                if node is root and root_unsafe:
                    legal_moves = [
                        (s, d) for s, d in legal_moves
                        if action_index(s, d) not in root_unsafe
                    ]
                node.edges = {
                    action_index(s, d): _Edge(
                        float(priors[action_index(s, d)]),
                        float(afterstate_q[action_index(s, d)]),
                    )
                    for s, d in legal_moves
                }
                break
            if node is root and forced_root_action is not None:
                action_id = forced_root_action
                edge = node.edges[action_id]
            else:
                underexplored = (
                    [item for item in node.edges.items() if item[1].visits < root_min_visits]
                    if node is root and root_min_visits > 0 else []
                )
            if node is not root or forced_root_action is None:
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
        else:
            if not state.dead:
                _priors, leaf_value, _afterstate_q = _evaluate(
                    net, state, device, cache, death_penalty,
                    shape_over_w, shape_low_w,
                )
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
    if root_sequential_halving > 0 and sequential_candidates:
        scores = root_scores(sequential_candidates)
        action_id = max(sequential_candidates, key=scores.get)
    else:
        action_id = max(root.edges.items(), key=lambda item: item[1].visits)[0]
    action = index_action(action_id)
    if not return_policy:
        return action
    if root_sequential_halving > 0:
        # halving 的访问计数按淘汰日程分配，末轮候选几乎等访问，不反映优劣；
        # 蒸馏目标改用 Gumbel MuZero 式改进策略 softmax(log prior + σ(Q))，
        # 未访问动作不加 Q 奖励，探索噪声不进入目标。
        q_means = {
            child_action: edge.value / edge.visits
            for child_action, edge in root.edges.items() if edge.visits > 0
        }
        low = min(q_means.values(), default=0.0)
        span = max(q_means.values(), default=0.0) - low
        logits = torch.full((N_ACTIONS,), float("-inf"))
        for child_action, edge in root.edges.items():
            bonus = (
                root_q_scale * (q_means[child_action] - low) / span
                if child_action in q_means and span > 1e-8 else 0.0
            )
            logits[child_action] = math.log(max(1e-12, edge.prior)) + bonus
        policy_dist = torch.softmax(logits, dim=0)
    else:
        visits = torch.zeros(N_ACTIONS)
        for child_action, edge in root.edges.items():
            visits[child_action] = edge.visits
        if visits.sum() == 0:
            visits[action_id] = 1.0
        policy_dist = visits / visits.sum()
    if not return_q:
        return action, policy_dist
    q_targets = torch.zeros(N_ACTIONS)
    q_mask = torch.zeros(N_ACTIONS)
    for child_action, edge in root.edges.items():
        if edge.visits > 0:
            q_targets[child_action] = edge.value / edge.visits
            q_mask[child_action] = 1.0
    return action, policy_dist, q_targets, q_mask
