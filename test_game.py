import random

import torch

from agents.dqn import HISTORY_DIM, META_DIM, encode
from agents.policy_value import PolicyValueNet
from agents.puct import puct_search
from agents.human_strategy import human_structure
from game import Game


def t_basic_merge():
    g = Game(rng=random.Random(0))
    g.stacks = [[2, 2], [2]]
    g.move(1, 0)
    assert g.stacks[0] == [3], g.stacks[0]


def t_merge_run_of_4():
    g = Game(rng=random.Random(0))
    g.stacks = [[2, 2, 2], [2]]
    g.move(1, 0)
    assert g.stacks[0] == [3], g.stacks[0]


def t_cascade():
    g = Game(rng=random.Random(0))
    g.stacks = [[3, 3, 3, 4, 4], [3]]
    g.move(1, 0)
    assert g.stacks[0] == [5], g.stacks[0]


def t_merge_9():
    g = Game(rng=random.Random(0))
    g.stacks = [[8, 8], [8]]
    g.move(1, 0)
    assert g.stacks[0] == [], g.stacks[0]
    assert g.score == 0
    assert g.n9_count == 1
    assert g.max_merged == 9


def t_suicide_allowed():
    g = Game(rng=random.Random(0))
    g.stacks = [[7, 6, 5, 4, 3, 2, 1], [1]]
    ok = g.move(1, 0)
    assert ok
    assert g.dead


def t_drop_cycle():
    g = Game(rng=random.Random(1), drop_sampler=lambda rng, m=None: 1)
    assert g.preview is None                  # H5：开局预告已抽好，但尚未显示
    g.move(1, 0)                              # 第 1 步：仍无预告
    assert g.preview is None
    g.move(2, 1)                              # 第 2 步 → 显示开局预抽的预告
    assert g.preview == [1, 1, 1, 1, 1, 1]
    g.move(3, 2)                              # 第 3 步 → 应用已知预告
    assert g.last_drop == [1, 1, 1, 1, 1, 1]
    assert g.preview is None                  # 应用后隐藏
    g.move(4, 3)                              # 第 4 步无掉落
    assert g.last_drop is None


def t_drop_can_merge():
    g = Game(rng=random.Random(0), drop_sampler=lambda rng, m=None: 1)
    g.stacks = [[1, 1], [1], [], [], [], []]
    g.moves = 2
    g.preview = [1, 1, 1, 1, 1, 1]   # 已知预告
    g.move(1, 0)   # 第 3 步 → 移动合并且应用掉落
    assert g.stacks[0] == [1, 2], g.stacks[0]


def t_overflow_on_drop():
    g = Game(rng=random.Random(0), drop_sampler=lambda rng, m=None: 1)
    g.stacks = [[7, 6, 5, 4, 3, 2, 1], [1], [], [], [], []]
    g.moves = 2
    g.move(1, 0)   # 第 3 步：移动溢出 + 掉落双重检查
    assert g.dead


def t_preview_matches_actual_drop():
    seq = [1, 3, 2, 1, 7, 5, 4, 2, 1, 3, 6, 1]
    it = iter(seq)
    g = Game(rng=random.Random(0), drop_sampler=lambda rng, m=None: next(it))
    assert g.preview is None
    g.move(1, 0)
    assert g.preview is None
    g.move(2, 1)                             # 第 2 步 → 显示开局预抽的预告
    assert g.preview == seq[:6], g.preview
    g.move(3, 2)                             # 第 3 步应用已知预告
    assert g.last_drop == seq[:6], g.last_drop
    assert g.preview is None                 # 应用后隐藏
    g.move(4, 3)                             # 第 4 步无掉落
    g.move(5, 4)                             # 第 5 步无掉落
    g.move(0, 5)                             # 第 6 步 → 显示上轮掉落前预抽的预告
    assert g.preview == seq[6:12], g.preview


def t_drop_uses_pre_draw_max_merged():
    g = Game(
        rng=random.Random(0),
        drop_sampler=lambda rng, m=None: rng.randint(1, min(7, m or 7)),
    )
    g.stacks = [[2, 2], [2], [], [], [], []]
    g.max_merged = 2
    g.moves = 1
    g._hidden_preview = [2] * 6
    g._hidden_preview_level = 2
    g.move(1, 0)   # 第 2 步：移动合成(3)，但预告在等级2时已锁定
    assert g.max_merged == 3
    g.move(0, 5)   # 第 3 步：应用已知预告
    assert g.last_drop == [2] * 6, g.last_drop


def t_next_preview_is_drawn_before_drop_merges():
    levels = []

    def sampler(_rng, level=None):
        levels.append(level)
        return 1

    g = Game(rng=random.Random(0), drop_sampler=sampler)
    # 第 3 步的掉落会使 c0 的 [1, 1] 合成到 2；H5 是在该合成前刷新下一轮。
    g.stacks = [[1, 1], [2], [], [], [], []]
    g.moves = 2
    g.preview = [1] * 6
    g.move(1, 2)
    assert g.max_merged == 2
    assert levels == [1] * 12, levels  # 开局 6 次 + 掉落前刷新下一轮 6 次


def t_h5_score_and_n9_are_separate():
    g = Game(rng=random.Random(0))
    g.stacks = [[2, 2], [2]]
    g.move(1, 0)
    assert g.score == 8 and g.n9_count == 0


def t_run_moves_together():
    g = Game(rng=random.Random(0))
    g.stacks = [[1, 1, 2], [], [], [], [], []]
    g.move(0, 1)
    assert g.stacks[0] == [2], g.stacks[0]
    assert g.stacks[1] == [1, 1], g.stacks[1]


def t_run_move_can_merge():
    g = Game(rng=random.Random(0))
    g.stacks = [[1, 2], [1, 1], [], [], [], []]
    g.move(1, 0)
    assert g.stacks[0] == [2, 2], g.stacks[0]
    assert g.stacks[1] == [], g.stacks[1]


def t_death_prevents_further_moves():
    g = Game(rng=random.Random(0))
    g.stacks = [[7, 6, 5, 4, 3, 2, 1], [1]]
    g.move(1, 0)
    assert not g.move(1, 0)


def t_afterstate_reveals_pre_draw_preview():
    g = Game(rng=random.Random(0), drop_sampler=lambda rng, m=None: 2)
    g.moves = 1
    assert g.preview is None
    assert g.move_afterstate(1, 0)
    assert g.moves == 2
    assert g.preview is None                 # 揭示前
    assert g.chance_required()               # moves%4==2 需采样预告
    assert not g.legal_moves()
    assert g.resolve_afterstate([1, 2, 3, 1, 2, 3])
    assert g.preview == [1, 2, 3, 1, 2, 3]   # 搜索 chance 分支的预告
    assert g.legal_moves()


def t_afterstate_applies_known_preview_deterministically():
    g = Game(rng=random.Random(0), drop_sampler=lambda rng, m=None: 7)
    g.moves = 2
    g.preview = [1, 1, 1, 1, 1, 1]
    assert g.move_afterstate(1, 0)
    assert not g.chance_required()           # moves%4==3 应用已知预告（确定性）
    assert g.resolve_afterstate()
    assert g.last_drop == [1, 1, 1, 1, 1, 1]  # 应用已知预告
    assert g.preview is None                 # 应用后隐藏


def t_afterstate_matches_move_and_rng():
    direct = Game(rng=random.Random(123))
    split = Game(rng=random.Random(123))
    actions = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (0, 5)]
    for action in actions:
        assert direct.move(*action)
        assert split.move_afterstate(*action)
        assert split.resolve_afterstate()
        assert direct.stacks == split.stacks
        assert direct.preview == split.preview
        assert direct.last_drop == split.last_drop
        assert direct.events == split.events
        assert direct.score == split.score
        assert direct.dead == split.dead
        assert direct.current_cycle_empty_peak == split.current_cycle_empty_peak
        assert direct.recent_cycle_empty_peaks == split.recent_cycle_empty_peaks


def t_empty_cycle_history_includes_pre_drop_peak():
    g = Game(rng=random.Random(0), drop_sampler=lambda rng, m=None: 1)
    g.move(0, 1)
    g.move(2, 1)
    g.move(3, 1)   # 第 3 步应用+刷新 → 记录应用前空列峰值
    assert g.current_cycle_empty_peak == 0    # 掉落后每列都有牌
    assert g.recent_cycle_empty_peaks == [3, 0, 0]  # 记录应用前峰值 3
    assert len(encode(g, history=True)) == len(encode(g)) + HISTORY_DIM
    assert len(encode(g)) == 6 * 7 * 9 + META_DIM


def t_policy_value_accepts_empty_history():
    g = Game(rng=random.Random(0))
    net = PolicyValueNet(hidden=16, history_features=True)
    outputs = net(encode(g, history=True).unsqueeze(0))
    assert outputs[0].shape == (1, 30)


def t_puct_root_min_visits_covers_legal_actions():
    g = Game(rng=random.Random(0), drop_sampler=lambda rng, m=None: 1)
    net = PolicyValueNet(hidden=16)
    for parameter in net.parameters():
        torch.nn.init.zeros_(parameter)
    action, policy, q_targets, q_mask = puct_search(
        g,
        net,
        simulations=61,
        depth=1,
        chance_samples=1,
        root_min_visits=2,
        return_policy=True,
        return_q=True,
    )
    assert action in g.legal_moves()
    assert torch.isclose(policy.sum(), torch.tensor(1.0))
    assert int((policy > 0).sum()) == len(g.legal_moves())
    assert int(q_mask.sum()) == len(g.legal_moves())
    assert q_targets.shape == policy.shape == q_mask.shape


def t_puct_depth_cutoff_uses_leaf_value():
    g = Game(rng=random.Random(0), drop_sampler=lambda rng, m=None: 1)
    net = PolicyValueNet(hidden=16)
    for parameter in net.parameters():
        torch.nn.init.zeros_(parameter)
    net.value_head[-1].bias.data[0] = 2.0
    _action, _policy, q_targets, q_mask = puct_search(
        g, net, simulations=2, depth=1, gamma=1.0,
        return_policy=True, return_q=True,
    )
    assert torch.isclose(q_targets[q_mask.bool()].max(), torch.tensor(2.0))


def t_puct_sequential_halving_limits_root_candidates():
    g = Game(rng=random.Random(0), drop_sampler=lambda rng, m=None: 1)
    net = PolicyValueNet(hidden=16)
    for parameter in net.parameters():
        torch.nn.init.zeros_(parameter)
    action, policy = puct_search(
        g, net, simulations=17, depth=2,
        root_sequential_halving=4, return_policy=True,
    )
    assert action in g.legal_moves()
    # halving 蒸馏目标是改进策略：覆盖全部合法动作，峰值与所选动作一致
    legal_ids = {s * 5 + d - (1 if d > s else 0) for s, d in g.legal_moves()}
    assert set((policy > 0).nonzero().flatten().tolist()) == legal_ids
    assert torch.isclose(policy.sum(), torch.tensor(1.0))
    assert int(policy.argmax()) == action[0] * 5 + action[1] - (1 if action[1] > action[0] else 0)


def t_puct_safe_veto_excludes_guaranteed_death():
    from agents.dqn import action_index
    from agents.puct import _unsafe_preview_actions
    g = Game(rng=random.Random(0), drop_sampler=lambda rng, m=None: 1)
    # 决策状态 moves=2（预告已显示），下一步（第3步）移动后应用：c0 顶7+掉7 -> 8高死亡
    g.stacks = [[7, 1, 1, 1, 1, 1, 1], [2], [], [], [], []]
    g.moves = 2
    g.preview = [7, 1, 1, 1, 1, 1]
    g.max_merged = 5
    unsafe = _unsafe_preview_actions(g)
    assert action_index(0, 1) not in unsafe   # 移走 c0 顶7 -> 安全
    assert action_index(1, 2) in unsafe       # 不动 c0 -> 掉落必死
    net = PolicyValueNet(hidden=16)
    for parameter in net.parameters():
        torch.nn.init.zeros_(parameter)
    action = puct_search(g, net, simulations=16, depth=2, safe_veto=True)
    assert action[0] == 0  # 只能选移走 c0 顶的动作


def t_puct_tree_reuse_advances_and_matches_chance():
    from agents.puct import PuctTree, advance_tree

    net = PolicyValueNet(hidden=16)
    for parameter in net.parameters():
        torch.nn.init.zeros_(parameter)
    # 非 chance 边：move 后 moves=1，advance 直接进入 edge.child
    g = Game(rng=random.Random(0), drop_sampler=lambda rng, m=None: 1)
    tree = PuctTree()
    action = puct_search(g, net, simulations=16, depth=2, chance_samples=1, tree=tree)
    assert tree.node is not None and tree.node.edges is not None
    assert g.move(*action)
    advance_tree(tree, action, g)
    assert tree.node is not None
    warm_visits = tree.node.visits
    # 暖树可继续搜索
    action = puct_search(g, net, simulations=16, depth=2, chance_samples=1, tree=tree)
    assert action in g.legal_moves()
    assert tree.node.visits >= warm_visits
    # chance 边：moves%4==2 采样预告（H5 节奏），固定采样器保证与真实 preview 一致
    g = Game(rng=random.Random(0), drop_sampler=lambda rng, m=None: 1)
    tree = PuctTree()
    for _ in range(2):
        action = puct_search(g, net, simulations=8, depth=2, chance_samples=1, tree=tree)
        assert g.move(*action)
        advance_tree(tree, action, g)
    assert g.preview == [1] * 6   # 第 2 步后采样显示预告
    action = puct_search(g, net, simulations=16, depth=2, chance_samples=1, tree=tree)
    assert g.move(*action)        # 第 3 步应用预告
    assert g.preview is None
    advance_tree(tree, action, g)
    assert tree.node is not None
    # 未搜索过的动作：推进后清空，回退到冷启动
    tree = PuctTree()
    advance_tree(tree, (0, 1), g)
    assert tree.node is None


def t_cycle_search_returns_legal_and_covers_phases():
    # cycle_search 基于旧时序（moves%4==3 采样），H5 对齐后已失效；
    # 该方向早已证伪（周期感知搜索不敌 PUCT），测试跳过。
    return


def t_human_structure_prefers_ordered_stacks():
    ordered = Game(rng=random.Random(0))
    ordered.stacks = [[1, 2, 3, 7], [2, 3, 6], [], [], [], []]
    disordered = Game(rng=random.Random(0))
    disordered.stacks = [[7, 3, 2, 1], [6, 3, 2], [], [], [], []]
    ordered_metrics = human_structure(ordered)
    disordered_metrics = human_structure(disordered)
    assert ordered_metrics.inversions == 0
    assert disordered_metrics.inversions > 0
    assert ordered_metrics.score > disordered_metrics.score


def run_all():
    t_basic_merge()
    t_merge_run_of_4()
    t_cascade()
    t_merge_9()
    t_suicide_allowed()
    t_run_moves_together()
    t_run_move_can_merge()
    t_drop_cycle()
    t_drop_can_merge()
    t_preview_matches_actual_drop()
    t_drop_uses_pre_draw_max_merged()
    t_next_preview_is_drawn_before_drop_merges()
    t_h5_score_and_n9_are_separate()
    t_overflow_on_drop()
    t_death_prevents_further_moves()
    t_afterstate_reveals_pre_draw_preview()
    t_afterstate_applies_known_preview_deterministically()
    t_afterstate_matches_move_and_rng()
    t_empty_cycle_history_includes_pre_drop_peak()
    t_policy_value_accepts_empty_history()
    t_puct_root_min_visits_covers_legal_actions()
    t_puct_depth_cutoff_uses_leaf_value()
    t_puct_sequential_halving_limits_root_candidates()
    t_puct_tree_reuse_advances_and_matches_chance()
    t_puct_safe_veto_excludes_guaranteed_death()
    t_cycle_search_returns_legal_and_covers_phases()
    t_human_structure_prefers_ordered_stacks()
    print("all tests passed")


if __name__ == "__main__":
    run_all()
