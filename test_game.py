import random

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
    assert g.score == 9
    assert g.max_merged == 9


def t_suicide_allowed():
    g = Game(rng=random.Random(0))
    g.stacks = [[7, 6, 5, 4, 3, 2, 1], [1]]
    ok = g.move(1, 0)
    assert ok
    assert g.dead


def t_drop_cycle():
    g = Game(rng=random.Random(1), drop_sampler=lambda rng, m=None: 1)
    g.move(1, 0)
    g.move(2, 1)
    g.move(3, 2)
    assert g.preview is not None and len(g.preview) == 6
    g.move(4, 3)
    assert g.preview is None
    assert g.last_drop == [1, 1, 1, 1, 1, 1]


def t_drop_can_merge():
    g = Game(rng=random.Random(0), drop_sampler=lambda rng, m=None: 1)
    g.stacks = [[1, 1], [1], [], [], [], []]
    g.moves = 3
    g.move(1, 0)
    assert g.stacks[0] == [1, 2], g.stacks[0]


def t_overflow_on_drop():
    g = Game(rng=random.Random(0), drop_sampler=lambda rng, m=None: 1)
    g.stacks = [[7, 6, 5, 4, 3, 2, 1], [1], [], [], [], []]
    g.moves = 3
    g.move(1, 0)
    assert g.dead


def t_preview_matches_actual_drop():
    seq = [1, 3, 2, 1, 7, 5, 4, 2, 1, 3, 6, 1]
    it = iter(seq)
    g = Game(rng=random.Random(0), drop_sampler=lambda rng, m=None: next(it))
    g.move(1, 0)
    g.move(2, 1)
    g.move(3, 2)
    assert g.preview == seq[:6], g.preview
    g.move(4, 3)
    assert g.preview is None
    assert g.last_drop == seq[:6], g.last_drop


def t_drop_respects_max_merged():
    g = Game(
        rng=random.Random(0),
        drop_sampler=lambda rng, m=None: rng.randint(1, min(7, m or 7)),
    )
    g.stacks = [[2, 2], [2], [], [], [], []]
    g.max_merged = 2
    g.moves = 3
    g.move(1, 0)
    assert g.max_merged == 3
    assert all(v <= g.max_merged for v in g.last_drop), g.last_drop


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


def t_afterstate_defers_preview_sampling():
    g = Game(rng=random.Random(0), drop_sampler=lambda rng, m=None: 2)
    g.moves = 2
    assert g.move_afterstate(1, 0)
    assert g.moves == 3
    assert g.preview is None
    assert g.chance_required()
    assert not g.legal_moves()
    assert g.resolve_afterstate([1, 2, 3, 1, 2, 3])
    assert g.preview == [1, 2, 3, 1, 2, 3]
    assert g.legal_moves()


def t_afterstate_applies_known_preview_deterministically():
    g = Game(rng=random.Random(0), drop_sampler=lambda rng, m=None: 7)
    g.moves = 3
    g.preview = [1, 1, 1, 1, 1, 1]
    assert g.move_afterstate(1, 0)
    assert not g.chance_required()
    assert g.resolve_afterstate()
    assert g.last_drop == [1, 1, 1, 1, 1, 1]


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
    t_drop_respects_max_merged()
    t_overflow_on_drop()
    t_death_prevents_further_moves()
    t_afterstate_defers_preview_sampling()
    t_afterstate_applies_known_preview_deterministically()
    t_afterstate_matches_move_and_rng()
    print("all tests passed")


if __name__ == "__main__":
    run_all()
