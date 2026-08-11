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
    print("all tests passed")


if __name__ == "__main__":
    run_all()
