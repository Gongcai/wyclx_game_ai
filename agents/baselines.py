import copy
import random

from game import Game


def clone(game):
    return copy.deepcopy(game)


def random_policy(game):
    return random.choice(game.legal_moves())


def immediate_reward(game):
    n9 = sum(1 for v in game.events if v >= 9)
    interm = sum(v for v in game.events if v < 9)
    return 10.0 * n9 + 1.0 * interm


def greedy_policy(game):
    best = None
    for move in game.legal_moves():
        sim = clone(game)
        sim.move(*move)
        if sim.dead:
            continue
        val = immediate_reward(sim) - 0.02 * sum(len(s) for s in sim.stacks)
        if best is None or val > best[0]:
            best = (val, move)
    return best[1] if best else random_policy(game)


def play(game, policy):
    if policy == "random":
        step = random_policy
    elif policy == "greedy":
        step = greedy_policy
    else:
        raise ValueError(policy)
    while not game.dead:
        src, dst = step(game)
        if not game.move(src, dst):
            break
    return game.score, game.moves, game.n9_count
