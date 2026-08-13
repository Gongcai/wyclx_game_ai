# Suggested title

[D] Planning/RL for a stochastic single-player merge puzzle: afterstates, previewed chance events, and long-horizon throughput

# Post body

I am working on an AI for a small single-player merge puzzle and would appreciate pointers to related algorithms, papers, or existing implementations. It resembles 2048 in its action -> afterstate -> random event structure, but has a larger action space, stack constraints, and a random event that is previewed one move before it is applied.

I have an exact simulator. I am not trying to learn the game dynamics from pixels at this stage; the current question is how best to learn values/policies and allocate a limited planning budget.

## Game rules

- The board contains 6 vertical stacks, each with a maximum height of 7. The first item in a stack is its top.
- An action chooses an ordered pair of different columns: 6 x 5 = 30 possible actions.
- The complete contiguous run of equal tiles at the top of the source stack is moved onto the destination stack. An action moves the whole run, not one tile.
- If the destination now has at least 3 equal tiles at its top, the complete run merges into one tile of value `n + 1`. Cascades are possible.
- A merged 9 disappears and gives one point. Tiles normally present on the board have values 1 through 8.
- Merging happens before overflow is checked. The game ends when any stack remains higher than 7.
- Every fourth player action is followed by one new random tile being added to every column.
- The six upcoming random values are revealed after the third action. The player can therefore choose the fourth action while knowing the exact six tiles that will then be added.
- A random tile is in `[1, min(7, highest value merged so far)]`. The real distribution is not yet known. It appears biased toward high values, and human players report runs of "simple" drops (one or two distinct values) alternating with more complex mixed drops.

One cycle is therefore:

```text
deterministic action 1
deterministic action 2
deterministic action 3 -> reveal a random six-tile preview
preview-conditioned action 4 -> apply the known six-tile drop
repeat
```

The random preview is the chance event. Applying an already revealed preview is deterministic.

## Objectives

There are two related objectives:

1. Maximize the number of 9s in one game.
2. Maximize the total number of 9s in 30 minutes. Death permits a restart, so this is closer to a continuing average-reward/throughput problem than a conventional episodic score problem.

The real interface is animation-limited to roughly one player action per second, so 30 minutes is approximately 1,800 actions. Human results in the timed mode are around 115 total 9s on the server I observed. In a separate untimed mode, strong humans can maintain a mature board for 1,000+ 9s, although that mode allows one limited revive.

The distinction between cold-start cost and mature-board efficiency seems important. In one of the current AI's best games, the first 9 took 48 actions, while subsequent 9s took 18.7 actions on average.

## Current representation and network

The state contains:

- a 6 x 7 x 9 one-hot board;
- the four-action cycle phase;
- the six preview values when known, plus a preview-present flag;
- the current random-tile value cap;
- the maximum number of empty columns reached in the current cycle and in each of the previous three cycles.

The current input has 394 features.

The Policy/Value network is column-permutation equivariant:

- one shared encoder processes each column;
- an ordered source/destination pair head scores the 30 actions;
- value heads predict future 9 count over a long horizon, normalized distance to the next 9, and short-term death risk.

The history features were motivated by a human rule of thumb: in long games, at least one of the last three drop cycles should have temporarily maintained two empty columns. The history is not required for Markov dynamics under the current IID simulator; it is intended as a strategic summary and may become predictive if real drops have temporal regimes.

## Current planning

I use the exact simulator with a stochastic PUCT search. The player action is separated into a deterministic afterstate and an explicit chance node.

Current configuration:

```text
128 simulations per real action
maximum tree depth: 32 player actions
c_puct: 1.5
gamma: 1.0
death-risk penalty: 0.5
maximum 8 fixed chance particles per chance node
chance progressive widening exponent: 0.5
minimum 2 visits for every legal root action
```

At the third action, simulations branch over sampled six-tile previews. Below each preview outcome, the tree can choose a different fourth action and applies that preview exactly. After every real action I currently rebuild the tree rather than reusing it.

Depth 32 is only a cap. With 30 root actions, 128 simulations, root coverage, and chance branching, most candidates receive only shallow explicit search; the learned Value network estimates most of the long horizon.

## Training process

The current process is a form of expert iteration/reanalyse:

1. Generate long games with beam search and then Policy/Value-guided PUCT.
2. Save full episodes, root visit distributions, 9-event positions, death, and optional root action values.
3. Train on column-permutation augmentation.
4. Give extra policy weight to states after the first 9, states containing 7/8 tiles, high-scoring episodes, and states with human-like long-game structure.
5. Generate new PUCT trajectories with the updated network and repeat.

I initially used DQN, behavior cloning, demonstration replay, and DAgger-style data aggregation. The Policy/Value + search route has been substantially better for long games.

## Current results

These are simulator results under one assumed high-value-biased drop distribution, not results from the real game distribution.

- An earlier explicit-chance PUCT model scored 81 total 9s in 16 episodes (mean 5.06, maximum 11, 2,365 actions).
- Search distillation later produced a game with 13 total 9s in 272 actions. This remains the single-game maximum.
- Adding human-structure weighting improved a small paired evaluation.
- Adding the four-cycle empty-column history produced 59 total 9s in 1,675 actions over 12 new episodes, versus 47 in 1,537 actions for its no-history teacher on the same seeds. This is 35.2 versus 30.6 9s per 1,000 actions, but 12 episodes is far too small for a reliable conclusion.
- Under the current assumed distribution, even 35.2 per 1,000 actions projects to only about 63 per 1,800 actions, still well below the observed human timed score.

I am moving toward paired evaluation on at least 64-128 untouched seeds with bootstrap confidence intervals. I track first-9 cost, subsequent-9 gaps, survival length, per-1,000-action throughput, and fixed-action-budget totals rather than only mean episodic score.

## Things that did not work

- A learned action/afterstate Q head achieved low offline MAE but made closed-loop search much worse. Ordinary reanalyse covered too few actions per state, while a full-action root target still suffered from extrapolation/calibration problems.
- Jointly fine-tuning the shared encoder for Q degraded the existing policy and value estimates.
- Increasing root minimum visits from 2 to 3 reduced performance.
- Increasing simulations from 128 to 192 did not improve the paired sample.
- Directly adding a handcrafted board-structure score to leaf values changed behavior but reduced overall performance. Using the structure only to weight policy training was better.
- Exhaustively maximizing over all preview-conditioned fourth actions at a leaf caused severe maximization bias because the learned Value was not one-step Bellman-consistent.
- Restricting search to exactly one four-action cycle had mixed results even after fixing depth-cutoff evaluation.
- Repeated policy-only self-distillation quickly saturated.

## Approaches I am considering

1. **2048-style afterstate TD / N-tuple value learning.** The deterministic action followed by a random event seems almost exactly the setting where afterstate TD is useful. I am unsure how best to combine it with the three deterministic actions, the preview chance node, and the preview-conditioned fourth action.
2. **Gumbel MuZero / sequential halving at the root.** With 30 legal actions and only 128 simulations, forcing every root action to receive two visits may waste half the budget.
3. **Persistent tree reuse.** Re-root after each selected action and, when the real preview appears, follow the matching chance outcome or add it if it was not sampled.
4. **Multi-horizon or distributional values.** Predict future 9s over 16/64/256 actions, survival, and perhaps return quantiles instead of one noisy long-horizon mean.
5. **Average-reward training.** Optimize fixed-action-budget throughput including restart/cold-start cost instead of episodic discounted return.
6. **A regime-switching drop model.** Fit an HMM or other conditional sampler if real preview logs confirm alternating simple/complex drop regimes, then condition the policy on recent previews or a distribution belief.
7. **A frozen base network plus residual adapters.** Learn history-dependent corrections to policy/value without damaging the already useful board encoder.

## Questions

- Is there an established algorithm or open-source project for a game with this action -> afterstate -> chance -> preview-conditioned action structure?
- Would an N-tuple afterstate value network plus expectimax be a better fit than a neural Policy/Value + PUCT system here?
- How would you allocate 128 simulations across 30 root actions and stochastic preview outcomes? Is Gumbel sequential halving the obvious next step?
- Is tree reuse across deterministic actions and observed chance outcomes likely to matter more than another round of self-play training?
- What is a sound way to train an afterstate value without the all-action extrapolation failure I saw with the Q head?
- For the 30-minute objective, would you formulate this as an average-reward continuing MDP, a fixed-horizon problem with automatic resets, or something else?
- Are there papers on 2048, SameGame, Tetris, stochastic packing/merge puzzles, or inventory-like stack planning that are especially relevant?
- Are there standard tests for deciding whether observed random drops are IID or generated by a hidden regime process before building a conditional model?

The most relevant work I have found so far is the 2048 N-tuple/afterstate TD literature, Single-Player MCTS for SameGame, Gumbel MuZero, and "Planning in Stochastic Environments with a Learned Model" (Stochastic MuZero). Pointers to stronger baselines, code, or terminology for this problem class would be very helpful.
