# Equity engine

## Overview

The equity engine calculates **expected pot share** for 2–6 players in Pot-Limit Omaha (PLO4 and PLO5). It supports weighted hand ranges, preflop and postflop boards, exact enumeration, and adaptive Monte Carlo estimation.

The package lives in `src/plorangelab/equity/`. Its public entry point is `compute_equity()`. Computationally intensive work is implemented with NumPy and Numba; the engine uses precomputed five-card ranking tables supplied by the separate `rank_combination` module.

**Scope:** This package validates and prepares calculation inputs, evaluates showdowns, selects and executes equity algorithms, and returns results with applicable sampling diagnostics. 

## Public interface and data contracts

The package exports the following names from `__init__.py`:

| Export | Purpose |
| --- | --- |
| `compute_equity()` | Validate inputs, select a method, and calculate player equities. |
| `sanitize_range()` | Return an encoded range with board-blocked hands removed. |
| `EquityResult` | Final result of an exact or Monte Carlo calculation. |
| `MonteCarloResult` | Intermediate or final Monte Carlo result and diagnostics. |
| `SamplingExhaustedError` | Indicates that no valid showdown was sampled within the attempt limit. |

`compute_equity()` accepts an ordered list of `EncodedRange` instances, `mode` (`4` or `5`), an optional encoded board, calculation options, and a required `RankLookup` containing precomputed ranking tables. Results preserve the input player order.

### Encoded input representation

| Data | Representation | Invariant |
| --- | --- | --- |
| Card | Integer ID in `[0, 51]` | Each ID represents one card. |
| Player hands | `uint8` array, shape `(n, mode)` | Four or five distinct cards per hand. |
| Hand masks | `uint64` array, shape `(n,)` | One bit per card; masks must match hands. |
| Hand weights | Floating-point array, shape `(n,)` | Finite and strictly positive, aligned with hands and masks. |
| Board | `uint8` array, shape `(b,)`, or `None` | `b` is 0, 3, 4, or 5; no duplicate cards. |

Ranges cannot contain duplicate hands, and each input range must be nonempty. Known board cards are removed from each range before computation. The filtering preserves the alignment of hands, masks, and weights and does not mutate the original range. An empty range after board filtering is an error.

Individual ranges are validated before execution; **cross-player compatibility is not established during initial validation**. Two valid ranges can still contain no non-overlapping combination of player hands.

## Calculation flow

```text
compute_equity(players, mode, board, options, rank_lookup)
    |
    +-- Validate arguments and encoded-range invariants
    |
    +-- Remove hands blocked by known board cards
    |
    +-- Choose method (if method="auto")
    |       +-- Exact enumeration for small postflop workloads
    |       +-- Monte Carlo otherwise
    |
    +-- Execute selected method
    |       +-- Exact: sequential or parallel enumeration
    |       +-- Monte Carlo: rejection or importance sampling
    |
    +-- Return EquityResult in input player order
```

The caller can explicitly request `"exact"` or `"monte-carlo"` instead of automatic method selection. The **automatic** method always chooses Monte Carlo preflop. Postflop, it chooses exact enumeration when the conservative estimate is at most **500,000 showdowns**; otherwise it chooses Monte Carlo. Explicit exact calculations are also subject to the workload limit.

## Omaha hand evaluation

A valid Omaha showdown hand must use **exactly two hole cards and exactly three community cards**. For a complete five-card board, this produces:

- **PLO4:** `C(4, 2) × C(5, 3) = 60` five-card candidates per player.
- **PLO5:** `C(5, 2) × C(5, 3) = 100` five-card candidates per player.

`evaluator.py` evaluates every permitted candidate using `_evaluate_hand_core()`. Candidates are sorted in descending card-ID order using an in-place, fixed five-element sorting network, then passed to `rank_combination_lookup()` with the supplied rank and binomial tables. The highest ranking is retained; equal rankings tie.

`numba_evaluate_hands()` evaluates a batch of player hands against one complete board **sequentially**. Higher-level calculation kernels parallelize independent showdowns or matchups where appropriate. The evaluation kernels assume inputs have already been validated and do not check for collisions.

## Exact enumeration

`exact.py` enumerates all **compatible player-hand matchups** and all valid completions of the board. A matchup is compatible when no player hands share cards and none conflict with the known board.

For each completed showdown, players with the highest ranking split the pot equally. The contribution of a matchup is proportional to the product of the players' input hand weights. Equivalently, for player `p`:

For each compatible matchup, the engine calculates a matchup weight
by multiplying the weights of all selected player hands.

Each valid board runout contributes to the calculation:

- Evaluate the showdown and determine each player's share of the pot.
- Multiply each player's pot share by the matchup weight.
- Accumulate these weighted pot shares across all matchups and runouts.

Each player's exact equity is the sum of their weighted pot shares
divided by the total accumulated matchup weight across all runouts.

Tied winners split the pot equally. The engine independently normalizes
each player's range weights before aggregation, which does not change
the resulting equity.

### Workload estimate

Before enumeration, the engine estimates number of showdowns is calculated in two steps:

1. Calculate the number of candidate matchups by multiplying the
   number of hands in each player's board-filtered range.

2. Multiply the candidate matchup count by the number of possible
   board completions.

The number of possible board completions is the number of ways to
choose the missing community cards from the remaining deck.

The remaining deck size is determined by subtracting the known board
cards and all players' hole cards from the standard 52-card deck.

The estimate assumes that all candidate matchups are compatible.

The estimate is a **conservative upper bound**: it assumes every candidate matchup is compatible. Overlapping hands are discarded during enumeration, so the actual number of evaluated showdowns can be lower.

### Execution strategy

The sequential Numba kernel iterates over the Cartesian product of player ranges and uses lexicographic combination indices to generate missing board cards. It evaluates complete river boards without constructing runouts.

For estimated workloads **above 1,000 showdowns**, the engine may use a parallel Numba kernel, provided there is more than one candidate matchup and more than one available Numba thread. Parallel work is divided into contiguous portions of the **hand-matchup Cartesian product**, not the board-runout space.

Each worker accumulates results privately. Worker result buffers are padded to reduce false sharing, then combined in a sequential reduction to avoid write races. Sequential and parallel results may differ by tiny floating-point rounding amounts.

## Monte Carlo estimation

For larger workloads, `monte_carlo.py` coordinates adaptive Monte Carlo batches. Player hands are sampled according to normalized input weights; missing board cards are sampled uniformly from the undealt cards. Each completed showdown contributes an observed pot share between 0 and 1.

The engine uses one of two samplers.

### Rejection sampling

1. Independently draw one weighted hand from each player's range.
2. Reject the attempt if any player hands overlap.
3. Remove the selected cards from the remaining deck.
4. Uniformly draw any missing board cards.
5. Evaluate the showdown and record the pot share for each player.

Equity is the arithmetic mean of each player's observed pot shares. The estimator uses sample variance rather than a Bernoulli formula because ties can produce fractional pot shares.

Sampling and dealing take place sequentially. **Completed showdown evaluations are parallelized** using Numba's `prange`; aggregation is performed sequentially to avoid write races and preserve a stable reduction order.

### Importance sampling

Independent sampling can waste many attempts when ranges heavily overlap. For calculations with **at least three players** and `max_samples >= 1,000`, the engine first probes **1,024** independently sampled matchups using a separate seeded random generator. If **fewer than 10** are collision-free, it selects the importance sampler. Otherwise it uses rejection sampling.

Importance sampling selects the first player's hand normally, then samples each subsequent player **conditionally on the cards already occupied**:

1. Determine the probability mass of hands compatible with the existing cards.
2. Sample a compatible hand proportionally to its original weight.
3. Multiply the showdown's importance weight by the compatible probability mass.
4. Continue until all players have hands, then draw the missing board cards and evaluate the showdown.

If a player has no compatible hand, that attempt is discarded.

The importance weight of a successful matchup is calculated by
multiplying the compatible probability masses of all players
except the first.

The first player is excluded because their hand is sampled directly
from the original probability distribution.

Each subsequent player's compatible probability mass represents
the total original probability of hands that do not overlap with
previously selected cards.

This compensates for the altered conditional sampling probabilities. Equity is estimated using the **self-normalized weighted mean**:

1. For each completed showdown, multiply the player's pot share
   by that showdown's importance weight.
2. Sum these weighted pot shares across all completed showdowns.
3. Divide the result by the sum of all showdown importance weights.

This ensures that the estimated equity accounts for differences
between the original and conditional sampling distributions.

#### Compatibility indexing

For indexed sampling, the engine builds:

- **Card-membership bitsets** identifying which hands contain each card.
- **Subset-mass lookup tables** containing the probability mass of all 256 possible subsets of each consecutive eight-hand block.

These tables accelerate the calculation of compatible probability mass. They are built when **every player range has at most 50,000 hands**. If any range exceeds that limit, the sampler falls back to direct compatibility scans for all players.

As with rejection sampling, conditional hand selection is sequential; completed showdown evaluations are parallelized using `prange`.

## Precision, stopping, and cancellation

### Margins of error

For rejection sampling, the approximate margin of error is calculated
as follows:

1. Calculate the sample variance of each player's observed pot shares.
2. Divide this variance by the number of completed showdowns.
3. Take the square root to obtain the estimated standard error.
4. Multiply the standard error by `confidence_multiplier`
   (1.96 by default).

The resulting margin of error is expressed as an absolute fraction
of the pot. For example, a margin of 0.001 corresponds to
0.1 percentage points.

Importance sampling uses a weighted residual-based standard-error estimate and also reports **effective sample size**:

Effective sample size is calculated using the following procedure:

1. Sum all importance weights.
2. Square that sum.
3. Separately, square each individual importance weight and sum
   the squared values.
4. Divide the squared total weight from step 2 by the sum of
   squared individual weights from step 3.

This is known as the Kish effective sample size estimator.

Unequal importance weights can make effective sample size much smaller than the raw number of showdowns. Precision-based stopping requires at least `min_samples` **completed showdowns** for rejection sampling or `min_samples` **effective samples** for importance sampling.

**Statistical limitation:** These margins are approximate plug-in estimates. Repeated precision checks and stopping based on observed results do not provide formal confidence-interval coverage guarantees.

### Stop conditions

Monte Carlo execution ends when one of the following occurs:

| `stop_reason` | Meaning |
| --- | --- |
| `target_reached` | Every player's estimated margin is within `target_moe` after the minimum sample threshold. |
| `max_simulations` | The configured maximum number of valid showdowns has been reached. |
| `attempt_limit` | The sampling attempt budget is exhausted. |
| `cancelled` | Cancellation is observed after at least one completed showdown. |

**Simulations** count completed, valid showdowns; **attempts** also include rejected selections. The global attempt limit used by `compute_equity()` is `max(10_000, max_samples * 20)`, and each batch performs at most **100,000 attempts**.

Cancellation is cooperative: the coordinator checks a `threading.Event` **between Monte Carlo batches**. If cancellation occurs before any simulation completes, it raises `InterruptedError`; otherwise it can return a partial estimate with `stop_reason="cancelled"`. A running exact Numba kernel cannot be interrupted this way, although a pre-set cancellation event is checked before calculation starts.

An optional `on_progress` callback receives intermediate `MonteCarloResult` snapshots after completed batches. Such snapshots have `stop_reason=None`. If the attempt budget produces **zero valid showdowns**, the engine raises `SamplingExhaustedError` rather than returning an estimate.

## Results

`compute_equity()` returns `EquityResult`, with one equity value per player:

| Field | Exact | Monte Carlo |
| --- | --- | --- |
| `equities` | Exact expected pot shares | Estimated expected pot shares |
| `method` | `"exact"` | `"monte-carlo"` |
| `margins_of_error` | `None` | Per-player approximate margins (possibly `None` individually) |
| `simulations`, `attempts` | `None` | Counts of completed showdowns and all sampling attempts |
| `stop_reason` | `None` | Final stopping reason |
| `sampler` | `None` | `"rejection"` or `"importance"` |
| `effective_samples` | `None` | Value for importance sampling; `None` for rejection |

`EquityResult` and `MonteCarloResult` are frozen dataclasses. The dataclass fields cannot be reassigned, although their contained lists remain mutable.

## Current limits and defaults

| Setting | Value |
| --- | --- |
| Players | 2–6 |
| Variants | PLO4, PLO5 |
| Known board cards | 0, 3, 4, or 5 |
| Default method | `"auto"` |
| Default exact showdown cap | 500,000 estimated evaluations |
| Parallel exact threshold | More than 1,000 estimated showdowns, with multiple candidate matchups and threads |
| Maximum Monte Carlo simulations | 1,200,000 |
| Default `min_samples` | 10,000 |
| Default `batch_size` | 10,000 |
| Maximum batch size | 100,000 |
| Maximum attempts per batch | 100,000 |
| Default `target_moe` | 0.001 (0.1 percentage points) |
| Default `confidence_multiplier` | 1.96 |
| Default random seed | 3 |
| Importance-index range limit | 50,000 hands per player |

These are current implementation settings, not universal performance recommendations. Runtime and memory consumption depend on range size, collision rate, board state, and available Numba threads.

## Package map

| Module | Responsibility |
| --- | --- |
| [`__init__.py`](../../src/plorangelab/equity/__init__.py) | Public package exports. |
| [`engine.py`](../../src/plorangelab/equity/engine.py) | Input validation, board filtering, method and sampler selection, result construction. |
| [`evaluator.py`](../../src/plorangelab/equity/evaluator.py) | Sequential Numba evaluation of Omaha hands using precomputed rank tables. |
| [`exact.py`](../../src/plorangelab/equity/exact.py) | Exact enumeration, showdown estimates, and sequential/parallel dispatch. |
| [`monte_carlo.py`](../../src/plorangelab/equity/monte_carlo.py) | Rejection sampler, batch coordination, and unweighted statistics. |
| [`importance_sampling.py`](../../src/plorangelab/equity/importance_sampling.py) | Conditional sampling, compatibility indexes, and weighted statistics. |
| [`results.py`](../../src/plorangelab/equity/results.py) | Calculation result dataclasses and stopping-reason type. |
| [`exceptions.py`](../../src/plorangelab/equity/exceptions.py) | Sampling-exhaustion exception. |

Related dependencies outside this package include `plorangelab.encoder` (`EncodedRange`, card masks) and `plorangelab.rank_combination` (`RankLookup`, five-card ranking). Their internals are documented separately from the equity engine.
