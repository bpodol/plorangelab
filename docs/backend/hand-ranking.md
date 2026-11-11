# Five-card hand ranking

## Overview

The hand-ranking subsystem assigns comparable integer strength scores to **every distinct five-card poker hand** in a standard 52-card deck. It generates a dense lookup table once per resource initialization and retrieves scores through combinatorial indexing rather than evaluating hand categories during every showdown.

The ranking algorithm is **not specific to PLO**: it evaluates an ordinary five-card high-poker hand. The equity engine enforces the Omaha rule of using **exactly two hole cards and three board cards** by enumerating permitted five-card candidates and choosing the strongest score. See [Equity engine](equity.md) for the showdown evaluation and equity-calculation flow.

This subsystem is also separate from the **six-handed ProPokerTools starting-hand orderings** used to select percentage-based PLO ranges. Those are loaded from packaged data by `rating.py`; see [Card and range encoding](encoding.md).

## Source files and responsibilities

| File | Responsibility |
| --- | --- |
| `src/plorangelab/rank_combination.py` | Encode five-card hand strengths, generate ranking tables, and perform indexed rank lookups. |
| `src/plorangelab/bootstrap.py` | Load starting-hand rating resources and construct the five-card ranking tables in an `EngineResources` container. |

The equity engine consumes the resulting `RankLookup` through its hand-evaluation kernels; it does not build the tables itself.

## Card representation and input contract

Each card uses an ID from `0` to `51`:

```text
card_id = 4 * rank_index + suit_index
```

The rank index runs from `2 = 0` through `A = 12`; suit indexes are `c = 0`, `d = 1`, `h = 2`, and `s = 3`.

The ranking routines assume a **one-dimensional array of exactly five distinct, valid card IDs sorted in descending order**. They do not validate those preconditions. The caller must supply valid cards and the correct lookup tables; violating these requirements may produce an incorrect result or an indexing error.

In the ranking implementation, shifting a card ID right by two bits extracts its rank index; masking with `3` extracts its suit index.

## Strength scores

`_numba_rank_combination()` evaluates a five-card combination. It identifies its category and relevant tie-breaking ranks, then passes them to `_numba_calculate_rank()`.

Categories are ordered from weakest to strongest:

| Category code | Hand category |
| ---: | --- |
| 0 | High card |
| 1 | One pair |
| 2 | Two pair |
| 3 | Three of a kind |
| 4 | Straight |
| 5 | Flush |
| 6 | Full house |
| 7 | Four of a kind |
| 8 | Straight flush |

The score packs the category followed by up to five tie-breaking rank values into a **base-13 integer**. The category has the highest significance; the remaining rank positions are ordered by tie-breaking priority, and unused positions default to zero.

Consequently:

- **Higher scores indicate stronger hands.**
- **Equal scores indicate equivalent hand strength**, including ties across different suits.
- Scores are strength encodings, **not** consecutive ordinals or probabilities. Their numerical distance does not measure how much stronger one hand is than another.

### Classification details

The evaluator compares adjacent ranks after sorting the five cards. A four-bit equality pattern captures the repetition structure and identifies pairs, trips, full houses, and four of a kind without counting ranks in a separate data structure.

When all five ranks differ, the evaluator checks for flushes and straights, including **A-5-4-3-2** (the ace-low *wheel*). A wheel is scored as a five-high straight rather than an ace-high straight. Straight flushes use the same straight-high-card convention.

For categories with kickers, the score stores the ranks required by normal five-card poker tie-breaking rules.

## Precomputed lookup tables

`build_lookup_tables()` constructs a `RankLookup` containing two NumPy arrays:

| Field | Shape | Dtype | Purpose |
| --- | --- | --- | --- |
| `rank_table` | `(2_598_960,)` | `uint32` | Strength score for each unique five-card combination. |
| `binomial_table` | `(52, 6)` | `uint32` | Binomial coefficients used to map five sorted card IDs to one unique table index. |

There are **2,598,960** distinct five-card combinations from 52 cards. The ranking table occupies approximately **9.9 MiB** of raw array data; the binomial table occupies **1,248 bytes**.

### Table generation

1. Build the binomial coefficient table for card IDs `0` through `51` and subset sizes `0` through `5`.
2. Enumerate every unique set of five card IDs in descending order.
3. Calculate a unique array index for that combination.
4. Evaluate the five-card hand and write its strength score at that index.

The five-card enumeration and score calculation run in a Numba-compiled kernel. A ranking table is generated **in memory**; the implementation does not load it from a packaged `.npy` file.

### Combinatorial indexing

The lookup uses the **combinatorial number system (colex indexing)**. For five card IDs in descending order, it sums five precomputed binomial coefficients:

```python
index = (
    binomial_table[cards[4], 1]
    + binomial_table[cards[3], 2]
    + binomial_table[cards[2], 3]
    + binomial_table[cards[1], 4]
    + binomial_table[cards[0], 5]
)
```

Every distinct five-card combination maps to exactly one index in the dense table. The same indexing operation is used both when generating the table and when retrieving a score.

`rank_combination_lookup()` therefore needs only a fixed number of table reads and arithmetic operations followed by one rank-table access. Once the tables are available, **lookup is constant time per five-card combination**.

## Integration with Omaha evaluation

The equity module's `_evaluate_hand_core()` enumerates all valid selections of exactly two hole cards and three board cards, sorts each candidate in descending card-ID order, and calls `rank_combination_lookup()`.

For a complete five-card board, each player's hand has:

- **PLO4:** 60 valid five-card candidates.
- **PLO5:** 100 valid five-card candidates.

The strongest returned score is the player's showdown rank. Equal highest scores share the pot when equity is calculated.

The five-card lookup does **not** check for illegal card overlaps or implement Omaha selection rules itself. Those responsibilities belong to the encoding, input-validation, and equity-evaluation layers.

## Resource initialization

`bootstrap.py` provides `load_resources()`, returning an `EngineResources` instance with:

- `ratings`: PLO4 and PLO5 starting-hand ordering arrays loaded from packaged `.npy` resources by `load_rating()`.
- `rank_lookup`: Five-card ranking and binomial tables generated by `build_lookup_tables()`.

```text
load_resources()
    |
    +-- load_rating()
    |       +-- Read PLO4/PLO5 starting-hand ordering resources
    |
    +-- build_lookup_tables()
            +-- Generate binomial coefficients
            +-- Generate all five-card strength scores
    |
    +-- Return EngineResources(ratings, rank_lookup)
```

The two resources have different meanings: **starting-hand ratings** determine how percentage expressions select hands; **five-card ranking tables** determine which hand wins at showdown.

`load_resources()` reconstructs these resources on **every call**. The function itself does not cache an `EngineResources` instance. Callers should retain and reuse the returned resources rather than repeatedly generating the ranking table. Numba's `cache=True` caches compiled code where supported; it does not cache the generated ranking-table contents across resource-loading calls.

## Performance and implementation trade-offs

- **Precomputation versus repeated evaluation:** Generating all five-card scores incurs initialization work and memory usage but avoids repeated category detection in equity simulations.
- **Dense arrays versus dictionaries:** Combinatorial indexing allows compact, predictable array access without storing tuple keys or performing hash lookups.
- **Integer comparisons:** Category and kicker ordering are encoded into one integer score, so showdown comparisons are simple numeric comparisons.
- **Numba kernels:** Table generation and lookup are compiled; the lookup function is marked for inlining into other compiled evaluation kernels.
- **Input assumptions:** The hot-path lookup omits validation. Correctness depends on valid, distinct, descending card IDs and tables generated by the same indexing scheme.

## Boundaries and limitations

This implementation provides standard **five-card high-poker rankings**. It does not implement low-hand evaluation, choose PLO hole-card/board combinations, estimate preflop starting-hand strength, or calculate equity by itself.

The lookup tables are generated at resource initialization, and `bootstrap.py` does not enforce singleton initialization or persistence of generated tables. Application-level resource lifetime and reuse are controlled by the caller.
