"""Five-card lookup generation, indexing, and exhaustive rank equivalence.

These are tests of plorangelab.rank_combination, not the Omaha evaluator.
"""

from math import comb

import numpy as np
import pytest
from numba import njit

from plorangelab.encoder import encode_board
from plorangelab.rank_combination import (
    RankLookup,
    _numba_rank_combination,
    rank_combination_lookup,
)


@njit(cache=True)
def _exhaustive_lookup_check(rank_table, binomial_table):
    """Check every distinct combination, including index uniqueness and coverage."""
    seen = np.zeros(len(rank_table), dtype=np.uint8)
    cards = np.empty(5, dtype=np.uint8)
    checked = 0

    for a in range(4, 52):
        cards[0] = a
        for b in range(3, a):
            cards[1] = b
            for c in range(2, b):
                cards[2] = c
                for d in range(1, c):
                    cards[3] = d
                    for e in range(d):
                        cards[4] = e
                        index = (
                            binomial_table[e, 1]
                            + binomial_table[d, 2]
                            + binomial_table[c, 3]
                            + binomial_table[b, 4]
                            + binomial_table[a, 5]
                        )
                        if index < 0 or index >= len(rank_table):
                            return checked, "index_out_of_range"
                        if seen[index]:
                            return checked, "duplicate_index"
                        seen[index] = 1
                        if rank_combination_lookup(cards, rank_table, binomial_table) != _numba_rank_combination(cards):
                            return checked, "incorrect_rank"
                        checked += 1

    if np.sum(seen) != len(rank_table):
        return checked, "missing_indices"
    return checked, "ok"


def test_lookup_tables_have_expected_shapes_and_dtypes(rank_lookup: RankLookup) -> None:
    assert rank_lookup.rank_table.shape == (comb(52, 5),)
    assert rank_lookup.rank_table.dtype == np.uint32
    assert rank_lookup.binomial_table.shape == (52, 6)
    assert rank_lookup.binomial_table.dtype == np.uint32


@pytest.mark.parametrize(
    ("n", "k"),
    [(0, 0), (0, 1), (4, 2), (16, 5), (51, 5), (51, 0)],
)
def test_binomial_coefficients_match_math_comb(
    rank_lookup: RankLookup, n: int, k: int
) -> None:
    assert int(rank_lookup.binomial_table[n, k]) == (comb(n, k) if k <= n else 0)


@pytest.mark.parametrize(
    ("cards", "expected_index"),
    [
        ([4, 3, 2, 1, 0], 0),
        ([5, 3, 2, 1, 0], 1),
        ([16, 15, 10, 5, 0], 5863),
        ([51, 50, 49, 48, 47], comb(52, 5) - 1),
    ],
)
def test_combinatorial_index_has_known_values(
    rank_lookup: RankLookup, cards: list[int], expected_index: int
) -> None:
    c0, c1, c2, c3, c4 = cards
    binomial = rank_lookup.binomial_table
    index = (
        binomial[c4, 1]
        + binomial[c3, 2]
        + binomial[c2, 3]
        + binomial[c1, 4]
        + binomial[c0, 5]
    )
    assert int(index) == expected_index


@pytest.mark.parametrize(
    "five_cards",
    [
        "AsKsQsJsTs",   # straight flush
        "AsAhAcAdKs",   # four of a kind
        "AsAhAcKsKh",   # full house
        "AsKs9s7s3s",   # flush
        "AsKsQcJcTc",   # straight
        "AsAhAcKd9s",   # three of a kind
        "AsAhKdKc9s",   # two pair
        "AsAhKd9c7s",   # pair
        "AsKdJh9c7s",   # high card
        "As5h4d3c2s",   # wheel
    ],
)
def test_lookup_matches_original_ranker_for_representative_hands(
    rank_lookup: RankLookup, five_cards: str
) -> None:
    cards = np.sort(encode_board(five_cards))[::-1].copy()
    expected = _numba_rank_combination(cards)
    actual = rank_combination_lookup(
        cards, rank_lookup.rank_table, rank_lookup.binomial_table
    )
    assert actual == expected


def test_every_five_card_hand_has_a_unique_index_and_identical_rank(
    rank_lookup: RankLookup,
) -> None:
    checked, status = _exhaustive_lookup_check(
        rank_lookup.rank_table, rank_lookup.binomial_table
    )
    assert status == "ok", f"Failed after {checked:,} combinations: {status}"
    assert checked == comb(52, 5)
