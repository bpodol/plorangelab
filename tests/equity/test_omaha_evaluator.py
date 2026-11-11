"""Omaha evaluator correctness: exactly two hole cards and three board cards.

The compiled evaluator expects valid encoded inputs. Card validation belongs to
its caller and is not part of this test module's contract.
"""

import numpy as np
import pytest

from plorangelab.encoder import encode_board, encode_range
from plorangelab.equity.evaluator import (
    numba_evaluate_hands,
)
from plorangelab.rank_combination import RankLookup, _numba_rank_combination


def _reference_rank(cards: np.ndarray) -> int:
    """Rank a known five-card hand with the original ranker as an oracle."""
    sorted_cards = np.sort(cards)[::-1].copy()
    return int(_numba_rank_combination(sorted_cards))


def _evaluate_one(hand_text: str, board_text: str, rank_lookup: RankLookup) -> int:
    """Evaluate one PLO4/PLO5 hand via the production batch evaluator."""
    mode = len(hand_text) // 2
    hands = encode_range(hand_text, mode).hands
    board = encode_board(board_text)
    scores = numba_evaluate_hands(
        hands,
        board,
        rank_lookup.rank_table,
        rank_lookup.binomial_table,
    )
    assert scores.shape == (1,)
    return int(scores[0])


# Omaha hand construction and comparisons


def test_evaluate_hand_uses_exactly_two_hole_and_three_board_cards(
    rank_lookup: RankLookup,
) -> None:
    best_five = encode_board("AsAhAcKdKh")

    assert _evaluate_one("AsAhQsJd", "AcKdKh2c3d", rank_lookup) == _reference_rank(
        best_five
    )


def test_evaluate_hand_supports_five_hole_cards(rank_lookup: RankLookup) -> None:
    best_five = encode_board("AsAhAcKdKh")

    assert _evaluate_one("AsAhQsJdTc", "AcKdKh2c3d", rank_lookup) == _reference_rank(
        best_five
    )


def test_evaluate_hand_requires_exactly_two_hole_cards_and_three_board_cards(
    rank_lookup: RankLookup,
) -> None:
    # The board is a spade flush, but there's only one spade in the hole cards.
    # Using exactly two hole cards means this player cannot make a flush.
    best_allowed = encode_board("AsKd6s5s4s")

    assert _evaluate_one("AsKdQhJc", "2s3s4s5s6s", rank_lookup) == _reference_rank(
        best_allowed
    )


def test_evaluate_hand_equal_rank_hands_with_different_suits_tie(
    rank_lookup: RankLookup,
) -> None:
    board = "2c3d5h7s9c"

    first = _evaluate_one("AsAhKdQc", board, rank_lookup)
    second = _evaluate_one("AdAcKsQh", board, rank_lookup)

    assert first == second


def test_evaluate_hand_compares_kickers(rank_lookup: RankLookup) -> None:
    board = "2c3d5h7s9c"

    higher = _evaluate_one("As9hKdQc", board, rank_lookup)
    lower = _evaluate_one("Kd9sJhTh", board, rank_lookup)

    assert higher > lower


@pytest.mark.parametrize(
    ("hand_text", "board_text", "best_five"),
    [
        ("As5h9d8d", "2c3d4hKs9s", "As5h4h3d2c"),  # Wheel (A-2-3-4-5)
        ("AsKd9d8d", "QhJcTc2s3s", "AsKdQhJcTc"),  # Broadway (T-J-Q-K-A)
    ],
)
def test_evaluate_hand_recognizes_wheel_and_ace_high_straights(
    hand_text: str,
    board_text: str,
    best_five: str,
    rank_lookup: RankLookup,
) -> None:
    assert _evaluate_one(hand_text, board_text, rank_lookup) == _reference_rank(
        encode_board(best_five)
    )


@pytest.mark.parametrize(
    ("hand_text", "board_text"),
    [
        ("AsAhKdQc", "2c2d5h7s9c"),  # Paired board
        ("AsKsQdJc", "2s5s8sJh9c"),  # Three spades on board
        ("AsKsQdJc", "2s5s8sTs9c"),  # Four spades on board
    ],
)
def test_evaluate_hand_handles_varied_board_textures(
    hand_text: str,
    board_text: str,
    rank_lookup: RankLookup,
) -> None:
    assert _evaluate_one(hand_text, board_text, rank_lookup) >= 0
