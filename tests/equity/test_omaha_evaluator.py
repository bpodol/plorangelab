"""Omaha evaluation rules: exactly two hole cards, three board cards, and invalid inputs."""

import numpy as np
import pytest

from plorangelab.encoder import encode_board, encode_range
from plorangelab.equity.evaluator import numba_evaluate_hand, numba_rank_combination

# Poker card-string helpers for constructing intentionally invalid inputs

RANKS_ENCODE = {
    "2": 0,
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "T": 8,
    "J": 9,
    "Q": 10,
    "K": 11,
    "A": 12,
}

SUITS_ENCODE = {
    "c": 0,
    "d": 1,
    "h": 2,
    "s": 3,
}

def encode_card(card: str) -> int:
    """Encode a two-character card as an integer from 0 to 51.

    Example: ``encode_card("As")`` returns ``51``.
    """
    rank, suit = card[0], card[1]
    try:
        return 4 * RANKS_ENCODE[rank] + SUITS_ENCODE[suit]
    except KeyError as exc:
        raise ValueError(f"invalid card: {card!r}") from exc


# Omaha hand construction and comparisons

def test_evaluate_hand_uses_exactly_two_hole_and_three_board_cards() -> None:
    hand = encode_range("AsAhQsJd", 4)
    board = encode_board("AcKdKh2c3d")
    best_five = encode_board("AsAhAcKdKh")

    assert numba_evaluate_hand(hand.hands[0], board) == numba_rank_combination(best_five)

def test_evaluate_hand_supports_five_hole_cards() -> None:
    hand = encode_range("AsAhQsJdTc", 5)
    board = encode_board("AcKdKh2c3d")

    assert numba_evaluate_hand(hand.hands[0], board) == numba_rank_combination(encode_board("AsAhAcKdKh"))

def test_evaluate_hand_requires_exactly_two_hole_cards_and_three_board_cards() -> None:
    # The board alone is a spade flush, but Omaha rules require two hole cards;
    # only one hole card is a spade, so the player cannot make a flush.
    hand = encode_range("AsKdQhJc", 4)
    board = encode_board("2s3s4s5s6s")
    best_allowed = encode_board("AsKd6s5s4s")

    assert numba_evaluate_hand(hand.hands[0], board) == numba_rank_combination(best_allowed)

def test_evaluate_hand_equal_rank_hands_with_different_suits_tie() -> None:
    board = encode_board("2c3d5h7s9c")
    first_hand = encode_range("AsAhKdQc", 4)
    second_hand = encode_range("AdAcKsQh", 4)

    assert numba_evaluate_hand(first_hand.hands[0], board) == numba_evaluate_hand(second_hand.hands[0], board)

def test_evaluate_hand_compares_kickers() -> None:
    board = encode_board("2c3d5h7s9c")
    higher_kickers = encode_range("As9hKdQc", 4)
    lower_kickers = encode_range("Kd9sJhTh", 4)

    assert numba_evaluate_hand(higher_kickers.hands[0], board) > numba_evaluate_hand(lower_kickers.hands[0], board)

@pytest.mark.parametrize(
    ("hand_string", "board_string", "best_five"),
    [
        # Wheel: A-2-3-4-5, with exactly two hole cards and three board cards.
        ("As5h9d8d", "2c3d4hKs9s", "As5h4h3d2c"),
        # Broadway: T-J-Q-K-A.
        ("AsKd9d8d", "QhJcTc2s3s", "AsKdQhJcTc"),
    ],
)
def test_evaluate_hand_recognizes_wheel_and_ace_high_straights(
    hand_string: str, board_string: str, best_five: str
) -> None:
    hand = encode_range(hand_string, 4)
    board = encode_board(board_string)

    assert numba_evaluate_hand(hand.hands[0], board) == numba_rank_combination(encode_board(best_five))

@pytest.mark.parametrize(
    ("hand_string", "board_string"),
    [
        ("AsAhKdQc", "2c2d5h7s9c"),  # paired board texture
        ("AsKsQdJc", "2s5s8sJh9c"),  # three board cards of one suit
        ("AsKsQdJc", "2s5s8sTs9c"),  # four board cards of one suit
    ],
)
def test_evaluate_hand_handles_varied_board_textures(
    hand_string: str, board_string: str
) -> None:
    hand = encode_range(hand_string, 4)
    board = encode_board(board_string)

    assert isinstance(numba_evaluate_hand(hand.hands[0], board), int)


# Invalid card sets

@pytest.mark.parametrize(
    ("hand_string", "board_string"),
    [
        ("AsAhKd", "2c3d5h7s9c"),  # too few hole cards
        ("AsAhKdQcJdTd", "2c3d5h7s9c"),  # too many hole cards
        ("AsAhKdQc", "2c3d5h7s"),  # board too short
        ("AsAhKdQc", "2c3d5h7s9cTd"),  # board too long
        ("AsAhKdQc", "2c3d5h7sAs"),  # hand/board overlap
        ("AsAhKdQc", "2c3d5h7h7h"),  # duplicate board card
    ],
)
def test_evaluate_hand_rejects_invalid_card_sets(
    hand_string: str, board_string: str
) -> None:
    hand = np.array(
        [encode_card(hand_string[index : index + 2]) for index in range(0, len(hand_string), 2)],
        dtype=np.uint8,
    )
    board = np.array(
        [encode_card(board_string[index : index + 2]) for index in range(0, len(board_string), 2)],
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        numba_evaluate_hand(hand, board)

def test_evaluate_hand_rejects_out_of_range_card_id() -> None:
    hand = encode_range("AsAhKdQc", 4)
    board = encode_board("2c3d5h7s9c")
    board[0] = np.uint8(52)

    with pytest.raises(ValueError):
        numba_evaluate_hand(hand.hands[0], board)
