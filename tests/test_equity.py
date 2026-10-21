import pytest
import numpy as np

from plorangelab.cards import encode_board, encode_card, encode_hand
from plorangelab.equity import evaluate_hand, rank_combination


# Each entry is ordered strongest, weaker, then (where applicable) the
# strongest made hand with a lower kicker. Card order is descending by ID.
COMBINATIONS = {
    "high_card": (
        "AsKdJh9c7s",
        "KsQdJh9c7s",
        "AsQdJh9c7s",
    ),
    "one_pair": (
        "AsAhKd9c7s",
        "KsKhQd9c7s",
        "AsAhQd9c7s",
    ),
    "two_pair": (
        "AsAhKdKc9s",
        "QsQhJdJc9s",
        "AsAhKdKc7s",
    ),
    "three_of_a_kind": (
        "AsAhAcKd9s",
        "KsKhKcQd9s",
        "AsAhAcQd7s",
    ),
    "straight": (
        "AsKsQcJcTc",
        "KsQcJcTh9s",
        "AhKsQcJsTh",
    ),
    "flush": (
        "AsKs9s7s3s",
        "QsJs8s6s2s",
        "AsQs9s7s3s",
    ),
    "full_house": (
        "AsAhAcKsKh",
        "QsQhQcJsJh",
        "AsAhAcQsQh",
    ),
    "four_of_a_kind": (
        "AsAhAcAdKs",
        "KsKhKcKdQs",
    ),
    "straight_flush": (
        "AsKsQsJsTs",
        "9s8s7s6s5s",
        "9h8h7h6h5h",
    ),
}


@pytest.mark.parametrize("category", COMBINATIONS)
def test_rank_combination_compares_hands_within_category(category: str) -> None:
    combinations = COMBINATIONS[category]
    stronger = rank_combination(encode_board(combinations[0]))
    weaker = rank_combination(encode_board(combinations[1]))

    assert stronger > weaker

    if len(combinations) == 3 and category != "straight":
        lower_kicker = rank_combination(encode_board(combinations[2]))
        assert stronger > lower_kicker


@pytest.mark.parametrize(
    ("stronger_category", "weaker_category"),
    [
        ("one_pair", "high_card"),
        ("two_pair", "one_pair"),
        ("three_of_a_kind", "two_pair"),
        ("straight", "three_of_a_kind"),
        ("flush", "straight"),
        ("full_house", "flush"),
        ("four_of_a_kind", "full_house"),
        ("straight_flush", "four_of_a_kind"),
    ],
)
def test_rank_combination_compares_category_strength(
    stronger_category: str, weaker_category: str
) -> None:
    stronger = rank_combination(encode_board(COMBINATIONS[stronger_category][0]))
    weaker = rank_combination(encode_board(COMBINATIONS[weaker_category][0]))

    assert stronger > weaker

def test_wheel_is_lower_than_six_high_straight():
    wheel = encode_board("As5h4d3c2s")
    six_high = encode_board("6s5h4d3c2s")

    assert rank_combination(wheel) < rank_combination(six_high)

def test_wheel_strflush_is_lower_than_six_high_straight():
    wheel = encode_board("As5s4s3s2s")
    six_high = encode_board("6s5s4s3s2s")

    assert rank_combination(wheel) < rank_combination(six_high)

@pytest.mark.parametrize(
    ("first", "same_rank_different_suits"),
    [
        ("AsAhAcKsKh", "AdAcAsKdKh"),
        ("AsAhKdKc9s", "AdAcKhKs9c"),
        ("AsAhAcKd9s", "AdAcAsKh9c"),
        ("AsAhAcAdKs", "AdAcAsAhKh"),
    ],
)
def test_rank_combination_equal_for_same_ranks_different_suits(
    first: str, same_rank_different_suits: str
) -> None:
    assert rank_combination(encode_board(first)) == rank_combination(
        encode_board(same_rank_different_suits)
    )


def test_evaluate_hand_uses_exactly_two_hole_and_three_board_cards() -> None:
    hand = encode_hand("AsAhQsJd", 4)
    board = encode_board("AcKdKh2c3d")
    best_five = encode_board("AsAhAcKdKh")

    assert evaluate_hand(hand, board) == rank_combination(best_five)


def test_evaluate_hand_supports_five_hole_cards() -> None:
    hand = encode_hand("AsAhQsJdTc", 5)
    board = encode_board("AcKdKh2c3d")

    assert evaluate_hand(hand, board) == rank_combination(encode_board("AsAhAcKdKh"))


def test_evaluate_hand_requires_exactly_two_hole_cards_and_three_board_cards() -> None:
    # The board alone is a spade flush, but Omaha rules require two hole cards;
    # only one hole card is a spade, so the player cannot make a flush.
    hand = encode_hand("AsKdQhJc", 4)
    board = encode_board("2s3s4s5s6s")
    best_allowed = encode_board("AsKd6s5s4s")

    assert evaluate_hand(hand, board) == rank_combination(best_allowed)


def test_evaluate_hand_equal_rank_hands_with_different_suits_tie() -> None:
    board = encode_board("2c3d5h7s9c")
    first_hand = encode_hand("AsAhKdQc", 4)
    second_hand = encode_hand("AdAcKsQh", 4)

    assert evaluate_hand(first_hand, board) == evaluate_hand(second_hand, board)


def test_evaluate_hand_compares_kickers() -> None:
    board = encode_board("2c3d5h7s9c")
    higher_kickers = encode_hand("As9hKdQc", 4)
    lower_kickers = encode_hand("Kd9sJhTh", 4)

    assert evaluate_hand(higher_kickers, board) > evaluate_hand(lower_kickers, board)


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
    hand = encode_hand(hand_string, 4)
    board = encode_board(board_string)

    assert evaluate_hand(hand, board) == rank_combination(encode_board(best_five))


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
    hand = encode_hand(hand_string, 4)
    board = encode_board(board_string)

    assert isinstance(evaluate_hand(hand, board), int)


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
        evaluate_hand(hand, board)


def test_evaluate_hand_rejects_out_of_range_card_id() -> None:
    hand = encode_hand("AsAhKdQc", 4)
    board = encode_board("2c3d5h7s9c")
    board[0] = np.uint8(52)

    with pytest.raises(ValueError):
        evaluate_hand(hand, board)

