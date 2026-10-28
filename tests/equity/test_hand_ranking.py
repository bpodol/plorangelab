"""Five-card hand rankings: within-category ordering, category strength, and ties."""

import pytest

from plorangelab.encoder import encode_board
from plorangelab.equity.evaluator import numba_rank_combination

# Representative five-card hands: strongest, weaker, then a lower-kicker hand.
# Each test chooses the relevant comparisons for its category.

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


# Category and tie-breaking behavior

@pytest.mark.parametrize("category", COMBINATIONS)
def test_rank_combination_compares_hands_within_category(category: str) -> None:
    combinations = COMBINATIONS[category]
    stronger = numba_rank_combination(encode_board(combinations[0]))
    weaker = numba_rank_combination(encode_board(combinations[1]))

    assert stronger > weaker

    if len(combinations) == 3 and category != "straight":
        lower_kicker = numba_rank_combination(encode_board(combinations[2]))
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
    stronger = numba_rank_combination(encode_board(COMBINATIONS[stronger_category][0]))
    weaker = numba_rank_combination(encode_board(COMBINATIONS[weaker_category][0]))

    assert stronger > weaker

def test_wheel_is_lower_than_six_high_straight():
    wheel = encode_board("As5h4d3c2s")
    six_high = encode_board("6s5h4d3c2s")

    assert numba_rank_combination(wheel) < numba_rank_combination(six_high)

def test_wheel_strflush_is_lower_than_six_high_straight():
    wheel = encode_board("As5s4s3s2s")
    six_high = encode_board("6s5s4s3s2s")

    assert numba_rank_combination(wheel) < numba_rank_combination(six_high)

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
    assert numba_rank_combination(encode_board(first)) == numba_rank_combination(
        encode_board(same_rank_different_suits)
    )
