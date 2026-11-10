import numpy as np
import pytest

from plorangelab.encoder import encode_board, encode_input, encode_range
from plorangelab.rating import HandRating


@pytest.fixture
def rating() -> HandRating:
    return HandRating(
        hands=np.arange(40, dtype=np.uint8).reshape(10, 4),
        masks=np.arange(10, dtype=np.uint64),
    )


def test_valid_input_range() -> None:
    encoded = encode_range("AsKsQhJhTs@100,AdKdQcJcTs@43,2c3c4c5c6c", 5)
    assert encoded.hands.dtype == np.uint8
    assert encoded.hands.shape == (3, 5)
    assert encoded.hands.tolist() == [
        [51, 47, 42, 38, 35],
        [49, 45, 40, 36, 35],
        [16, 12, 8, 4, 0],
    ]
    assert encoded.weights.dtype == np.float32
    np.testing.assert_allclose(encoded.weights, [1.0, 0.43, 1.0])

    board = encode_board("AsKdQhJcTs")
    assert board.dtype == np.uint8
    assert board.shape == (5,)
    assert board.tolist() == [51, 45, 42, 36, 35]


def test_valid_plo4_single_hand() -> None:
    encoded = encode_range("AsKsQhJh", 4)

    assert encoded.hands.shape == (1, 4)
    assert encoded.hands.tolist() == [[51, 47, 42, 38]]
    np.testing.assert_array_equal(encoded.weights, [1.0])


@pytest.mark.parametrize(
    ("encoder", "value", "mode"),
    [
        (encode_range, "", 4),  # empty range entry
        (encode_range, ",AsKsQhJh", 4),  # empty entry before comma
        (encode_range, "AsKsQhJh@", 4),  # missing weight
        (encode_range, "AsKsQhJh@abc", 4),  # non-numeric weight
        (encode_range, "AsKsQhJh@1@2", 4),  # multiple weight separators
        (encode_range, "AsKsQhJh@-1", 4),  # negative weight
        (encode_range, "AsKsQhJh@101", 4),  # weight exceeds 100 percent
        (encode_range, "AsKsQhJh@nan", 4),  # non-finite weight
        (encode_range, "AsKcQd", 4),  # too few cards
        (encode_range, "AsKcQdJhTs", 4),  # too many cards
        (encode_range, "AsKcQdJx", 4),  # invalid card
        (encode_range, "AsKcQdAs", 4),  # duplicate card
        (encode_range, "AsKcQdJh", 3),  # mode must be 4 or 5
        (encode_range, "AsKsQhJh,JhQhKsAs", 4),  # duplicate hand
        (encode_board, "", None),  # no board cards
        (encode_board, "AsKd", None),  # too few cards
        (encode_board, "AsKdQhJcTs2d", None),  # too many cards
        (encode_board, "AsKdQx", None),  # invalid card
        (encode_board, "AsKdAs", None),  # duplicate card
    ],
)
def test_invalid_inputs(encoder, value: str, mode: int | None) -> None:
    with pytest.raises(ValueError):
        if mode is None:
            encoder(value)
        else:
            encoder(value, mode)


def test_duplicate_hands_with_different_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate hand in range"):
        encode_range("AsKsQhJh@12,JhQhKsAs@100", 4)


def test_multiple_hands_keep_weights_aligned_with_rows() -> None:
    encoded = encode_range("AsKsQhJh@25,AdKdQcJc@100,2c3c4c5c@50", 4)

    assert encoded.hands.tolist() == [
        [51, 47, 42, 38],
        [49, 45, 40, 36],
        [12, 8, 4, 0],
    ]
    np.testing.assert_array_equal(encoded.weights, [0.25, 1.0, 0.5])


import pytest

from plorangelab.encoder import encode_range


def test_duplicate_hand_different_order():
    with pytest.raises(ValueError, match="duplicate hand"):
        encode_range(
            "AsKdQcJhTs,TsJhQcKdAs",
            mode=5,
        )


def test_duplicate_card_in_hand():
    with pytest.raises(ValueError, match="duplicate cards"):
        encode_range("AsAsQcJhTs", mode=5)


def test_invalid_card():
    with pytest.raises(ValueError, match="invalid card"):
        encode_range("ZsKdQcJhTs", mode=5)


def test_encoded_hand_and_mask():
    result = encode_range("AsKdQcJhTs@75", mode=5)

    np.testing.assert_array_equal(
        result.hands[0],
        np.array([51, 45, 40, 38, 35], dtype=np.uint8),
    )

    assert result.masks[0] == sum(1 << int(card) for card in result.hands[0])
    assert result.weights[0] == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("notation", "start", "end"),
    [
        ("10%", 0, 1),
        ("50%", 0, 5),
        ("100%", 0, 10),
        ("50%-20%", 2, 5),
        ("75%-25%", 2, 7),
        (" 50% ", 0, 5),
    ],
)
def test_valid_percentage_input(
    notation: str,
    start: int,
    end: int,
    rating: HandRating,
) -> None:
    encoded = encode_input(notation, mode=4, rating=rating)

    np.testing.assert_array_equal(
        encoded.hands,
        rating.hands[start:end],
    )
    np.testing.assert_array_equal(
        encoded.masks,
        rating.masks[start:end],
    )
    np.testing.assert_array_equal(
        encoded.weights,
        np.ones(end - start, dtype=np.float32),
    )

    assert encoded.hands.dtype == np.uint8
    assert encoded.masks.dtype == np.uint64
    assert encoded.weights.dtype == np.float32


@pytest.mark.parametrize(
    "notation",
    [
        "0%",
        "101%",
        "-5%",
        "50%-60%",
        "50%-50%",
        "50%-101%",
        "abc%",
        "50%%",
        "50%-",
    ],
)
def test_invalid_percentage_input(
    notation: str,
    rating: HandRating,
) -> None:
    with pytest.raises(ValueError):
        encode_input(notation, mode=4, rating=rating)


def test_encode_input_with_literal_hands(
    rating: HandRating,
) -> None:
    notation = "AsKsQhJh@80,AdKdQcJc@50"

    result = encode_input(notation, mode=4, rating=rating)
    expected = encode_range(notation, mode=4)

    np.testing.assert_array_equal(result.hands, expected.hands)
    np.testing.assert_array_equal(result.masks, expected.masks)
    np.testing.assert_array_equal(result.weights, expected.weights)


def test_percentage_input_shares_ranking_memory(
    rating: HandRating,
) -> None:
    encoded = encode_input("50%", mode=4, rating=rating)

    assert np.shares_memory(encoded.hands, rating.hands)
    assert np.shares_memory(encoded.masks, rating.masks)


def test_percentage_input_does_not_modify_rating(
    rating: HandRating,
) -> None:
    original_hands = rating.hands.copy()
    original_masks = rating.masks.copy()

    encode_input("50%-20%", mode=4, rating=rating)

    np.testing.assert_array_equal(rating.hands, original_hands)
    np.testing.assert_array_equal(rating.masks, original_masks)
