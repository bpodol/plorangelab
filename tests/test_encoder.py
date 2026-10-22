import numpy as np
import pytest

from plorangelab.encoder import encode_board, encode_range


def test_valid_input_range() -> None:
    encoded = encode_range(
        "AsKsQhJhTs@100,AdKdQcJcTs@43,2c3c4c5c6c", 5
    )
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


def test_zero_and_full_weights() -> None:
    encoded = encode_range(
        "AsKsQhJh@0,AdKdQcJc@100", 4
    )

    assert encoded.hands.tolist() == [[51, 47, 42, 38], [49, 45, 40, 36]]
    np.testing.assert_array_equal(encoded.weights, [0.0, 1.0])


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
        encode_range("AsKsQhJh@0,JhQhKsAs@100", 4)


def test_multiple_hands_keep_weights_aligned_with_rows() -> None:
    encoded = encode_range(
        "AsKsQhJh@25,AdKdQcJc@100,2c3c4c5c@0", 4
    )

    assert encoded.hands.tolist() == [
        [51, 47, 42, 38],
        [49, 45, 40, 36],
        [12, 8, 4, 0],
    ]
    np.testing.assert_array_equal(encoded.weights, [0.25, 1.0, 0.0])


