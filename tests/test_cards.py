import pytest

from plorangelab.cards import (
    decode_card,
    encode_board,
    encode_card,
    encode_hand,
    decode_hand,
)
import numpy as np

RANKS = "23456789TJQKA"
SUITS = "cdhs"


@pytest.mark.parametrize(
    ("card", "expected_id"),
    [
        (rank + suit, 4 * rank_index + suit_index)
        for rank_index, rank in enumerate(RANKS)
        for suit_index, suit in enumerate(SUITS)
    ],
)
def test_encode_card(card: str, expected_id: int) -> None:
    assert encode_card(card) == expected_id


@pytest.mark.parametrize("card_id", range(52))
def test_decode_card(card_id: int) -> None:
    expected_card = RANKS[card_id // 4] + SUITS[card_id % 4]
    assert decode_card(card_id) == expected_card


@pytest.mark.parametrize("card", ["", "A", "Asx", "1s", "Ax"])
def test_encode_card_rejects_invalid_cards(card: str) -> None:
    with pytest.raises(ValueError):
        encode_card(card)


@pytest.mark.parametrize("card_id", [-1, 52, 1.5, True, "1"])
def test_decode_card_rejects_invalid_ids(card_id: object) -> None:
    with pytest.raises(ValueError):
        decode_card(card_id)  

@pytest.mark.parametrize("card_id", range(52))
def test_round_trip(card_id: int) -> None:
    assert encode_card(decode_card(card_id)) == card_id

@pytest.mark.parametrize(
    ("hand", "mode", "expected_ids"),
    [
        ("AsKcQdJh", 4, [51, 44, 41, 38]),
        ("AsKcQdJhTs", 5, [51, 44, 41, 38, 35]),
    ],
)
def test_encode_hand(hand: str, mode: int, expected_ids: list[int]) -> None:
    encoded = encode_hand(hand, mode)  

    assert encoded.dtype == np.uint8
    assert encoded.tolist() == expected_ids


@pytest.mark.parametrize(
    ("hand_ids", "mode", "expected_hand"),
    [
        ([51, 44, 41, 38], 4, "AsKcQdJh"),
        ([51, 44, 41, 38, 35], 5, "AsKcQdJhTs"),
    ],
)
def test_decode_hand(hand_ids: list[int], mode: int, expected_hand: str) -> None:
    decoded = decode_hand(np.array(hand_ids, dtype=np.uint8), mode)  

    assert decoded == expected_hand


@pytest.mark.parametrize(
    ("hand", "mode"),
    [
        ("AsKcQd", 4),  # too few cards
        ("AsKcQdJhTs", 4),  # too many cards
        ("AsKcQdJx", 4),  # invalid card
        ("AsKcQdAs", 4),  # duplicate card
    ],
)
def test_encode_hand_rejects_invalid_hands(hand: str, mode: int) -> None:
    with pytest.raises(ValueError):
        encode_hand(hand, mode)  


def test_encode_hand_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError):
        encode_hand("AsKcQdJh", 3)  


@pytest.mark.parametrize(
    ("hand_ids", "mode"),
    [
        ([51, 44, 40], 4),  # too few card IDs
        ([51, 44, 40, 35, 34], 4),  # too many card IDs
    ],
)
def test_decode_hand_rejects_wrong_number_of_cards(hand_ids: list[int], mode: int) -> None:
    with pytest.raises(ValueError):
        decode_hand(np.array(hand_ids, dtype=np.uint8), mode)  


def test_decode_hand_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError):
        decode_hand(np.array([51, 44, 40, 35], dtype=np.uint8), 3)  


def test_decode_hand_rejects_out_of_range_card_id() -> None:
    with pytest.raises(ValueError):
        decode_hand(np.array([51, 44, 40, 52], dtype=np.uint8), 4)


@pytest.mark.parametrize(
    ("board", "expected_ids"),
    [
        ("AsKdQh", [51, 45, 42]),
        ("AsKdQhJc", [51, 45, 42, 36]),
        ("AsKdQhJcTs", [51, 45, 42, 36, 35]),
    ],
)
def test_encode_board(board: str, expected_ids: list[int]) -> None:
    encoded = encode_board(board)

    assert encoded.dtype == np.uint8
    assert encoded.shape == (len(expected_ids),)
    assert encoded.tolist() == expected_ids


@pytest.mark.parametrize(
    "board",
    ["", "AsKd", "AsKdQhJcTs2d", "AsKdQx", "AsKdAs"],
)
def test_encode_board_rejects_invalid_boards(board: str) -> None:
    with pytest.raises(ValueError):
        encode_board(board)

