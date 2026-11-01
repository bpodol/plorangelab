"""Small shared equity test data builders (not production fixtures)."""

import numpy as np

from plorangelab.encoder import EncodedRange, build_mask, encode_board, encode_range

RIVER = "2c3d5h7s9c"
TURN = "2c3d5h7s"


def _make_range_from_ids(card_rows: list[tuple[int, ...]], mode: int) -> EncodedRange:
    hands = np.asarray(card_rows, dtype=np.uint8).reshape((-1, mode))
    masks = np.asarray([build_mask(hand) for hand in hands], dtype=np.uint64)
    weights = np.ones(len(hands), dtype=np.float32)
    return EncodedRange(hands=hands, masks=masks, weights=weights)


def _configured_ranges(
    player_count: int, mode: int, multi_hand: bool
) -> tuple[list[EncodedRange], np.ndarray]:
    board = encode_board(RIVER)
    blocked = {int(card) for card in board}
    available = [card for card in range(52) if card not in blocked]
    cards_per_player = mode + 1 if multi_hand else mode
    players = []

    for player_index in range(player_count):
        start = player_index * cards_per_player
        group = available[start : start + cards_per_player]
        hands = [tuple(group[:mode])]
        if multi_hand:
            hands.append(tuple(group[1 : mode + 1]))
        players.append(_make_range_from_ids(hands, mode))

    return players, board


def _guaranteed_split_players() -> tuple[list[EncodedRange], np.ndarray]:
    board = encode_board("3d4h5s9cTc")
    players = [
        encode_range("6c7cAsAh", 4),
        encode_range("6d7dKsKh", 4),
    ]
    return players, board


def _weighted_players(mode: int) -> list[EncodedRange]:
    """Three small, overlapping weighted ranges for PLO4 or PLO5."""
    if mode == 4:
        texts = (
            "AsAhKdQc@80,QsQhJdTd@20",
            "KsKhJcTc@35,AsQsJcTc@65",
            "AdAcTsTh@60,KdKh8c8d@40",
        )
    else:
        texts = (
            "AsAhKdQc6c@80,QsQhJdTd6d@20",
            "KsKhJcTc8s@35,AsQsJcTc8h@65",
            "AdAcTsTh4s@60,KdKh8c8d4d@40",
        )

    return [encode_range(text, mode) for text in texts]
