import numpy as np
from numpy.typing import NDArray
from typing import Literal

Mode = Literal[4, 5]

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

RANKS_DECODE = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
SUITS_DECODE = ("c", "d", "h", "s")

def encode_card(card: str) -> int:
    """Encode a two-character card as an integer from 0 to 51.

    Example: ``encode_card("As")`` returns ``51``.
    """
    if len(card) != 2:
        raise ValueError("card must be a two-character rank and suit, such as 'As'")

    rank, suit = card[0], card[1]
    try:
        return 4 * RANKS_ENCODE[rank] + SUITS_ENCODE[suit]
    except KeyError as exc:
        raise ValueError(f"invalid card: {card!r}") from exc

def decode_card(card_id: int) -> str:
    """Decode an integer from 0 to 51 into its two-character card representation.

    Example: ``decode_card(51)`` returns ``"As"``.
    """
    if not isinstance(card_id, int) or isinstance(card_id, bool) or not 0 <= card_id < 52:
        raise ValueError("card_id must be an integer from 0 to 51")

    rank, suit = divmod(card_id, 4)
    return RANKS_DECODE[rank] + SUITS_DECODE[suit]

def encode_hand(hand: str, mode: Mode) -> NDArray[np.uint8]:
    """Encode a concatenated 4- or 5-card hand as a uint8 array of card IDs.

    Example: ``encode_hand("AsKcQdJh", 4)`` returns ``[51, 44, 41, 38]``.
    """
    if mode not in (4, 5):
        raise ValueError("mode must be 4 or 5")
    if len(hand) != mode * 2:
        raise ValueError(f"hand must contain exactly {mode} two-character cards")

    cards = [hand[index : index + 2] for index in range(0, len(hand), 2)]
    encoded_cards = [encode_card(card) for card in cards]
    if len(set(encoded_cards)) != mode:
        raise ValueError("hand cannot contain duplicate cards")

    encoded_cards.sort(reverse=True)
    return np.array(encoded_cards, dtype=np.uint8)


def decode_hand(hand: NDArray[np.uint8], mode: Mode) -> str:
    """Decode a 4- or 5-card array of card IDs into a concatenated card string.

    Example: ``decode_hand(np.array([51, 44, 41, 38], dtype=np.uint8), 4)``
    returns ``"AsKcQdJh"``.
    """
    if mode not in (4, 5):
        raise ValueError("mode must be 4 or 5")
    if hand.ndim != 1 or len(hand) != mode:
        raise ValueError(f"hand must be a one-dimensional array of exactly {mode} card IDs")

    return "".join(decode_card(int(card_id)) for card_id in hand)

def encode_board(board: str) -> NDArray[np.uint8]:
    if len(board) not in (6, 8, 10):  # for flop: 3*2, turn: 4*2, and river: 5*2
            raise ValueError("board must contain 3, 4, or 5 two-character cards")
    cards = [board[index : index + 2] for index in range(0, len(board), 2)]
    encoded_cards = [encode_card(card) for card in cards]

    if len(set(encoded_cards)) != len(encoded_cards):
        raise ValueError("board cannot contain duplicate cards")

    return np.array(encoded_cards, dtype=np.uint8)