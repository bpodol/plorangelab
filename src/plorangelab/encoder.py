import numpy as np
from numpy.typing import NDArray
from typing import Literal
from dataclasses import dataclass

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

@dataclass
class EncodedRange:
    """Encoded hands, masks, and strictly positive per-hand weights."""

    hands: NDArray[np.uint8]
    masks: NDArray[np.uint64]
    weights: NDArray[np.float32]

def build_mask(cards: np.uint8) -> np.uint64:
    """
    Encode a set of cards as a 64-bit bitmask.

    Each card is represented by its ID (0–51), with the corresponding
    bit set to 1. The resulting mask allows efficient card overlap
    detection using bitwise AND.

    Args:
        hand: A 1D NumPy array of card IDs (np.uint8).

    Returns:
        np.uint64: A bitmask representing all cards in the hand.
    """
    mask = 0

    for card in cards:
        mask |= 1 << int(card)

    return np.uint64(mask)

def _encode_card(card: str) -> int:
    """Encode a two-character card as an integer from 0 to 51.

    Example: ``encode_card("As")`` returns ``51``.
    """
    rank, suit = card[0], card[1]
    try:
        return 4 * RANKS_ENCODE[rank] + SUITS_ENCODE[suit]
    except KeyError as exc:
        raise ValueError(f"invalid card: {card!r}") from exc

def _encode_hand(hand: str, mode: Mode) -> list[int]:
    """Encode a concatenated 4- or 5-card hand as a list of card IDs.

    Example: ``encode_hand("AsKcQdJh", 4)`` returns ``[51, 44, 41, 38]``.
    """
    cards = [hand[index : index + 2] for index in range(0, len(hand), 2)]
    encoded_cards = [_encode_card(card) for card in cards]
    if len(set(encoded_cards)) != mode:
        raise ValueError("hand cannot contain duplicate cards")

    encoded_cards.sort(reverse=True)
    return encoded_cards

def encode_board(board: str) -> NDArray[np.uint8]:
    if len(board) not in (6, 8, 10):  # for flop: 3*2, turn: 4*2, and river: 5*2
            raise ValueError("board must contain 3, 4, or 5 two-character cards")
    cards = [board[index : index + 2] for index in range(0, len(board), 2)]
    encoded_cards = [_encode_card(card) for card in cards]

    if len(set(encoded_cards)) != len(encoded_cards):
        raise ValueError("board cannot contain duplicate cards")

    return np.array(encoded_cards, dtype=np.uint8)


def decode_card(card_id: int) -> str:
    """Decode an integer from 0 to 51 into its two-character card representation.

    Example: ``decode_card(51)`` returns ``"As"``.
    """
    if not isinstance(card_id, int) or isinstance(card_id, bool) or not 0 <= card_id < 52:
        raise ValueError("card_id must be an integer from 0 to 51")

    rank, suit = divmod(card_id, 4)
    return RANKS_DECODE[rank] + SUITS_DECODE[suit]

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


def encode_range(
    hands: str, mode: Mode
) -> EncodedRange:
    """Encode comma-separated hands and optional weights.

    Each entry is a hand optionally followed by ``@weight`` in range @1-@100. Missing weights
    default to 1.0. Returns the encoded hands and a parallel weight array.

    Example: ``encode_range("AsKsQhJhTh@100,AdKdQcJcTc@43", 5)`` returns
    ``(hands, weights)`` with shapes ``(2, 5)`` and ``(2,)``.
    """
    if mode not in (4, 5):
        raise ValueError("mode must be 4 or 5")

    hand_strings: list[str] = []
    weight_values: list[float] = []
    for entry in hands.split(","):
        parts = entry.split("@")
        if len(parts) > 2 or not parts[0]:
            raise ValueError(f"invalid range entry: {entry!r}")

        hand = parts[0]
        if len(hand) != mode * 2:
            raise ValueError(f"hand must contain exactly {mode} two-character cards")
        if len(parts) == 1:
            weight = 1.0
        else:
            try:
                weight = float(parts[1])/100
            except ValueError as exc:
                raise ValueError(f"invalid weight in range entry: {entry!r}") from exc
            if not np.isfinite(weight) or not 0 < weight <= 1:
                raise ValueError(f"weight must be in range 1-100: {entry!r}")

        hand_strings.append(hand)
        weight_values.append(weight)
    encoded_hands = np.empty((len(hand_strings), mode), dtype=np.uint8)   
    masks = np.empty((len(hand_strings)), dtype=np.uint64)  
    seen = set()
    for i, hand in enumerate(hand_strings):
        encoded_hand = _encode_hand(hand, mode)
        key = tuple(encoded_hand)
        if key in seen:
            raise ValueError(f"duplicate hand in range: {hand!r}")
        seen.add(key)
        encoded_hands[i] = encoded_hand
        masks[i] = build_mask(encoded_hand)
    weights = np.asarray(weight_values, dtype=np.float32)
    return EncodedRange(hands=encoded_hands, masks=masks, weights=weights)
