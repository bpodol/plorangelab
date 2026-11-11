"""Encode and decode PLO cards, hands, boards, and weighted ranges.

Converts human-readable card notation into compact NumPy arrays
and bitmasks used by the equity and filtering engines.

Supports explicit weighted hand ranges and percentage-based selection
from precomputed starting-hand orderings.
"""

import re
from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

from plorangelab.rating import HandRating
from plorangelab.types import Mode

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
    """Compact representation of a weighted PLO hand range.

    The three arrays are aligned by hand index. Each hand is stored
    in descending card-ID order, and its mask identifies the cards
    contained in that hand.

    Attributes:
        hands: uint8 array of shape (n, mode) containing encoded hands.
        masks: uint64 array of shape (n,) with one card bitmask per hand.
        weights: float32 array of shape (n,) containing positive
            per-hand weights.
    """

    hands: NDArray[np.uint8]
    masks: NDArray[np.uint64]
    weights: NDArray[np.float32]


def build_mask(cards: NDArray[np.uint8]) -> np.uint64:
    """
    Encode a set of cards as a 64-bit bitmask.

    Each card is represented by its ID (0–51), with the corresponding
    bit set to 1. The resulting mask allows efficient card overlap
    detection using bitwise AND.

    Args:
        cards: A 1D NumPy array of encoded card IDs (np.uint8).

    Returns:
        np.uint64: A bitmask representing all cards in the hand.
    """
    mask = 0

    for card in cards:
        mask |= 1 << int(card)

    return np.uint64(mask)


def _encode_card(card: str) -> int:
    """Encode a two-character card notation into an integer ID (0-51).

    Card IDs are calculated as rank * 4 + suit.

    Example:
        _encode_card("As") returns 51.
    """
    rank, suit = card[0], card[1]
    try:
        return 4 * RANKS_ENCODE[rank] + SUITS_ENCODE[suit]
    except KeyError as exc:
        raise ValueError(f"invalid card: {card!r}") from exc


def encode_board(board: str) -> NDArray[np.uint8]:
    """Encode a flop, turn, or river board into card IDs.

    Preserves the order of the supplied cards and rejects duplicates.

    Args:
        board: Concatenated two-character card notations containing
            exactly 3, 4, or 5 cards.

    Returns:
        One-dimensional uint8 array of encoded community cards.

    Raises:
        ValueError: If the board length, card notation, or card
            uniqueness is invalid.
    """
    if len(board) not in (6, 8, 10):
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
    if (
        not isinstance(card_id, int)
        or isinstance(card_id, bool)
        or not 0 <= card_id < 52
    ):
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
        raise ValueError(
            f"hand must be a one-dimensional array of exactly {mode} card IDs"
        )

    return "".join(decode_card(int(card_id)) for card_id in hand)


@njit(cache=True)
def _encode_hands_kernel(
    packed_chars: NDArray[np.uint8],
    mode: int,
) -> tuple[NDArray[np.uint8], NDArray[np.uint64], int, int]:
    """Encode packed ASCII card notation into hand arrays and masks.

    Processes each hand in a Numba-compiled loop, validates card IDs
    and uniqueness, and sorts encoded cards in descending order.

    A hand's bitmask also serves as its order-independent identity,
    allowing duplicate combinations to be detected.

    Args:
        packed_chars: uint8 array of shape (n, mode * 2) containing
            ASCII-encoded card notation.
        mode: Number of hole cards per hand (4 or 5).

    Returns:
        A tuple containing encoded hands, card masks, a status code,
        and the index of the first invalid hand (-1 on success).

    Status codes:
        0: Success.
        1: Invalid card notation.
        2: Duplicate card within a hand.
        3: Duplicate hand within the range.

    Notes:
        On failure, the output arrays may be partially initialized
        and must not be used.
    """
    n = packed_chars.shape[0]

    encoded_hands = np.empty((n, mode), dtype=np.uint8)
    masks = np.empty(n, dtype=np.uint64)

    seen = set()

    for i in range(n):
        mask = np.uint64(0)

        for j in range(mode):
            rank_char = packed_chars[i, 2 * j]
            suit_char = packed_chars[i, 2 * j + 1]

            # ASCII rank -> integer 0-12
            if 50 <= rank_char <= 57:  # 2-9
                rank = int(rank_char) - 50
            elif rank_char == 84:  # T
                rank = 8
            elif rank_char == 74:  # J
                rank = 9
            elif rank_char == 81:  # Q
                rank = 10
            elif rank_char == 75:  # K
                rank = 11
            elif rank_char == 65:  # A
                rank = 12
            else:
                return encoded_hands, masks, 1, i

            # ASCII suit -> integer 0-3
            if suit_char == 99:  # c
                suit = 0
            elif suit_char == 100:  # d
                suit = 1
            elif suit_char == 104:  # h
                suit = 2
            elif suit_char == 115:  # s
                suit = 3
            else:
                return encoded_hands, masks, 1, i

            card_id = 4 * rank + suit
            bit = np.uint64(1) << np.uint64(card_id)

            # Detect duplicate cards within one hand.
            if mask & bit:
                return encoded_hands, masks, 2, i

            mask |= bit
            encoded_hands[i, j] = card_id

        # The mask uniquely identifies a valid hand,
        # regardless of the order of its cards.
        if mask in seen:
            return encoded_hands, masks, 3, i

        seen.add(mask)
        masks[i] = mask

        # Insertion sort, descending. Only 4 or 5 cards.
        for j in range(1, mode):
            value = encoded_hands[i, j]
            k = j

            while k > 0 and encoded_hands[i, k - 1] < value:
                encoded_hands[i, k] = encoded_hands[i, k - 1]
                k -= 1

            encoded_hands[i, k] = value

    return encoded_hands, masks, 0, -1


def _encode_hands(
    hand_strings: list[str],
    weight_values: list[float],
    mode: Mode,
) -> EncodedRange:
    """Encode validated-length hand strings and their associated weights.

    Packs all hand strings into a contiguous ASCII byte array and
    delegates card encoding and duplicate detection to the Numba kernel.

    Converts kernel status codes into Python exceptions and returns
    the encoded arrays with float32 weights.

    Args:
        hand_strings: List of concatenated card strings.
        weight_values: Per-hand weights aligned with hand_strings.
        mode: Number of hole cards per hand (4 or 5).

    Returns:
        EncodedRange containing sorted hands, card masks, and weights.

    Raises:
        ValueError: If a hand contains invalid or duplicate cards,
            a duplicate combination is found, or the input contains
            non-ASCII characters.
    """

    try:
        packed_chars = np.frombuffer(
            "".join(hand_strings).encode("ascii"),
            dtype=np.uint8,
        ).reshape(len(hand_strings), mode * 2)

    except UnicodeEncodeError as exc:
        raise ValueError("Hand contains a non-ASCII card character") from exc

    encoded_hands, masks, status, error_index = _encode_hands_kernel(packed_chars, mode)

    if status == 1:
        raise ValueError(f"invalid card in hand: {hand_strings[error_index]!r}")

    if status == 2:
        raise ValueError(
            f"hand cannot contain duplicate cards: {hand_strings[error_index]!r}"
        )

    if status == 3:
        raise ValueError(f"duplicate hand in range: {hand_strings[error_index]!r}")

    weights = np.asarray(weight_values, dtype=np.float32)

    return EncodedRange(
        hands=encoded_hands,
        masks=masks,
        weights=weights,
    )


def _encode_percentage(notation: str, hands_rating: HandRating):
    """Select a percentage interval from a preordered PLO hand range.

    Percentage notation selects a contiguous slice of the supplied
    hand ordering, which is assumed to run from strongest to weakest.

    For example, "50%" selects the first 50% of hands, while
    "50%-7%" selects hands between the 7% and 50% boundaries.

    Boundaries are converted to integer indices by truncation.
    All selected combinations receive a weight of 1.0.

    Args:
        notation: Percentage range expression.
        hands_rating: Precomputed ordered hands and corresponding masks.

    Returns:
        EncodedRange containing the selected hands with uniform weights.

    Raises:
        ValueError: If the notation or percentage boundaries are
            invalid, or the selected interval contains no hands.
    """
    pattern = r"(\d+(?:\.\d+)?)%(?:-(\d+(?:\.\d+)?)%)?"

    match = re.fullmatch(pattern, notation.strip())

    if match is None:
        raise ValueError(f"Invalid percentage notation: {notation!r}")

    upper = float(match.group(1))
    lower = float(match.group(2)) if match.group(2) else 0.0

    if not 0 <= lower < upper <= 100:
        raise ValueError("Percentages must satisfy 0 <= lower < upper <= 100")
    # Truncate percentage boundaries to whole-hand indices rather
    # than including fractional combinations.
    start = int(len(hands_rating.hands) * lower / 100)
    end = int(len(hands_rating.hands) * upper / 100)

    if start == end:
        raise ValueError("Percentage range contains no hands")

    return EncodedRange(
        hands=hands_rating.hands[start:end],
        masks=hands_rating.masks[start:end],
        weights=np.ones(end - start, dtype=np.float32),
    )


def encode_range(hands: str, mode: Mode) -> EncodedRange:
    """Parse explicit PLO hand combinations with optional weights.

    Accepts comma-separated hands containing 4 or 5 cards, depending
    on the selected mode. Each hand may include an @weight suffix
    representing a percentage.

    Weights are stored as fractions: @50 becomes 0.5, and omitted
    weights default to 1.0.

    Args:
        hands: Comma-separated hand combinations with optional
            percentage weights, e.g. "AsKsQhJh@100,AdKdQcJc@50".
        mode: Number of hole cards per hand (4 or 5).

    Returns:
        EncodedRange containing sorted hands, masks, and float32 weights.

    Raises:
        ValueError: If the mode, hand notation, weights, or card
            combinations are invalid.
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
                weight = float(parts[1]) / 100
            except ValueError as exc:
                raise ValueError(f"invalid weight in range entry: {entry!r}") from exc
            if not np.isfinite(weight) or not 0 < weight <= 1:
                raise ValueError(f"weight must be in range 1-100: {entry!r}")

        hand_strings.append(hand)
        weight_values.append(weight)

    encoded_range = _encode_hands(hand_strings, weight_values, mode)

    return encoded_range


def encode_input(input_str: str, mode: Mode, rating: HandRating) -> EncodedRange:
    """Encode either percentage-based or explicit PLO range notation.

    Inputs containing a percent sign are interpreted as percentage
    ranges and resolved using the supplied hand ordering. Other
    inputs are parsed as explicit weighted combinations.

    Args:
        input_str: Percentage notation or explicit hand combinations.
        mode: Number of hole cards per hand (4 or 5).
        rating: Precomputed hand ordering used for percentage selection.

    Returns:
        EncodedRange containing the selected hands, masks, and weights.

    Raises:
        ValueError: If the supplied range notation is invalid.
    """
    input_str = input_str.strip()

    if "%" in input_str:
        return _encode_percentage(input_str, rating)

    return encode_range(input_str, mode)
