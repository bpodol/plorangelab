from plorangelab.cards import encode_hand, Mode
import numpy as np
from numpy.typing import NDArray

def encode_range(
    hands: str, mode: Mode
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Encode comma-separated hands and optional weights.

    Each entry is a hand optionally followed by ``@weight``. Missing weights
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
        if len(parts) == 1:
            weight = 1.0
        else:
            try:
                weight = float(parts[1])/100
            except ValueError as exc:
                raise ValueError(f"invalid weight in range entry: {entry!r}") from exc
            if not np.isfinite(weight) or not 0 <= weight <= 1:
                raise ValueError(f"weight must be in range 1-100: {entry!r}")

        hand_strings.append(hand)
        weight_values.append(weight)

    encoded_hands = np.stack([encode_hand(hand, mode) for hand in hand_strings])
    weights = np.asarray(weight_values, dtype=np.float32)
    return encoded_hands, weights
