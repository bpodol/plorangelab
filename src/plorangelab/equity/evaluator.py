"""PLO4/PLO5 hand evaluation using precomputed five-card ranking tables.

Provides Numba-compiled evaluation of multiple Omaha hands against a
complete board, using exactly two hole cards and three community cards.
"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from plorangelab.rank_combination import rank_combination_lookup


@njit(cache=True)
def _evaluate_hand_core(
    hand: NDArray[np.uint8],
    board: NDArray[np.uint8],
    rank_table: NDArray[np.uint32],
    binomial_table: NDArray[np.uint32],
) -> int:
    """Return the best Omaha hand rank without performing input validation.

    Assumes valid encoded cards, a complete five-card board, and compatible
    ranking tables. Intended for repeated calls from Numba-compiled kernels.
    """
    best_score = -1
    cards = np.empty(5, dtype=np.uint8)

    # Enumerate C(4, 2) * C(5, 3) = 60 combinations for PLO4,
    # or C(5, 2) * C(5, 3) = 100 combinations for PLO5.
    for i in range(len(hand) - 1):
        for j in range(i + 1, len(hand)):
            for a in range(3):
                for b in range(a + 1, 4):
                    for c in range(b + 1, 5):
                        cards[0] = hand[i]
                        cards[1] = hand[j]
                        cards[2] = board[a]
                        cards[3] = board[b]
                        cards[4] = board[c]

                        sort5_desc(cards)
                        score = rank_combination_lookup(
                            cards, rank_table, binomial_table
                        )

                        best_score = max(best_score, score)

    return best_score


@njit(inline="always")
def sort5_desc(cards: NDArray[np.uint8]) -> None:
    """Sort five card IDs in descending order using a fixed sorting network.

    Operates in place using a predetermined sequence of comparisons and swaps,
    without allocating an additional array.
    """

    if cards[0] < cards[1]:
        cards[0], cards[1] = cards[1], cards[0]

    if cards[3] < cards[4]:
        cards[3], cards[4] = cards[4], cards[3]

    if cards[2] < cards[4]:
        cards[2], cards[4] = cards[4], cards[2]

    if cards[2] < cards[3]:
        cards[2], cards[3] = cards[3], cards[2]

    if cards[1] < cards[4]:
        cards[1], cards[4] = cards[4], cards[1]

    if cards[0] < cards[3]:
        cards[0], cards[3] = cards[3], cards[0]

    if cards[0] < cards[2]:
        cards[0], cards[2] = cards[2], cards[0]

    if cards[1] < cards[3]:
        cards[1], cards[3] = cards[3], cards[1]

    if cards[1] < cards[2]:
        cards[1], cards[2] = cards[2], cards[1]


@njit(cache=True)
def numba_evaluate_hands(
    hands: NDArray[np.uint8],
    board: NDArray[np.uint8],
    rank_table: NDArray[np.uint32],
    binomial_table: NDArray[np.uint32],
) -> NDArray[np.uint32]:
    """Evaluate multiple Omaha hands against the same board sequentially.

    Args:
        hands: Two-dimensional uint8 array of shape (n, 4) or (n, 5).
        board: Complete five-card community board.
        rank_table: Precomputed five-card hand ranking table.
        binomial_table: Precomputed table used for combination indexing.

    Returns:
        One uint32 hand ranking per input hand, preserving input order.

    Notes:
        Inputs are assumed valid and are not checked for card collisions.
    """
    evaluations = np.empty(hands.shape[0], dtype=np.uint32)
    for i in range(hands.shape[0]):
        evaluations[i] = _evaluate_hand_core(
            hands[i], board, rank_table, binomial_table
        )

    return evaluations
