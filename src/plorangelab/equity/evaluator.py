"""PLO4/PLO5 hand evaluation using precomputed five-card ranking tables.

Supports validated single-hand evaluation and Numba-compiled batch
evaluation, including parallel execution.
"""
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from plorangelab.rank_combination import RankLookup, rank_combination_lookup


def evaluate_hand(
    hand: NDArray[np.uint8], board: NDArray[np.uint8], rank_lookup: RankLookup
) -> int:
    """Evaluate the strongest Omaha hand using exactly two hole cards.

    Validates the hand and board before calling the Numba-compiled evaluator.
    The best ranking is determined from all combinations of exactly two hole
    cards and three community cards.

    Args:
        hand: One-dimensional array containing 4 or 5 encoded hole cards.
        board: One-dimensional array containing exactly 5 community cards.
        rank_lookup: Precomputed tables used to rank five-card combinations.

    Returns:
        Integer ranking of the strongest valid five-card combination.
        Higher scores represent stronger hands.

    Raises:
        ValueError: If the hand or board has an invalid length, contains
            invalid or duplicate card IDs, or shares cards with the other.
    """
    if hand.ndim != 1 or len(hand) not in (4, 5):
        raise ValueError("hand must be a one-dimensional array of 4 or 5 cards")
    if board.ndim != 1 or len(board) != 5:
        raise ValueError("board must be a one-dimensional array of exactly 5 cards")
    if any(int(card) >= 52 for card in hand) or any(int(card) >= 52 for card in board):
        raise ValueError("card IDs must be between 0 and 51")
    if len({int(card) for card in hand}) != len(hand):
        raise ValueError("hand cannot contain duplicate cards")
    if len({int(card) for card in board}) != len(board):
        raise ValueError("board cannot contain duplicate cards")
    if {int(card) for card in hand} & {int(card) for card in board}:
        raise ValueError("hand and board cannot contain the same card")

    return _evaluate_hand_core(
        hand, board, rank_lookup.rank_table, rank_lookup.binomial_table
    )


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


@njit(parallel=True, cache=True)
def numba_evaluate_hands_parallel(
    hands: NDArray[np.uint8],
    board: NDArray[np.uint8],
    rank_table: NDArray[np.uint32],
    binomial_table: NDArray[np.uint32],
) -> NDArray[np.uint32]:
    """Evaluate multiple independent Omaha hands in parallel using Numba.

    Produces the same rankings and output order as numba_evaluate_hands(),
    but distributes evaluations across Numba threads using prange.

    Each input row is evaluated independently against the same board.
    Inputs are assumed valid and are not checked for card collisions.
    """
    evaluations = np.empty(hands.shape[0], dtype=np.uint32)

    for i in prange(hands.shape[0]):
        evaluations[i] = _evaluate_hand_core(
            hands[i], board, rank_table=rank_table, binomial_table=binomial_table
        )

    return evaluations
