from typing import Literal

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from plorangelab.rank_combination import rank_combination_lookup, RankLookup

#0-high card
#1-one pair
#2-two pair
#3-three of a kind
#4-straight
#5-flush
#6-full house
#7-four of a kind
#8-straight flush
CombinationStrength = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]


def evaluate_hand(hand: NDArray[np.uint8], board: NDArray[np.uint8], rank_lookup: RankLookup) -> int:
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

    return _evaluate_hand_core(hand, board, rank_lookup.rank_table, rank_lookup.binomial_table)



@njit(cache=True)
def _evaluate_hand_core(hand: NDArray[np.uint8], board: NDArray[np.uint8], rank_table: NDArray[np.uint32], binomial_table: NDArray[np.uint32]) -> int:
    best_score = -1
    cards = np.empty(5, dtype=np.uint8)

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
                        score = rank_combination_lookup(cards, rank_table, binomial_table)

                        best_score = max(best_score, score)

    return best_score 

@njit(inline="always")
def sort5_desc(cards: NDArray[np.uint8]) -> None:
    """Sort exactly 5 elements in descending order, in-place."""

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
    binomial_table: NDArray[np.uint32]
) -> NDArray[np.uint32]:
    evaluations = np.empty(hands.shape[0], dtype=np.uint32)
    for i in range(hands.shape[0]):
        evaluations[i] = _evaluate_hand_core(hands[i], board, rank_table, binomial_table)

    return evaluations

@njit(parallel=True, cache=True)
def numba_evaluate_hands_parallel(
    hands: NDArray[np.uint8], 
    board: NDArray[np.uint8], 
    rank_table: NDArray[np.uint32], 
    binomial_table: NDArray[np.uint32]
) -> NDArray[np.uint32]:
    evaluations = np.empty(hands.shape[0], dtype=np.uint32)

    for i in prange(hands.shape[0]):
        evaluations[i] = _evaluate_hand_core(hands[i], board, rank_table=rank_table, binomial_table=binomial_table)

    return evaluations