from typing import Literal

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

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


@njit(cache=True)
def numba_calculate_rank(
    c: CombinationStrength,
    r1: int = 0,
    r2: int = 0,
    r3: int = 0,
    r4: int = 0,
    r5: int = 0,
) -> int:
    """Encode a category and five rank values as a base-13 integer."""
    return (((((c * 13 + int(r1)) * 13 + int(r2)) * 13 + int(r3)) * 13 + int(r4)) * 13 + int(r5))

@njit(cache=True)
def numba_rank_combination(cards: NDArray[np.uint8]) -> int:
    """Evaluate five cards.

    Preconditions:
        - Exactly 5 valid, distinct cards.
        - Cards sorted by ID in descending order.
    """
    # Card encoding: card_id = (rank * 4) + suit.
    # The lowest 2 bits store the suit (0-3); the remaining bits store the rank (0-12).
    #
    # card & 3  -> extracts the suit by masking all bits except the lowest 2 (binary 11).
    # card >> 2 -> extracts the rank by shifting right 2 bits (equivalent to card // 4).
    #
    # These bitwise operations avoid division and modulo
 
    r0 = int(cards[0]) >> 2
    r1 = int(cards[1]) >> 2
    r2 = int(cards[2]) >> 2
    r3 = int(cards[3]) >> 2
    r4 = int(cards[4]) >> 2
    # Encode adjacent rank equalities as a 4-bit mask.
    # Bit i is 1 if ranks[i] == ranks[i+1], otherwise 0.
    # Since ranks are sorted descending, each pattern uniquely identifies
    # the rank repetition structure (pair, two pair, trips, full house, quads).
    # Example: AAAKK -> 0b1011 (equal, equal, different, equal).
    pattern = (
        (int(r0 == r1) << 0)
        | (int(r1 == r2) << 1)
        | (int(r2 == r3) << 2)
        | (int(r3 == r4) << 3)
        )

    if pattern == 0b0000:
        suit = cards[0] & 3
        is_flush = (
            (cards[1] & 3) == suit
            and (cards[2] & 3) == suit
            and (cards[3] & 3) == suit
            and (cards[4] & 3) == suit
        )

        is_straight = (
            r0 == r1 + 1
            and r1 == r2 + 1
            and r2 == r3 + 1
            and r3 == r4 + 1
        )
        is_wheel = (
            r0 == 12 and r1 == 3 and r2 == 2 and r3 == 1 and r4 == 0
        )
        if is_wheel:
            is_straight = True

        if is_flush and is_straight:
            straight_high = 3 if r0 == 12 and r1 == 3 else r0
            return numba_calculate_rank(c=8, r1=straight_high)

        if is_flush:
                return numba_calculate_rank(
                    c=5, 
                    r1=r0, 
                    r2=r1, 
                    r3=r2,
                    r4=r3,
                    r5=r4
                )

        if is_straight:
            straight_high = 3 if r0 == 12 and r1 == 3 else r0
            return numba_calculate_rank(c=4, r1=straight_high)
        
        # HIGH CARD
        return numba_calculate_rank(c=0, r1=r0, r2=r1, r3=r2, r4=r3, r5=r4)
    
    # FOUR OF A KIND
    if pattern == 0b0111:  # AAAAK
        return numba_calculate_rank(7, r0, r4)

    if pattern == 0b1110:  # AKKKK
        return numba_calculate_rank(7, r1, r0)
    
    # FULL HOUSE
    if pattern == 0b1011:  # AAAKK
        return numba_calculate_rank(6, r0, r3)

    if pattern == 0b1101:  # AAKKK
        return numba_calculate_rank(6, r2, r0)
 
    # THREE OF A KIND
    if pattern == 0b0011:  # AAAKQ
        return numba_calculate_rank(3, r0, r3, r4)

    if pattern == 0b0110:  # KAAAQ
        return numba_calculate_rank(3, r1, r0, r4)

    if pattern == 0b1100:  # KQAAA
        return numba_calculate_rank(3, r2, r0, r1)

    # TWO PAIR
    if pattern == 0b0101:  # AAKKQ
        return numba_calculate_rank(2, r0, r2, r4)

    if pattern == 0b1001:  # AAQKK
        return numba_calculate_rank(2, r0, r3, r2)

    if pattern == 0b1010:  # AKKQQ
        return numba_calculate_rank(2, r1, r3, r0)

    # ONE PAIR
    if pattern == 0b0001:  # AAKQJ
        return numba_calculate_rank(1, r0, r2, r3, r4)

    if pattern == 0b0010:  # KAAQJ
        return numba_calculate_rank(1, r1, r0, r3, r4)

    if pattern == 0b0100:  # KQAAJ
        return numba_calculate_rank(1, r2, r0, r1, r4)

    if pattern == 0b1000:  # KQJAA
        return numba_calculate_rank(1, r3, r0, r1, r2)

    raise ValueError("Invalid five-card rank pattern")
    


def numba_evaluate_hand(hand: NDArray[np.uint8], board: NDArray[np.uint8]) -> int:
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

    return _evaluate_hand_core(hand, board)



@njit(cache=True)
def _evaluate_hand_core(hand: NDArray[np.uint8], board: NDArray[np.uint8]) -> int:
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
                        score = numba_rank_combination(cards)

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
def numba_evaluate_hands(hands: NDArray[np.uint8], board: NDArray[np.uint8]) -> NDArray[np.uint32]:
    evaluations = np.empty(hands.shape[0], dtype=np.uint32)
    for i in range(hands.shape[0]):
        evaluations[i] = _evaluate_hand_core(hands[i], board)

    return evaluations

@njit(parallel=True, cache=True)
def numba_evaluate_hands_parallel(hands: NDArray[np.uint8], board: NDArray[np.uint8]) -> NDArray[np.uint32]:
    evaluations = np.empty(hands.shape[0], dtype=np.uint32)

    for i in prange(hands.shape[0]):
        evaluations[i] = _evaluate_hand_core(hands[i], board)

    return evaluations