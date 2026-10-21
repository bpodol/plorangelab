from numpy.typing import NDArray
import numpy as np
from typing import Literal
from itertools import combinations

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

def calculate_rank(
    c: CombinationStrength,
    r1: int = 0,
    r2: int = 0,
    r3: int = 0,
    r4: int = 0,
    r5: int = 0,
) -> int:
    """Encode a category and five rank values as a base-13 integer."""
    return (((((c * 13 + int(r1)) * 13 + int(r2)) * 13 + int(r3)) * 13 + int(r4)) * 13 + int(r5))

def rank_combination(cards: NDArray[np.uint8]) -> int:
    """Evaluate five cards.

    Preconditions:
        - Exactly 5 valid, distinct cards.
        - Cards sorted by ID in descending order.
    """
    ranks = cards // 4
    suits = cards % 4

    is_flush = True
    for suit in suits[1:]:
        if suit != suits[0]:
            is_flush = False
            break

    is_straight = True
    for index in range(4):
        if ranks[index] != ranks[index + 1] + np.uint8(1):
            is_straight = False
            break
    # Ace-low wheel: A-5-4-3-2 (rank values 12, 3, 2, 1, 0).
    is_wheel = (
    ranks[0] == 12
    and ranks[1] == 3
    and ranks[2] == 2
    and ranks[3] == 1
    and ranks[4] == 0
)
    if is_wheel:
        is_straight = True

    if is_flush and is_straight:
        straight_high = 3 if ranks[0] == 12 and ranks[1] == 3 else ranks[0]
        return calculate_rank(c=8, r1=straight_high)

    # Find repeated ranks by walking the already sorted ranks once.
    ranks_count: list[tuple[int, int]] = []
    current_rank = ranks[0]
    count = 1
    for rank in ranks[1:]:
        if rank == current_rank:
            count += 1
        else:
            ranks_count.append((count, current_rank))
            current_rank = rank
            count = 1
    ranks_count.append((count, current_rank))
    ranks_count.sort(reverse=True)
    
    if ranks_count[0][0] == 4:
        quad_rank = ranks_count[0][1]
        kicker = ranks_count[1][1]
        return calculate_rank(c=7, r1=quad_rank, r2=kicker)
    
    if ranks_count[0][0] == 3 and ranks_count[1][0] == 2:
        trips = ranks_count[0][1]
        pair = ranks_count[1][1]
        return calculate_rank(c=6, r1=trips, r2=pair)

    if is_flush:
        return calculate_rank(
            c=5, 
            r1=ranks[0], 
            r2=ranks[1], 
            r3=ranks[2],
            r4=ranks[3],
            r5=ranks[4]
        )

    if is_straight:
        straight_high = 3 if ranks[0] == 12 and ranks[1] == 3 else ranks[0]
        return calculate_rank(c=4, r1=straight_high)

    if ranks_count[0][0] == 3 and ranks_count[1][0] == 1:
        kickers = []
        for rank in ranks:
            if rank == ranks_count[0][1]:
                continue
            else:
                kickers.append(rank)
                
        return calculate_rank(c=3, r1=ranks_count[0][1], r2=kickers[0], r3=kickers[1])

    if ranks_count[0][0] == 2 and ranks_count[1][0] == 2:
        first_pair = 0 
        for count, rank in ranks_count:
            if count == 2 and rank > first_pair:
                first_pair = rank
            elif count == 2 and rank < first_pair:
                second_pair = rank
            else:
                kicker = rank

        return calculate_rank(c=2, r1=first_pair, r2=second_pair, r3=kicker)

    if ranks_count[0][0] == 2 and ranks_count[1][0] == 1:
        kickers = []
        for rank in ranks:
            if rank == ranks_count[0][1]:
                pair = rank
            else:
                kickers.append(rank)

        return calculate_rank(c=1, r1=pair, r2=kickers[0], r3=kickers[1], r4=kickers[2])

    return calculate_rank(c=0, r1=ranks[0], r2=ranks[1], r3=ranks[2], r4=ranks[3], r5=ranks[4])

    


def evaluate_hand(hand: NDArray[np.uint8], board: NDArray[np.uint8]) -> int:
    if hand.ndim != 1 or len(hand) not in (4, 5):
        raise ValueError("hand must be a one-dimensional array of 4 or 5 cards")
    if board.ndim != 1 or len(board) != 5:
        raise ValueError("board must be a one-dimensional array of exactly 5 cards")
    if any(int(card) >= 52 for card in hand) or any(int(card) >= 52 for card in board):
        raise ValueError("card IDs must be between 0 and 51")
    if len(set(int(card) for card in hand)) != len(hand):
        raise ValueError("hand cannot contain duplicate cards")
    if len(set(int(card) for card in board)) != len(board):
        raise ValueError("board cannot contain duplicate cards")
    if set(int(card) for card in hand) & set(int(card) for card in board):
        raise ValueError("hand and board cannot contain the same card")

    best_score = -1

    for hole_cards in combinations(hand, 2):
        for board_cards in combinations(board, 3):
            cards = np.array((*hole_cards, *board_cards), dtype=np.uint8)
            cards.sort()
            cards = cards[::-1]  # rank_combination expects descending card IDs

            score = rank_combination(cards)
            if score > best_score:
                best_score = score

    return best_score    

def evaluate_hands(hands: NDArray[np.uint8], board: NDArray[np.uint8]) -> NDArray[np.uint32]:
    evaluations = np.empty(len(hands), dtype=np.uint32)
    for i, hand in enumerate(hands):
        evaluations[i] = evaluate_hand(hand, board)

    return evaluations