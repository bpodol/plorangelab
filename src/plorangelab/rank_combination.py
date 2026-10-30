from math import comb
from typing import Literal
from dataclasses import dataclass

import numpy as np
from numba import njit
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

@dataclass
class RankLookup:
    rank_table: NDArray[np.uint32]
    binomial_table: NDArray[np.uint32]

@njit(cache=True)
def _numba_calculate_rank(
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
def _numba_rank_combination(cards: NDArray[np.uint8]) -> int:
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
            return _numba_calculate_rank(c=8, r1=straight_high)

        if is_flush:
                return _numba_calculate_rank(
                    c=5, 
                    r1=r0, 
                    r2=r1, 
                    r3=r2,
                    r4=r3,
                    r5=r4
                )

        if is_straight:
            straight_high = 3 if r0 == 12 and r1 == 3 else r0
            return _numba_calculate_rank(c=4, r1=straight_high)
        
        # HIGH CARD
        return _numba_calculate_rank(c=0, r1=r0, r2=r1, r3=r2, r4=r3, r5=r4)
    
    # FOUR OF A KIND
    if pattern == 0b0111:  # AAAAK
        return _numba_calculate_rank(7, r0, r4)

    if pattern == 0b1110:  # AKKKK
        return _numba_calculate_rank(7, r1, r0)
    
    # FULL HOUSE
    if pattern == 0b1011:  # AAAKK
        return _numba_calculate_rank(6, r0, r3)

    if pattern == 0b1101:  # AAKKK
        return _numba_calculate_rank(6, r2, r0)
 
    # THREE OF A KIND
    if pattern == 0b0011:  # AAAKQ
        return _numba_calculate_rank(3, r0, r3, r4)

    if pattern == 0b0110:  # KAAAQ
        return _numba_calculate_rank(3, r1, r0, r4)

    if pattern == 0b1100:  # KQAAA
        return _numba_calculate_rank(3, r2, r0, r1)

    # TWO PAIR
    if pattern == 0b0101:  # AAKKQ
        return _numba_calculate_rank(2, r0, r2, r4)

    if pattern == 0b1001:  # AAQKK
        return _numba_calculate_rank(2, r0, r3, r2)

    if pattern == 0b1010:  # AKKQQ
        return _numba_calculate_rank(2, r1, r3, r0)

    # ONE PAIR
    if pattern == 0b0001:  # AAKQJ
        return _numba_calculate_rank(1, r0, r2, r3, r4)

    if pattern == 0b0010:  # KAAQJ
        return _numba_calculate_rank(1, r1, r0, r3, r4)

    if pattern == 0b0100:  # KQAAJ
        return _numba_calculate_rank(1, r2, r0, r1, r4)

    if pattern == 0b1000:  # KQJAA
        return _numba_calculate_rank(1, r3, r0, r1, r2)

    raise ValueError("Invalid five-card rank pattern")
    


@njit(cache=True)
def _generate_ranking_table(
    binomial: NDArray[np.int32],
) -> NDArray[np.uint32]:
    """Generate ranks for all 2,598,960 unique five-card combinations."""

    ranking_table = np.empty(2_598_960, dtype=np.uint32)
    cards = np.empty(5, dtype=np.uint8)

    # Enumerate every five-card combination in descending card-ID order.
    for a in range(4, 52):
        cards[0] = a

        for b in range(3, a):
            cards[1] = b

            for c in range(2, b):
                cards[2] = c

                for d in range(1, c):
                    cards[3] = d

                    for e in range(d):
                        cards[4] = e

                        index = (
                            binomial[e, 1]
                            + binomial[d, 2]
                            + binomial[c, 3]
                            + binomial[b, 4]
                            + binomial[a, 5]
                        )

                        ranking_table[index] = _numba_rank_combination(cards)

    return ranking_table


def build_lookup_tables() -> RankLookup:
    """Generate the five-card ranking and binomial lookup tables.

    Returns:
        ranking_table: Precomputed five-card strength scores.
        binomial: Coefficients used to calculate combination indices.
    """
    binomial = np.zeros((52, 6), dtype=np.uint32)

    for n in range(52):
        for k in range(min(n, 5) + 1):
            binomial[n, k] = comb(n, k)

    rank_table = _generate_ranking_table(binomial)

    return RankLookup(rank_table=rank_table, binomial_table=binomial)



@njit(inline="always")
def rank_combination_lookup(
    cards: NDArray[np.uint8],
    rank_table: NDArray[np.uint32],
    binomial_table: NDArray[np.uint32]
) -> np.uint32:
    """Retrieve the precomputed strength of a five-card combination.

    Preconditions:
        - Exactly 5 valid, distinct cards.
        - Cards sorted by ID in descending order.
    """
    index = (
        binomial_table[cards[4], 1]
        + binomial_table[cards[3], 2]
        + binomial_table[cards[2], 3]
        + binomial_table[cards[1], 4]
        + binomial_table[cards[0], 5]
    )

    return rank_table[index]