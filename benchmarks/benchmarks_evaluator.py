"""Reproducible benchmarks for PLO RangeLab.

Run from the repository root:
    uv run python benchmarks/benchmarks_evaluator.py
"""

import random
import statistics
import timeit
from collections.abc import Callable
from itertools import combinations, islice

import numpy as np
from numba import njit
from numpy.typing import NDArray
from plorangelab.encoder import encode_board, encode_range
from plorangelab.equity.evaluator import (
    numba_evaluate_hands,
    numba_evaluate_hands_parallel,
    numba_rank_combination,
)


def benchmark(
    name: str,
    func: Callable[[], object],
    *,
    items_per_call: int = 1,
    iterations: int = 1,
    repeats: int = 11,
) -> None:
    """Benchmark a zero-argument callable, excluding initial Numba compilation.

    Report median time and median absolute deviation (MAD) over repeats.
    Each repeat executes the callable `iterations` times.
    """
    if items_per_call <= 0 or iterations <= 0 or repeats <= 0:
        raise ValueError("Benchmark counts must be positive")

    func()  # Warm up Numba, including any first-call specialization.

    elapsed = timeit.repeat(func, number=iterations, repeat=repeats)
    us_per_item = [
        seconds * 1_000_000 / (iterations * items_per_call)
        for seconds in elapsed
    ]

    median_us = statistics.median(us_per_item)
    mad_us = statistics.median(abs(x - median_us) for x in us_per_item)
    min_us = min(us_per_item)
    max_us = max(us_per_item)

    unit = "hand" if items_per_call > 1 else "call"
    print(
        f"{name}\n"
        f"  Median: {median_us:.2f} µs/{unit} | "
        f"MAD: {mad_us:.2f} µs/{unit}\n"
        f"  Min/Max: {min_us:.2f} / {max_us:.2f} µs/{unit}\n"
        f"  Throughput: {1_000_000 / median_us:,.0f} {unit}s/sec\n"
    )


def generate_range(n: int, mode: int = 4) -> str:
    deck = [rank + suit for rank in "23456789TJQKA" for suit in "cdhs"]
    hands = islice(combinations(deck, mode), n)
    return ",".join("".join(hand) for hand in hands)


def benchmark_encode_range() -> None:
    for n in (100, 1_000, 10_000, 100_000):
        range_string = generate_range(n)
        benchmark(
            f"encode_range | {n:,} hands",
            lambda range_string=range_string: encode_range(range_string, 4),
            items_per_call=n,
            iterations=5 if n <= 1_000 else 1,
        )


def generate_hands(
    n: int, mode: int, board: NDArray[np.uint8]
) -> NDArray[np.uint8]:
    rng = random.Random(42)
    blocked = set(board.tolist())
    deck = [card for card in range(52) if card not in blocked]

    hands: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    while len(hands) < n:
        hand = tuple(sorted(rng.sample(deck, mode), reverse=True))
        if hand not in seen:
            seen.add(hand)
            hands.append(hand)

    return np.array(hands, dtype=np.uint8)


def benchmark_numba_evaluate_hands() -> None:
    board = encode_board("2c3d4h5s6c")

    for mode in (4, 5):
        for n in (1_000, 10_000):
            hands = generate_hands(n, mode, board)
            benchmark(
                f"Numba PLO{mode} | {n:,} hands",
                lambda hands=hands, board=board: numba_evaluate_hands(hands, board),
                items_per_call=n,
                iterations=5 if n == 1_000 else 1,
            )


def benchmark_numba_evaluate_hands_parallel() -> None:
    board = encode_board("2c3d4h5s6c")

    for mode in (4, 5):
        for n in (1_000, 10_000, 100_000):
            hands = generate_hands(n, mode, board)
            benchmark(
                f"Numba PLO{mode} | {n:,} hands",
                lambda hands=hands, board=board: numba_evaluate_hands_parallel(hands, board),
                items_per_call=n,
                iterations=5 if n == 1_000 else 1,
            )


# -------- Evaluator microbenchmarks (sequential JIT, no threading) --------
# The checksum makes all five card writes observable to the compiler.
# We use the exact same combinatorial iteration and checksum in every stage.
# Consequently, the incremental differences are more useful than the
# absolute costs. These are *diagnostic* kernels, not production evaluators.


@njit(cache=True)
def _checksum_cards(cards: NDArray[np.uint8]) -> int:
    # Order-sensitive: sorting changes the result, so the sort cannot be elided.
    return (
        int(cards[0])
        + 3 * int(cards[1])
        + 5 * int(cards[2])
        + 7 * int(cards[3])
        + 11 * int(cards[4])
    )


@njit(cache=True)
def _generate_only(hand: NDArray[np.uint8], board: NDArray[np.uint8]) -> int:
    cards = np.empty(5, dtype=np.uint8)
    checksum = 0

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

                        checksum += _checksum_cards(cards)

    return checksum

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
def _generate_and_sort(hand: NDArray[np.uint8], board: NDArray[np.uint8]) -> int:
    cards = np.empty(5, dtype=np.uint8)
    checksum = 0

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
                        checksum += _checksum_cards(cards)

    return checksum


@njit(cache=True)
def _generate_sort_and_rank(hand: NDArray[np.uint8], board: NDArray[np.uint8]) -> int:
    cards = np.empty(5, dtype=np.uint8)
    checksum = 0
    best_score = -1

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
                        checksum += _checksum_cards(cards)
                        score = numba_rank_combination(cards)
                        best_score = max(best_score, score)

    return checksum + best_score


@njit(cache=True)
def _run_stage(
    hands: NDArray[np.uint8],
    board: NDArray[np.uint8],
    stage: int,
) -> NDArray[np.int64]:
    results = np.empty(hands.shape[0], dtype=np.int64)
    for i in range(hands.shape[0]):
        if stage == 1:
            results[i] = _generate_only(hands[i], board)
        elif stage == 2:
            results[i] = _generate_and_sort(hands[i], board)
        else:
            results[i] = _generate_sort_and_rank(hands[i], board)
    return results


def benchmark_evaluator_stages() -> None:
    board = encode_board("2c3d4h5s6c")
    print("\n=== EVALUATOR STAGES (sequential Numba) ===")

    for mode in (4, 5):
        n = 10_000
        hands = generate_hands(n, mode, board)
        print(f"\nPLO{mode}: {10 * (mode * (mode - 1) // 2)} combinations/hand")

        for stage, title in (
            (1, "1. Generate combinations only"),
            (2, "2. Generate + sort"),
            (3, "3. Generate + sort + rank"),
        ):
            benchmark(
                f"PLO{mode} | {title}",
                lambda hands=hands, board=board, stage=stage: _run_stage(
                    hands, board, stage
                ),
                items_per_call=n,
                iterations=1,
            )

        # Separately measure the real function to quantify checksum/profiling
        # overhead. Do not conflate its result with stage 3's result.
        benchmark(
            f"PLO{mode} | Production numba_evaluate_hands (reference)",
            lambda hands=hands, board=board: numba_evaluate_hands(hands, board),
            items_per_call=n,
            iterations=1,
        )


if __name__ == "__main__":
    # benchmark_evaluator_stages()
    benchmark_numba_evaluate_hands_parallel()
    # benchmark_numba_evaluate_hands()
    # benchmark_encode_range()
