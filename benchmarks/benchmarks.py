"""Reproducible benchmarks for PLO RangeLab.

Run from the repository root:
    uv run python benchmarks/benchmarks.py
"""

import random
import statistics
import timeit
from collections.abc import Callable
from itertools import combinations, islice

import numpy as np
from numpy.typing import NDArray

from plorangelab.encoder import encode_board, encode_range
from plorangelab.equity import evaluate_hands
from plorangelab.equity_numba import numba_evaluate_hands


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
            lambda: encode_range(range_string, 4),
            items_per_call=n,
            iterations=5 if n <= 1_000 else 1,
            repeats=11,
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


def benchmark_evaluate_hands() -> None:
    """Slow pure-Python reference benchmark; intentionally limited sizes."""
    board = encode_board("2c3d4h5s6c")

    for mode in (4, 5):
        for n in (100, 1_000):
            hands = generate_hands(n, mode, board)
            benchmark(
                f"Python PLO{mode} | {n:,} hands",
                lambda: evaluate_hands(hands, board),
                items_per_call=n,
                iterations=1,
                repeats=5,
            )


def benchmark_numba_evaluate_hands() -> None:
    board = encode_board("2c3d4h5s6c")

    for mode in (4, 5):
        for n in (1_000, 10_000):
            hands = generate_hands(n, mode, board)
            benchmark(
                f"Numba PLO{mode} | {n:,} hands",
                lambda: numba_evaluate_hands(hands, board),
                items_per_call=n,
                iterations=5 if n == 1_000 else 1,
                repeats=11,
            )


if __name__ == "__main__":
    benchmark_numba_evaluate_hands()
    # Uncomment when needed:
    #benchmark_evaluate_hands()
    # benchmark_encode_range()
