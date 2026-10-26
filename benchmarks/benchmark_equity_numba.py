"""Small, deterministic end-to-end benchmarks for the Numba equity API.

Run with ``uv run python benchmarks/benchmark_equity_numba.py``. Exact and
Monte Carlo methods are timed in separate sections; setup and first-call JIT
compilation are excluded from each timing.
"""

from __future__ import annotations

import argparse
import platform
import random
import statistics
import sys
import timeit
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations, islice

import numba
import numpy as np

from plorangelab.encoder import EncodedRange, build_mask
from plorangelab.equity import EquityResult, compute_equity


@dataclass(frozen=True)
class Scenario:
    mode: int
    board_cards: int
    hands_per_range: int
    overlap: bool = False

    @property
    def label(self) -> str:
        street = {3: "flop", 4: "turn", 5: "river"}[self.board_cards]
        profile = "overlapping" if self.overlap else "disjoint"
        return (
            f"PLO{self.mode} | {street} | {profile} | "
            f"{self.hands_per_range} hands/range"
        )


SCENARIOS = (
    Scenario(4, 3, 3),
    Scenario(4, 4, 8, overlap=True),
    Scenario(4, 5, 16, overlap=True),
    Scenario(5, 3, 2),
    Scenario(5, 4, 5),
    Scenario(5, 5, 12, overlap=True),
)


def make_range(hands: list[tuple[int, ...]]) -> EncodedRange:
    cards = np.asarray(hands, dtype=np.uint8)
    masks = np.asarray([build_mask(hand) for hand in cards], dtype=np.uint64)
    weights = np.ones(len(hands), dtype=np.float32)
    return EncodedRange(hands=cards, masks=masks, weights=weights)


def make_inputs(scenario: Scenario) -> tuple[list[EncodedRange], np.ndarray]:
    """Build deterministic ranges outside the timed public API call."""
    seed = (
        2026
        + scenario.mode * 1_000
        + scenario.board_cards * 100
        + scenario.hands_per_range * 10
        + int(scenario.overlap)
    )
    rng = random.Random(seed)
    deck = list(range(52))
    rng.shuffle(deck)
    board = np.asarray(deck[: scenario.board_cards], dtype=np.uint8)
    available = deck[scenario.board_cards :]

    if scenario.overlap:
        shared_pool = available[:16]
        candidates = list(combinations(shared_pool, scenario.mode))
        first_anchor = tuple(shared_pool[: scenario.mode])
        second_anchor = tuple(shared_pool[scenario.mode : 2 * scenario.mode])
        first_hands = [first_anchor] + rng.sample(
            [hand for hand in candidates if hand != first_anchor],
            scenario.hands_per_range - 1,
        )
        second_hands = [second_anchor] + rng.sample(
            [hand for hand in candidates if hand != second_anchor],
            scenario.hands_per_range - 1,
        )
    else:
        pool_size = len(available) // 2
        first_pool, second_pool = (
            available[:pool_size],
            available[pool_size : 2 * pool_size],
        )
        first_hands = list(
            islice(combinations(first_pool, scenario.mode), scenario.hands_per_range)
        )
        second_hands = list(
            islice(combinations(second_pool, scenario.mode), scenario.hands_per_range)
        )

    return [make_range(first_hands), make_range(second_hands)], board


def benchmark(
    name: str,
    call: Callable[[], EquityResult],
    *,
    repeats: int,
) -> tuple[float, float]:
    """Warm up Numba, then print median and median absolute deviation."""
    result = call()
    if not np.isclose(sum(result.equities.values()), 1.0, atol=1e-10):
        raise AssertionError(f"Invalid equity total in {name}: {result}")

    seconds = timeit.repeat(call, number=1, repeat=repeats)
    median = statistics.median(seconds)
    mad = statistics.median(abs(sample - median) for sample in seconds)
    print(f"  {name}: median {median * 1000:.2f} ms | MAD {mad * 1000:.2f} ms")
    return median, mad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.samples <= 0 or args.repeats <= 0:
        parser.error("--samples and --repeats must be positive")

    print(
        f"Python {sys.version.split()[0]} | NumPy {np.__version__} | "
        f"Numba {numba.__version__} | {platform.platform()}\n"
        f"samples={args.samples}, repeats={args.repeats}"
    )
    prepared = [(scenario, *make_inputs(scenario)) for scenario in SCENARIOS]

    for method in ("exact", "monte-carlo"):
        print(f"\n=== {method.upper()} | public compute_equity API ===")
        for scenario, players, board in prepared:
            benchmark(
                scenario.label,
                lambda players=players, board=board, scenario=scenario, method=method: (
                    compute_equity(
                        players,
                        mode=scenario.mode,
                        board=board,
                        min_samples=args.samples,
                        max_samples=args.samples,
                        target_moe=0.0,
                        seed=42,
                        method=method,
                    )
                ),
                repeats=args.repeats,
            )


if __name__ == "__main__":
    main()
