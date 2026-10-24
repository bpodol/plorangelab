
from __future__ import annotations

import argparse
import platform
import random
import statistics
import sys
import timeit
from dataclasses import dataclass
from itertools import combinations, islice
from math import comb, prod
from collections.abc import Callable

import numba
import numpy as np

from plorangelab.encoder import EncodedRange, build_mask
from plorangelab.equity import (
    compute_equity,
    prepare_probabilities,
    prepare_ranges,
    run_monte_carlo,
)


@dataclass(frozen=True)
class Scenario:
    mode: int
    players: int
    board_cards: int  # 0 (preflop), 3 (flop), 4 (turn), or 5 (river)
    hands_per_range: int
    overlap: bool = False

    @property
    def label(self) -> str:
        street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[
            self.board_cards
        ]
        suffix = " | shared card pool" if self.overlap else " | disjoint pools"
        return (
            f"PLO{self.mode} | {self.players} players | {street} | "
            f"{self.hands_per_range} hands/range{suffix}"
        )


def make_range(
    hands: list[tuple[int, ...]], rng: random.Random
) -> EncodedRange:
    """Build an encoded range with deterministic, nonuniform positive weights."""
    cards = np.asarray(hands, dtype=np.uint8)
    masks = np.asarray([build_mask(row) for row in cards], dtype=np.uint64)
    weights = np.asarray(
        [0.25 + 0.75 * rng.random() for _ in hands], dtype=np.float32
    )
    return EncodedRange(hands=cards, masks=masks, weights=weights)


def make_inputs(s: Scenario) -> tuple[list[EncodedRange], np.ndarray | None]:
    """Prepare valid, reproducible inputs outside the timed region.

    Disjoint pools guarantee no cross-player collisions. The overlapping
    scenario deliberately introduces collisions to measure rejection costs.
    """
    rng = random.Random(2026 + s.mode + 10 * s.players + s.board_cards)
    deck = list(range(52))
    rng.shuffle(deck)
    board = (
        None
        if s.board_cards == 0
        else np.asarray(deck[: s.board_cards], dtype=np.uint8)
    )
    available = deck[s.board_cards :]

    if s.overlap:
        if s.players != 2:
            raise ValueError("Overlapping benchmark currently supports 2 players")
        shared_pool = available[:16]
        candidates = list(combinations(shared_pool, s.mode))
        players = [
            make_range(rng.sample(candidates, s.hands_per_range), rng)
            for _ in range(s.players)
        ]
    else:
        pool_size = len(available) // s.players
        if comb(pool_size, s.mode) < s.hands_per_range:
            raise ValueError(f"Not enough distinct hands for {s.label}")
        players = []
        for i in range(s.players):
            pool = available[i * pool_size : (i + 1) * pool_size]
            hands = list(islice(combinations(pool, s.mode), s.hands_per_range))
            players.append(make_range(hands, rng))

    return players, board


def weighted_acceptance(players: list[EncodedRange]) -> float:
    """Exact weighted compatibility probability for a two-player scenario."""
    a, b = players
    compatible = (a.masks[:, None] & b.masks[None, :]) == 0
    wa = a.weights.astype(np.float64)
    wb = b.weights.astype(np.float64)
    joint = wa[:, None] * wb[None, :]
    return float(joint[compatible].sum() / joint.sum())


def benchmark(
    name: str,
    func: Callable[[], dict[str, float]],
    *,
    work_per_call: int,
    unit: str,
    repeats: int,
) -> None:
    """Report robust wall time and throughput, excluding initial compilation."""
    result = func()  # Warm up Numba; do not time first-call JIT compilation.
    if not np.isclose(sum(result.values()), 1.0, atol=1e-10):
        raise AssertionError(f"Invalid equity total in {name}: {result}")

    seconds = timeit.repeat(func, number=1, repeat=repeats)
    median_s = statistics.median(seconds)
    mad_s = statistics.median(abs(t - median_s) for t in seconds)
    print(
        f"{name}\n"
        f"  Median: {median_s * 1000:.2f} ms/call | "
        f"MAD: {mad_s * 1000:.2f} ms | "
        f"Min/Max: {min(seconds) * 1000:.2f}/{max(seconds) * 1000:.2f} ms\n"
        f"  Throughput: {work_per_call / median_s:,.0f} {unit}/sec\n"
    )


def mc_scenarios(suite: str) -> list[Scenario]:
    quick = [
        Scenario(4, 2, 3, 10),
        Scenario(4, 2, 5, 10),
        Scenario(5, 2, 3, 10),
        Scenario(5, 2, 5, 10),
        Scenario(5, 6, 5, 10),
        Scenario(5, 2, 5, 100, overlap=True),
    ]
    if suite == "quick":
        return quick

    full = [
        # Street comparison on identical-sized ranges.
        *(Scenario(mode, 2, board_cards, 10)
          for mode in (4, 5) for board_cards in (0, 3, 4, 5)),
        # Multiway scaling.
        *(Scenario(mode, player_count, 5, 10)
          for mode in (4, 5) for player_count in (4, 6)),
        # Range-size scaling.
        *(Scenario(mode, 2, 5, count)
          for mode in (4, 5) for count in (1, 100)),
        # Collision rejection overhead at larger range sizes.
        *(Scenario(mode, 2, 5, 100, overlap=True) for mode in (4, 5)),
    ]
    # Preserve order and avoid duplicating identical cases.
    return list(dict.fromkeys(full))


def exact_scenarios(suite: str) -> list[Scenario]:
    quick = [
        Scenario(4, 2, 5, 1),
        Scenario(4, 2, 5, 10),
        Scenario(5, 2, 5, 1),
        Scenario(5, 2, 5, 10),
        Scenario(4, 2, 4, 1),
        Scenario(5, 2, 3, 1),
    ]
    if suite == "quick":
        return quick

    return quick + [
        Scenario(4, 2, 5, 100),
        Scenario(5, 2, 5, 100),
        Scenario(4, 2, 3, 5),
        Scenario(5, 2, 4, 5),
        Scenario(5, 6, 5, 1),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("quick", "full"), default="quick")
    parser.add_argument("--samples", type=int, default=1_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--core", action="store_true",
        help="Also benchmark run_monte_carlo() with preparation excluded",
    )
    args = parser.parse_args()
    if args.samples <= 0 or args.repeats <= 0:
        parser.error("--samples and --repeats must be positive")

    print(
        f"Python {sys.version.split()[0]} | NumPy {np.__version__} | "
        f"Numba {numba.__version__} | threads {numba.get_num_threads()}\n"
        f"Platform: {platform.platform()} | "
        f"suite={args.suite}, samples={args.samples}, repeats={args.repeats}\n"
    )

    print("=== MONTE CARLO: public API (validation + preparation + simulation) ===")
    for s in mc_scenarios(args.suite):
        players, board = make_inputs(s)
        if s.overlap:
            print(f"  Weighted hand-tuple acceptance probability: "
                  f"{weighted_acceptance(players):.1%}")
        benchmark(
            s.label,
            lambda players=players, board=board, s=s: compute_equity(
                players, mode=s.mode, board=board,
                samples=args.samples, seed=42, method="monte-carlo"
            ),
            work_per_call=args.samples, unit="valid simulations",
            repeats=args.repeats,
        )

        if args.core:
            valid_players = prepare_ranges(players, board)
            probabilities = prepare_probabilities(valid_players)
            benchmark(
                s.label + " | simulation core only",
                lambda valid_players=valid_players, probabilities=probabilities,
                board=board, s=s: run_monte_carlo(
                    players=valid_players, probabilities=probabilities,
                    mode=s.mode, board=board,
                    samples=args.samples, seed=42,
                ),
                work_per_call=args.samples, unit="valid simulations",
                repeats=args.repeats,
            )

    print("=== EXACT: public API (disjoint ranges, no collision rejection) ===")
    for s in exact_scenarios(args.suite):
        players, board = make_inputs(s)
        # In the disjoint-pool scenarios every tuple is compatible.
        showdowns = (
            prod(len(p.hands) for p in players)
            * comb(52 - s.board_cards - s.players * s.mode, 5 - s.board_cards)
        )
        if showdowns > 100_000:
            print(f"SKIP: {s.label} ({showdowns:,} showdowns exceeds current limit)\n")
            continue
        benchmark(
            s.label + f" | {showdowns:,} exhaustive showdowns",
            lambda players=players, board=board, s=s: compute_equity(
                players, mode=s.mode, board=board, method="exact"
            ),
            work_per_call=showdowns, unit="showdowns", repeats=args.repeats,
        )


if __name__ == "__main__":
    main()
