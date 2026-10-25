"""Benchmark the Numba PLO equity APIs and batched Monte Carlo core.

Run with ``uv run python benchmarks/benchmark_equity_numba.py``; use
``--suite full`` for the larger scenario set, or ``--core`` to include
coordinator-only timings. Use ``--exact-streets`` to benchmark exact equity
across flop, turn, and river only, or ``--exact-parallel`` to compare the
sequential and worker-local parallel exact kernels.
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
from math import comb, isqrt, prod

import numba
import numpy as np
from numba.typed import List

from plorangelab.encoder import EncodedRange, build_mask
from plorangelab.equity_numba import (
    _run_batched_monte_carlo,
    _run_exact_equity_kernel,
    _run_exact_equity_kernel_parallel,
    compute_equity,
    prepare_probabilities,
    prepare_ranges,
    run_exact_equity,
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
    print_result: bool = True,
) -> tuple[float, float]:
    """Report robust wall time and throughput, excluding initial compilation."""
    result = func()  # Warm up Numba; do not time first-call JIT compilation.
    if not np.isclose(sum(result.values()), 1.0, atol=1e-10):
        raise AssertionError(f"Invalid equity total in {name}: {result}")

    seconds = timeit.repeat(func, number=1, repeat=repeats)
    median_s = statistics.median(seconds)
    mad_s = statistics.median(abs(t - median_s) for t in seconds)
    if print_result:
        print(
            f"{name}\n"
            f"  Median: {median_s * 1000:.2f} ms/call | "
            f"MAD: {mad_s * 1000:.2f} ms | "
            f"Min/Max: {min(seconds) * 1000:.2f}/{max(seconds) * 1000:.2f} ms\n"
            f"  Throughput: {work_per_call / median_s:,.0f} {unit}/sec\n"
        )
    return median_s, mad_s


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


def make_exact_target_inputs(
    mode: int,
    target_showdowns: int,
    overlap: bool,
) -> tuple[Scenario, list[EncodedRange], np.ndarray, int, int]:
    """Create deterministic river inputs near a target of actual evaluations.

    For overlapping ranges, the estimated workload is the Cartesian product
    of candidate hands while the actual count includes only collision-free
    matchups. Shared hands are selected from a 16-card pool to make collisions
    frequent.
    """
    rng = random.Random(2026 + mode * 100 + target_showdowns + overlap)
    deck = list(range(52))
    rng.shuffle(deck)
    board = np.asarray(deck[:5], dtype=np.uint8)
    available = deck[5:]

    if overlap:
        candidates = list(combinations(available[:16], mode))
        rng.shuffle(candidates)
        masks = [sum(1 << card for card in hand) for hand in candidates]

        compatible_pairs = 0
        best_range_size = 1
        best_delta = target_showdowns
        for range_size, mask in enumerate(masks, start=1):
            compatible_previous = sum(
                not (mask & previous_mask)
                for previous_mask in masks[: range_size - 1]
            )
            compatible_pairs += 2 * compatible_previous
            delta = abs(compatible_pairs - target_showdowns)
            if delta < best_delta:
                best_range_size = range_size
                best_delta = delta
            if compatible_pairs >= target_showdowns:
                break

        hands = candidates[:best_range_size]
        players = [make_range(hands, rng), make_range(hands, rng)]
        range_size = best_range_size
    else:
        range_size = isqrt(target_showdowns)
        if (range_size + 1) ** 2 - target_showdowns < (
            target_showdowns - range_size**2
        ):
            range_size += 1
        pool_size = len(available) // 2
        players = []
        for player_index in range(2):
            pool = available[player_index * pool_size : (player_index + 1) * pool_size]
            hands = list(islice(combinations(pool, mode), range_size))
            players.append(make_range(hands, rng))

    scenario = Scenario(mode, 2, 5, range_size, overlap=overlap)
    hand_counts = [len(player.hands) for player in players]
    runouts_per_matchup = comb(52 - len(board) - 2 * mode, 0)
    estimated_showdowns = prod(hand_counts) * runouts_per_matchup
    compatible_matchups = int(
        np.count_nonzero(
            (players[0].masks[:, None] & players[1].masks[None, :]) == 0
        )
    )
    actual_showdowns = compatible_matchups * runouts_per_matchup
    return scenario, players, board, estimated_showdowns, actual_showdowns


def choose_exact_street_range_sizes(
    mode: int,
    board_cards: int,
    target_showdowns: int,
) -> tuple[int, int, int]:
    """Choose disjoint range sizes near a target without exceeding the API cap."""
    available_cards = 52 - board_cards
    pool_size = available_cards // 2
    max_hands = comb(pool_size, mode)
    runouts = comb(52 - board_cards - 2 * mode, 5 - board_cards)
    max_product = min(max_hands**2, 100_000 // runouts)

    best_counts = (1, 1)
    best_actual = runouts
    best_key = (abs(best_actual - target_showdowns), 0)
    for first_count in range(1, min(max_hands, max_product) + 1):
        max_second = min(max_hands, max_product // first_count)
        floor_second = target_showdowns // (runouts * first_count)
        candidates = {
            max(1, min(max_second, floor_second)),
            max(1, min(max_second, floor_second + 1)),
        }
        for second_count in candidates:
            actual = first_count * second_count * runouts
            key = (abs(actual - target_showdowns), abs(first_count - second_count))
            if key < best_key:
                best_counts = first_count, second_count
                best_actual = actual
                best_key = key

    return best_counts[0], best_counts[1], best_actual


def make_exact_street_inputs(
    mode: int,
    board_cards: int,
    target_showdowns: int,
) -> tuple[list[EncodedRange], np.ndarray, tuple[int, int], int]:
    """Build deterministic, collision-free ranges for an exact street case."""
    first_count, second_count, actual_showdowns = choose_exact_street_range_sizes(
        mode, board_cards, target_showdowns
    )
    rng = random.Random(2026 + 100 * mode + 10 * board_cards + target_showdowns)
    deck = list(range(52))
    rng.shuffle(deck)
    board = np.asarray(deck[:board_cards], dtype=np.uint8)
    available = deck[board_cards:]
    pool_size = len(available) // 2
    players = []
    for player_index, hand_count in enumerate((first_count, second_count)):
        pool = available[player_index * pool_size : (player_index + 1) * pool_size]
        hands = list(islice(combinations(pool, mode), hand_count))
        players.append(make_range(hands, rng))

    return players, board, (first_count, second_count), actual_showdowns


def make_unbalanced_exact_inputs(
    mode: int,
) -> tuple[list[EncodedRange], np.ndarray, int]:
    """Build a deterministic collision-free 1-by-1,000 river workload."""
    rng = random.Random(2026 + mode * 1000)
    deck = list(range(52))
    rng.shuffle(deck)
    board = np.asarray(deck[:5], dtype=np.uint8)
    available = deck[5:]
    pool_size = len(available) // 2
    players = []
    for player_index, hand_count in enumerate((1, 1_000)):
        pool = available[player_index * pool_size : (player_index + 1) * pool_size]
        hands = list(islice(combinations(pool, mode), hand_count))
        players.append(make_range(hands, rng))
    return players, board, 1_000


def make_exact_parallel_size_inputs(
    mode: int,
    target_showdowns: int,
    collision_profile: str,
) -> tuple[list[EncodedRange], np.ndarray, int, int]:
    """Build two deterministic river ranges near a candidate-matchup target."""
    first_count = isqrt(target_showdowns)
    second_count = max(1, round(target_showdowns / first_count))
    profile_seed = {
        "disjoint": 11,
        "moderate overlap": 23,
        "heavy overlap": 37,
    }[collision_profile]
    rng = random.Random(20261025 + mode * 100_000 + target_showdowns + profile_seed)
    deck = list(range(52))
    rng.shuffle(deck)
    board = np.asarray(deck[:5], dtype=np.uint8)
    available = deck[5:]

    if collision_profile == "disjoint":
        pool_size = len(available) // 2
        players = []
        for index, count in enumerate((first_count, second_count)):
            pool = available[index * pool_size : (index + 1) * pool_size]
            hands = list(islice(combinations(pool, mode), count))
            players.append(make_range(hands, rng))
    else:
        shared_pool_size = 24 if collision_profile == "moderate overlap" else 16
        shared_pool = available[:shared_pool_size]
        candidate_hands = list(combinations(shared_pool, mode))
        reserved_hands = [
            tuple(shared_pool[:mode]),
            tuple(shared_pool[mode : 2 * mode]),
        ]
        remaining_hands = [
            hand for hand in candidate_hands if hand not in reserved_hands
        ]
        players = [
            make_range(
                [reserved_hands[index]]
                + rng.sample(remaining_hands, count - 1),
                rng,
            )
            for index, count in enumerate((first_count, second_count))
        ]

    estimated = len(players[0].hands) * len(players[1].hands)
    compatible = int(
        np.count_nonzero(
            (players[0].masks[:, None] & players[1].masks[None, :]) == 0
        )
    )
    return players, board, estimated, compatible


def run_exact_street_benchmarks(repeats: int) -> None:
    """Compare public exact equity with the prebuilt Numba kernel by street."""
    street_names = {3: "flop", 4: "turn", 5: "river"}
    print("=== EXACT STREETS: public API and Numba kernel ===")
    for mode in (4, 5):
        for board_cards in (3, 4, 5):
            for target in (10_000, 100_000):
                players, board, range_sizes, actual = make_exact_street_inputs(
                    mode, board_cards, target
                )
                runouts = comb(52 - board_cards - 2 * mode, 5 - board_cards)
                estimated = prod(range_sizes) * runouts
                if estimated != actual or estimated > 100_000:
                    raise AssertionError(
                        "Exact street scenario must have a predictable workload "
                        "within the public API limit"
                    )

                label = (
                    f"PLO{mode} | 2 players | {street_names[board_cards]} | "
                    f"ranges {range_sizes[0]}x{range_sizes[1]} | disjoint pools"
                )
                print(
                    f"{label}\n"
                    f"  Showdowns: {actual:,} actual; "
                    f"{estimated:,} estimated upper bound"
                )
                benchmark(
                    label + " | public API",
                    lambda players=players, board=board, mode=mode: compute_equity(
                        players, mode=mode, board=board, method="exact"
                    ),
                    work_per_call=actual,
                    unit="showdowns",
                    repeats=repeats,
                )

                valid_players = prepare_ranges(players, board)
                hands = List([player.hands for player in valid_players])
                masks = List([player.masks for player in valid_players])
                weights = List()
                for player in valid_players:
                    normalized = player.weights.astype(np.float64)
                    normalized /= normalized.sum(dtype=np.float64)
                    weights.append(normalized)
                board_mask = build_mask(board)

                def kernel_call(
                    hands=hands,
                    masks=masks,
                    weights=weights,
                    board=board,
                    board_mask=board_mask,
                    mode=mode,
                ) -> dict[str, float]:
                    pot_shares, total_weight = _run_exact_equity_kernel(
                        hands, masks, weights, board, board_mask, mode
                    )
                    return {
                        f"player {index + 1}:": float(share / total_weight)
                        for index, share in enumerate(pot_shares)
                    }

                benchmark(
                    label + " | Numba kernel",
                    kernel_call,
                    work_per_call=actual,
                    unit="showdowns",
                    repeats=repeats,
                )


def run_exact_parallel_benchmarks(repeats: int) -> None:
    """Compare sequential and worker-local exact kernels on prepared inputs."""
    thread_counts = [
        count for count in (1, 2, 4, 8)
        if count <= numba.config.NUMBA_NUM_THREADS
    ]
    original_threads = numba.get_num_threads()
    print(
        "=== EXACT PARALLEL: sequential vs worker-local ===\n"
        "Estimated counts are Cartesian candidate tuples; actual counts exclude "
        "card collisions."
    )
    try:
        scenarios = [
            (mode, target, profile)
            for mode in (4, 5)
            for target in (1_000, 5_000, 10_000, 20_000, 50_000, 100_000)
            for profile in ("disjoint", "moderate overlap", "heavy overlap")
        ]
        for mode, target, profile in scenarios:
            players, board, estimate, actual = make_exact_parallel_size_inputs(
                mode, target, profile
            )
            label = f"PLO{mode} | river | {profile} | target {target:,}"
            valid_players = prepare_ranges(players, board)
            hands = List([player.hands for player in valid_players])
            masks = List([player.masks for player in valid_players])
            weights = List()
            for player in valid_players:
                normalized = player.weights.astype(np.float64)
                normalized /= normalized.sum(dtype=np.float64)
                weights.append(normalized)
            board_mask = build_mask(board)

            def sequential_call(
                hands=hands,
                masks=masks,
                weights=weights,
                board=board,
                board_mask=board_mask,
                mode=mode,
            ) -> dict[str, float]:
                shares, total = _run_exact_equity_kernel(
                    hands, masks, weights, board, board_mask, mode
                )
                return {
                    f"player {index + 1}:": float(share / total)
                    for index, share in enumerate(shares)
                }

            def worker_local_call(
                hands=hands,
                masks=masks,
                weights=weights,
                board=board,
                board_mask=board_mask,
                mode=mode,
            ) -> dict[str, float]:
                shares, total = _run_exact_equity_kernel_parallel(
                    hands,
                    masks,
                    weights,
                    board,
                    board_mask,
                    mode,
                    numba.get_num_threads(),
                )
                return {
                    f"player {index + 1}:": float(share / total)
                    for index, share in enumerate(shares)
                }

            sequential_result = sequential_call()
            parallel_result = worker_local_call()
            if not np.allclose(
                list(parallel_result.values()),
                list(sequential_result.values()),
                rtol=0,
                atol=1e-12,
            ):
                raise AssertionError(f"Kernel mismatch in {label}")

            print(
                f"\n{label} | {estimate:,} estimated candidates, "
                f"{actual:,} actual evaluations "
                f"({actual / estimate:.1%} collision-free)"
            )
            for thread_count in thread_counts:
                numba.set_num_threads(thread_count)
                sequential = benchmark(
                    f"  sequential | threads={thread_count}",
                    sequential_call,
                    work_per_call=actual,
                    unit="actual showdowns",
                    repeats=repeats,
                    print_result=False,
                )
                parallel = benchmark(
                    f"  worker-local parallel | threads={thread_count}",
                    worker_local_call,
                    work_per_call=actual,
                    unit="actual showdowns",
                    repeats=repeats,
                    print_result=False,
                )
                speedup = sequential[0] / parallel[0]
                comparison = (
                    f"{speedup:.2f}x speedup"
                    if speedup >= 1
                    else f"{1 / speedup:.2f}x regression"
                )
                print(
                    f"  threads={thread_count}: "
                    f"sequential {sequential[0] * 1000:.2f} ms "
                    f"(MAD {sequential[1] * 1000:.2f}); "
                    f"worker-local parallel {parallel[0] * 1000:.2f} ms "
                    f"(MAD {parallel[1] * 1000:.2f}, "
                    f"{actual / parallel[0]:,.0f}/sec) | {comparison}"
                )

        # Retain the previous highly unbalanced workloads as a dispatch check.
        for mode in (4, 5):
            players, board, actual = make_unbalanced_exact_inputs(mode)
            valid_players = prepare_ranges(players, board)
            hands = List([player.hands for player in valid_players])
            masks = List([player.masks for player in valid_players])
            weights = List()
            for player in valid_players:
                normalized = player.weights.astype(np.float64)
                normalized /= normalized.sum(dtype=np.float64)
                weights.append(normalized)
            board_mask = build_mask(board)

            def unbalanced_sequential(
                hands=hands,
                masks=masks,
                weights=weights,
                board=board,
                board_mask=board_mask,
                mode=mode,
            ) -> dict[str, float]:
                shares, total = _run_exact_equity_kernel(
                    hands, masks, weights, board, board_mask, mode
                )
                return {
                    f"player {index + 1}:": float(share / total)
                    for index, share in enumerate(shares)
                }

            def unbalanced_worker_local(
                hands=hands,
                masks=masks,
                weights=weights,
                board=board,
                board_mask=board_mask,
                mode=mode,
            ) -> dict[str, float]:
                shares, total = _run_exact_equity_kernel_parallel(
                    hands,
                    masks,
                    weights,
                    board,
                    board_mask,
                    mode,
                    numba.get_num_threads(),
                )
                return {
                    f"player {index + 1}:": float(share / total)
                    for index, share in enumerate(shares)
                }

            label = f"PLO{mode} | river | unbalanced 1x1,000"
            sequential_result = unbalanced_sequential()
            parallel_result = unbalanced_worker_local()
            if not np.allclose(
                list(parallel_result.values()),
                list(sequential_result.values()),
                rtol=0,
                atol=1e-12,
            ):
                raise AssertionError(f"Kernel mismatch in {label}")

            print(f"\n{label} | {actual:,} estimated and actual evaluations")
            for thread_count in thread_counts:
                numba.set_num_threads(thread_count)
                sequential = benchmark(
                    f"  sequential | threads={thread_count}",
                    unbalanced_sequential,
                    work_per_call=actual,
                    unit="actual showdowns",
                    repeats=repeats,
                    print_result=False,
                )
                parallel = benchmark(
                    f"  worker-local parallel | threads={thread_count}",
                    unbalanced_worker_local,
                    work_per_call=actual,
                    unit="actual showdowns",
                    repeats=repeats,
                    print_result=False,
                )
                speedup = sequential[0] / parallel[0]
                comparison = (
                    f"{speedup:.2f}x speedup"
                    if speedup >= 1
                    else f"{1 / speedup:.2f}x regression"
                )
                print(
                    f"  threads={thread_count}: "
                    f"sequential {sequential[0] * 1000:.2f} ms "
                    f"(MAD {sequential[1] * 1000:.2f}); "
                    f"worker-local parallel {parallel[0] * 1000:.2f} ms "
                    f"(MAD {parallel[1] * 1000:.2f}, "
                    f"{actual / parallel[0]:,.0f}/sec) | {comparison}"
                )
    finally:
        numba.set_num_threads(original_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("quick", "full"), default="quick")
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--core", action="store_true",
        help="Also benchmark the batched simulation coordinator without preparation",
    )
    parser.add_argument(
        "--exact-streets",
        action="store_true",
        help="Run only targeted exact-equity benchmarks across flop, turn, and river",
    )
    parser.add_argument(
        "--exact-parallel",
        action="store_true",
        help=(
            "Compare sequential and worker-local parallel exact kernels "
            "on targeted workloads"
        ),
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.samples <= 0 or args.repeats <= 0:
        parser.error("--samples and --repeats must be positive")
    if args.exact_streets and args.exact_parallel:
        parser.error("--exact-streets and --exact-parallel cannot be combined")
    if not 1 <= args.threads <= numba.config.NUMBA_NUM_THREADS:
        parser.error("Invalid Numba thread count")

    numba.set_num_threads(args.threads)
    print(
        f"Python {sys.version.split()[0]} | NumPy {np.__version__} | "
        f"Numba {numba.__version__} | threads {numba.get_num_threads()}\n"
        f"Platform: {platform.platform()} | "
        f"suite={args.suite}, samples={args.samples}, repeats={args.repeats}\n"
    )

    if args.exact_streets:
        run_exact_street_benchmarks(args.repeats)
        return

    if args.exact_parallel:
        run_exact_parallel_benchmarks(args.repeats)
        return

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
            # Match compute_equity()'s Numba argument types. Construct all
            # preparation arrays outside the timed simulation-core region.
            valid_players = prepare_ranges(players, board)
            probabilities = prepare_probabilities(valid_players)
            hands = List([p.hands for p in valid_players])
            masks = List([p.masks for p in valid_players])
            cumulative_arrays = []
            for p in probabilities:
                cdf = np.cumsum(p)
                cdf[-1] = 1.0  # Prevent out-of-bounds sampling from rounding.
                cumulative_arrays.append(cdf)
            cumulative = List(cumulative_arrays)
            board_mask = np.uint64(0) if board is None else build_mask(board)
            rng = np.random.default_rng(42)

            def core_call(
                hands=hands,
                masks=masks,
                cumulative=cumulative,
                board=board,
                board_mask=board_mask,
                mode=s.mode,
                samples=args.samples,
                rng=rng,
            ) -> dict[str, float]:
                # Reuse rng to exclude generator creation from core timing.
                # Its state advances, so repeated calls remain independent.
                result = _run_batched_monte_carlo(
                    hands,
                    masks,
                    cumulative,
                    mode,
                    board,
                    board_mask,
                    rng,
                    requested_samples=samples,
                    batch_size=10_000,
                    attempt_limit=max(10_000, samples * 20),
                    adaptive=False,
                    target_moe=0.0,
                    min_samples=samples,
                    confidence_multiplier=1.96,
                )
                return result.equities

            benchmark(
                s.label + " | simulation core only",
                core_call,
                work_per_call=args.samples,
                unit="valid simulations",
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

    print("=== EXACT: targeted actual showdown counts ===")
    for mode in (4, 5):
        for target_showdowns in (10_000, 100_000):
            for overlap in (False, True):
                s, players, board, estimated, actual = make_exact_target_inputs(
                    mode, target_showdowns, overlap
                )
                print(
                    f"{s.label}\n"
                    f"  Showdowns: {estimated:,} estimated candidate evaluations; "
                    f"{actual:,} actual collision-free evaluations"
                )
                benchmark(
                    s.label + f" | {actual:,} actual evaluations",
                    lambda players=players, board=board, s=s: run_exact_equity(
                        players,
                        mode=s.mode,
                        board=board,
                        max_showdowns=None,
                    ),
                    work_per_call=actual,
                    unit="actual showdowns",
                    repeats=args.repeats,
                )

if __name__ == "__main__":
    main()
