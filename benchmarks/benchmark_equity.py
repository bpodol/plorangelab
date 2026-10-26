"""Benchmark real-world-sized PLO4/PLO5 equity workloads through the public API.

Run from the repository root:

    uv run python benchmarks/benchmark_equity.py
    uv run python benchmarks/benchmark_equity.py --suite full
    uv run python benchmarks/benchmark_equity.py --suite full --production-precision
    uv run python benchmarks/benchmark_equity.py --suite stress --repeats 2
    uv run python benchmarks/benchmark_equity.py --csv benchmarks/results.csv
    uv run python benchmarks/benchmark_equity.py --only-ranges --ranges-file benchmarks/real_ranges.txt --range-mode 5 --range-board JhTd6c

All results include validation, blocker filtering, algorithm setup, and calculation.
Range generation and first-call Numba JIT compilation are excluded from timings.
"""

from __future__ import annotations

import argparse
import csv
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from math import comb, prod
from pathlib import Path
from typing import Literal

import numba
import numpy as np
from numba import njit

from plorangelab.encoder import EncodedRange, encode_board, encode_range
from plorangelab.equity import EquityResult, compute_equity
from plorangelab.equity.exact import _EXACT_MAX_SHOWDOWNS

Street = Literal[0, 3, 4, 5]
RangeProfile = Literal["broad", "tight"]
BenchmarkGroup = Literal["paired", "scale"]


@dataclass(frozen=True)
class Scenario:
    name: str
    mode: Literal[4, 5]
    board_cards: Street
    sizes: tuple[int, ...]
    profile: RangeProfile = "broad"
    group: BenchmarkGroup = "paired"

    @property
    def label(self) -> str:
        street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[self.board_cards]
        hands = "x".join(f"{count:,}" for count in self.sizes)
        return f"{self.name} | PLO{self.mode} {street} {len(self.sizes)}p {hands} ({self.profile})"


# Paired cases are small enough for exact, but span realistic hand counts and
# several streets. Both methods receive exactly the same hands and board.
QUICK_PAIRED = (
    Scenario("HU flop", 4, 3, (20, 20)),
    Scenario("HU turn", 4, 4, (100, 100)),
    Scenario("HU river", 4, 5, (600, 600)),
    Scenario("HU flop", 5, 3, (20, 20)),
    Scenario("HU river", 5, 5, (600, 600)),
    Scenario("Three-way river", 4, 5, (35, 35, 35)),
)

# Scale cases exercise work that would be unreasonably expensive to enumerate.
# Broad ranges intentionally include board blockers, like application inputs.
QUICK_SCALE = (
    Scenario("HU large preflop", 4, 0, (10_000, 10_000), group="scale"),
    Scenario("HU large flop", 5, 3, (20_000, 20_000), group="scale"),
    Scenario("HU shared-card pressure", 5, 5, (3_000, 3_000), "tight", "scale"),
    Scenario("Six-way collision pressure", 5, 5, (2_000,) * 6, group="scale"),
    Scenario("Six-way preflop", 5, 0, (2_000,) * 6, group="scale"),
)

FULL_PAIRED = (
    Scenario("HU concentrated flop", 4, 3, (20, 20), "tight"),
    Scenario("HU concentrated turn", 5, 4, (100, 100), "tight"),
    Scenario("HU turn", 5, 4, (110, 110)),
    Scenario("HU river", 5, 5, (700, 700)),
    Scenario("Unbalanced river", 4, 5, (1, 100_000)),
    Scenario("Three-way river", 5, 5, (40, 40, 40)),
)
FULL_SCALE = (
    Scenario("HU large preflop", 5, 0, (100_000, 100_000), group="scale"),
    Scenario("HU large flop", 5, 3, (100_000, 80_000), group="scale"),
    Scenario("HU large turn", 4, 4, (100_000, 60_000), group="scale"),
    Scenario("Three-way flop", 5, 3, (20_000,) * 3, group="scale"),
)
STRESS_SCALE = (
    Scenario("Million-hand preflop", 5, 0, (1_000_000, 600_000), group="scale"),
    Scenario("Large PLO4 preflop", 4, 0, (200_000, 200_000), group="scale"),
    Scenario("Six-way large river", 5, 5, (30_000,) * 6, group="scale"),
)


def _binomial_table() -> np.ndarray:
    table = np.zeros((53, 6), dtype=np.int64)
    for n in range(53):
        for k in range(min(n, 5) + 1):
            table[n, k] = comb(n, k)
    return table


BINOMIAL = _binomial_table()


@njit(cache=True)
def _unrank_combinations(
    ranks: np.ndarray, cards: np.ndarray, mode: int, binomial: np.ndarray
) -> np.ndarray:
    """Map unique integer combination IDs to unique hole-card hands efficiently."""
    rows = np.empty((len(ranks), mode), dtype=np.uint8)
    n = len(cards)
    for row in range(len(ranks)):
        remaining = int(ranks[row])
        start = 0
        for position in range(mode):
            for card_index in range(start, n):
                subset_count = binomial[n - card_index - 1, mode - position - 1]
                if remaining >= subset_count:
                    remaining -= subset_count
                else:
                    rows[row, position] = cards[card_index]
                    start = card_index + 1
                    break
    return rows


def _make_range(
    pool: np.ndarray,
    mode: int,
    count: int,
    anchor: np.ndarray,
    rng: np.random.Generator,
) -> EncodedRange:
    """Construct unique, nonuniformly weighted hands without Python tuples."""
    population = comb(len(pool), mode)
    if count > population:
        raise ValueError(f"Requested {count:,} hands, but pool has only {population:,}")

    # Unique ranks guarantee there are no duplicate combinations within a range.
    indices = rng.choice(population, size=count, replace=False).astype(np.int64)
    hands = _unrank_combinations(indices, pool, mode, BINOMIAL)

    # Guarantee at least one compatible multi-player matchup without making
    # the rest of the range artificially disjoint.
    hits = np.flatnonzero(np.all(hands == anchor, axis=1))
    if len(hits):
        offset = int(hits[0])
        hands[[0, offset]] = hands[[offset, 0]]
    else:
        hands[0] = anchor

    masks = np.zeros(count, dtype=np.uint64)
    for card_position in range(mode):
        masks |= np.uint64(1) << hands[:, card_position].astype(np.uint64)
    weights = rng.lognormal(mean=0.0, sigma=0.7, size=count).astype(np.float32)
    return EncodedRange(hands=hands, masks=masks, weights=weights)


def make_inputs(scenario: Scenario) -> tuple[list[EncodedRange], np.ndarray | None]:
    """Create realistic overlapping ranges and a fixed board outside timings."""
    import hashlib

    # Stable across machines and Python hash randomization.
    seed_data = (
        f"{scenario.mode}|{scenario.board_cards}|{scenario.sizes}|"
        f"{scenario.profile}|{scenario.name}"
    ).encode()
    seed = int.from_bytes(hashlib.blake2b(seed_data, digest_size=8).digest(), "little")
    rng = np.random.default_rng(seed)
    deck = rng.permutation(52).astype(np.uint8)
    board = deck[: scenario.board_cards].copy() if scenario.board_cards else None
    available = deck[scenario.board_cards :]
    if scenario.profile == "tight":
        if len(scenario.sizes) != 2:
            raise ValueError("Tight-pool scenarios currently require two players")
        pool = available[:20]  # High but not pathological hand-overlap pressure.
    else:
        pool = deck  # Includes board cards; public API must filter these.

    players = []
    for index, count in enumerate(scenario.sizes):
        anchor = available[index * scenario.mode : (index + 1) * scenario.mode]
        players.append(_make_range(pool, scenario.mode, count, anchor, rng))
    return players, board


def conservative_showdowns(
    ranges: list[EncodedRange], mode: int, board: np.ndarray | None
) -> tuple[int, int, tuple[int, ...]]:
    """Estimate eligible work after blockers, ignoring collisions between players."""
    board_count = 0 if board is None else len(board)
    if board is None:
        sizes = tuple(len(player.hands) for player in ranges)
    else:
        board_mask = np.uint64(0)
        for card in board:
            board_mask |= np.uint64(1) << np.uint64(card)
        sizes = tuple(int(np.count_nonzero((p.masks & board_mask) == 0)) for p in ranges)
    matchups = prod(sizes)
    runouts = comb(52 - board_count - len(ranges) * mode, 5 - board_count)
    return matchups, matchups * runouts, sizes


def actual_exact_evaluations(
    ranges: list[EncodedRange], mode: int, board: np.ndarray | None
) -> int | None:
    """Count legal exact showdowns for 2-3 player comparisons (outside timings)."""
    if len(ranges) not in (2, 3):
        return None
    board_mask = np.uint64(0)
    if board is not None:
        for card in board:
            board_mask |= np.uint64(1) << np.uint64(card)
    masks = [player.masks[(player.masks & board_mask) == 0] for player in ranges]
    if prod(len(row) for row in masks) > 1_000_000:
        return None  # Limit analysis allocations, not the production algorithm.

    combined = masks[0][:, None] | masks[1][None, :]
    compatible = (masks[0][:, None] & masks[1][None, :]) == 0
    if len(masks) == 3:
        compatible = compatible[:, :, None] & (
            (combined[:, :, None] & masks[2][None, None, :]) == 0
        )
    board_count = 0 if board is None else len(board)
    runouts = comb(52 - board_count - len(ranges) * mode, 5 - board_count)
    return int(np.count_nonzero(compatible)) * runouts


@dataclass(frozen=True)
class Timing:
    seconds: float
    mad: float
    result: EquityResult


def measure(call, repeats: int) -> Timing:
    """Measure warm-compiled end-to-end calls with identical seeded inputs."""
    first = call()  # Excludes compilation and first-call effects from timing.
    _verify_result(first)
    elapsed: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        elapsed.append(time.perf_counter() - start)
        _verify_result(result)
        if result != first:
            raise AssertionError("Seeded benchmark result changed between repetitions")
    median = statistics.median(elapsed)
    mad = statistics.median(abs(duration - median) for duration in elapsed)
    return Timing(median, mad, first)


def _verify_result(result: EquityResult) -> None:
    equity_total = sum(result.equities.values())
    if not np.isclose(equity_total, 1.0, atol=1e-8):
        raise AssertionError(f"Equities do not sum to 1: {result}")


def display_run(
    scenario: Scenario, timing: Timing, requested: str, estimate: int,
    exact_evaluations: int | None = None,
) -> dict:
    result = timing.result
    median_ms = timing.seconds * 1_000
    mad_ms = timing.mad * 1_000
    print(f"  {requested:<13} {median_ms:>9.2f} +/- {mad_ms:>7.2f} ms", end="")
    simulations = result.simulations
    attempts = result.attempts
    moe = result.margins_of_error
    if requested == "auto":
        print(f" -> {result.method}", end="")
    if result.method == "monte-carlo":
        max_moe = max((v for v in (moe or {}).values() if v is not None), default=None)
        sampler = result.sampler or "rejection"
        rate_name = "completion" if sampler == "importance" else "acceptance"

        details = (
            f" | {sampler} sampler"
            f" | {simulations:>8,} valid / {attempts:>9,} attempts"
            f" | {rate_name} {simulations / attempts:>8.4%}"
            f" | {simulations / timing.seconds:>10,.0f} valid/s"
        )

        if result.effective_samples is not None:
            details += (
                f" | ESS {result.effective_samples:,.0f}"
                f" | {result.effective_samples / timing.seconds:,.0f} ESS/s"
            )
        details += f" | MOE {max_moe:.5f}" if max_moe is not None else " | MOE n/a"
        print(f"{details} | {result.stop_reason}")
    elif exact_evaluations is None:
        print(f" | <= {estimate:,} showdowns (conservative bound)")
    else:
        print(
            f" | {exact_evaluations:,} legal showdowns"
            f" | {exact_evaluations / timing.seconds:,.0f} exact eval/s"
        )
    return {
        "scenario": scenario.name,
        "profile": scenario.profile,
        "mode": scenario.mode,
        "players": len(scenario.sizes),
        "board_cards": scenario.board_cards,
        "sizes": ":".join(str(n) for n in scenario.sizes),
        "method_requested": requested,
        "method_selected": result.method,
        "estimated_showdowns": estimate,
        "exact_evaluations": exact_evaluations if result.method == "exact" else None,
        "exact_evaluations_per_second": (
            exact_evaluations / timing.seconds
            if result.method == "exact" and exact_evaluations is not None else None
        ),
        "median_ms": round(median_ms, 4),
        "mad_ms": round(mad_ms, 4),
        "valid_simulations": simulations,
        "attempts": attempts,
        "sampler": result.sampler,
        "effective_samples": result.effective_samples,
        "effective_samples_per_second": (
            result.effective_samples / timing.seconds
            if result.effective_samples is not None
            else None
        ),
        "acceptance": (
            simulations / attempts
            if attempts and result.sampler != "importance"
            else None
        ),
        "completion_rate": (
            simulations / attempts
            if attempts and result.sampler == "importance"
            else None
        ),
        "valid_per_second": (simulations / timing.seconds if simulations else None),
        "max_moe": max((v for v in (moe or {}).values() if v is not None), default=None),
        "stop_reason": result.stop_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("quick", "full", "stress"), default="quick")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--target-moe", type=float, default=0.01)
    parser.add_argument("--min-samples", type=int, default=2_000)
    parser.add_argument("--max-samples", type=int, default=50_000)
    parser.add_argument("--fixed-samples", type=int, default=0,
                        help="Also benchmark fixed-budget MC (min=max=N) on paired cases")
    parser.add_argument("--production-precision", action="store_true",
                        help="Use real API MC defaults: MOE=.001, min=10k, max=1.2M")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--csv", type=Path, help="Write machine-readable results CSV")
    parser.add_argument("--ranges-file", type=Path,
                        help="Also benchmark one player range per line from a real range file")
    parser.add_argument("--only-ranges", action="store_true",
                        help="Skip synthetic scenarios and benchmark only --ranges-file")
    parser.add_argument("--range-mode", type=int, choices=(4, 5),
                        help="PLO4 or PLO5 for --ranges-file")
    parser.add_argument("--range-board", type=str, default=None,
                        help="Board for --ranges-file (e.g. 2c3d5h), default preflop")
    args = parser.parse_args()

    if args.only_ranges and args.ranges_file is None:
        parser.error("--only-ranges requires --ranges-file")
    if args.ranges_file is not None and args.range_mode is None:
        parser.error("--range-mode is required with --ranges-file")
    if args.production_precision:
        args.target_moe = 0.001
        args.min_samples = 10_000
        args.max_samples = 1_200_000
    if args.repeats < 1 or args.fixed_samples < 0:
        parser.error("--repeats must be positive; --fixed-samples must be nonnegative")
    if not (0 <= args.target_moe < float("inf")):
        parser.error("--target-moe must be finite and nonnegative")
    if not (1 <= args.min_samples <= args.max_samples <= 1_200_000):
        parser.error("Require 1 <= min-samples <= max-samples <= 1,200,000")
    if args.fixed_samples > 1_200_000:
        parser.error("--fixed-samples must not exceed 1,200,000")
    if args.threads is not None:
        if not 1 <= args.threads <= numba.config.NUMBA_NUM_THREADS:
            parser.error("--threads must be between 1 and available Numba threads")
        numba.set_num_threads(args.threads)

    paired = [] if args.only_ranges else list(QUICK_PAIRED)
    scale = [] if args.only_ranges else list(QUICK_SCALE)
    if not args.only_ranges and args.suite in ("full", "stress"):
        paired.extend(FULL_PAIRED)
        scale.extend(FULL_SCALE)
    if not args.only_ranges and args.suite == "stress":
        scale.extend(STRESS_SCALE)

    print(
        f"Python {sys.version.split()[0]} | NumPy {np.__version__} | "
        f"Numba {numba.__version__} | {platform.platform()}"
    )
    print(
        f"threads={numba.get_num_threads()} | suite={args.suite} | repeats={args.repeats} "
        f"| adaptive MOE={args.target_moe:g}, min={args.min_samples:,}, "
        f"max={args.max_samples:,}"
    )
    print("Timings include API validation, range filtering and calculation; exclude input generation and JIT warmup.")
    print("Synthetic weighted ranges sample distinct hands uniformly; they are NOT solver-derived strategy ranges.")
    print("Exact legal evaluation counts exclude colliding tuples; MC acceptance measures rejected collisions.\n")
    records = []

    if paired:
        print("=== PAIRED: EXACT vs ADAPTIVE MONTE CARLO (identical input ranges) ===")
    for scenario in paired:
        ranges, board = make_inputs(scenario)
        _, estimate, survivors = conservative_showdowns(ranges, scenario.mode, board)
        print(f"\n{scenario.label} | unblocked={survivors}")
        if estimate <= _EXACT_MAX_SHOWDOWNS:
            exact_call = lambda: compute_equity(ranges, scenario.mode, board=board, method="exact")
            exact_timing = measure(exact_call, args.repeats)
            exact_evaluations = actual_exact_evaluations(ranges, scenario.mode, board)
            records.append(display_run(scenario, exact_timing, "exact", estimate, exact_evaluations))
        else:
            print(f"  exact         skipped: {estimate:,} > {_EXACT_MAX_SHOWDOWNS:,} workload cap")
            exact_timing = None
        mc_call = lambda: compute_equity(
            ranges, scenario.mode, board=board, method="monte-carlo",
            target_moe=args.target_moe, min_samples=args.min_samples,
            max_samples=args.max_samples, seed=42,
        )
        mc_timing = measure(mc_call, args.repeats)
        records.append(display_run(scenario, mc_timing, "adaptive MC", estimate))
        if exact_timing is not None:
            diff = max(
                abs(mc_timing.result.equities[key] - exact_timing.result.equities[key])
                for key in exact_timing.result.equities
            )
            print(f"  equity difference vs exact: max absolute {diff:.5f}")
        if args.fixed_samples:
            fixed_call = lambda: compute_equity(
                ranges, scenario.mode, board=board, method="monte-carlo",
                target_moe=0.0, min_samples=args.fixed_samples,
                max_samples=args.fixed_samples, seed=42,
            )
            fixed_timing = measure(fixed_call, args.repeats)
            records.append(display_run(scenario, fixed_timing, "fixed MC", estimate))

    if scale:
        print("\n=== SCALE: LARGE-RANGE AUTO DISPATCH AND MONTE CARLO ===")
    for scenario in scale:
        ranges, board = make_inputs(scenario)
        _, estimate, survivors = conservative_showdowns(ranges, scenario.mode, board)
        print(f"\n{scenario.label} | unblocked={survivors} | work estimate={estimate:,}")
        call = lambda: compute_equity(
            ranges, scenario.mode, board=board, method="auto",
            target_moe=args.target_moe, min_samples=args.min_samples,
            max_samples=args.max_samples, seed=42,
        )
        try:
            timing = measure(call, args.repeats)
        except RuntimeError as exc:
            # Multiway sampling can exhaust its attempt budget before finding
            # even one collision-free tuple. This is a meaningful result.
            if "Sampling limit exceeded" not in str(exc):
                raise
            print(f"  AUTO FAILED: {exc}")
            records.append({
                "scenario": scenario.name, "mode": scenario.mode,
                "players": len(scenario.sizes), "board_cards": scenario.board_cards,
                "sizes": ":".join(map(str, scenario.sizes)), "profile": scenario.profile,
                "method_requested": "auto", "method_selected": "monte-carlo",
                "estimated_showdowns": estimate, "stop_reason": "zero_valid_samples",
            })
            continue
        records.append(display_run(scenario, timing, "auto", estimate))


    if args.ranges_file is not None:
        # Use actual hand frequencies and collisions rather than relying on
        # synthetic uniform-within-pool hand selection.
        lines = [
            line.strip()
            for line in args.ranges_file.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not 2 <= len(lines) <= 6:
            parser.error("--ranges-file must contain 2 to 6 nonempty player ranges")
        ranges = [encode_range(line, args.range_mode) for line in lines]
        board = encode_board(args.range_board) if args.range_board else None
        board_cards = 0 if board is None else len(board)
        scenario = Scenario(
            args.ranges_file.name, args.range_mode, board_cards,
            tuple(len(r.hands) for r in ranges), group="scale",
        )
        _, estimate, survivors = conservative_showdowns(ranges, scenario.mode, board)
        print(f"\n=== ACTUAL RANGES: {scenario.label} | unblocked={survivors} ===")
        if estimate <= _EXACT_MAX_SHOWDOWNS:
            exact_call = lambda: compute_equity(ranges, scenario.mode, board=board, method="exact")
            exact_timing = measure(exact_call, args.repeats)
            exact_evaluations = actual_exact_evaluations(ranges, scenario.mode, board)
            records.append(display_run(scenario, exact_timing, "exact", estimate, exact_evaluations))
        else:
            exact_timing = None
            print(f"  exact skipped: {estimate:,} > {_EXACT_MAX_SHOWDOWNS:,} workload cap")
        mc_call = lambda: compute_equity(
            ranges, scenario.mode, board=board, method="monte-carlo",
            target_moe=args.target_moe, min_samples=args.min_samples,
            max_samples=args.max_samples, seed=42,
        )
        try:
            mc_timing = measure(mc_call, args.repeats)
        except RuntimeError as exc:
            if "Sampling limit exceeded" not in str(exc):
                raise
            print(f"  Monte Carlo failed: {exc}")
        else:
            records.append(display_run(scenario, mc_timing, "adaptive MC", estimate))
            if exact_timing is not None:
                diff = max(
                    abs(mc_timing.result.equities[k] - exact_timing.result.equities[k])
                    for k in exact_timing.result.equities
                )
                print(f"  equity difference vs exact: max absolute {diff:.5f}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "scenario", "profile", "mode", "players", "board_cards", "sizes",
            "method_requested", "method_selected", "estimated_showdowns",
            "exact_evaluations", "exact_evaluations_per_second",
            "median_ms", "mad_ms", "valid_simulations", "attempts",
            "acceptance", "valid_per_second", "max_moe", "stop_reason","completion_rate",
            "sampler", "effective_samples", "effective_samples_per_second",
        ]
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        print(f"\nSaved {len(records)} result rows to {args.csv}")


if __name__ == "__main__":
    main()
