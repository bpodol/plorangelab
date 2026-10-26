"""Benchmark public PLO range encoding, excluding input generation and JIT warmup.

From the repository root:
    uv run python benchmarks/benchmark_encoder.py
    uv run python benchmarks/benchmark_encoder.py --mode 5 --sizes 10000,100000,1000000
    uv run python benchmarks/benchmark_encoder.py --mode 5 --ranges-file benchmarks/real_ranges.txt
    uv run python benchmarks/benchmark_encoder.py --suite full --csv benchmarks/encoder_results.csv

All timings include string parsing, ASCII packing, encoding, duplicate checks,
mask construction, weights, and EncodedRange creation. Timings exclude synthetic
input generation, reading text files, explicit garbage collection and Numba JIT
compilation. This is a performance benchmark, not a substitute for pytest.
"""

from __future__ import annotations

import argparse
import csv
import gc
import platform
import random
import statistics
import sys
from dataclasses import dataclass
from itertools import combinations, islice
from math import comb
from pathlib import Path
from time import perf_counter

import numba
import numpy as np

from plorangelab.encoder import RANKS_DECODE, SUITS_DECODE, encode_range


@dataclass(frozen=True)
class Case:
    name: str
    mode: int
    hands: int
    text: str


def make_synthetic_range(count: int, mode: int, *, weighted: bool) -> str:
    """Build deterministic, unique explicit hands outside the timed section."""
    if count < 1 or count > comb(52, mode):
        raise ValueError(f"PLO{mode} supports 1..{comb(52, mode):,} distinct hands")

    card_strings = [
        rank + suit for rank in RANKS_DECODE for suit in SUITS_DECODE
    ]
    deck = list(range(52))
    random.Random(2026 + mode).shuffle(deck)
    weights = (25, 50, 75, 100)

    entries = []
    for i, card_ids in enumerate(islice(combinations(deck, mode), count)):
        hand = "".join(card_strings[card_id] for card_id in card_ids)
        if weighted:
            hand += f"@{weights[i % len(weights)]}"
        entries.append(hand)

    return ",".join(entries)


def load_real_cases(path: Path, mode: int) -> list[Case]:
    """Each nonempty, non-comment line holds one comma-separated range."""
    cases = []
    for line_num, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        cases.append(Case(f"{path.name}, line {line_num}", mode, text.count(",") + 1, text))

    if not cases:
        raise ValueError(f"No ranges found in {path}")
    return cases


def verify_result(result: object, case: Case) -> None:
    """Cheap structural checks only, outside the timed section."""
    hands = result.hands
    masks = result.masks
    weights = result.weights
    if hands.shape != (case.hands, case.mode):
        raise AssertionError(f"Unexpected hands shape: {hands.shape}")
    if masks.shape != (case.hands,) or weights.shape != (case.hands,):
        raise AssertionError("Unexpected masks/weights shape")
    if hands.dtype != np.uint8 or masks.dtype != np.uint64:
        raise AssertionError("Unexpected encoding dtypes")

    for i in {0, case.hands // 2, case.hands - 1}:
        expected = 0
        for card in hands[i]:
            expected |= 1 << int(card)
        if int(masks[i]) != expected:
            raise AssertionError(f"Incorrect bitmask at hand index {i}")


def benchmark_case(case: Case, repeats: int) -> dict[str, object]:
    """Report steady-state wall time for one complete encode_range() call."""
    # The small warmup in main() compiles the Numba kernel. Each case is then
    # checked once outside timing, rather than paying verification cost per run.
    result = encode_range(case.text, case.mode)
    verify_result(result, case)
    del result
    gc.collect()

    durations = []
    for _ in range(repeats):
        gc.collect()  # Keep prior-run garbage collection outside the timer.
        start = perf_counter()
        result = encode_range(case.text, case.mode)
        durations.append(perf_counter() - start)
        del result

    median = statistics.median(durations)
    mad = statistics.median(abs(t - median) for t in durations)
    print(
        f"{case.name} | PLO{case.mode} | {case.hands:,} hands\n"
        f"  median {median * 1000:,.2f} ms | MAD {mad * 1000:,.2f} ms | "
        f"min/max {min(durations) * 1000:,.2f}/{max(durations) * 1000:,.2f} ms\n"
        f"  throughput {case.hands / median:,.0f} hands/s"
    )
    return {
        "scenario": case.name,
        "mode": case.mode,
        "hands": case.hands,
        "repeats": repeats,
        "median_ms": round(median * 1000, 3),
        "mad_ms": round(mad * 1000, 3),
        "min_ms": round(min(durations) * 1000, 3),
        "max_ms": round(max(durations) * 1000, 3),
        "hands_per_second": round(case.hands / median),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=int, choices=(4, 5), default=5)
    parser.add_argument("--suite", choices=("quick", "full"), default="quick")
    parser.add_argument(
        "--sizes", help="Synthetic hand counts, e.g. 1000,10000,100000 (overrides suite)"
    )
    parser.add_argument("--weighted", action="store_true", help="Add @25/@50/@75/@100 weights")
    parser.add_argument("--ranges-file", type=Path, help="One comma-separated player range per line")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--csv", type=Path, help="Write results as CSV after benchmarking")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be positive")

    if args.ranges_file:
        cases = load_real_cases(args.ranges_file, args.mode)
    else:
        try:
            sizes = (
                [int(s.replace("_", "")) for s in args.sizes.split(",")]
                if args.sizes is not None
                else ([1_000, 10_000, 100_000] if args.suite == "quick"
                      else [1_000, 10_000, 100_000, 250_000, 1_000_000])
            )
            cases = [
                Case(f"synthetic{' weighted' if args.weighted else ''}", args.mode,
                     n, make_synthetic_range(n, args.mode, weighted=args.weighted))
                for n in sizes
                if n <= comb(52, args.mode)
            ]
            if not cases or any(n < 1 for n in sizes):
                raise ValueError("Need positive sizes compatible with the PLO mode")
        except ValueError as exc:
            parser.error(str(exc))
        skipped = [n for n in sizes if n > comb(52, args.mode)]
        if skipped:
            print(f"Skipped sizes exceeding PLO{args.mode} capacity: {skipped}")

    # Warming the public API also compiles any Numba encoder it invokes.
    warm_hand = "AsKdQcJh" if args.mode == 4 else "AsKdQcJhTs"
    encode_range(warm_hand, args.mode)

    print(
        f"Python {sys.version.split()[0]} | NumPy {np.__version__} | "
        f"Numba {numba.__version__} | {platform.platform()}\n"
        f"mode=PLO{args.mode}, repeats={args.repeats}; input generation, file IO, "
        "and JIT compilation excluded\n"
    )

    rows = [benchmark_case(case, args.repeats) for case in cases]

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV: {args.csv}")


if __name__ == "__main__":
    main()
