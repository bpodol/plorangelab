# PLO RangeLab

**A Python equity engine for Pot-Limit Omaha (PLO4 and PLO5).**

This repository is an engineering-focused rebuild of an existing poker-analysis desktop application. The current milestone is a standalone, tested equity engine; the user interface and broader range-analysis features are not yet part of this rewrite.

## Features

- **2–6 players**, with PLO4 and PLO5 hole-card rules.
- **Weighted ranges** of explicitly specified hands (for example, `AsAhKdQc@75`).
- **Preflop, flop, turn, and river** equity calculations.
- **Exact enumeration** of compatible hands and board runouts, or **adaptive Monte Carlo** simulation.
- Automatic method selection, card-blocker handling, split pots, seeded reproducibility, and Monte Carlo cancellation.
- NumPy arrays, 64-bit card masks, and Numba-compiled evaluation kernels, including parallel exact enumeration.

## Quick start

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

```python
from plorangelab.encoder import encode_board, encode_range
from plorangelab.equity import compute_equity

players = [
    encode_range("AsAhKdQc@75,QsQhJdTd@25", mode=4),
    encode_range("KsKhJcTc", mode=4),
]
board = encode_board("2c3d5h7s9c")

result = compute_equity(players, mode=4, board=board)
print(result.method)    # "exact" for this small river workload
print(result.equities)  # Per-player pot equity, from 0.0 to 1.0
```

`@75` assigns a hand a 75% relative weight; omitted weights default to 100%. Use `mode=5` for PLO5. Omit `board` for preflop calculations.

### Calculation methods

| Method | Behavior |
| --- | --- |
| `auto` (default) | Monte Carlo preflop; exact postflop for an estimated ≤500,000 showdowns; otherwise Monte Carlo |
| `exact` | Enumerates compatible hand selections and remaining board cards, subject to the workload limit |
| `monte-carlo` | Samples weighted, non-colliding hands and board runouts until the precision target or a resource limit is reached |

Monte Carlo can be configured with `target_moe`, `min_samples`, `max_samples`, `batch_size`, and `seed`. Its result also includes per-player estimated margins of error, simulation/attempt counts, and the stopping reason. These margins are approximate, not guaranteed confidence bounds. Exact results have no sampling diagnostics.

## Architecture

```text
src/plorangelab/
├── encoder.py              # Shared card, board, and weighted-range encoding
└── equity/
    ├── __init__.py         # Public compute_equity and EquityResult exports
    ├── api.py              # Validation, preparation, and algorithm selection
    ├── exact.py            # Sequential/parallel exhaustive enumeration
    ├── monte_carlo.py      # Adaptive sampling and uncertainty estimates
    └── evaluator.py        # Numba-compiled Omaha hand ranking

tests/
├── equity/                 # API, exact, Monte Carlo, and evaluator tests
└── test_encoder.py
```

Both algorithms use the same evaluator, which enforces Omaha's **exactly two hole cards and three board cards** rule. The public API keeps selection and validation separate from the computational kernels.

## Tests and benchmarks

```bash
uv run pytest
uv run ruff check .
uv run python benchmarks/benchmark_equity_numba.py
uv run python benchmarks/benchmarks_evaluator.py
```

Benchmarks warm up Numba before timing to exclude initial JIT compilation. The automatic-selection policy is informed by measured exact-vs-Monte-Carlo performance; see [benchmark findings](benchmarks/exact_vs_mc_findings.md) for the methodology and results.

## Status

The equity engine and its modular test suite are implemented. This is **not yet a complete desktop poker-analysis application**; the interface and additional range-analysis tools will be developed in subsequent milestones.
