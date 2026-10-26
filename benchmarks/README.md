# Equity benchmark notes

## Previous exact-kernel optimization

The worker-local parallel exact kernel was compared with the sequential exact
kernel on two-player, disjoint river ranges. Ranges were prepared before
timing; each kernel was warmed once to exclude Numba compilation, then timed
three times. The table reports median speedup on Windows 11 with Python 3.14.7,
NumPy 2.5.4, Numba 0.68.0, and eight Numba threads.

| Estimated showdowns | PLO4 speedup | PLO5 speedup |
| ---: | ---: | ---: |
| About 1K (992) | 3.30x | 3.35x |
| 10K | 3.34x | 4.00x |
| About 100K (99,856) | 3.41x | 3.63x |

The dispatch threshold is 1,000 estimated showdowns. Parallel work starts above
that point when there is more than one candidate matchup and multiple threads
are available. The threshold reflects the point where worker-local parallel
execution showed a benefit on the measured machine; it is hardware-dependent
and was benchmarked primarily with river workloads, where each matchup has
one board runout. Flop and turn workloads may have different crossover points.

## Current benchmark method

Run `uv run python benchmarks/benchmark_equity_numba.py`. The benchmark uses a
small deterministic set of PLO4/PLO5 flop, turn, river, disjoint, and
overlapping ranges. It measures the public `compute_equity()` API end to end,
with exact and Monte Carlo results in separate sections. A warm-up call checks
that each result sums to one and excludes first-call JIT compilation. Timings
report the median and median absolute deviation; inputs are built outside the
timed calls. Defaults are 10,000 Monte Carlo samples and three repetitions.
