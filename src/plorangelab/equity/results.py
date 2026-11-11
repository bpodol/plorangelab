"""Result models and stopping diagnostics for equity calculations."""

from dataclasses import dataclass
from typing import Literal

StopReason = Literal["target_reached", "max_simulations", "cancelled", "attempt_limit"]


@dataclass(frozen=True)
class EquityResult:
    """Result of an exact or Monte Carlo equity calculation.

    Equity values represent expected pot shares, including split pots,
    and are aligned with the input player order.

    Monte Carlo diagnostics are unavailable for exact calculations.

    Attributes:
        equities: Per-player equity values as fractions from 0 to 1.
        method: Calculation method used.
        margins_of_error: Approximate per-player Monte Carlo margins of
            error, or None for exact calculations.
        simulations: Number of completed Monte Carlo showdowns, or None
            for exact calculations.
        attempts: Number of sampling attempts, including rejected ones,
            or None for exact calculations.
        stop_reason: Reason the Monte Carlo calculation stopped, or None
            for exact calculations.
        sampler: Monte Carlo sampling strategy, or None for exact calculations.
        effective_samples: Effective sample size for importance sampling,
            or None when not applicable.
    """

    equities: list[float]
    method: Literal["exact", "monte-carlo"]
    margins_of_error: list[float | None] | None
    simulations: int | None
    attempts: int | None
    stop_reason: StopReason | None
    sampler: Literal["rejection", "importance"] | None = None
    effective_samples: float | None = None


@dataclass(frozen=True)
class MonteCarloResult:
    """Result or progress snapshot from an adaptive Monte Carlo calculation.

    Attributes:
        equities: Per-player estimated pot shares as fractions from 0 to 1.
        margins_of_error: Approximate per-player margins of error.
            Individual values may be None when insufficient samples exist.
        simulations: Number of completed valid showdowns.
        attempts: Number of sampling attempts, including rejected ones.
        stop_reason: Reason for stopping, or None for intermediate results.
        sampler: Sampling strategy, either "rejection" or "importance".
        effective_samples: Effective sample size for importance sampling,
            or None for rejection sampling.
    """

    equities: list[float]
    margins_of_error: list[float | None]
    simulations: int
    attempts: int
    stop_reason: StopReason | None
    sampler: Literal["rejection", "importance"] = "rejection"
    effective_samples: float | None = None
