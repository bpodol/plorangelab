"""Result models and stopping diagnostics for equity calculations."""

from dataclasses import dataclass
from typing import Literal

StopReason = Literal["target_reached", "max_simulations", "cancelled", "attempt_limit"]


@dataclass(frozen=True)
class EquityResult:
    """Result of an exact or Monte Carlo equity calculation.

    Equity values represent expected pot shares, including split pots,
    and are keyed by the original, zero-based player slot IDs.

    Monte Carlo diagnostics are unavailable for exact calculations.

    Attributes:
        equities: Expected pot shares from 0 to 1, keyed by player slot ID.
        method: Calculation method used.
        margins_of_error: Approximate Monte Carlo margins of error keyed by
            player slot ID, or None for exact calculations.
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

    equities: dict[int, float]
    method: Literal["exact", "monte-carlo"]
    margins_of_error: dict[int, float | None] | None
    simulations: int | None
    attempts: int | None
    stop_reason: StopReason | None
    sampler: Literal["rejection", "importance"] | None = None
    effective_samples: float | None = None
