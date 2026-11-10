from dataclasses import dataclass
from typing import Literal

StopReason = Literal["target_reached", "max_simulations", "cancelled", "attempt_limit"]


@dataclass(frozen=True)
class EquityResult:
    """Equity values and, for Monte Carlo, stopping diagnostics."""

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
    """Equity estimates and stopping diagnostics for an adaptive run."""

    equities: list[float]
    margins_of_error: list[float | None]
    simulations: int
    attempts: int
    stop_reason: StopReason | None
    sampler: Literal["rejection", "importance"] = "rejection"
    effective_samples: float | None = None
