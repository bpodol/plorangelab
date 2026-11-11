"""Public interface for PLO4/PLO5 equity calculations.

Exports the main calculation function, range sanitization utility,
result models, and sampling-specific exceptions.
"""

from .engine import compute_equity, sanitize_range
from .exceptions import SamplingExhaustedError
from .results import EquityResult, MonteCarloResult

__all__ = [
    "EquityResult",
    "MonteCarloResult",
    "SamplingExhaustedError",
    "compute_equity",
    "sanitize_range",
]
