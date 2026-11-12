"""Public interface for parsing, compiling, and evaluating PLO card filters.

Provides functions for converting filter expressions into immutable
syntax trees, compiling them into reusable instructions, and evaluating
them against encoded hand ranges.
"""

from .compiler import compile_filter
from .evaluator import evaluate_filter
from .parser import encode_filter

__all__ = ["compile_filter", "encode_filter", "evaluate_filter"]
