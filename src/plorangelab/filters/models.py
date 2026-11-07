"""Data models for card-based filter expressions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CardRequirement:
    """Require ``count`` distinct cards from the card set ``mask``."""

    mask: np.uint64
    count: int


@dataclass(frozen=True, slots=True)
class AtomicFilter:
    """All requirements within an atom must be satisfied by distinct cards."""

    requirements: tuple[CardRequirement, ...]


@dataclass(frozen=True, slots=True)
class And:
    left: FilterExpression
    right: FilterExpression


@dataclass(frozen=True, slots=True)
class Or:
    left: FilterExpression
    right: FilterExpression


@dataclass(frozen=True, slots=True)
class Not:
    child: FilterExpression


FilterExpression = AtomicFilter | And | Or | Not
