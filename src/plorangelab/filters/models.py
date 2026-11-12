"""Data models for card-based filter expressions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CardRequirement:
    """Represent a requirement for distinct cards from an eligible card set.

    Attributes:
        mask: uint64 bitmask identifying the cards that may satisfy
            the requirement.
        count: Number of distinct matching cards required.
    """

    mask: np.uint64
    count: int


@dataclass(frozen=True, slots=True)
class AtomicFilter:
    """Represent a conjunction of card requirements within one atom.

    All requirements must be satisfied using distinct hole cards.
    A single card cannot satisfy multiple requirements within the
    same atom.

    Distinct-card assignment applies only within an individual atom,
    not across separate atoms combined by logical operators.
    """

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
