"""Parse card-based poker filters into immutable expression trees.

Operator precedence (highest to lowest):
    prefix !          NOT
    : and binary !    AND / AND NOT
    |                 OR

Parentheses override operator precedence. An atom is one or more card
requirements with no logical operators; for example, ``Ass`` requires
As and a *different* spade, whereas ``As:s`` has two independent conditions.
"""

import numpy as np

from plorangelab.encoder import (
    RANKS_ENCODE,
    SUITS_ENCODE,
    _encode_card,
    build_mask,
)
from plorangelab.types import Mode

from .models import And, AtomicFilter, CardRequirement, FilterExpression, Not, Or


def _validate_mode(mode: Mode) -> None:
    """Accept only the two supported Omaha variants."""
    if not isinstance(mode, int) or isinstance(mode, bool) or mode not in (4, 5):
        raise ValueError("mode must be 4 (PLO4) or 5 (PLO5)")


def _rank_mask(rank: str) -> int:
    """All four cards of a rank, using the shared card encoder."""
    cards = np.array(
        [_encode_card(rank + suit) for suit in SUITS_ENCODE],
        dtype=np.uint8,
    )
    return int(build_mask(cards))


def _suit_mask(suit: str) -> int:
    """All thirteen cards of a suit, using the shared card encoder."""
    cards = np.array(
        [_encode_card(rank + suit) for rank in RANKS_ENCODE],
        dtype=np.uint8,
    )
    return int(build_mask(cards))


def encode_atomic_filter(text: str, mode: Mode) -> AtomicFilter:
    """Encode a card atom, ensuring it fits the PLO4/PLO5 hand size."""
    _validate_mode(mode)
    if not text:
        raise ValueError("Atomic filter must not be empty")

    counts: dict[int, int] = {}
    i = 0
    while i < len(text):
        char = text[i]
        if char in RANKS_ENCODE:
            # An uppercase rank followed by a lowercase suit is an exact card.
            if i + 1 < len(text) and text[i + 1] in SUITS_ENCODE:
                mask = 1 << _encode_card(text[i : i + 2])
                i += 2
            else:
                mask = _rank_mask(char)
                i += 1
        elif char in SUITS_ENCODE:
            mask = _suit_mask(char)
            i += 1
        else:
            raise ValueError(f"Invalid character {char!r} in atomic filter {text!r}")

        counts[mask] = counts.get(mask, 0) + 1

    required_cards = sum(counts.values())
    if required_cards > mode:
        raise ValueError(
            f"Atomic filter {text!r} requires {required_cards} cards; "
            f"PLO{mode} allows at most {mode}"
        )

    # Reject impossible repeated exact-card / rank requirements.
    for mask, count in counts.items():
        if count > mask.bit_count():
            raise ValueError(f"Atomic filter {text!r} requires too many matching cards")

    return AtomicFilter(
        tuple(
            CardRequirement(mask=np.uint64(mask), count=count)
            for mask, count in counts.items()
        )
    )


class _Parser:
    def __init__(self, text: str, mode: Mode) -> None:
        _validate_mode(mode)
        self.text = text
        self.mode = mode
        self.pos = 0

    def _skip_spaces(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _peek(self) -> str | None:
        self._skip_spaces()
        return self.text[self.pos] if self.pos < len(self.text) else None

    def _accept(self, char: str) -> bool:
        if self._peek() == char:
            self.pos += 1
            return True
        return False

    def parse(self) -> FilterExpression:
        if self._peek() is None:
            raise ValueError("Filter expression must not be empty")
        result = self._parse_or()
        if self._peek() is not None:
            raise ValueError(
                f"Unexpected character at position {self.pos}: {self.text[self.pos]!r}"
            )
        return result

    def _parse_or(self) -> FilterExpression:
        left = self._parse_and()
        while self._accept("|"):
            left = Or(left, self._parse_and())
        return left

    def _parse_and(self) -> FilterExpression:
        left = self._parse_unary()
        while True:
            if self._accept(":"):
                left = And(left, self._parse_unary())
            elif self._accept("!"):
                # Binary ! means set difference: left AND NOT right.
                left = And(left, Not(self._parse_unary()))
            else:
                return left

    def _parse_unary(self) -> FilterExpression:
        if self._accept("!"):
            return Not(self._parse_unary())
        if self._accept("("):
            inner = self._parse_or()
            if not self._accept(")"):
                raise ValueError(f"Expected ')' at position {self.pos}")
            return inner

        self._skip_spaces()
        start = self.pos
        while (
            self.pos < len(self.text)
            and (
                self.text[self.pos] in RANKS_ENCODE
                or self.text[self.pos] in SUITS_ENCODE
            )
        ):
            self.pos += 1

        if self.pos == start:
            raise ValueError(f"Expected a card expression at position {self.pos}")
        return encode_atomic_filter(self.text[start : self.pos], self.mode)


def encode_filter(expression: str, mode: Mode) -> FilterExpression:
    """Parse a PLO4/PLO5 filter string into an immutable expression tree.

    Each atomic expression must require at most ``mode`` distinct cards.
    Examples: ``8!88``, ``(AsKK|AA):!ss``, and ``Asss|Ks``.
    """
    return _Parser(expression, mode).parse()
