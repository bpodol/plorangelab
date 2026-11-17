from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from plorangelab.encoder import RANKS_DECODE, RANKS_ENCODE


@dataclass
class FilterNode:
    name: str
    expression: str


@dataclass
class GroupNode:
    name: str
    filters: list[FilterNode | GroupNode] | None = None


def _rank_order(ranks: Counter[str]) -> list[str]:
    """Distinct board ranks, strongest first."""
    return [rank for rank in reversed(RANKS_DECODE) if rank in ranks]


def _can_hold(token: str, board_ranks: Counter[str]) -> bool:
    """Check whether the required hole-card ranks are available in the deck."""
    return all(board_ranks[rank] + count <= 4 for rank, count in Counter(token).items())


def _contains_token(token: str, other: str) -> bool:
    """Whether requiring `token` also necessarily requires `other`."""
    available = Counter(token)
    return all(available[rank] >= count for rank, count in Counter(other).items())


def _add_patterns(
    result: list[FilterNode | GroupNode],
    higher: list[str],
    category: str,
    tokens: list[str],
) -> None:
    """Append a group of mutually exclusive rank filters.

    Tokens are processed in descending priority order. Each expression
    excludes previously accepted tokens to prevent overlapping matches.

    The ``higher`` list is updated in place and may be shared across
    categories to maintain their exclusion priority.

    Empty groups are omitted.
    """
    nodes: list[FilterNode] = []
    for token in tokens:
        if token in higher or any(_contains_token(token, prior) for prior in higher):
            # E.g. on AAAK, AK necessarily makes quads, not a full house.
            continue

        expression = token + "".join(f"!{prior}" for prior in higher)
        nodes.append(FilterNode(token, expression))
        higher.append(token)

    if nodes:
        result.append(GroupNode(category, nodes))


def _quad_patterns(board_ranks: Counter[str], ordered: list[str]) -> list[str]:
    patterns: list[str] = []
    for rank in ordered:
        count = board_ranks[rank]
        if count == 3:
            # Exactly three board cards plus one hole card of this rank.
            patterns.append(rank)
        elif count == 2:
            # Two board cards and two hole cards of this rank.
            patterns.append(rank * 2)
        # Four on the board cannot make quads: Omaha uses only three board cards.
    return patterns


def _full_house_patterns(board_ranks: Counter[str], ordered: list[str]) -> list[str]:
    """Build two-hole-rank templates from 3+0 and 2+1 board patterns.

    Store the best (trips rank, pair rank) for each template so that a
    turn/river board offering several sets of three board cards still
    produces one expression per hole-rank requirement.
    """
    candidates: dict[tuple[str, ...], tuple[str, tuple[int, int]]] = {}

    def add(token: str, trips: str, pair: str) -> None:
        if not _can_hold(token, board_ranks):
            return
        key = tuple(sorted(token, key=RANKS_ENCODE.__getitem__, reverse=True))
        strength = (RANKS_ENCODE[trips], RANKS_ENCODE[pair])
        previous = candidates.get(key)
        if previous is None:
            candidates[key] = (token, strength)
        elif strength > previous[1]:
            # Preserve the original spelling, e.g. KA on a KK/A board.
            candidates[key] = (previous[0], strength)

    for paired in ordered:
        if board_ranks[paired] < 2:
            continue

        # Three board cards of one rank, plus a pocket pair of another.
        if board_ranks[paired] >= 3:
            for other in reversed(RANKS_DECODE):
                if other != paired:
                    add(other * 2, paired, other)

        # Two board cards of one rank and one of another:
        #   pp + s, hole ps -> pppss
        #   pp + s, hole ss -> ssspp
        for single in ordered:
            if single == paired:
                continue
            add(paired + single, paired, single)
            add(single * 2, single, paired)

    return [
        token
        for token, _ in sorted(
            candidates.values(), key=lambda entry: entry[1], reverse=True
        )
    ]


def _set_patterns(ordered: list[str]) -> list[str]:
    return [rank * 2 for rank in ordered]


def _two_pair_patterns(ordered: list[str]) -> list[str]:
    return [first + second for first, second in combinations(ordered, 2)]


def _overpair_patterns(board_ranks: Counter[str], ordered: list[str]) -> list[str]:
    if not ordered or any(count >= 3 for count in board_ranks.values()):
        return []
    top = RANKS_ENCODE[ordered[0]]
    return [
        rank * 2
        for rank in reversed(RANKS_DECODE)
        if rank not in board_ranks and RANKS_ENCODE[rank] > top
    ]


def generate_filters(board: NDArray[np.uint8]) -> list[GroupNode]:
    """Generate rank-based filter groups for the given board.

    Generate expressions for rank-based made-hand categories, including
    quads, full houses, sets, two pairs, trips, overpairs, and board pairs.

    Expressions are ordered by priority, with exclusions preventing
    overlapping matches across categories. Overpairs are independent
    of other rank categories.

    The generated patterns follow Omaha's requirement to use exactly
    two hole cards and three board cards.

    Only expressions are generated; no hands are evaluated, and no
    filters are parsed, compiled, or executed. Straight and flush
    categories are not supported.

    Args:
        board: Encoded flop, turn, or river containing 3, 4, or 5 cards.
            Cards are expected to be valid and unique.

    Returns:
        Non-empty groups containing generated filter expressions.

    Raises:
        ValueError: If the board contains an unsupported number of cards.
    """
    if len(board) not in (3, 4, 5):
        raise ValueError("board must contain 3, 4, or 5 cards")

    board_ranks = Counter(RANKS_DECODE[int(card) >> 2] for card in board)
    ordered = _rank_order(board_ranks)
    has_trips = any(count >= 3 for count in board_ranks.values())
    is_unpaired = len(board_ranks) == len(board)

    filters: list[GroupNode] = []
    higher: list[str] = []

    _add_patterns(filters, higher, "quad", _quad_patterns(board_ranks, ordered))
    _add_patterns(
        filters, higher, "full house", _full_house_patterns(board_ranks, ordered)
    )

    if is_unpaired:
        _add_patterns(filters, higher, "set", _set_patterns(ordered))
        _add_patterns(filters, higher, "two pair", _two_pair_patterns(ordered))
    else:
        # A paired board + one matching hole card makes trips. If the
        # board already has trips of a higher rank, this is not an upgrade.
        best_board_trips = max(
            (RANKS_ENCODE[rank] for rank in ordered if board_ranks[rank] >= 3),
            default=-1,
        )
        trips = [
            rank
            for rank in ordered
            if board_ranks[rank] == 2 and RANKS_ENCODE[rank] > best_board_trips
        ]
        _add_patterns(filters, higher, "trips", trips)

    _add_patterns(filters, [], "overpair", _overpair_patterns(board_ranks, ordered))

    if not has_trips:
        # On paired boards, a singleton board rank produces two pair with
        # the board pair. On unpaired boards it produces one pair.
        singles = [rank for rank in ordered if is_unpaired or board_ranks[rank] == 1]
        _add_patterns(filters, higher, "board pair", singles)

    return filters
