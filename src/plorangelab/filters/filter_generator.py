from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from plorangelab.encoder import RANKS_DECODE, RANKS_ENCODE, SUITS_DECODE


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
    """Check rank availability in a real deck (not a player hand)."""
    return all(board_ranks[rank] + count <= 4 for rank, count in Counter(token).items())


def _contains_token(token: str, other: str) -> bool:
    """Whether requiring `token` also necessarily requires `other`."""
    available = Counter(token)
    return all(available[rank] >= count for rank, count in Counter(other).items())


def _add_patterns(
    result: list[GroupNode],
    higher: list[str],
    category: str,
    tokens: list[str],
) -> None:
    """Add mutually exclusive rank expressions in descending priority order.

    A hand may contain multiple qualifying patterns (particularly in PLO5).
    Earlier patterns win: later expressions exclude them with `!` tokens.
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


# All ten straights, strongest first. Ace is low only in the wheel window.
# Each window is a set of five distinct ranks (suits are irrelevant here).
STRAIGHT_WINDOWS = (
    "AKQJT",
    "KQJT9",
    "QJT98",
    "JT987",
    "T9876",
    "98765",
    "87654",
    "76543",
    "65432",
    "5432A",
)


def _straight_patterns(
    board_ranks: Counter[str],
) -> list[tuple[str, list[str]]]:
    """Group possible two-hole-rank tokens by their best straight strength.

    For each five-rank straight, select three distinct ranks from the board;
    the remaining two must come from the hole cards (Omaha's 3+2 rule).
    Account for rank blockers, and retain only the strongest straight window
    when the same two-card token can complete more than one straight.
    """
    best_by_token: dict[str, int] = {}

    for strength, window in enumerate(STRAIGHT_WINDOWS):
        for board_three in combinations(window, 3):
            if not all(rank in board_ranks for rank in board_three):
                continue

            token = "".join(rank for rank in window if rank not in board_three)
            token = "".join(sorted(token, key=RANKS_ENCODE.__getitem__, reverse=True))
            if _can_hold(token, board_ranks):
                # Windows are visited strongest first, so the first wins.
                best_by_token.setdefault(token, strength)

    strengths: list[tuple[str, list[str]]] = []
    for strength, window in enumerate(STRAIGHT_WINDOWS):
        tokens = sorted(
            (token for token, best in best_by_token.items() if best == strength),
            key=lambda token: tuple(RANKS_ENCODE[rank] for rank in token),
            reverse=True,
        )
        if tokens:
            strengths.append((f"{window[0]}-high", tokens))

    return strengths


def _add_straight_group(
    result: list[GroupNode],
    higher: list[str],
    board_ranks: Counter[str],
) -> None:
    """Add one OR-combined filter per achievable straight strength.

    Each straight strength excludes previously generated stronger straight
    patterns. Equal-strength requirements are combined using OR without
    excluding one another.

    The exclusion list is independent of rank-based and suited categories.
    """
    nodes: list[FilterNode] = []

    for name, tokens in _straight_patterns(board_ranks):
        accepted = [
            token
            for token in tokens
            if not any(_contains_token(token, prior) for prior in higher)
        ]
        if not accepted:
            continue

        exclusions = "".join(f"!{prior}" for prior in higher)
        expression = "|".join(token + exclusions for token in accepted)
        nodes.append(FilterNode(name, expression))
        higher.extend(accepted)

    if nodes:
        result.append(GroupNode("straight", nodes))


SUITED_RANK_NAMES = ("nut", "2nd nut", "3rd nut", "low")
DOUBLE_DRAW_NAMES = ("nut+nut", "nut+noNut", "noNut+noNut")


def _suits_by_board_strength(board: NDArray[np.uint8]) -> list[str]:
    """Distinct board suits ordered by their highest board card.

    Sorting the encoded cards keeps OR-branch ordering independent of the
    original board-card order.
    """
    return list(
        dict.fromkeys(
            SUITS_DECODE[card & 3]
            for card in sorted((int(card) for card in board), reverse=True)
        )
    )


def _available_suited_ranks(board: NDArray[np.uint8], suit: str) -> list[str]:
    """Highest three ranks still available in ``suit`` after board blockers."""
    blocked = {
        RANKS_DECODE[int(card) >> 2]
        for card in board
        if SUITS_DECODE[int(card) & 3] == suit
    }
    return [rank for rank in reversed(RANKS_DECODE) if rank not in blocked][:3]


def _single_suit_expressions(suit: str, ranks: list[str]) -> list[str]:
    """Generate nut, second-nut, third-nut, and low suited expressions."""
    base = suit * 2
    expressions: list[str] = []
    excluded: list[str] = []

    for rank in ranks:
        expressions.append(f"{rank}{base}" + "".join(f"!{card}" for card in excluded))
        excluded.append(f"{rank}{suit}")

    expressions.append(base + "".join(f"!{card}" for card in excluded))
    return expressions


def _add_suited_group(
    result: list[GroupNode],
    category: str,
    suits: list[str],
    available: dict[str, list[str]],
) -> None:
    """Generate suited-strength filters, excluding double draws from single draws.

    For FD and BDFD, exclude hands containing two cards of another
    eligible suit. This keeps single and double draw categories disjoint.
    """

    if not suits:
        return

    expressions_by_suit = []

    for suit in suits:
        expressions = _single_suit_expressions(suit, available[suit])

        if category in ("fd", "bdfd"):
            exclusions = "".join(f"!{other * 2}" for other in suits if other != suit)
            expressions = [expr + exclusions for expr in expressions]

        expressions_by_suit.append(expressions)

    result.append(
        GroupNode(
            category,
            [
                FilterNode(
                    name, "|".join(parts[index] for parts in expressions_by_suit)
                )
                for index, name in enumerate(SUITED_RANK_NAMES)
            ],
        )
    )


def _add_double_draw_group(
    result: list[GroupNode | FilterNode],
    category: str,
    suits: list[str],
    available: dict[str, list[str]],
) -> None:
    """Generate double-draw patterns for all pairs of eligible suits.

    Two cards of each suit are required, with branches partitioned by
    whether each suit contains its highest available card.

    Double-draw groups are mutually exclusive with their corresponding
    single-draw groups.
    """
    if len(suits) < 2:
        return

    branches: dict[str, list[str]] = {name: [] for name in DOUBLE_DRAW_NAMES}
    for first, second in combinations(suits, 2):
        first_base = first * 2
        second_base = second * 2
        first_nut = f"{available[first][0]}{first}"
        second_nut = f"{available[second][0]}{second}"

        branches["nut+nut"].append(f"{first_nut}{first}{second_nut}{second}")
        branches["nut+noNut"].extend(
            (
                f"{first_nut}{first}{second_base}!{second_nut}",
                f"{first_base}{second_nut}{second}!{first_nut}",
            )
        )
        branches["noNut+noNut"].append(
            f"{first_base}{second_base}!{first_nut}!{second_nut}"
        )

    result.append(
        GroupNode(
            category,
            [FilterNode(name, "|".join(branches[name])) for name in DOUBLE_DRAW_NAMES],
        )
    )


def generate_filters(board: NDArray[np.uint8]) -> list[GroupNode]:
    """Generate rank-based, straight, and suited filters for a flop, turn, or river.

    Rank categories share priority exclusions, while straights and suited
    categories are independent. Straight filters exclude stronger straight
    patterns only.

    The two-hole-card/three-board-card Omaha rule determines straight, flush,
    and flush-draw eligibility. Only expressions are generated; no hands are
    evaluated and no filters are parsed, compiled, or executed. Straight-draw
    generation is not yet supported.

    The caller must supply a validated board of unique encoded cards.
    """
    if len(board) not in (3, 4, 5):
        raise ValueError("board must contain 3, 4, or 5 cards")

    board_ranks = Counter(RANKS_DECODE[int(card) >> 2] for card in board)
    ordered = _rank_order(board_ranks)
    has_trips = any(count >= 3 for count in board_ranks.values())
    is_unpaired = len(board_ranks) == len(board)

    filters: list[GroupNode] = []
    higher: list[str] = []

    suit_counts = Counter(SUITS_DECODE[int(card) & 3] for card in board)
    ordered_suits = _suits_by_board_strength(board)
    available = {suit: _available_suited_ranks(board, suit) for suit in ordered_suits}

    flush_suits = [suit for suit in ordered_suits if suit_counts[suit] >= 3]
    fd_suits = (
        [suit for suit in ordered_suits if suit_counts[suit] == 2]
        if len(board) < 5
        else []
    )
    bdfd_suits = (
        [suit for suit in ordered_suits if suit_counts[suit] == 1]
        if len(board) == 3
        else []
    )

    _add_patterns(filters, higher, "quad", _quad_patterns(board_ranks, ordered))
    _add_patterns(
        filters, higher, "full house", _full_house_patterns(board_ranks, ordered)
    )

    _add_suited_group(filters, "flush", flush_suits, available)
    _add_straight_group(filters, [], board_ranks)

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

    _add_double_draw_group(filters, "2fd", fd_suits, available)
    _add_double_draw_group(filters, "2bdfd", bdfd_suits, available)
    _add_suited_group(filters, "fd", fd_suits, available)
    _add_suited_group(filters, "bdfd", bdfd_suits, available)

    return filters
