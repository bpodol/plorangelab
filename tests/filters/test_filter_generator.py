"""Unit tests for automatic rank, straight, and suited filter generation.

These tests inspect the generated tree and strings directly. They deliberately
avoid the filter parser, hand evaluator, and range/coverage calculations.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pytest
from numpy.typing import NDArray

from plorangelab.encoder import RANKS_DECODE, SUITS_DECODE, encode_board
from plorangelab.filters.filter_generator import (
    STRAIGHT_WINDOWS,
    FilterNode,
    GroupNode,
    _straight_patterns,
    generate_filters,
)


def _board(ranks: str, *, suit_offset: int = 0) -> NDArray[np.uint8]:
    """Build a valid board from rank-only notation using the production encoder.

    Assign different suits to repeated ranks; rotate suits when requested.
    """
    used_suits: dict[str, int] = {}
    cards: list[str] = []

    for rank in ranks:
        index = used_suits.get(rank, 0)
        if index >= len(SUITS_DECODE):
            raise ValueError(f"Too many cards of rank {rank!r}")

        cards.append(rank + SUITS_DECODE[(index + suit_offset) % len(SUITS_DECODE)])
        used_suits[rank] = index + 1

    return encode_board("".join(cards))


SUITED_CATEGORIES = {"flush", "fd", "bdfd", "2fd", "2bdfd"}


def _all_groups(board: NDArray[np.uint8]) -> dict[str, list[FilterNode | GroupNode]]:
    result: dict[str, list[FilterNode | GroupNode]] = {}
    for node in generate_filters(board):
        assert isinstance(node, GroupNode), "Every category must be a group"
        assert node.name not in result, "Category names must be unique"
        assert node.filters, "Empty categories must not be returned"
        assert all(isinstance(child, FilterNode) for child in node.filters)
        result[node.name] = node.filters
    return result


def _rank_groups(board: NDArray[np.uint8]) -> dict[str, list[FilterNode]]:
    groups: dict[str, list[FilterNode]] = {}
    for name, nodes in _all_groups(board).items():
        if name not in SUITED_CATEGORIES | {"straight"}:
            assert all(isinstance(node, FilterNode) for node in nodes)
            groups[name] = nodes  # type: ignore[assignment]
    return groups


def _straight_groups(board: NDArray[np.uint8]) -> dict[str, FilterNode]:
    """Return the single OR-combined filter for each straight strength."""
    children = _all_groups(board).get("straight", [])
    assert all(isinstance(child, FilterNode) for child in children)
    return {child.name: child for child in children}


def _required_straight_tokens(node: FilterNode) -> list[str]:
    """Extract each required hole-card pair without parsing expressions."""
    return [branch.split("!", 1)[0] for branch in node.expression.split("|")]


def _groups(ranks: str) -> dict[str, list[FilterNode]]:
    """Preserve the existing rank-only assertions independently of suits."""
    return _rank_groups(_board(ranks))


def _names(nodes: list[FilterNode]) -> list[str]:
    return [node.name for node in nodes]


def _expressions(nodes: list[FilterNode]) -> list[str]:
    return [node.expression for node in nodes]


# All possible rank-multiplicity partitions for valid flops, turns and rivers.
@pytest.mark.parametrize(
    ("ranks", "categories"),
    [
        ("AKQ", ["set", "two pair", "board pair"]),
        ("KK2", ["quad", "full house", "trips", "overpair", "board pair"]),
        ("AAA", ["quad", "full house"]),
        ("KQJ2", ["set", "two pair", "overpair", "board pair"]),
        ("KK23", ["quad", "full house", "trips", "overpair", "board pair"]),
        ("AAKK", ["quad", "full house", "trips"]),
        ("AAAK", ["quad", "full house"]),
        ("AAAA", ["full house"]),
        ("AKQJT", ["set", "two pair", "board pair"]),
        ("AAKQJ", ["quad", "full house", "trips", "board pair"]),
        ("AAKKQ", ["quad", "full house", "trips", "board pair"]),
        ("KKKQJ", ["quad", "full house"]),
        ("KKKAA", ["quad", "full house", "trips"]),
        ("AAAAK", ["full house"]),
    ],
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_categories_for_each_rank_pattern(ranks: str, categories: list[str]) -> None:
    assert list(_groups(ranks)) == categories


def test_unpaired_flop_has_exact_priority_exclusions() -> None:
    groups = _groups("AKQ")

    assert _expressions(groups["set"]) == [
        "AA",
        "KK!AA",
        "QQ!AA!KK",
    ]
    assert _expressions(groups["two pair"]) == [
        "AK!AA!KK!QQ",
        "AQ!AA!KK!QQ!AK",
        "KQ!AA!KK!QQ!AK!AQ",
    ]
    assert _expressions(groups["board pair"]) == [
        "A!AA!KK!QQ!AK!AQ!KQ",
        "K!AA!KK!QQ!AK!AQ!KQ!A",
        "Q!AA!KK!QQ!AK!AQ!KQ!A!K",
    ]


def test_paired_flop_includes_quads_full_houses_and_trips() -> None:
    groups = _groups("KK2")

    assert _expressions(groups["quad"]) == ["KK"]
    assert _expressions(groups["full house"]) == [
        "K2!KK",
        "22!KK!K2",
    ]
    assert _expressions(groups["trips"]) == ["K!KK!K2!22"]
    assert _expressions(groups["board pair"]) == ["2!KK!K2!22!K"]


def test_tripled_flop_only_generates_quads_and_pocket_pair_full_houses() -> None:
    groups = _groups("AAA")

    assert _expressions(groups["quad"]) == ["A"]
    assert _names(groups["full house"]) == [
        "KK",
        "QQ",
        "JJ",
        "TT",
        "99",
        "88",
        "77",
        "66",
        "55",
        "44",
        "33",
        "22",
    ]
    assert _expressions(groups["full house"])[0] == "KK!A"


def test_unpaired_turn_has_all_set_and_two_pair_combinations() -> None:
    groups = _groups("KQJ2")

    assert _names(groups["set"]) == ["KK", "QQ", "JJ", "22"]
    assert _names(groups["two pair"]) == ["KQ", "KJ", "K2", "QJ", "Q2", "J2"]
    assert _names(groups["board pair"]) == ["K", "Q", "J", "2"]


def test_single_paired_turn_uses_all_available_board_kickers() -> None:
    groups = _groups("KK23")

    assert _names(groups["full house"]) == ["K3", "K2", "33", "22"]
    assert _names(groups["board pair"]) == ["3", "2"]


def test_double_paired_turn_has_two_quad_patterns_and_one_full_house() -> None:
    groups = _groups("AAKK")

    assert _names(groups["quad"]) == ["AA", "KK"]
    assert _expressions(groups["full house"]) == ["AK!AA!KK"]
    assert _names(groups["trips"]) == ["A", "K"]
    assert "board pair" not in groups


def test_tripled_turn_excludes_impossible_hole_ace_patterns() -> None:
    groups = _groups("AAAK")

    assert _expressions(groups["quad"]) == ["A"]
    assert _names(groups["full house"])[0] == "KK"
    assert all("A" not in name for name in _names(groups["full house"]))
    assert "trips" not in groups


def test_four_of_a_kind_board_cannot_make_quads_in_omaha() -> None:
    for ranks in ("AAAA", "AAAAK"):
        groups = _groups(ranks)
        assert "quad" not in groups
        assert _names(groups["full house"])[0] == "KK"
        assert "trips" not in groups


def test_unpaired_river_generates_all_rank_pairs() -> None:
    groups = _groups("AKQJT")

    assert _names(groups["set"]) == ["AA", "KK", "QQ", "JJ", "TT"]
    # Straights no longer suppress rank-based two-pair filters.
    assert _names(groups["two pair"]) == [
        "AK",
        "AQ",
        "AJ",
        "AT",
        "KQ",
        "KJ",
        "KT",
        "QJ",
        "QT",
        "JT",
    ]
    assert _names(groups["board pair"]) == ["A", "K", "Q", "J", "T"]


def test_single_paired_river_has_ordered_full_house_patterns() -> None:
    groups = _groups("AAKQJ")

    assert _expressions(groups["quad"]) == ["AA"]
    assert _names(groups["full house"]) == ["AK", "AQ", "AJ", "KK", "QQ", "JJ"]
    assert _names(groups["trips"]) == ["A"]
    assert _names(groups["board pair"]) == ["K", "Q", "J"]


def test_double_paired_river_deduplicates_full_house_requirements() -> None:
    groups = _groups("AAKKQ")

    assert _names(groups["quad"]) == ["AA", "KK"]
    assert _names(groups["full house"]) == ["AK", "AQ", "KQ", "QQ"]
    assert _names(groups["trips"]) == ["A", "K"]
    assert _names(groups["board pair"]) == ["Q"]


def test_tripled_and_paired_river_respects_exactly_three_board_cards() -> None:
    groups = _groups("KKKAA")

    assert _names(groups["quad"]) == ["AA", "K"]
    assert _names(groups["full house"])[0] == "QQ"
    assert _names(groups["trips"]) == ["A"]
    assert "board pair" not in groups


def test_overpairs_are_independent_of_other_categories() -> None:
    groups = _groups("Q72")

    # Overpairs exclude earlier overpairs only (not sets/two-pair/board-pair).
    assert _expressions(groups["overpair"]) == ["AA", "KK!AA"]

    # Their tokens must not enter the exclusions of any other category.
    for category, nodes in groups.items():
        if category != "overpair":
            assert all("!AA" not in node.expression for node in nodes)
            assert all("!KK" not in node.expression for node in nodes)


def test_overpairs_also_exist_on_paired_boards() -> None:
    groups = _groups("QQ2")
    assert _expressions(groups["overpair"]) == ["AA", "KK!AA"]


@pytest.mark.parametrize(
    "ranks",
    [
        "AKQ",
        "KK2",
        "AAA",
        "KQJ2",
        "KK23",
        "AAKK",
        "AAAK",
        "AAAA",
        "AKQJT",
        "AAKQJ",
        "AAKKQ",
        "KKKQJ",
        "KKKAA",
        "AAAAK",
    ],
)
def test_exclusions_follow_priority_without_self_contradictions(ranks: str) -> None:
    """Rank/straight categories keep separate exclusion histories."""
    seen: list[str] = []
    board_count = Counter(ranks)
    for category in generate_filters(_board(ranks)):
        if category.name in SUITED_CATEGORIES:
            continue

        local_seen = [] if category.name in ("overpair", "straight") else seen
        assert category.filters
        assert all(isinstance(child, FilterNode) for child in category.filters)

        for node in category.filters:
            # All OR branches in one straight-strength filter have the same
            # exclusions. Tokens within the strength never exclude siblings.
            required_tokens: list[str] = []
            branches = node.expression.split("|")
            if category.name != "straight":
                assert len(branches) == 1

            for branch in branches:
                required, *excluded = branch.split("!")
                if category.name != "straight":
                    assert required == node.name
                assert excluded == local_seen
                assert required not in excluded
                assert required not in local_seen

                for rank, count in Counter(required).items():
                    assert board_count[rank] + count <= 4

                required_count = Counter(required)
                for token in excluded:
                    assert not all(
                        required_count[rank] >= count
                        for rank, count in Counter(token).items()
                    )
                required_tokens.append(required)

            # Update exclusions only after finishing this OR filter.
            local_seen.extend(required_tokens)


def test_suits_do_not_affect_generated_expressions() -> None:
    first = _board("AAKQJ")
    different_suits = _board("AAKQJ", suit_offset=2)
    assert _rank_groups(first) == _rank_groups(different_suits)


def test_board_card_order_does_not_affect_generated_expressions() -> None:
    board = _board("AAKKQ")
    assert generate_filters(board) == generate_filters(board[::-1])


def test_generate_filters_does_not_modify_input_board() -> None:
    board = _board("Q2AAK")
    original = board.copy()
    generate_filters(board)
    np.testing.assert_array_equal(board, original)


@pytest.mark.parametrize("size", [0, 1, 2, 6])
def test_invalid_board_length_raises_value_error(size: int) -> None:
    with pytest.raises(ValueError, match="3, 4, or 5"):
        generate_filters(np.arange(size, dtype=np.uint8))


# Straight generation (no evaluation or expression parsing).
@pytest.mark.parametrize(
    ("ranks", "expected"),
    [
        ("T98", ["Q-high", "J-high", "T-high"]),
        ("JT98", ["K-high", "Q-high", "J-high", "T-high"]),
        ("AKQ", ["A-high"]),
        ("A23", ["5-high"]),
        ("A234", ["6-high", "5-high"]),
        ("A2345", ["7-high", "6-high", "5-high"]),
        ("AAKQJ", ["A-high", "K-high"]),
        ("AAAKQ", ["A-high"]),
        ("AAAAK", []),
        ("AAKK", []),
        ("AKQJT", ["A-high", "K-high", "Q-high"]),
    ],
)
def test_straight_strengths_by_board(ranks: str, expected: list[str]) -> None:
    assert list(_straight_groups(_board(ranks))) == expected


def test_straight_flop_uses_two_hole_and_three_board_cards() -> None:
    groups = _straight_groups(_board("T98"))
    assert groups["Q-high"].expression == "QJ"
    assert groups["J-high"].expression == "J7!QJ"
    assert groups["T-high"].expression == "76!QJ!J7"


def test_turn_multiple_hole_pairs_form_same_straight_strength() -> None:
    groups = _straight_groups(_board("JT98"))

    assert groups["K-high"].expression == "KQ"
    assert groups["Q-high"].expression == ("QJ!KQ|QT!KQ|Q9!KQ|Q8!KQ")
    assert _required_straight_tokens(groups["J-high"]) == ["J7", "T7", "97", "87"]
    assert groups["T-high"].expression == ("76!KQ!QJ!QT!Q9!Q8!J7!T7!97!87")


def test_same_hole_pair_is_assigned_to_its_strongest_straight() -> None:
    # KJT98: QJ makes K-high and Q-high via different board triples.
    # Keep the QJ requirement only in the K-high strength filter.
    groups = _straight_groups(_board("KJT98"))
    assert "QJ" in _required_straight_tokens(groups["K-high"])
    assert "QJ" not in _required_straight_tokens(groups["Q-high"])


@pytest.mark.parametrize(
    "ranks",
    [
        "T98",
        "A23",
        "543",
        "AKQ",
        "JT98",
        "A234",
        "KKQJ",
        "A2345",
        "KJT98",
        "AAKQJ",
        "AAAKQ",
        "AAAAK",
        "KKKAA",
    ],
)
def test_straight_windows_match_independent_pair_first_enumeration(ranks: str) -> None:
    """Cross-check window-first enumeration against two-hole-ranks-first logic.

    The reference enumerates every distinct hole-rank pair and all ways to
    choose three distinct ranks from the board, then compares five-rank sets
    to the ten possible straights. No parser or hand evaluator is involved.
    """
    board_ranks = Counter(ranks)
    expected: dict[str, set[str]] = {}
    for first, second in combinations(reversed(RANKS_DECODE), 2):
        # Do not require a hole card whose entire rank is on the board.
        if board_ranks[first] == 4 or board_ranks[second] == 4:
            continue

        pair = {first, second}
        strengths = [
            index
            for board_three in combinations(board_ranks, 3)
            for index, window in enumerate(STRAIGHT_WINDOWS)
            if pair | set(board_three) == set(window)
        ]
        if not strengths:
            continue

        strength_name = f"{STRAIGHT_WINDOWS[min(strengths)][0]}-high"
        expected.setdefault(strength_name, set()).add(first + second)

    actual = {
        strength_name: set(tokens)
        for strength_name, tokens in _straight_patterns(board_ranks)
    }
    assert actual == expected


def test_straight_strengths_are_single_filter_nodes() -> None:
    root = generate_filters(_board("JT98"))
    straight = next(group for group in root if group.name == "straight")

    assert isinstance(straight, GroupNode)
    assert straight.filters
    assert all(isinstance(node, FilterNode) for node in straight.filters)
    assert [node.name for node in straight.filters] == [
        "K-high",
        "Q-high",
        "J-high",
        "T-high",
    ]
    assert "|" in straight.filters[1].expression


def test_straight_wheel_treats_ace_as_low_only() -> None:
    groups = _straight_groups(_board("A23"))
    assert _required_straight_tokens(groups["5-high"]) == ["54"]
    assert "A-high" not in groups


def test_paired_river_can_provide_three_distinct_straight_ranks() -> None:
    groups = _straight_groups(_board("AAKQJ"))
    assert _required_straight_tokens(groups["A-high"]) == ["AT", "KT", "QT", "JT"]
    assert _required_straight_tokens(groups["K-high"]) == ["T9"]
    assert sum(len(_required_straight_tokens(node)) for node in groups.values()) == 5


def test_tripled_board_can_make_straight_using_distinct_board_ranks() -> None:
    groups = _straight_groups(_board("AAAKQ"))
    assert _required_straight_tokens(groups["A-high"]) == ["JT"]
    # With four of a kind plus a kicker, the board has too few distinct ranks.
    assert not _straight_groups(_board("AAAAK"))


def test_straights_are_independent_of_rank_filters() -> None:
    # Rank filters exclude other rank patterns, not straight requirements.
    assert _expressions(_groups("AKQ")["set"]) == [
        "AA",
        "KK!AA",
        "QQ!AA!KK",
    ]

    # Straight filters do not exclude set, full-house or other rank tokens.
    assert _straight_groups(_board("AKQ"))["A-high"].expression == "JT"
    assert _straight_groups(_board("AAKQJ"))["A-high"].expression == ("AT|KT|QT|JT")


def test_straight_strengths_still_exclude_stronger_straights() -> None:
    groups = _straight_groups(_board("T98"))
    assert groups["Q-high"].expression == "QJ"
    assert groups["J-high"].expression == "J7!QJ"
    assert groups["T-high"].expression == "76!QJ!J7"


def test_straights_are_independent_of_suited_filters() -> None:
    rainbow = encode_board("AsKhQc")
    monotone = encode_board("AsKsQs")
    assert _straight_groups(rainbow) == _straight_groups(monotone)
    assert _straight_groups(rainbow)["A-high"].expression == "JT"
    assert "bdfd" in _all_groups(rainbow)
    assert "flush" in _all_groups(monotone)


def test_straight_strengths_ignore_suits_and_board_order() -> None:
    board = encode_board("JsTh9c8d")
    swapped = encode_board("JhTc9d8s")
    assert _straight_groups(board) == _straight_groups(swapped)
    assert _straight_groups(board) == _straight_groups(board[::-1])


def test_straight_generation_does_not_mutate_board() -> None:
    board = _board("AAKQJ")
    original = board.copy()
    _straight_groups(board)
    np.testing.assert_array_equal(board, original)


def test_straight_draw_group_is_not_generated() -> None:
    assert "sd" not in _all_groups(_board("JT98"))
    assert "straight draw" not in _all_groups(_board("JT98"))


# Suit-based filter generation (no evaluation or expression parsing).
@pytest.mark.parametrize(
    ("board", "expected_categories"),
    [
        ("AsKhQc", {"bdfd", "2bdfd"}),  # rainbow flop
        ("AsQs7h", {"fd", "bdfd"}),  # two-tone flop
        ("AsQs7s", {"flush"}),  # monotone flop
        ("AsQsKhJh", {"fd", "2fd"}),  # two FDs on turn
        ("AsQs7h2c", {"fd"}),
        ("AsQs7s2c", {"flush"}),
        ("AsQs7s2s", {"flush"}),
        ("AsKhQc2d", set()),  # rainbow turn
        ("AsQs7s2h3c", {"flush"}),
        ("AsQsKhJh2c", set()),
        ("AsQs7s2s3s", {"flush"}),
        ("AsKhQc2d3s", set()),
    ],
)
def test_suit_categories_by_texture(board: str, expected_categories: set[str]) -> None:
    groups = _all_groups(encode_board(board))
    assert set(groups) & SUITED_CATEGORIES == expected_categories


@pytest.mark.parametrize("category", ["flush", "fd", "bdfd"])
def test_single_suit_categories_contain_four_strengths(category: str) -> None:
    board = {
        "flush": encode_board("AsQs7s"),
        "fd": encode_board("AsQs7h"),
        "bdfd": encode_board("AsKhQc"),
    }[category]
    assert _names(_all_groups(board)[category]) == ["nut", "2nd nut", "3rd nut", "low"]


def test_flop_fd_accounts_for_all_board_suited_cards() -> None:
    groups = _all_groups(encode_board("AsQs7h"))
    assert _expressions(groups["fd"]) == [
        "Kss",
        "Jss!Ks",
        "Tss!Ks!Js",
        "ss!Ks!Js!Ts",
    ]
    assert _expressions(groups["bdfd"]) == [
        "Ahh",
        "Khh!Ah",
        "Qhh!Ah!Kh",
        "hh!Ah!Kh!Qh",
    ]


def test_monotone_flop_flush_has_nut_ranks_excluding_board_cards() -> None:
    groups = _all_groups(encode_board("AsQs7s"))
    assert _expressions(groups["flush"]) == [
        "Kss",
        "Jss!Ks",
        "Tss!Ks!Js",
        "ss!Ks!Js!Ts",
    ]


def test_rainbow_flop_merges_three_bdfd_suits_by_or() -> None:
    groups = _all_groups(encode_board("AsKhQc"))
    assert _expressions(groups["bdfd"]) == [
        "Kss!hh!cc|Ahh!ss!cc|Acc!ss!hh",
        "Qss!Ks!hh!cc|Qhh!Ah!ss!cc|Kcc!Ac!ss!hh",
        "Jss!Ks!Qs!hh!cc|Jhh!Ah!Qh!ss!cc|Jcc!Ac!Kc!ss!hh",
        "ss!Ks!Qs!Js!hh!cc|hh!Ah!Qh!Jh!ss!cc|cc!Ac!Kc!Jc!ss!hh",
    ]


def test_turn_2fd_combines_both_suit_draws() -> None:
    groups = _all_groups(encode_board("AsQsKhJh"))
    assert _expressions(groups["fd"]) == [
        "Kss!hh|Ahh!ss",
        "Jss!Ks!hh|Qhh!Ah!ss",
        "Tss!Ks!Js!hh|Thh!Ah!Qh!ss",
        "ss!Ks!Js!Ts!hh|hh!Ah!Qh!Th!ss",
    ]
    assert _names(groups["2fd"]) == ["nut+nut", "nut+noNut", "noNut+noNut"]
    assert _expressions(groups["2fd"]) == [
        "KssAhh",
        "Ksshh!Ah|ssAhh!Ks",
        "sshh!Ks!Ah",
    ]


def test_rainbow_flop_2bdfd_has_three_suit_pairs() -> None:
    groups = _all_groups(encode_board("AsKhQc"))
    assert _names(groups["2bdfd"]) == ["nut+nut", "nut+noNut", "noNut+noNut"]
    assert _expressions(groups["2bdfd"]) == [
        "KssAhh|KssAcc|AhhAcc",
        ("Ksshh!Ah|ssAhh!Ks|Ksscc!Ac|ssAcc!Ks|Ahhcc!Ac|hhAcc!Ah"),
        "sshh!Ks!Ah|sscc!Ks!Ac|hhcc!Ah!Ac",
    ]


def test_four_suited_board_still_generates_flush_not_fd() -> None:
    groups = _all_groups(encode_board("AsKsQsJs"))
    assert _expressions(groups["flush"]) == [
        "Tss",
        "9ss!Ts",
        "8ss!Ts!9s",
        "ss!Ts!9s!8s",
    ]
    assert "fd" not in groups


def test_five_suited_board_still_generates_flush() -> None:
    groups = _all_groups(encode_board("AsKsQsJsTs"))
    assert _expressions(groups["flush"]) == [
        "9ss",
        "8ss!9s",
        "7ss!9s!8s",
        "ss!9s!8s!7s",
    ]


def test_suit_specific_categories_do_not_affect_rank_patterns() -> None:
    rainbow = encode_board("AsKhQc")
    monochrome = encode_board("AsKsQs")
    assert _rank_groups(rainbow) == _rank_groups(monochrome)
    assert "bdfd" in _all_groups(rainbow)
    assert "flush" in _all_groups(monochrome)


def test_suited_expressions_are_invariant_to_board_order() -> None:
    board = encode_board("AsKhQc")
    assert generate_filters(board) == generate_filters(board[[2, 0, 1]])
    board = encode_board("AsQsKhJh")
    assert generate_filters(board) == generate_filters(board[::-1])


def test_generator_does_not_mutate_suited_board() -> None:
    board = encode_board("AsQsKhJh")
    original = board.copy()
    generate_filters(board)
    np.testing.assert_array_equal(board, original)


@pytest.mark.parametrize(
    "board",
    [
        "AsAhKs",
        "AsAhQsKh",
        "AsAhQsKhJc",
    ],
)
def test_suited_categories_support_paired_ranks(board: str) -> None:
    groups = _all_groups(encode_board(board))

    if len(board) == 6:
        assert _expressions(groups["fd"])[0] == "Qss"
    elif len(board) == 8:
        # Both aces have the same rank; suit tie ordering is not significant.
        assert set(_expressions(groups["fd"])[0].split("|")) == {
            "Kss!hh",
            "Qhh!ss",
        }
    else:
        assert "fd" not in groups
