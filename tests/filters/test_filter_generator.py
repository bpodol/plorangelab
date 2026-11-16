
"""Unit tests for rank-only automatic filter-expression generation.

These tests inspect the generated tree and strings directly. They deliberately
avoid the filter parser, hand evaluator, and range/coverage calculations.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pytest
from numpy.typing import NDArray

from plorangelab.encoder import RANKS_ENCODE
from plorangelab.filters.filter_generator import FilterNode, GroupNode, generate_filters


def _board(ranks: str, *, suit_offset: int = 0) -> NDArray[np.uint8]:
    """Encode ranks as distinct physical cards; suits do not affect the tests."""
    used_suits: defaultdict[str, int] = defaultdict(int)
    cards: list[int] = []
    for rank in ranks:
        suit = used_suits[rank]
        assert suit < 4, "Test boards must contain physically possible cards"
        cards.append((RANKS_ENCODE[rank] << 2) | ((suit + suit_offset) % 4))
        used_suits[rank] += 1
    return np.array(cards, dtype=np.uint8)


def _groups(ranks: str) -> dict[str, list[FilterNode]]:
    result: dict[str, list[FilterNode]] = {}
    for node in generate_filters(_board(ranks)):
        assert isinstance(node, GroupNode), "Every category must be a group"
        assert node.name not in result, "Category names must be unique"
        assert node.filters, "Empty categories must not be returned"
        assert all(isinstance(child, FilterNode) for child in node.filters)
        result[node.name] = node.filters
    return result


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
        "KK", "QQ", "JJ", "TT", "99", "88",
        "77", "66", "55", "44", "33", "22",
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
    assert _names(groups["two pair"]) == [
        "AK", "AQ", "AJ", "AT", "KQ", "KJ", "KT", "QJ", "QT", "JT",
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
    ["AKQ", "KK2", "AAA", "KQJ2", "KK23", "AAKK", "AAAK", "AAAA",
     "AKQJT", "AAKQJ", "AAKKQ", "KKKQJ", "KKKAA", "AAAAK"],
)
def test_exclusions_follow_priority_without_self_contradictions(ranks: str) -> None:
    """Each category except overpairs excludes every previously emitted token."""
    seen: list[str] = []
    board_count = Counter(ranks)
    for category, nodes in _groups(ranks).items():
        local_seen = [] if category == "overpair" else seen
        for node in nodes:
            required, *excluded = node.expression.split("!")
            assert required == node.name
            assert excluded == local_seen
            assert node.name not in excluded
            assert node.name not in local_seen

            # A generated token must leave enough cards of every rank in deck.
            for rank, count in Counter(required).items():
                assert board_count[rank] + count <= 4

            # A requirement must never entail one of its own exclusions.
            required_count = Counter(required)
            for token in excluded:
                assert not all(
                    required_count[rank] >= count
                    for rank, count in Counter(token).items()
                )
            local_seen.append(required)


def test_suits_do_not_affect_generated_expressions() -> None:
    first = _board("AAKQJ")
    different_suits = _board("AAKQJ", suit_offset=2)
    assert generate_filters(first) == generate_filters(different_suits)


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
