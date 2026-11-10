"""Tests for atomic requirement encoding and logical-expression parsing."""

import numpy as np
import pytest

from plorangelab.encoder import _encode_card, build_mask
from plorangelab.filters import encode_filter
from plorangelab.filters.models import And, Not, Or
from plorangelab.filters.parser import encode_atomic_filter


def expected_mask(*cards: str) -> int:
    """Compute expected masks independently from filter parsing logic."""
    card_ids = np.array([_encode_card(card) for card in cards], dtype=np.uint8)
    return int(build_mask(card_ids))


def requirements(atom: str) -> list[tuple[int, int]]:
    return [
        (int(req.mask), req.count) for req in encode_atomic_filter(atom, 5).requirements
    ]


@pytest.mark.parametrize(
    ("atom", "expected"),
    [
        ("A", [(expected_mask("Ac", "Ad", "Ah", "As"), 1)]),
        ("AA", [(expected_mask("Ac", "Ad", "Ah", "As"), 2)]),
        ("As", [(expected_mask("As"), 1)]),
        ("AsKs", [(expected_mask("As"), 1), (expected_mask("Ks"), 1)]),
        (
            "AsKK",
            [(expected_mask("As"), 1), (expected_mask("Kc", "Kd", "Kh", "Ks"), 2)],
        ),
        ("s", [(expected_mask(*(r + "s" for r in "23456789TJQKA")), 1)]),
        ("ss", [(expected_mask(*(r + "s" for r in "23456789TJQKA")), 2)]),
        (
            "Ass",
            [
                (expected_mask("As"), 1),
                (expected_mask(*(r + "s" for r in "23456789TJQKA")), 1),
            ],
        ),
        (
            "Asss",
            [
                (expected_mask("As"), 1),
                (expected_mask(*(r + "s" for r in "23456789TJQKA")), 2),
            ],
        ),
        ("KKK", [(expected_mask("Kc", "Kd", "Kh", "Ks"), 3)]),
        (
            "AcAdAhAs",
            [
                (expected_mask("Ac"), 1),
                (expected_mask("Ad"), 1),
                (expected_mask("Ah"), 1),
                (expected_mask("As"), 1),
            ],
        ),
    ],
)
def test_atomic_requirements(atom: str, expected: list[tuple[int, int]]) -> None:
    assert requirements(atom) == expected


def test_atomic_uses_uint64_masks() -> None:
    assert all(
        isinstance(req.mask, np.uint64)
        for req in encode_atomic_filter("AsKK", 5).requirements
    )


def test_atomic_repeated_rank_merges_counts_in_first_appearance_order() -> None:
    assert requirements("AKAA") == [
        (expected_mask("Ac", "Ad", "Ah", "As"), 3),
        (expected_mask("Kc", "Kd", "Kh", "Ks"), 1),
    ]


def test_atomic_allows_cross_requirement_overlap_for_later_evaluator() -> None:
    # 'AsA' requires two distinct Aces, but is syntactically valid.
    tree = encode_atomic_filter("AsA", 5)
    assert len(tree.requirements) == 2


@pytest.mark.parametrize(
    "atom", ["", "AsAs", "AAAAA", "AKQJT9", "A?", "a", "Xs", "AsAsK"]
)
def test_invalid_atomic_expressions(atom: str) -> None:
    with pytest.raises(ValueError):
        encode_atomic_filter(atom, 5)


def test_single_atom_returns_atomic_filter() -> None:
    assert encode_filter("AA", 5) == encode_atomic_filter("AA", 5)


def test_binary_not_is_and_not() -> None:
    assert encode_filter("8!88", 5) == And(
        encode_atomic_filter("8", 5), Not(encode_atomic_filter("88", 5))
    )


def test_unary_not() -> None:
    assert encode_filter("!AA", 5) == Not(encode_atomic_filter("AA", 5))


def test_and_precedes_or() -> None:
    assert encode_filter("A|K:ss", 5) == Or(
        encode_atomic_filter("A", 5),
        And(encode_atomic_filter("K", 5), encode_atomic_filter("ss", 5)),
    )


def test_parentheses_override_precedence() -> None:
    assert encode_filter("(A|K):ss", 5) == And(
        Or(encode_atomic_filter("A", 5), encode_atomic_filter("K", 5)),
        encode_atomic_filter("ss", 5),
    )


def test_prefix_not_and_nested_parentheses() -> None:
    assert encode_filter("!(A|K):ss", 5) == And(
        Not(Or(encode_atomic_filter("A", 5), encode_atomic_filter("K", 5))),
        encode_atomic_filter("ss", 5),
    )


def test_complex_expression_from_design_discussion() -> None:
    assert encode_filter("((A|K):ss)|((Q|J):!AA)", 5) == Or(
        And(
            Or(encode_atomic_filter("A", 5), encode_atomic_filter("K", 5)),
            encode_atomic_filter("ss", 5),
        ),
        And(
            Or(encode_atomic_filter("Q", 5), encode_atomic_filter("J", 5)),
            Not(encode_atomic_filter("AA", 5)),
        ),
    )


def test_or_and_are_left_associative() -> None:
    assert encode_filter("A|K|Q", 5) == Or(
        Or(encode_atomic_filter("A", 5), encode_atomic_filter("K", 5)),
        encode_atomic_filter("Q", 5),
    )
    assert encode_filter("A:K:Q", 5) == And(
        And(encode_atomic_filter("A", 5), encode_atomic_filter("K", 5)),
        encode_atomic_filter("Q", 5),
    )


def test_leading_trailing_and_operator_whitespace() -> None:
    assert encode_filter("  ( A | K ) : ! AA  ", 5) == encode_filter("(A|K):!AA", 5)


def test_binary_not_and_colon_same_precedence() -> None:
    assert encode_filter("A!K:Q", 5) == And(
        And(encode_atomic_filter("A", 5), Not(encode_atomic_filter("K", 5))),
        encode_atomic_filter("Q", 5),
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "A|",
        ":A",
        "A:",
        "A!",
        "A||K",
        "A::K",
        "(A|K",
        "A|K)",
        "()",
        "A:()",
        "A||",
        "A K",
        "A&K",
        "A,B",
        "A:B #",
        "A:(K|)",
        "!(A",
    ],
)
def test_invalid_expressions(text: str) -> None:
    with pytest.raises(ValueError):
        encode_filter(text, 5)


def test_extra_closing_parenthesis_is_not_ignored() -> None:
    with pytest.raises(ValueError, match="Unexpected character"):
        encode_filter("A)", 5)


def test_missing_closing_parenthesis_has_meaningful_error() -> None:
    with pytest.raises(ValueError, match=r"Expected '\)'"):
        encode_filter("(A|K", 5)


@pytest.mark.parametrize("atom", ["AKQJ", "AsKsQsJs", "ssss", "AsKKKs", "Assss"])
def test_four_card_atoms_allowed_in_both_modes(atom: str) -> None:
    assert encode_atomic_filter(atom, 4) == encode_atomic_filter(atom, 5)


@pytest.mark.parametrize("atom", ["AKQJT", "AsKsQsJsTs", "sssss", "Asssss"])
def test_five_card_atoms_only_allowed_in_plo5(atom: str) -> None:
    with pytest.raises(ValueError, match=r"PLO4 allows at most 4"):
        encode_atomic_filter(atom, 4)
    assert len(encode_atomic_filter(atom, 5).requirements) > 0


@pytest.mark.parametrize("mode", [4, 5])
def test_six_card_atoms_invalid_in_both_modes(mode: int) -> None:
    with pytest.raises(ValueError, match=rf"PLO{mode} allows at most {mode}"):
        encode_atomic_filter("AKQJT9", mode)


@pytest.mark.parametrize(
    "expression",
    [
        "AKQJT",
        "!AKQJT",
        "(A|AKQJT)",
        "A:(AKQJT|K)",
    ],
)
def test_parser_enforces_mode_inside_nested_expressions(expression: str) -> None:
    with pytest.raises(ValueError, match=r"PLO4 allows at most 4"):
        encode_filter(expression, 4)
    assert encode_filter(expression, 5) is not None


@pytest.mark.parametrize("mode", [4, 5])
def test_complex_logical_filter_valid_in_both_modes(mode: int) -> None:
    expression = "((A|K):ss)|((Q|J):!AA)"
    assert encode_filter(expression, mode) == encode_filter(expression, 5)


@pytest.mark.parametrize("mode", [0, 3, 6, True, 4.0, "4", None])
def test_invalid_mode_rejected_at_public_entry_points(mode: object) -> None:
    with pytest.raises(ValueError, match="mode must be 4"):
        encode_atomic_filter("A", mode)
    with pytest.raises(ValueError, match="mode must be 4"):
        encode_filter("A", mode)
