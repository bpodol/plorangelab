"""Check the Numba evaluator against an independent pure-Python matcher."""

from itertools import combinations

import numpy as np
import pytest

from plorangelab.encoder import _encode_card, build_mask
from plorangelab.filters.compiler import compile_filter
from plorangelab.filters.evaluator import evaluate_filter
from plorangelab.filters.models import And, AtomicFilter, Not, Or
from plorangelab.filters.parser import encode_filter


def hands_mask(*hands: str) -> np.ndarray:
    result = []
    for hand in hands:
        cards = [_encode_card(hand[j : j + 2]) for j in range(0, len(hand), 2)]
        assert len(set(cards)) == len(cards)
        result.append(build_mask(np.array(cards, dtype=np.uint8)))
    return np.array(result, dtype=np.uint64)


def reference_atomic(atom: AtomicFilter, cards: tuple[int, ...]) -> bool:
    slots = [int(requirement.mask) for req in atom.requirements for requirement in [req] for _ in range(req.count)]

    def backtrack(i: int, used: set[int]) -> bool:
        if i == len(slots):
            return True
        return any(
            backtrack(i + 1, used | {card})
            for card in cards
            if card not in used and (slots[i] & (1 << card))
        )

    return backtrack(0, set())


def reference_ast(node, cards: tuple[int, ...]) -> bool:
    if isinstance(node, AtomicFilter):
        return reference_atomic(node, cards)
    if isinstance(node, Not):
        return not reference_ast(node.child, cards)
    if isinstance(node, And):
        return reference_ast(node.left, cards) and reference_ast(node.right, cards)
    if isinstance(node, Or):
        return reference_ast(node.left, cards) or reference_ast(node.right, cards)
    raise TypeError(node)


@pytest.mark.parametrize("mode", [4, 5])
def test_against_independent_reference_randomized(mode: int) -> None:
    expressions = [
        "A", "AA", "AAA", "AAAA", "As", "AsA", "Ass", "Asss",
        "sAs", "AsKs", "AsKK", "s", "ss", "sss", "ssss", "As:s",
        "A:!ss", "8!88", "(A|K):!ss", "((A|K):ss)|((Q|J):!AA)",
        "!ss", "!(AA|KK)", "(s|h):!(A|K)", "(As|Ks):(s|c)",
        "As:s!ss", "A:K:Q", "s!ss", "AsK", "A:s:!AA",
    ]
    if mode == 5:
        expressions += ["AsKsQsJsTs", "sssss", "Assss", "AKQJT"]

    rng = np.random.default_rng(1827)
    hands = [tuple(int(x) for x in rng.choice(52, size=mode, replace=False)) for _ in range(400)]
    masks = np.array(
        [build_mask(np.array(cards, dtype=np.uint8)) for cards in hands],
        dtype=np.uint64,
    )

    for expression in expressions:
        ast = encode_filter(expression, mode)
        expected = np.array([reference_ast(ast, cards) for cards in hands], dtype=np.bool_)
        assert np.array_equal(evaluate_filter(expression, masks, mode), expected), expression
        compiled = compile_filter(ast, mode)
        assert np.array_equal(evaluate_filter(compiled, masks), expected), expression
        assert np.array_equal(evaluate_filter(ast, masks, mode), expected), expression


@pytest.mark.parametrize(
    ("filter_str", "hand", "expected"),
    [
        ("Ass", "AsKdQhJc", False),
        ("Ass", "AsKsQhJc", True),
        ("As:s", "AsKdQhJc", True),
        ("AsA", "AsKdQhJc", False),
        ("AsA", "AsAcQhJc", True),
        ("AsKs", "AsKdQhJc", False),
        ("AsKs", "AsKsQhJc", True),
        ("8!88", "8sKdQhJc", True),
        ("8!88", "8s8dQhJc", False),
        ("8!88", "AsKdQhJc", False),
        ("A!ss", "AsKsQhJc", False),
        ("A!ss", "AsKdQhJc", True),
        ("Asss", "AsKsTsJc", True),
        ("Asss", "AsKsQhJc", False),
        ("s!ss", "AsKdQhJc", True),
        ("s!ss", "AsKsQhJc", False),
    ],
)
def test_known_hands(filter_str: str, hand: str, expected: bool) -> None:
    selected = evaluate_filter(filter_str, hands_mask(hand), mode=4)
    assert selected.tolist() == [expected]


def test_empty_masks_still_returns_empty_bool_array() -> None:
    result = evaluate_filter("A", np.empty(0, dtype=np.uint64), 5)
    assert result.dtype == np.bool_
    assert result.shape == (0,)


def test_source_unchanged_and_result_is_independent() -> None:
    source = hands_mask("AsKdQhJc", "AsKsQhJc")
    before = source.copy()
    selection = evaluate_filter("Ass", source, 4)
    selection[0] = True
    assert np.array_equal(source, before)
    assert selection.shape == source.shape


@pytest.mark.parametrize(
    "bad_masks", [
        np.array([1], dtype=np.uint32),
        np.array([[1]], dtype=np.uint64),
        [np.uint64(1)],
    ],
)
def test_invalid_hand_masks(bad_masks) -> None:
    with pytest.raises(TypeError, match="1D NumPy array"):
        evaluate_filter("A", bad_masks, 4)


def test_mode_required_for_raw_expression() -> None:
    with pytest.raises(ValueError, match="mode is required"):
        evaluate_filter("A", hands_mask("AsKdQhJc"))


def test_compiled_mode_mismatch() -> None:
    compiled = compile_filter(encode_filter("A", 5), 5)
    with pytest.raises(ValueError, match="mode does not match"):
        evaluate_filter(compiled, hands_mask("AsKdQhJc"), mode=4)


def test_noncontiguous_input() -> None:
    source = hands_mask("AsKdQhJc", "AsKsQhJc", "8sKdQhJc", "8s8dQhJc")
    assert evaluate_filter("A", source[::2], 4).tolist() == [True, False]


def test_all_four_card_combinations_selected_by_rank() -> None:
    # Reference over all 270,725 PLO4 hands; Numba must not lose results.
    card_combos = combinations(range(52), 4)
    raw_masks = np.fromiter((sum(1 << c for c in cards) for cards in card_combos), dtype=np.uint64)
    selected = evaluate_filter("AA", raw_masks, 4)
    # Number of four-card hands with >= 2 Aces, derived combinatorially.
    from math import comb
    expected = sum(comb(4, n) * comb(48, 4-n) for n in range(2,5))
    assert selected.sum() == expected
