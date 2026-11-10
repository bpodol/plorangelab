"""Tests for compiling parsed ASTs into Numba-friendly instructions.

These tests inspect the compiled program without evaluating poker hands.
"""

from collections.abc import Iterator
from itertools import product

import numpy as np
import pytest

from plorangelab.encoder import _encode_card, build_mask
from plorangelab.filters.compiler import (
    AND,
    MATCH,
    NOT,
    OR,
    CompiledFilter,
    compile_filter,
)
from plorangelab.filters.models import And, AtomicFilter, CardRequirement, Not, Or
from plorangelab.filters.parser import encode_filter


def mask(*cards: str) -> int:
    card_ids = np.asarray([_encode_card(card) for card in cards], dtype=np.uint8)
    return int(build_mask(card_ids))


def compile_text(text: str, mode: int = 5) -> CompiledFilter:
    return compile_filter(encode_filter(text, mode), mode)


def test_compiled_filter_array_types_and_dimensions() -> None:
    program = compile_text("AsKK|AA", 4)

    assert program.mode == 4
    assert program.operations.dtype == np.int8
    assert program.operands.dtype == np.int32
    assert program.slot_masks.dtype == np.uint64
    assert program.slot_counts.dtype == np.int32
    assert program.operations.shape == program.operands.shape == (3,)
    assert program.slot_masks.shape == (2, 4)
    assert program.slot_counts.shape == (2,)


def test_single_atomic_filter_compiles_to_match_instruction() -> None:
    program = compile_text("AA", 4)

    assert program.operations.tolist() == [MATCH]
    assert program.operands.tolist() == [0]
    assert program.slot_counts.tolist() == [2]
    aces = mask("Ac", "Ad", "Ah", "As")
    assert program.slot_masks.tolist() == [[aces, aces, 0, 0]]


def test_count_expansion_and_specificity_sort() -> None:
    program = compile_text("KKAs", 5)

    kings = mask("Kc", "Kd", "Kh", "Ks")
    ace_spades = mask("As")
    # The exact-card slot is checked first, even though it occurred last.
    assert program.slot_masks[0].tolist() == [ace_spades, kings, kings, 0, 0]
    assert program.slot_counts.tolist() == [3]


def test_suit_count_expands_and_pads() -> None:
    program = compile_text("sss", 4)
    spades = mask(*(rank + "s" for rank in "23456789TJQKA"))

    assert program.slot_masks.tolist() == [[spades, spades, spades, 0]]
    assert program.slot_counts.tolist() == [3]


def test_atomic_requirements_with_overlapping_masks_are_preserved() -> None:
    program = compile_text("Ass", 5)
    spades = mask(*(rank + "s" for rank in "23456789TJQKA"))

    assert program.slot_masks[0].tolist() == [mask("As"), spades, 0, 0, 0]
    assert program.slot_counts.tolist() == [2]


@pytest.mark.parametrize(
    ("expression", "opcodes", "operands"),
    [
        ("A|K", [MATCH, MATCH, OR], [0, 1, -1]),
        ("A:K", [MATCH, MATCH, AND], [0, 1, -1]),
        ("!AA", [MATCH, NOT], [0, -1]),
        ("8!88", [MATCH, MATCH, NOT, AND], [0, 1, -1, -1]),
        ("A|K:ss", [MATCH, MATCH, MATCH, AND, OR], [0, 1, 2, -1, -1]),
        ("(A|K):ss", [MATCH, MATCH, OR, MATCH, AND], [0, 1, -1, 2, -1]),
        ("(AsKK|AA):!ss", [MATCH, MATCH, OR, MATCH, NOT, AND], [0, 1, -1, 2, -1, -1]),
        (
            "((A|K):ss)|((Q|J):!AA)",
            [MATCH, MATCH, OR, MATCH, AND, MATCH, MATCH, OR, MATCH, NOT, AND, OR],
            [0, 1, -1, 2, -1, 3, 4, -1, 5, -1, -1, -1],
        ),
    ],
)
def test_postfix_compilation(
    expression: str, opcodes: list[int], operands: list[int]
) -> None:
    program = compile_text(expression)

    assert program.operations.tolist() == opcodes
    assert program.operands.tolist() == operands
    assert program.slot_counts.size == opcodes.count(MATCH)


def test_nested_expression_preserves_atom_order() -> None:
    program = compile_text("(As|KK):!ss")
    as_mask = mask("As")
    kings_mask = mask("Kc", "Kd", "Kh", "Ks")
    spades_mask = mask(*(rank + "s" for rank in "23456789TJQKA"))

    assert program.slot_masks[0, 0] == as_mask
    assert program.slot_masks[1, :2].tolist() == [kings_mask, kings_mask]
    assert program.slot_masks[2, :2].tolist() == [spades_mask, spades_mask]
    assert program.slot_counts.tolist() == [1, 2, 2]


def test_compiled_arrays_cannot_be_mutated_directly() -> None:
    program = compile_text("A|K")
    for array in (
        program.operations,
        program.operands,
        program.slot_masks,
        program.slot_counts,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 1


def test_compilations_have_separate_outputs() -> None:
    a = compile_text("A")
    b = compile_text("K")
    assert not np.shares_memory(a.slot_masks, b.slot_masks)
    assert a.slot_masks[0, 0] != b.slot_masks[0, 0]


def test_four_and_five_card_modes_change_slot_width() -> None:
    p4 = compile_text("AKQJ", 4)
    p5 = compile_text("AKQJ", 5)
    assert p4.slot_masks.shape == (1, 4)
    assert p5.slot_masks.shape == (1, 5)
    assert p5.slot_masks[0, -1] == 0
    assert p4.slot_counts.tolist() == p5.slot_counts.tolist() == [4]


def test_five_card_atomic_filter_allowed_in_plo5() -> None:
    program = compile_text("AKQJT", 5)
    assert program.slot_counts.tolist() == [5]


def test_four_card_limit_is_enforced_even_for_manual_ast() -> None:
    ast = encode_filter("AKQJT", 5)
    with pytest.raises(ValueError, match="PLO4"):
        compile_filter(ast, 4)


@pytest.mark.parametrize("bad_mode", [0, 3, 6, True, None, 4.0, "5"])
def test_invalid_modes(bad_mode: object) -> None:
    with pytest.raises(ValueError, match="mode"):
        compile_filter(encode_filter("A", 5), bad_mode)


@pytest.mark.parametrize(
    ("requirements", "message"),
    [
        ((), "at least one"),
        ((CardRequirement(np.uint64(0), 1),), "bits"),
        ((CardRequirement(np.uint64(1 << 52), 1),), "bits"),
        ((CardRequirement(np.uint64(1), 0),), "positive"),
        ((CardRequirement(np.uint64(1), -1),), "positive"),
        ((CardRequirement(np.uint64(1), True),), "positive"),
        ((CardRequirement(np.uint64(1), 2),), "more cards"),
        ((CardRequirement(np.uint64(15), 5),), "more cards"),
    ],
)
def test_invalid_manually_built_atoms(
    requirements: tuple[CardRequirement, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        compile_filter(AtomicFilter(requirements), 5)


def test_manually_constructed_ast_with_operators() -> None:
    a = encode_filter("A", 4)
    k = encode_filter("K", 4)
    program = compile_filter(Not(Or(And(a, k), a)), 4)

    assert program.operations.tolist() == [MATCH, MATCH, AND, MATCH, OR, NOT]
    assert program.operands.tolist() == [0, 1, -1, 2, -1, -1]


def test_unsupported_ast_node() -> None:
    with pytest.raises(TypeError, match="Unsupported"):
        compile_filter("A", 5)


@pytest.mark.parametrize(
    "expression",
    [
        "A|K:ss",
        "8!88",
        "(AsKK|AA):!ss",
        "((A|K):ss)|((Q|J):!AA)",
    ],
)
def test_postfix_program_preserves_ast_boolean_semantics(expression: str) -> None:
    """Check all atom truth assignments, independently of card matching."""
    ast = encode_filter(expression, 5)
    compiled = compile_filter(ast, 5)
    atoms = len(compiled.slot_counts)

    def eval_ast(node: object, flags: Iterator[bool]) -> bool:
        if isinstance(node, AtomicFilter):
            return bool(next(flags))
        if isinstance(node, Not):
            return not eval_ast(node.child, flags)
        if isinstance(node, And):
            left = eval_ast(node.left, flags)
            right = eval_ast(node.right, flags)
            return left and right
        if isinstance(node, Or):
            left = eval_ast(node.left, flags)
            right = eval_ast(node.right, flags)
            return left or right
        raise AssertionError("Invalid AST type")

    def eval_program(flags: tuple[bool, ...]) -> bool:
        stack: list[bool] = []
        for opcode, operand in zip(compiled.operations, compiled.operands, strict=True):
            if opcode == MATCH:
                stack.append(flags[int(operand)])
            elif opcode == NOT:
                stack[-1] = not stack[-1]
            else:
                right = stack.pop()
                left = stack.pop()
                stack.append(left and right if opcode == AND else left or right)
        assert len(stack) == 1
        return stack[0]

    for flags in product((False, True), repeat=atoms):
        assert eval_program(flags) == eval_ast(ast, iter(flags))
