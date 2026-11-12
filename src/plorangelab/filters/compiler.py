"""Compile poker filter ASTs into NumPy instructions for the evaluator.

Postfix notation eliminates Python AST traversal during hand evaluation.
For example, ``(A|K):!ss`` becomes::

    MATCH(A), MATCH(K), OR, MATCH(ss), NOT, AND

Every MATCH instruction references one row in ``slot_masks``. Repeated
requirements expand into individual slots, so ``AsKK`` has three slots:
one exact As and two king slots. Slots are sorted by specificity to make
subsequent distinct-card matching more efficient.

The compiler does NOT evaluate hands or check whether overlapping slots
can all be assigned distinct cards; that belongs to the evaluator.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from plorangelab.types import Mode

from .models import And, AtomicFilter, FilterExpression, Not, Or

MATCH = 0
NOT = 1
AND = 2
OR = 3

_DECK_MASK = (1 << 52) - 1


@dataclass(frozen=True, slots=True)
class CompiledFilter:
    """NumPy-only, postfix representation of a filter expression.

    Each operation has one operand. For MATCH, that operand is a row index
    into ``slot_masks``; for other operations the operand is -1.

    Each row of ``slot_masks`` contains ``slot_counts[row]`` requirements,
    padded with zeros to ``mode`` columns. Each set bit in a slot mask
    denotes a card that may satisfy that requirement.

    Array fields are read-only, including their contents.
    """

    operations: NDArray[np.int8]
    operands: NDArray[np.int32]
    slot_masks: NDArray[np.uint64]
    slot_counts: NDArray[np.int32]
    mode: Mode


def compile_filter(expression: FilterExpression, mode: Mode) -> CompiledFilter:
    """Compile a filter AST into postfix instructions for Numba evaluation.

    Converts logical expression nodes into MATCH, NOT, AND, and OR
    instructions. Each atomic filter is expanded into card requirement
    slots, ordered from most restrictive to least restrictive.

    The resulting program uses read-only NumPy arrays and can be
    reused to evaluate multiple ranges without recompilation.

    Args:
        expression: Parsed filter expression AST.
        mode: Number of hole cards per hand (4 or 5). Atomic
            requirements are validated against this limit.

    Returns:
        CompiledFilter containing postfix operations, operands,
        card requirement masks, slot counts, and the PLO mode.

    Raises:
        ValueError: If the mode or atomic requirements are invalid.
        TypeError: If the AST contains an unsupported node type.

    Notes:
        Compilation validates individual requirements but does not
        determine whether overlapping card sets can be satisfied
        using distinct cards. That is handled by the evaluator.
    """

    if not isinstance(mode, int) or isinstance(mode, bool) or mode not in (4, 5):
        raise ValueError("mode must be 4 (PLO4) or 5 (PLO5)")

    operations: list[int] = []
    operands: list[int] = []
    atom_slots: list[list[int]] = []

    def emit(node: FilterExpression) -> None:
        if isinstance(node, AtomicFilter):
            if not node.requirements:
                raise ValueError("Atomic filter must contain at least one requirement")

            slots: list[int] = []
            for requirement in node.requirements:
                mask = int(requirement.mask)
                count = requirement.count
                if mask <= 0 or mask & ~_DECK_MASK:
                    raise ValueError("Requirement mask must use card bits 0 through 51")
                if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                    raise ValueError("Requirement count must be a positive integer")
                if count > mask.bit_count():
                    raise ValueError(
                        "Requirement asks for more cards than its mask contains"
                    )
                slots.extend([mask] * count)
                if len(slots) > mode:
                    raise ValueError(
                        f"Atomic filter requires {len(slots)} cards; "
                        f"PLO{mode} allows at most {mode}"
                    )

            # Try the rarest card sets first during backtracking. Sorting is
            # stable: equally restrictive slots keep their input order.
            slots.sort(key=int.bit_count)
            operands.append(len(atom_slots))
            operations.append(MATCH)
            atom_slots.append(slots)
        elif isinstance(node, Not):
            emit(node.child)
            operations.append(NOT)
            operands.append(-1)
        elif isinstance(node, And):
            emit(node.left)
            emit(node.right)
            operations.append(AND)
            operands.append(-1)
        elif isinstance(node, Or):
            emit(node.left)
            emit(node.right)
            operations.append(OR)
            operands.append(-1)
        else:
            raise TypeError(f"Unsupported filter expression node: {type(node)!r}")

    emit(expression)

    slot_masks = np.zeros((len(atom_slots), mode), dtype=np.uint64)
    for index, slots in enumerate(atom_slots):
        slot_masks[index, : len(slots)] = slots

    arrays = (
        np.asarray(operations, dtype=np.int8),
        np.asarray(operands, dtype=np.int32),
        slot_masks,
        np.asarray([len(slots) for slots in atom_slots], dtype=np.int32),
    )
    # frozen=True prevents field reassignment but does not protect
    # NumPy array contents. Mark the compiled arrays read-only.
    for array in arrays:
        array.setflags(write=False)

    return CompiledFilter(*arrays, mode=mode)
