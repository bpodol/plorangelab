"""Evaluate compiled card filters against PLO4/PLO5 hand bitmasks.

Matching uses iterative backtracking to ensure requirements within an
atomic filter refer to *distinct* cards. Logical operations use a postfix
Boolean stack. Both loops run in Numba, without Python work per hand.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

from plorangelab.types import Mode

from .compiler import AND, MATCH, NOT, OR, CompiledFilter, compile_filter
from .models import FilterExpression
from .parser import encode_filter


@njit(cache=True)
def _matches_slots(
    hand_mask: np.uint64,
    slot_masks: NDArray[np.uint64],
    atom_index: int,
    count: int,
    available: NDArray[np.uint64],
    used: NDArray[np.uint64],
) -> bool:
    """Assign a different card to each required slot, if possible.

    ``available`` and ``used`` are reusable scratch arrays of length
    ``mode``. The compiler guarantees 1 <= count <= mode.
    """
    depth = 0
    used[0] = np.uint64(0)
    available[0] = hand_mask & slot_masks[atom_index, 0]

    while depth >= 0:
        candidates = available[depth]
        if candidates == 0:
            depth -= 1  # No candidate: retry the previous slot.
            continue

        # Choose the lowest set bit and remove it from this slot's options.
        remaining = candidates & (candidates - np.uint64(1))
        chosen = candidates ^ remaining
        available[depth] = remaining

        if depth + 1 == count:
            return True  # All slots have distinct matching cards.

        used[depth + 1] = used[depth] | chosen
        depth += 1
        available[depth] = hand_mask & slot_masks[atom_index, depth] & ~used[depth]

    return False


@njit(cache=True)
def _evaluate_compiled(
    hand_masks: NDArray[np.uint64],
    operations: NDArray[np.int8],
    operands: NDArray[np.int32],
    slot_masks: NDArray[np.uint64],
    slot_counts: NDArray[np.int32],
    mode: int,
) -> NDArray[np.bool_]:
    """Evaluate postfix instructions for every hand, reusing scratch space."""
    selection = np.empty(len(hand_masks), dtype=np.bool_)
    stack = np.empty(len(operations), dtype=np.bool_)
    available = np.empty(mode, dtype=np.uint64)
    used = np.empty(mode, dtype=np.uint64)

    for i in range(len(hand_masks)):
        top = 0
        for j in range(len(operations)):
            opcode = operations[j]
            if opcode == MATCH:
                atom = operands[j]
                stack[top] = _matches_slots(
                    hand_masks[i],
                    slot_masks,
                    atom,
                    slot_counts[atom],
                    available,
                    used,
                )
                top += 1
            elif opcode == NOT:
                stack[top - 1] = not stack[top - 1]
            elif opcode == AND:
                stack[top - 2] = stack[top - 2] and stack[top - 1]
                top -= 1
            elif opcode == OR:
                stack[top - 2] = stack[top - 2] or stack[top - 1]
                top -= 1

        selection[i] = stack[0]

    return selection


def evaluate_filter(
    expression: str | FilterExpression | CompiledFilter,
    hand_masks: NDArray[np.uint64],
    mode: Mode | None = None,
) -> NDArray[np.bool_]:
    """Return a Boolean selection aligned with ``hand_masks``.

    ``expression`` may be a filter string, a parsed AST, or a compiled
    program. ``mode`` (4 or 5) is required unless a compiled program is
    supplied, since ``CompiledFilter`` already stores its mode.

    Hand masks must be a 1D uint64 NumPy array holding valid, distinct
    PLO4/PLO5 card combinations as produced by the range encoder.

    This function does not modify the range or apply cumulative filters.
    To avoid recompilation on repeated evaluations, pass a CompiledFilter.

    Examples::

        selection = evaluate_filter("8!88", encoded_range.masks, mode=5)
        program = compile_filter(encode_filter("Ass", 5), 5)
        selection = evaluate_filter(program, encoded_range.masks)
    """
    if not isinstance(hand_masks, np.ndarray) or (
        hand_masks.ndim != 1 or hand_masks.dtype != np.uint64
    ):
        raise TypeError("hand_masks must be a 1D NumPy array of dtype uint64")

    if isinstance(expression, CompiledFilter):
        if mode is not None and mode != expression.mode:
            raise ValueError("mode does not match the compiled filter")
        compiled = expression
    else:
        if mode is None:
            raise ValueError("mode is required for filter strings and ASTs")
        ast = (
            encode_filter(expression, mode)
            if isinstance(expression, str)
            else expression
        )
        compiled = compile_filter(ast, mode)

    return _evaluate_compiled(
        hand_masks,
        compiled.operations,
        compiled.operands,
        compiled.slot_masks,
        compiled.slot_counts,
        compiled.mode,
    )
