"""Structural tests for immutable filter AST nodes."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from plorangelab.filters.models import And, AtomicFilter, CardRequirement, Not, Or


def test_filter_models_can_form_nested_ast() -> None:
    ace = AtomicFilter((CardRequirement(np.uint64(1 << 51), 1),))
    king = AtomicFilter((CardRequirement(np.uint64(1 << 47), 1),))

    tree = And(Or(ace, king), Not(ace))

    assert tree.left.left is ace
    assert tree.left.right is king
    assert tree.right.child is ace


@pytest.mark.parametrize(
    "value",
    [CardRequirement(np.uint64(1), 1), AtomicFilter(()), Not(AtomicFilter(()))],
)
def test_models_are_frozen(value: object) -> None:
    field_name = next(iter(value.__dataclass_fields__))
    with pytest.raises(FrozenInstanceError):
        setattr(value, field_name, getattr(value, field_name))
