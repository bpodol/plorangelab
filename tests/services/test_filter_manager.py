"""Unit tests for filter definitions, hierarchical activation, coverage, and cloning."""

import numpy as np
import pytest

from plorangelab.encoder import EncodedRange, encode_range
from plorangelab.services.filter_manager import (
    FilterAddress,
    FilterGroup,
    FilterManager,
)

HANDS = (
    "AsKsQhJc@100,"
    "AdKhQcJd@50,"
    "AcAhQhJc@25,"
    "KsKhQhJh@80,"
    "9s8s7s6s@90,"
    "2c3d4h5s@60"
)


@pytest.fixture
def base() -> EncodedRange:
    return encode_range(HANDS, 4)


@pytest.fixture
def manager(base: EncodedRange) -> FilterManager:
    return FilterManager(base, 4)


def indices(base: EncodedRange, current: EncodedRange) -> set[int]:
    lookup = {int(mask): i for i, mask in enumerate(base.masks)}
    return {lookup[int(mask)] for mask in current.masks}


def mask_indices(selection: np.ndarray) -> set[int]:
    return set(np.flatnonzero(selection))


def test_default_has_valid_original_range(manager, base):
    assert manager.current_range is base
    assert manager.active_selection is None
    assert manager.filtered_coverage == 1.0


def test_root_filters_start_inactive_and_can_be_activated(manager, base):
    coverage = manager.create_filter("A", name="aces")
    expected = float(
        np.sum(base.weights[:3], dtype=np.float64)
        / np.sum(base.weights, dtype=np.float64)
    )

    assert coverage == pytest.approx(expected)
    assert manager.filtered_coverage == 1.0
    assert manager.current_range is base
    leaf = manager.get_node(FilterAddress("aces"))
    assert leaf.active is False
    assert leaf.selection.flags.writeable is False
    assert mask_indices(leaf.selection) == {0, 1, 2}

    manager.set_active(FilterAddress("aces"), True)
    assert indices(base, manager.current_range) == {0, 1, 2}
    assert manager.filtered_coverage == pytest.approx(
        float(
            np.sum(base.weights[:3], dtype=np.float64)
            / np.sum(base.weights, dtype=np.float64)
        )
    )


def test_union_and_deactivation_fallback(manager, base):
    manager.create_filter("A")
    manager.create_filter("K")
    manager.set_active(FilterAddress("A"), True)
    manager.set_active(FilterAddress("K"), True)
    assert indices(base, manager.current_range) == {0, 1, 2, 3}
    manager.set_active(FilterAddress("A"), False)
    assert indices(base, manager.current_range) == {0, 1, 3}
    manager.set_active(FilterAddress("K"), False)
    assert manager.current_range is base


def test_groups_membership_and_child_activation(manager, base):
    manager.create_group("g")
    manager.create_filter("A", path=("g",))
    manager.create_filter("K", path=("g",))
    manager.create_group("inner", path=("g",))
    manager.create_filter("ss", path=("g", "inner"))
    g = FilterAddress("g")
    a = FilterAddress("A", ("g",))
    k = FilterAddress("K", ("g",))
    inner = FilterAddress("inner", ("g",))

    assert indices(base, base) == set(range(6))
    assert mask_indices(manager.get_selection(g)) == {0, 1, 2, 3, 4}
    assert mask_indices(manager.get_selection(g, active_only=True)) == set()
    manager.set_active(g, True)
    assert indices(base, manager.current_range) == {0, 1, 2, 3, 4}
    assert manager.get_node(a).active and manager.get_node(k).active
    assert isinstance(manager.get_node(inner), FilterGroup)

    manager.set_active(k, False)
    assert indices(base, manager.current_range) == {0, 1, 2, 4}
    manager.set_active(g, False)
    assert manager.current_range is base
    assert not manager.get_node(a).active
    assert not manager.get_node(inner).active

    manager.set_active(inner, True)  # Automatically opens the ancestor gate.
    assert manager.get_node(g).active
    assert not manager.get_node(a).active
    assert indices(base, manager.current_range) == {0, 4}

    manager.set_active(g, True)  # Idempotent: siblings stay disabled.
    assert indices(base, manager.current_range) == {0, 4}
    manager.set_active(g, False)
    manager.set_active(g, True)  # Fresh activation enables all children.
    assert indices(base, manager.current_range) == {0, 1, 2, 3, 4}


def test_add_to_active_group_auto_activates_leaf(manager, base):
    manager.create_group("g")
    manager.create_filter("A", path=("g",))
    manager.set_active(FilterAddress("g"), True)
    coverage = manager.create_filter("K", path=("g",))

    assert coverage == pytest.approx(
        manager.get_coverage(FilterAddress("K", ("g",)))
    )
    assert manager.filtered_coverage > coverage
    assert manager.get_node(FilterAddress("K", ("g",))).active
    assert indices(base, manager.current_range) == {0, 1, 2, 3}


def test_empty_group_rejected_without_mutation(manager, base):
    manager.create_group("empty")
    with pytest.raises(ValueError, match="zero hands"):
        manager.set_active(FilterAddress("empty"), True)
    assert manager.current_range is base
    assert not manager.get_node(FilterAddress("empty")).active


def test_zero_match_filter_stays_inactive(manager, base):
    coverage = manager.create_filter("TT")
    assert coverage == 0.0
    with pytest.raises(ValueError, match="zero hands"):
        manager.set_active(FilterAddress("TT"), True)
    assert manager.current_range is base


def test_clear_retains_definitions_and_deactivates_nested_children(manager, base):
    manager.create_group("g")
    manager.create_filter("A", path=("g",))
    manager.set_active(FilterAddress("g"), True)
    manager.clear_all_filters()
    assert manager.filtered_coverage == 1.0
    assert manager.active_selection is None
    assert manager.current_range is base
    assert not manager.get_node(FilterAddress("g")).active
    assert not manager.get_node(FilterAddress("A", ("g",))).active
    manager.set_active(FilterAddress("g"), True)
    assert indices(base, manager.current_range) == {0, 1, 2}


def test_delete_group_removes_entire_subtree(manager, base):
    manager.create_group("g")
    manager.create_filter("A", path=("g",))
    manager.set_active(FilterAddress("g"), True)
    manager.delete(FilterAddress("g"))
    assert manager.current_range is base
    with pytest.raises(ValueError, match="Unknown group"):
        manager.get_node(FilterAddress("A", ("g",)))


def test_duplicate_names_bad_paths_and_invalid_expression_do_not_mutate(manager, base):
    manager.create_group("g")
    manager.create_filter("A")
    for action in [
        lambda: manager.create_filter("K", name="A"),
        lambda: manager.create_group("g"),
        lambda: manager.create_filter("A", path=["missing"]),
        lambda: manager.create_filter("A|"),
        lambda: manager.create_filter("AA", path=["A"]),
    ]:
        with pytest.raises(ValueError):
            action()
        assert manager.current_range is base
    assert list(manager.filters) == ["g", "A"]


def test_clone_isolates_tree_but_reuses_immutable_arrays(manager, base):
    manager.create_group("g")
    manager.create_filter("A", path=("g",))
    manager.set_active(FilterAddress("g"), True)
    cloned = manager.clone()
    node = FilterAddress("A", ("g",))
    assert cloned.get_node(node) is not manager.get_node(node)
    assert cloned.get_node(node).selection is manager.get_node(node).selection
    assert cloned.current_range is manager.current_range
    cloned.set_active(node, False)
    assert cloned.current_range is base
    assert indices(base, manager.current_range) == {0, 1, 2}


def test_group_coverage_excludes_overlap(manager, base):
    manager.create_group("g")
    manager.create_filter("A", path=("g",))
    manager.create_filter("K", path=("g",))
    expected = sum(map(float, base.weights[:4])) / sum(map(float, base.weights))
    assert manager.get_coverage(FilterAddress("g")) == pytest.approx(expected)


def test_full_nested_expression(manager, base):
    manager.create_filter("(A|K):!ss", name="no-double-spades")
    manager.set_active(FilterAddress("no-double-spades"), True)
    assert indices(base, manager.current_range) == {1, 2, 3}


def test_active_group_allows_saving_nonmatching_filter(manager, base):
    manager.create_group("g")
    manager.create_filter("A", path=("g",))
    manager.set_active(FilterAddress("g"), True)
    manager.create_filter("TT", path=("g",))
    assert not manager.get_node(FilterAddress("TT", ("g",))).active
    assert indices(base, manager.current_range) == {0, 1, 2}


def test_five_card_game_mode_filters_work_with_real_evaluator():
    base = encode_range("AsKsQsJhTc,AdKdQhJcTs,AcAhKdQdJs", 5)
    manager = FilterManager(base, 5)
    manager.create_filter("Ass", name="two-spades")
    manager.set_active(FilterAddress("two-spades"), True)
    assert indices(base, manager.current_range) == {0}
    with pytest.raises(ValueError):
        manager.create_filter("AKQJT9")


def test_filter_address_value_equality_and_hashability() -> None:
    a = FilterAddress("A", ("Premium", "Aggressive"))
    b = FilterAddress("A", ("Premium", "Aggressive"))
    c = FilterAddress("A", ("Premium",))
    assert a == b
    assert a != c
    assert {a, b, c} == {a, c}
    assert a.path == ("Premium", "Aggressive")


def test_get_selection_returns_independent_read_only_snapshot(
    manager: FilterManager,
) -> None:
    manager.create_filter("A")
    address = FilterAddress("A")
    selection = manager.get_selection(address)
    assert not selection.flags.writeable
    assert mask_indices(selection) == {0, 1, 2}
    manager.set_active(address, True)
    assert mask_indices(selection) == {0, 1, 2}
    assert manager.get_selection(address, active_only=True).flags.writeable is False


def test_active_only_group_coverage_and_membership(manager: FilterManager) -> None:
    manager.create_group("premium")
    manager.create_filter("A", path=("premium",))
    manager.create_filter("K", path=("premium",))
    address = FilterAddress("premium")
    assert mask_indices(manager.get_selection(address)) == {0, 1, 2, 3}
    assert mask_indices(manager.get_selection(address, active_only=True)) == set()
    manager.set_active(address, True)
    manager.set_active(FilterAddress("K", ("premium",)), False)
    assert mask_indices(manager.get_selection(address, active_only=True)) == {0, 1, 2}
    assert manager.get_coverage(address, active_only=True) < manager.get_coverage(address)


def test_delete_active_root_restores_base(manager: FilterManager, base: EncodedRange) -> None:
    manager.create_filter("A")
    manager.set_active(FilterAddress("A"), True)
    manager.delete(FilterAddress("A"))
    assert manager.current_range is base
    assert not manager.filters


def test_root_filter_activation_is_idempotent(manager: FilterManager) -> None:
    manager.create_filter("A")
    address = FilterAddress("A")
    manager.set_active(address, True)
    original_range = manager.current_range
    manager.set_active(address, True)
    assert manager.current_range is original_range
    manager.set_active(address, False)
    manager.set_active(address, False)
    assert manager.active_selection is None

