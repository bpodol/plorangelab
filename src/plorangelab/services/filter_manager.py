from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from plorangelab.encoder import EncodedRange
from plorangelab.filters.evaluator import evaluate_filter
from plorangelab.types import Mode


@dataclass(slots=True)
class Filter:
    """A saved predicate evaluated against the street's original hand order.

    ``selection`` is immutable by convention and made read-only on creation.
    ``active`` is managed exclusively by FilterManager.
    """

    expression: str
    selection: NDArray[np.bool_]
    coverage: float
    active: bool = False


@dataclass(slots=True)
class FilterGroup:
    """A named collection of filters and/or nested groups.

    An active group is a gate: only its active descendants contribute. Turning
    a group on activates all descendants; toggling a child off then removes
    that child's contribution without disabling its siblings.
    """

    name: str
    filters: dict[str, Filter | FilterGroup] = field(default_factory=dict)
    active: bool = False


@dataclass(frozen=True, slots=True)
class FilterAddress:
    """Node name and path of *group names* from the root to its parent."""

    name: str
    path: tuple[str, ...] = ()


class FilterManager:
    """Own all filtering state for exactly one street.

    Invariants:
      * Leaf selections always align with ``base_range``.
      * A group's membership selection is the union of all its leaves.
      * Its *effective* selection uses only active descendants when the group
        and every ancestor are active.
      * Root-level active nodes are combined with OR, never AND.
      * No active leaf means ``current_range is base_range`` (unfiltered).
      * The current range and its coverage are derived caches, never edited
        independently by PlayerRangeState or RangeManager.
    """

    def __init__(self, base_range: EncodedRange, mode: Mode) -> None:
        self.mode = mode
        self.base_range = base_range
        self.base_weights_sum = float(np.sum(base_range.weights, dtype=np.float64))
        self._filters: dict[str, Filter | FilterGroup] = {}
        self._active_selection: NDArray[np.bool_] | None = None
        self._current_range = base_range
        self._filtered_coverage = 1.0

    @property
    def filters(self) -> Mapping[str, Filter | FilterGroup]:
        """Read-only view of root entries. Mutate only through manager methods."""
        return MappingProxyType(self._filters)

    @property
    def current_range(self) -> EncodedRange:
        """Always an EncodedRange; the unfiltered base if nothing is active."""
        return self._current_range

    @property
    def active_selection(self) -> NDArray[np.bool_] | None:
        """Read-only mask, or None when filtering is not applied."""
        return self._active_selection

    @property
    def filtered_coverage(self) -> float:
        """Fraction of original *weighted* range currently included."""
        return self._filtered_coverage

    def create_filter(
        self,
        filter_expression: str,
        name: str | None = None,
        path: tuple[str, ...] = (),
    ) -> float:
        key = filter_expression if name is None else name
        self._check_name(key)
        children, ancestors = self._parent(path)
        self._check_new_name(children, key)

        selection = evaluate_filter(filter_expression, self.base_range.masks, mode=self.mode)
        selection.setflags(write=False)

        # A zero-match filter can be saved, but cannot become active.
        active = bool(ancestors and ancestors[-1].active and selection.any())
        coverage = self._coverage(selection)
        children[key] = Filter(filter_expression, selection, coverage, active)
        self._refresh()
        return coverage

    def create_group(self, name: str, path: tuple[str, ...] = ()) -> None:
        self._check_name(name)
        children, _ = self._parent(path)
        self._check_new_name(children, name)
        # New groups start inactive, even inside an active group. Users may
        # activate them once they contain useful filters.
        children[name] = FilterGroup(name=name)

    def delete(self, address: FilterAddress) -> None:
        children, _ = self._parent(address.path)
        if address.name not in children:
            raise ValueError(f"Unknown filter or group: {address!r}")
        del children[address.name]
        # No separate address list exists, so deleting a group cannot leave
        # dangling references to its descendants.
        self._refresh()

    def set_active(self, address: FilterAddress, value: bool) -> None:
        node, ancestors = self._find(address)
        if node.active == value:
            return  # Idempotent: do not reset individually toggled children.
        if value:
            # Prevent activating expressions that cannot match any hands.
            if not self._membership_selection(node).any():
                raise ValueError("Cannot activate a filter/group matching zero hands")
            # Activating a child makes its parents traversable but does not
            # activate siblings. Activating a group activates its whole subtree.
            for group in ancestors:
                group.active = True
        self._set_subtree_active(node, value)
        self._refresh()

    def clear_all_filters(self) -> None:
        """Deactivate everything, preserving saved definitions and groups."""
        for node in self._filters.values():
            self._set_subtree_active(node, False)
        self._refresh()

    def get_node(self, address: FilterAddress) -> Filter | FilterGroup:
        """Access saved metadata (treat returned object as read-only)."""
        node, _ = self._find(address)
        return node

    def get_selection(
        self, address: FilterAddress, *, active_only: bool = False
    ) -> NDArray[np.bool_]:
        """Group membership union, or only currently active descendants.

        For ``active_only=True``, inactive ancestor groups also gate results.
        This returns a new, read-only Boolean array, not shared manager state.
        """
        node, ancestors = self._find(address)
        selection = np.zeros(len(self.base_range.masks), dtype=np.bool_)
        if not active_only or all(group.active for group in ancestors):
            self._union_into(selection, node, active_only=active_only)
        selection.setflags(write=False)
        return selection

    def get_coverage(
        self, address: FilterAddress, *, active_only: bool = False
    ) -> float:
        """Weighted coverage for a filter or a group, relative to its street."""
        return self._coverage(self.get_selection(address, active_only=active_only))

    def clone(self) -> FilterManager:
        """Copy mutable configuration, share immutable masks and range arrays.

        New filtering operations only replace cached arrays; they never mutate
        arrays shared with the original manager.
        """
        cloned = copy(self)
        cloned._filters = self._clone_nodes(self._filters)
        return cloned

    @staticmethod
    def _check_name(name: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Filter and group names must be non-empty strings")

    @staticmethod
    def _check_new_name(
        children: dict[str, Filter | FilterGroup], name: str
    ) -> None:
        if name in children:
            raise ValueError(f"Filter or group {name!r} already exists")

    def _parent(
        self, path: tuple[str, ...] = ()
    ) -> tuple[dict[str, Filter | FilterGroup], list[FilterGroup]]:
        children = self._filters
        ancestors: list[FilterGroup] = []
        for name in path:
            node = children.get(name)
            if not isinstance(node, FilterGroup):
                # Invalid path, not an invalid argument type.
                raise ValueError(f"Unknown group in path: {name!r}") # noqa: TRY004
            ancestors.append(node)
            children = node.filters
        return children, ancestors

    def _find(
        self, address: FilterAddress
    ) -> tuple[Filter | FilterGroup, list[FilterGroup]]:
        children, ancestors = self._parent(address.path)
        node = children.get(address.name)
        if node is None:
            raise ValueError(f"Unknown filter or group: {address!r}")
        return node, ancestors

    @classmethod
    def _set_subtree_active(cls, node: Filter | FilterGroup, value: bool) -> None:
        node.active = value
        if isinstance(node, FilterGroup):
            for child in node.filters.values():
                cls._set_subtree_active(child, value)

    @classmethod
    def _union_into(
        cls,
        destination: NDArray[np.bool_],
        node: Filter | FilterGroup,
        *,
        active_only: bool,
    ) -> None:
        if active_only and not node.active:
            return
        if isinstance(node, Filter):
            destination |= node.selection
        else:
            for child in node.filters.values():
                cls._union_into(destination, child, active_only=active_only)

    def _membership_selection(self, node: Filter | FilterGroup) -> NDArray[np.bool_]:
        result = np.zeros(len(self.base_range.masks), dtype=np.bool_)
        self._union_into(result, node, active_only=False)
        return result

    def _coverage(self, selection: NDArray[np.bool_]) -> float:
        numerator = np.sum(self.base_range.weights[selection], dtype=np.float64)
        return float(numerator / self.base_weights_sum)

    def _refresh(self) -> None:
        selection = np.zeros(len(self.base_range.masks), dtype=np.bool_)
        for node in self._filters.values():
            self._union_into(selection, node, active_only=True)

        if not selection.any():
            # All filter definitions may be inactive (or empty groups).
            self._active_selection = None
            self._current_range = self.base_range
            self._filtered_coverage = 1.0
            return

        new_range = EncodedRange(
            hands=self.base_range.hands[selection],
            masks=self.base_range.masks[selection],
            weights=self.base_range.weights[selection],
        )
        coverage = self._coverage(selection)
        selection.setflags(write=False)
        self._active_selection = selection
        self._current_range = new_range
        self._filtered_coverage = coverage

    @classmethod
    def _clone_nodes(
        cls, children: dict[str, Filter | FilterGroup]
    ) -> dict[str, Filter | FilterGroup]:
        result: dict[str, Filter | FilterGroup] = {}
        for name, node in children.items():
            if isinstance(node, Filter):
                result[name] = Filter(
                    expression=node.expression,
                    selection=node.selection,  # Read-only, shared intentionally.
                    coverage=node.coverage,
                    active=node.active,
                )
            else:
                result[name] = FilterGroup(
                    name=node.name,
                    filters=cls._clone_nodes(node.filters),
                    active=node.active,
                )
        return result
