"""Manage player ranges, street states, and equity calculations.

Coordinates up to six player ranges across poker streets, preserving
compatible street states and propagating filtered ranges forward.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from threading import Event

import numpy as np
from numpy.typing import NDArray

from plorangelab.bootstrap import EngineResources
from plorangelab.encoder import build_mask, encode_board, encode_input
from plorangelab.equity import (
    EquityResult,
    MonteCarloResult,
    compute_equity,
    sanitize_range,
)
from plorangelab.types import Mode

from .player_range import (
    FilterAddress,
    PlayerRange,
)


class Street(IntEnum):
    PREFLOP = 0
    FLOP = 3
    TURN = 4
    RIVER = 5


@dataclass(frozen=True, slots=True)
class PlayerFilterSnapshot:
    """Serialized view of a player's current street state.

    Attributes:
        base_hands: Number of hands in the current street's base range,
            which may incorporate filtering from earlier streets.
        filtered_hands: Number of hands remaining after active filters.
        filtered_coverage: Fraction of base-range weight retained.
        filters: Serialized tree of filter and group metadata.
    """

    base_hands: int
    filtered_hands: int
    filtered_coverage: float
    filters: dict[str, dict[str, object]]


@dataclass(frozen=True, eq=False, slots=True)
class Board:
    cards: NDArray[np.uint8]
    mask: np.uint64

    @classmethod
    def from_string(cls, board: str) -> Board:
        """Create an immutable board representation from card notation.

        Encodes the cards and builds an order-independent bitmask used
        for board identity and compatibility checks.

        Args:
            board: Concatenated notation of 3, 4, or 5 cards.

        Returns:
            Board with read-only encoded cards and a combined bitmask.

        Raises:
            ValueError: If the notation is invalid or contains duplicate cards.
        """
        cards = encode_board(board)
        mask = build_mask(cards)
        cards.setflags(write=False)

        return cls(cards=cards, mask=mask)


@dataclass(slots=True)
class StreetState:
    board: Board | None
    ranges: list[PlayerRange | None] = field(default_factory=lambda: [None] * 6)


class RangeManager:
    """Manage player ranges, shared board states, and equity calculations.

    Maintains up to six player ranges per street. Each street owns
    independent filtering state, while filtered ranges are propagated
    forward when creating subsequent streets.

    The preflop state always exists. Postflop states are created on
    demand and preserved while their boards remain compatible.

    All players share a single PLO mode, initially PLO4.
    """

    def __init__(
        self,
        resources: EngineResources,
    ) -> None:
        self.resources = resources
        self._streets: dict[Street, StreetState] = {Street.PREFLOP: StreetState(None)}
        self._active_street = Street.PREFLOP
        self.mode: Mode = 4

    @property
    def _current_state(self) -> StreetState:
        return self._streets[self._active_street]

    def change_mode(self, mode: Mode) -> None:
        """Change the PLO mode. There should be no active ranges
        in order to change the mode.

        Does nothing if the requested mode is already active.

        Raises:
            ValueError: If the mode is not PLO4 or PLO5.
            RuntimeError: if active ranges are present
        """
        if mode not in (4, 5):
            raise ValueError("Invalid Mode")
        if self.mode == mode:
            return
        else:
            if any(self._current_state.ranges):
                raise RuntimeError("Cannot change mode with active player ranges")

            self.mode = mode

    def add_range(self, range_input: str, slot_index: int) -> PlayerFilterSnapshot:
        """Add or replace a player's range across all existing streets.

        Encodes the input range and propagates it through each existing board,
        removing hands that conflict with community cards.

        Replaces the player's previous range and filters on every street.
        Changes are committed only after all street ranges are created.

        Returns:
            A number of hands in the new added range after removing the board card(if present)

        Raises:
            ValueError: If the slot, range input, or resulting range is invalid.
        """
        self._validate_slot(slot_index)

        encoded_range = encode_input(
            range_input,
            self.mode,
            self.resources.ratings[self.mode],
        )

        parent_range = PlayerRange(encoded_range, self.mode)

        replacements = {Street.PREFLOP: parent_range}

        for street in sorted(self._streets):
            if street == Street.PREFLOP:
                continue

            state = self._streets[street]

            sanitized_range = sanitize_range(
                parent_range.current_range,
                state.board.mask,
            )

            parent_range = PlayerRange(
                sanitized_range,
                self.mode,
            )

            replacements[street] = parent_range

        new_streets = self._streets.copy()

        for street, player_range in replacements.items():
            old_state = self._streets[street]
            new_ranges = old_state.ranges.copy()
            new_ranges[slot_index] = player_range

            new_streets[street] = StreetState(
                board=old_state.board,
                ranges=new_ranges,
            )

        new_state = new_streets[self._active_street]
        if new_state.board is not None:
            new_state.ranges[slot_index].autogenerate_filters(new_state.board.cards)
        snapshot = self._get_player_snapshot(new_state.ranges[slot_index])

        self._streets = new_streets

        return snapshot

    def remove_range(self, slot_index: int) -> None:
        """Remove a player's range and filters from all existing streets.

        Preserves the ranges and filter states of other players.
        """
        self._validate_slot(slot_index)

        new_streets = {}

        for street, state in self._streets.items():
            new_ranges = state.ranges.copy()
            new_ranges[slot_index] = None

            new_streets[street] = StreetState(
                board=state.board,
                ranges=new_ranges,
            )

        self._streets = new_streets

    def change_board(
        self, input_board: str | None
    ) -> list[PlayerFilterSnapshot | None]:
        """Change the active board while preserving compatible street states.

        Reuses existing street states when their boards match, regardless
        of card order. Preserves compatible earlier streets and discards
        incompatible or later states.

        New street ranges are derived from the nearest preserved parent's
        filtered ranges, excluding hands that conflict with the new board.

        Passing None resets to preflop and discards all postflop states.

        The change is committed only after all new ranges are constructed.

        Returns:
            Six player snapshots in slot order, with None for empty slots.

        Raises:
            ValueError: If the board is invalid or sanitization leaves
                any player's range empty.
        """
        if input_board is None:
            self._streets = {Street.PREFLOP: self._streets[Street.PREFLOP]}
            self._active_street = Street.PREFLOP
            return self._get_players_snapshot()

        new_board = Board.from_string(input_board)
        target_street = Street(len(new_board.cards))

        # Existing identical board: reuse the state and its filters.
        existing = self._streets.get(target_street)

        if (
            existing is not None
            and existing.board is not None
            and existing.board.mask == new_board.mask
        ):
            self._streets = {
                street: state
                for street, state in self._streets.items()
                if street <= target_street
            }

            self._active_street = target_street
            return self._get_players_snapshot()

        # Preflop always survives.
        new_streets = {Street.PREFLOP: self._streets[Street.PREFLOP]}

        # Preserve compatible earlier streets.
        for street in (Street.FLOP, Street.TURN):
            if street >= target_street:
                break

            old_state = self._streets.get(street)

            if old_state is None or old_state.board is None:
                continue

            if self._extends(old_state.board, new_board):
                new_streets[street] = old_state

        # Find the nearest surviving parent.
        parent_street = max(new_streets)
        parent_state = new_streets[parent_street]

        # Create the target street state from its parent.
        new_state = StreetState(new_board)

        for slot, parent_range in enumerate(parent_state.ranges):
            if parent_range is None:
                continue

            parent_range = parent_range.current_range

            sanitized_range = sanitize_range(parent_range, new_board.mask)

            new_player = PlayerRange(
                sanitized_range,
                self.mode,
            )

            new_player.autogenerate_filters(new_board.cards)

            new_state.ranges[slot] = new_player

        new_streets[target_street] = new_state

        # Commit only after successful construction.
        self._streets = new_streets
        self._active_street = target_street

        return self._get_players_snapshot()

    def add_filter(
        self,
        expression: str,
        slot_index: int,
        name: str | None = None,
        path: tuple[str, ...] = (),
    ) -> PlayerFilterSnapshot:
        return self._edit_player(
            slot_index, lambda player: player.create_filter(expression, name, path)
        )

    def add_group(
        self, name: str, slot_index: int, path: tuple[str, ...] = ()
    ) -> PlayerFilterSnapshot:
        return self._edit_player(
            slot_index, lambda player: player.create_group(name, path)
        )

    def delete_filter(
        self, name: str, slot_index: int, path: tuple[str, ...] = ()
    ) -> PlayerFilterSnapshot:
        return self._edit_player(
            slot_index, lambda player: player.delete(FilterAddress(name, path))
        )

    def set_filter_active(
        self,
        name: str,
        slot_index: int,
        active: bool,
        path: tuple[str, ...] = (),
    ) -> PlayerFilterSnapshot:
        return self._edit_player(
            slot_index,
            lambda player: player.set_active(FilterAddress(name, path), active),
        )

    def clear_filters(self, slot_index: int) -> PlayerFilterSnapshot:
        return self._edit_player(slot_index, lambda player: player.clear_all_filters())

    def calculate_equity(
        self,
        on_progress: Callable[[MonteCarloResult], None] | None = None,
        cancel_event: Event | None = None,
        target_moe: float = 0.001,
    ) -> tuple[list[int], EquityResult]:
        """Calculate equity for all occupied player slots.

        Collects the current filtered range from each occupied player,
        preserving their slot order, and delegates the calculation to
        the equity engine.

        Uses the shared board and precomputed ranking tables.
        Monte Carlo progress callbacks, cancellation, and precision
        settings are forwarded to compute_equity().

        Args:
            on_progress: Optional callback receiving intermediate
                MonteCarloResult instances.
            cancel_event: Optional event requesting calculation cancellation.
            target_moe: Target absolute Monte Carlo margin of error.

        Returns:
            A tuple containing:
                - Player slot indices in calculation order.
                - EquityResult with corresponding per-player equities.

        Raises:
            ValueError: If the player configuration or equity inputs
                are invalid.
            RuntimeError: If the delegated calculation fails or exceeds
                an execution limit.

        Notes:
            This method does not modify player ranges or filter states.
            Equity values are aligned with the returned player indices,
            not necessarily with all six available slots.
        """
        state = self._current_state
        active_players = [
            (slot, player_range)
            for slot, player_range in enumerate(state.ranges)
            if player_range is not None
        ]

        ids = [slot for slot, _ in active_players]
        ranges = [player.current_range for _, player in active_players]

        equity_result = compute_equity(
            players=ranges,
            mode=self.mode,
            board=state.board.cards if state.board is not None else None,
            on_progress=on_progress,
            cancel_event=cancel_event,
            target_moe=target_moe,
            rank_lookup=self.resources.rank_lookup,
        )

        return ids, equity_result

    @staticmethod
    def _validate_slot(slot_index: int) -> None:
        if not 0 <= slot_index < 6:
            raise ValueError("Out of range slot index")

    @staticmethod
    def _extends(previous: Board, next_board: Board) -> bool:
        """Check whether all cards of the previous board exist in the next board.

        Uses bitmasks to compare card membership regardless of card order.
        """
        return bool((previous.mask & next_board.mask) == previous.mask)

    @staticmethod
    def _get_player_snapshot(player: PlayerRange) -> PlayerFilterSnapshot:
        return PlayerFilterSnapshot(
            base_hands=len(player.base_range.hands),
            filtered_hands=len(player.current_range.hands),
            filtered_coverage=player.filtered_coverage,
            filters=player.get_tree(),
        )

    def _get_player(self, slot_index: int) -> PlayerRange:
        self._validate_slot(slot_index)
        player = self._current_state.ranges[slot_index]
        if player is None:
            raise ValueError("Player slot is empty")
        return player

    def _get_players_snapshot(self) -> list[PlayerFilterSnapshot | None]:
        """Return filter snapshots for all six player slots.

        Preserves slot order, representing empty slots with None.
        """
        players = self._current_state.ranges
        return [
            self._get_player_snapshot(player) if player is not None else None
            for player in players
        ]

    def _edit_player(
        self,
        slot_index: int,
        change: Callable[[PlayerRange], None],
    ) -> PlayerFilterSnapshot:
        """Apply a filter change using copy-on-write.

        Clone the player's range, apply the change, and build its snapshot
        before committing. If any step fails, the original state is preserved.
        """
        updated = self._get_player(slot_index).clone()
        change(updated)

        snapshot = self._get_player_snapshot(updated)

        self._current_state.ranges[slot_index] = updated
        return snapshot
