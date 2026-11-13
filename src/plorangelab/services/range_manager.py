"""Coordinate player ranges, board state, filters, and equity calculations.

Manages six optional player slots and delegates street-specific state
to PlayerRangeState instances.

Uses clone-based updates to prevent failed operations from publishing
partially modified player or board state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event

from plorangelab.bootstrap import EngineResources
from plorangelab.encoder import encode_input
from plorangelab.equity import (
    EquityResult,
    MonteCarloResult,
    compute_equity,
)
from plorangelab.types import Mode

from .player_range_state import Board, PlayerRangeState


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


class RangeManager:
    """Manage player ranges and shared board state for one calculation context.

    Owns up to six player slots and a shared community board. Delegates
    range parsing to the encoder, street transitions to PlayerRangeState,
    and filter operations to each player's active FilterManager.

    Coordinates equity calculations using the currently selected ranges
    and shared ranking resources.

    State-changing operations prepare replacement objects before
    publishing them, preserving the previous state if preparation fails.
    """

    def __init__(
        self,
        resources: EngineResources,
    ) -> None:
        self.ratings = resources.ratings
        self.rank_lookup = resources.rank_lookup
        self.players: list[PlayerRangeState | None] = [None] * 6
        self.board: Board | None = None

    def _get_player(self, player_index: int) -> PlayerRangeState:
        if not 0 <= player_index < len(self.players):
            raise ValueError("Out of range player index")
        player = self.players[player_index]
        if player is None:
            raise ValueError("Player slot is empty")
        return player

    def _edit_player[T](
        self, player_index: int, change: Callable[[PlayerRangeState], T]
    ) -> T:
        """Atomic manager-level updates: publish only after a successful edit."""
        updated = self._get_player(player_index).clone()
        result = change(updated)
        self.players[player_index] = updated
        return result

    def _get_player_snapshot(self, player_index: int) -> PlayerFilterSnapshot:
        """Build a snapshot of an occupied player's current street.

        Args:
            player_index: Zero-based player slot index.

        Returns:
            Current street hand counts, coverage, and serialized filter tree.

        Raises:
            ValueError: If the player index is invalid or the slot is empty.
        """

        player = self._get_player(player_index)
        active_street = player.active_street

        return PlayerFilterSnapshot(
            base_hands=len(active_street.filter_manager.base_range.hands),
            filtered_hands=len(active_street.current_range.hands),
            filtered_coverage=active_street.filtered_coverage,
            filters=self.get_filters(player_index),
        )

    def _get_players_snapshot(self) -> list[PlayerFilterSnapshot | None]:
        """Return filter snapshots for all six player slots.

        Preserves slot order, representing empty slots with None.
        """

        return [
            self._get_player_snapshot(player_index) if player is not None else None
            for player_index, player in enumerate(self.players)
        ]

    def add_range(self, input_range: str, mode: Mode, player_index: int) -> None:
        """Create or replace a player range in the specified slot.

        Parses the supplied range notation and initializes a new
        PlayerRangeState. If a board is already selected, applies that
        board before publishing the new player state.

        Replacing a player discards the previous range and filter states
        associated with that slot.

        Args:
            input_range: Explicit weighted hands or percentage-based range.
            mode: PLO variant (4 or 5).
            player_index: Zero-based player slot index (0-5).

        Raises:
            ValueError: If the slot index or range is invalid, or board
                filtering removes all hands.
        """

        if not 0 <= player_index <= 5:
            raise ValueError("Out of range player index")
        encoded_range = encode_input(input_range, mode, self.ratings[mode])
        new_player = PlayerRangeState(hands=encoded_range, id=player_index, mode=mode)
        if self.board is not None:
            new_player.apply_board_change(self.board)
        self.players[player_index] = new_player

    def remove_range(self, player_index: int) -> None:
        if not 0 <= player_index <= 5:
            raise ValueError("Out of range player index")
        self.players[player_index] = None

    def change_board(
        self, input_board: str | None
    ) -> list[PlayerFilterSnapshot | None]:
        """Update the shared board and all occupied player slots atomically.

        Creates a validated Board instance and applies the change to
        clones of every occupied PlayerRangeState.

        Each player independently preserves compatible street states
        and updates its current range according to the new board.

        The updated player filter snapshot list and board are published only after
        every player has been updated successfully.

        Passing None clears the board and returns all players to preflop.

        Args:
            input_board: Concatenated notation of 3, 4, or 5 community
                cards, or None to clear the board.

        Returns:
            A six-element list in player slot order. Each occupied slot
            contains its current PlayerFilterSnapshot; empty slots contain None

        Raises:
            ValueError: If the board is invalid or the change leaves
                any player's applicable range empty.
        """

        if input_board is None:
            self.clear_board()
            return self._get_players_snapshot()

        new_board = Board.from_string(input_board)
        # Stage changes to every player before publishing the new state.
        # If any player update fails, the original players and board remain intact.
        updated_players = self.players.copy()

        for i, player in enumerate(self.players):
            if player is not None:
                updated_player = player.clone()
                updated_player.apply_board_change(new_board)

                updated_players[i] = updated_player

        self.players = updated_players
        self.board = new_board

        return self._get_players_snapshot()

    def clear_board(self) -> None:
        """Clear the community board and reset all players to preflop.

        Clones each occupied player and discards its postflop street
        states while preserving its preflop range and filters.

        Publishes the updated player list and clears the shared board
        after all player updates have completed.
        """

        updated_players = self.players.copy()

        for index, player in enumerate(self.players):
            if player is not None:
                updated_player = player.clone()
                updated_player.clear_board()
                updated_players[index] = updated_player

        self.players = updated_players
        self.board = None

    def add_filter(
        self,
        expression: str,
        player_index: int,
        name: str | None = None,
        path: tuple[str, ...] = (),
    ) -> PlayerFilterSnapshot:
        self._edit_player(
            player_index, lambda player: player.add_filter(expression, name, path)
        )
        return self._get_player_snapshot(player_index)

    def add_group(
        self, name: str, player_index: int, path: tuple[str, ...] = ()
    ) -> PlayerFilterSnapshot:
        self._edit_player(player_index, lambda player: player.add_group(name, path))
        return self._get_player_snapshot(player_index)

    def delete_filter(
        self, name: str, player_index: int, path: tuple[str, ...] = ()
    ) -> PlayerFilterSnapshot:
        self._edit_player(player_index, lambda player: player.delete_filter(name, path))
        return self._get_player_snapshot(player_index)

    def set_filter_active(
        self,
        name: str,
        player_index: int,
        active: bool,
        path: tuple[str, ...] = (),
    ) -> PlayerFilterSnapshot:
        self._edit_player(
            player_index, lambda player: player.set_filter_active(name, active, path)
        )
        return self._get_player_snapshot(player_index)

    def clear_filters(self, player_index: int) -> PlayerFilterSnapshot:
        self._edit_player(player_index, lambda player: player.clear_filters())
        return self._get_player_snapshot(player_index)

    def get_filters(self, player_index: int) -> dict[str, dict[str, object]]:
        return self._get_player(player_index).get_filters()

    def get_filter(
        self,
        name: str,
        player_index: int,
        path: tuple[str, ...] = (),
    ) -> dict[str, object]:
        return self._get_player(player_index).get_filter(name, path)

    def calculate_equity(
        self,
        mode: Mode,
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
            mode: PLO variant used for the equity calculation (4 or 5).
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

        active_players = [player for player in self.players if player is not None]
        ranges = [player.current_range for player in active_players]

        equity_result = compute_equity(
            players=ranges,
            mode=mode,
            board=self.board.cards if self.board is not None else None,
            on_progress=on_progress,
            cancel_event=cancel_event,
            target_moe=target_moe,
            rank_lookup=self.rank_lookup,
        )

        ids = [player.id for player in active_players]

        return ids, equity_result
