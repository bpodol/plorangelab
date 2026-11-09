from __future__ import annotations

from collections.abc import Callable
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


class RangeManager:
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

    def _edit_player(
        self, player_index: int, change: Callable[[PlayerRangeState], None]
    ) -> None:
        """Atomic manager-level updates: publish only after a successful edit."""
        updated = self._get_player(player_index).clone()
        change(updated)
        self.players[player_index] = updated

    def add_range(self, input_range: str, mode: Mode, player_index: int) -> None:
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

    def change_board(self, input_board: str | None):
        if input_board is None:
            self.clear_board()
            return
        new_board = Board.from_string(input_board)
        updated_players = self.players.copy()
        
        for i, player in enumerate(self.players):
            if player is not None:
                updated_player = player.clone()
                updated_player.apply_board_change(new_board)

                updated_players[i] = updated_player

        self.players = updated_players
        self.board = new_board

    def clear_board(self) -> None:
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
    ) -> None:
        self._edit_player(
            player_index, lambda player: player.add_filter(expression, name, path)
        )

    def add_group(
        self, name: str, player_index: int, path: tuple[str, ...] = ()
    ) -> None:
        self._edit_player(player_index, lambda player: player.add_group(name, path))

    def delete_filter(
        self, name: str, player_index: int, path: tuple[str, ...] = ()
    ) -> None:
        self._edit_player(
            player_index, lambda player: player.delete_filter(name, path)
        )

    def set_filter_active(
        self,
        name: str,
        player_index: int,
        active: bool,
        path: tuple[str, ...] = (),
    ) -> None:
        def change(player: PlayerRangeState) -> None:
            if active:
                player.activate_filter(name, path)
            else:
                player.deactivate_filter(name, path)

        self._edit_player(player_index, change)

    def clear_filters(self, player_index: int) -> None:
        self._edit_player(player_index, lambda player: player.clear_filters())

    def calculate_equity(
            self, 
            mode: Mode, 
            on_progress: Callable[[MonteCarloResult], None] | None = None, 
            cancel_event: Event | None = None, 
            target_moe: float = 0.001
        ) -> tuple[list[int], EquityResult]:
   
        active_players = [player for player in self.players if player is not None]
        ranges = [player.current_range for player in active_players]
       
        equity_result = compute_equity(
            players=ranges, 
            mode=mode, 
            board=self.board.cards if self.board is not None else None,
            on_progress=on_progress, 
            cancel_event=cancel_event, 
            target_moe=target_moe,
            rank_lookup=self.rank_lookup
        )
    
        ids = [player.id for player in active_players]
     
        return ids, equity_result

