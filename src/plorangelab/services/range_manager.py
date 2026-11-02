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
        ):
        self.ratings = resources.ratings
        self.rank_lookup = resources.rank_lookup
        self.players: list[PlayerRangeState | None] = [None] * 6
        self.board: Board | None = None
        

    def add_range(self, input_range: str, mode: Mode, player_index: int) -> None:
        if not 0 <= player_index <= 5:
            raise ValueError("Out of range player index")
        encoded_range = encode_input(input_range, mode, self.ratings[mode])
        new_player = PlayerRangeState(hands=encoded_range, mode=mode, id=player_index)
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

    def calculate_equity(
            self, 
            mode: Mode, 
            on_progress: Callable[[MonteCarloResult], None] | None = None, 
            cancel_event: Event | None = None, 
            target_moe: float = 0.001
        ) -> tuple[list[int], EquityResult]:
   
        active_players = [player for player in self.players if player is not None]
        ranges = [player.active_street.ranges[-1] for player in active_players]
       
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

