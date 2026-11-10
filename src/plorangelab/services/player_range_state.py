from __future__ import annotations

from copy import copy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from plorangelab.encoder import EncodedRange, build_mask, encode_board
from plorangelab.equity import sanitize_range
from plorangelab.types import Mode

from .filter_manager import (
    FilterAddress,
    FilterManager,
)


@dataclass(frozen=True, eq=False, slots=True)
class Board:
    cards: NDArray[np.uint8] 
    mask: np.uint64 

    @classmethod
    def from_string(cls, board: str) -> Board:
        cards = encode_board(board)
        mask = build_mask(cards)
        cards.setflags(write=False)

        return cls(cards=cards, mask=mask)


@dataclass
class StreetState: 
    board: Board | None
    filter_manager: FilterManager

    @classmethod
    def from_range(
        cls, base_range: EncodedRange, board: Board | None, mode: Mode
    ) -> StreetState:
        return cls(board=board, filter_manager=FilterManager(base_range, mode))

    @property
    def current_range(self) -> EncodedRange:
        return self.filter_manager.current_range

    @property
    def filtered_coverage(self) -> float:
        return self.filter_manager.filtered_coverage

    def clone(self) -> StreetState:
        return StreetState(board=self.board, filter_manager=self.filter_manager.clone())


def _extends(previous: Board, next_board: Board) -> bool:
    """True if the next board adds only cards, without replacing any."""
    return bool((previous.mask & next_board.mask) == previous.mask)
    

class PlayerRangeState:
    def __init__(self, hands: EncodedRange, id: int, mode: Mode):
        self.mode: Mode = mode
        self.id: int = id
        self.preflop = StreetState.from_range(hands, None, mode)
        self.flop: StreetState | None = None
        self.turn: StreetState | None = None
        self.river: StreetState | None = None
        self.active_street: StreetState = self.preflop

    @property
    def current_range(self) -> EncodedRange:
        return self.active_street.current_range

    @property
    def filtered_coverage(self) -> float:
        return self.active_street.filtered_coverage

    def apply_board_change(self, board: Board) -> None:
        street_size = len(board.cards)
        if street_size == 3:
            if self.flop is not None and board.mask == self.flop.board.mask:
                next_flop = self.flop
            else:
                sanitized_range = sanitize_range(self.preflop.current_range, board.mask)
                next_flop = StreetState.from_range(sanitized_range, board, self.mode)
            self.flop = next_flop
            self.turn = None
            self.river = None
            self.active_street = next_flop
            return

        elif street_size == 4:
            if self.turn is not None and board.mask == self.turn.board.mask:
                next_turn = self.turn
                keep_flop = self.flop
            else:
                keep_flop = (
                    self.flop
                    if self.flop is not None and _extends(self.flop.board, board)
                    else None
                )
                parent = keep_flop if keep_flop is not None else self.preflop
                sanitized_range = sanitize_range(parent.current_range, board.mask)
                next_turn = StreetState.from_range(sanitized_range, board, self.mode)
            self.flop = keep_flop
            self.turn = next_turn
            self.river = None
            self.active_street = next_turn
            return
        
        elif street_size == 5:
            if self.river is not None and board.mask == self.river.board.mask:
                next_river = self.river
                keep_turn = self.turn
                keep_flop = self.flop
            else:
                keep_turn = (
                    self.turn
                    if self.turn is not None and _extends(self.turn.board, board)
                    else None
                )
                keep_flop = (
                    self.flop
                    if self.flop is not None and _extends(self.flop.board, board)
                    else None
                )
                parent = keep_turn or keep_flop or self.preflop
                sanitized_range = sanitize_range(parent.current_range, board.mask)
                next_river = StreetState.from_range(sanitized_range, board, self.mode)
            self.flop = keep_flop
            self.turn = keep_turn
            self.river = next_river
            self.active_street = next_river
            return
        
        raise ValueError("Board can only contain 3, 4, or 5 cards")

    def clear_board(self) -> None:
        self.active_street = self.preflop
        self.flop = None
        self.turn = None
        self.river = None

    def add_filter(
        self, expression: str, name: str | None = None, path: tuple[str, ...] = ()
    ) -> float:
        return self.active_street.filter_manager.create_filter(expression, name, path)

    def add_group(self, name: str, path: tuple[str, ...] = ()) -> None:
        self.active_street.filter_manager.create_group(name, path)

    def delete_filter(self, name: str, path: tuple[str, ...] = ()) -> float:
        self.active_street.filter_manager.delete(FilterAddress(name, path))
        return self.filtered_coverage

    def set_filter_active(self, name: str, active: bool, path: tuple[str, ...] = ()) -> float:
        self.active_street.filter_manager.set_active(FilterAddress(name, path), active)
        return self.filtered_coverage

    def clear_filters(self) -> float:
        self.active_street.filter_manager.clear_all_filters()
        return self.filtered_coverage

    def get_filters(self) -> dict[str, dict[str, object]]:
        return self.active_street.filter_manager.get_tree()

    def get_filter(
        self,
        name: str,
        path: tuple[str, ...] = (),
    ) -> dict[str, object]:
        return self.active_street.filter_manager.get_node_info(FilterAddress(name, path))
    
    def clone(self) -> PlayerRangeState:
        cloned = copy(self)
        for name in ("preflop", "flop", "turn", "river"):
            old_street = getattr(self, name)
            if old_street is None:
                setattr(cloned, name, None)
                continue
            new_street = old_street.clone()
            setattr(cloned, name, new_street)
            if old_street is self.active_street:
                cloned.active_street = new_street
        return cloned