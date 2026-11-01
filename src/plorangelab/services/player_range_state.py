from __future__ import annotations

from copy import copy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from plorangelab.encoder import EncodedRange, build_mask, encode_board
from plorangelab.equity import sanitize_range
from plorangelab.types import Mode


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
    ranges: list[EncodedRange] 
    board: Board | None

def count_different_cards(board1: np.uint64, board2: np.uint64) -> int:  
    diff = board1 ^ board2
    return diff.bit_count()
    

class PlayerRangeState:
    def __init__(self, hands: EncodedRange, id: int, mode: Mode):
        self.mode: Mode = mode
        self.id: int = id
        self.preflop = StreetState(ranges=[hands], board=None)
        self.flop: StreetState | None = None
        self.turn: StreetState | None = None
        self.river: StreetState | None = None
        self.active_street: StreetState = self.preflop

    def apply_board_change(self, board: Board) -> None:
        if len(board.cards) == 3:
            if self.flop is not None and board.mask == self.flop.board.mask:
                self.active_street = self.flop
            else:
                sanitized_range = sanitize_range(self.preflop.ranges[-1], board.mask)
                self.flop = StreetState(ranges=[sanitized_range], board=board)
                self.active_street = self.flop

            self.turn = None
            self.river = None
        elif len(board.cards) == 4:
            if self.turn is not None and board.mask == self.turn.board.mask:
                self.active_street = self.turn
            else:
                if self.flop is not None and count_different_cards(self.flop.board.mask, board.mask) == 1:
                    parent = self.flop
                else:
                    parent = self.preflop
                    self.flop = None
                sanitized_range = sanitize_range(parent.ranges[-1], board.mask)
                self.turn = StreetState(ranges=[sanitized_range], board=board)
                self.active_street = self.turn
            self.river = None
        elif len(board.cards) == 5:
            if self.river is not None and board.mask == self.river.board.mask:
                self.active_street = self.river
            else:
                if self.turn is not None and count_different_cards(self.turn.board.mask, board.mask) == 1:
                    parent = self.turn
                elif self.flop is not None and count_different_cards(self.flop.board.mask, board.mask) == 2:
                    parent = self.flop
                    self.turn = None
                else:
                    parent = self.preflop
                    self.flop = None
                    self.turn = None

                sanitized_range = sanitize_range(parent.ranges[-1], board.mask)
                self.river = StreetState(ranges=[sanitized_range], board=board)
                self.active_street = self.river
        else: 
            raise ValueError("Board can only contain up to 5 cards")

    def clear_board(self) -> None:
        self.active_street = self.preflop
        self.flop = None
        self.turn = None
        self.river = None

    def clone(self) -> PlayerRangeState:
        cloned = copy(self)  
        for name in ("preflop", "flop", "turn", "river"):
            street = getattr(self, name)

            if street is None:
                setattr(cloned, name, None)
                continue

            cloned_street = StreetState(
                ranges=street.ranges.copy(),
                board=street.board
            )

            setattr(cloned, name, cloned_street)

            if street is self.active_street:
                cloned.active_street = cloned_street

        return cloned