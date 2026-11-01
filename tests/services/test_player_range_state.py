"""Tests for per-player board transitions, history, and copy-on-write state."""

import pytest
from plorangelab.encoder import EncodedRange, encode_range
from plorangelab.services.player_range_state import Board, PlayerRangeState


@pytest.fixture
def base_range() -> EncodedRange:
    return encode_range("AsAhKdQc,KsKhJdTd,QsQhJcTc,9s9h8d8c", 4)


@pytest.fixture
def player(base_range: EncodedRange) -> PlayerRangeState:
    return PlayerRangeState(hands=base_range, id=0, mode=4)


def _masks(hands: EncodedRange) -> set[int]:
    return set(map(int, hands.masks))


def test_initial_state(player: PlayerRangeState, base_range: EncodedRange) -> None:
    assert player.active_street is player.preflop
    assert player.preflop.ranges[0] is base_range
    assert player.preflop.board is None
    assert player.flop is player.turn is player.river is None


def test_board_changes_select_street_and_remove_blocked_hands(
    player: PlayerRangeState, base_range: EncodedRange
) -> None:
    # The turn blocks the first hand; the river blocks the second as well.
    for text, street_name, surviving_indices in (
        ("2c3d5h", "flop", (0, 1, 2, 3)),
        ("2c3d5hAs", "turn", (1, 2, 3)),
        ("2c3d5hAsKs", "river", (2, 3)),
    ):
        board = Board.from_string(text)
        player.apply_board_change(board)
        street = getattr(player, street_name)

        assert player.active_street is street
        assert street.board is board
        assert _masks(street.ranges[-1]) == {
            int(base_range.masks[i]) for i in surviving_indices
        }


@pytest.mark.parametrize(
    "board_text", ["2c3d5hAs", "2c3d5hAsKs"],
)
def test_direct_board_jumps_from_preflop(
    player: PlayerRangeState,
    base_range: EncodedRange,
    board_text: str,
) -> None:
    board = Board.from_string(board_text)
    player.apply_board_change(board)

    assert player.active_street.board is board
    assert _masks(player.active_street.ranges[-1]) == {
        int(mask)
        for mask in base_range.masks
        if not (mask & board.mask)
    }


@pytest.mark.parametrize(
    ("board_text", "reapplied_text"),
    [
        ("2c3d5h", "5h2c3d"),  # Flop order is irrelevant.
        ("2c3d5hAs", "2c3d5hAs"),
        ("2c3d5hAsKs", "2c3d5hAsKs"),
    ],
)
def test_same_board_preserves_range_history(
    player: PlayerRangeState, board_text: str, reapplied_text: str
) -> None:
    board = Board.from_string(board_text)
    player.apply_board_change(board)
    saved_street = player.active_street
    filtered = encode_range("QsQhJcTc", 4)
    saved_street.ranges.append(filtered)

    player.apply_board_change(Board.from_string(reapplied_text))

    assert player.active_street is saved_street
    assert player.active_street.ranges[-1] is filtered


def test_new_flop_rebuilds_from_preflop_and_discards_later_streets(
    player: PlayerRangeState, base_range: EncodedRange
) -> None:
    player.apply_board_change(Board.from_string("2c3d5h"))
    player.apply_board_change(Board.from_string("2c3d5hAs"))
    player.apply_board_change(Board.from_string("2c3d5hAsKs"))

    player.apply_board_change(Board.from_string("Ks2c3d"))

    assert player.active_street is player.flop
    assert player.turn is player.river is None
    assert _masks(player.flop.ranges[-1]) == {
        int(base_range.masks[i]) for i in (0, 2, 3)
    }  # First hand, blocked by the old turn, is restored.


def test_turn_inherits_filtered_flop_range(player: PlayerRangeState) -> None:
    player.apply_board_change(Board.from_string("2c3d5h"))
    filtered = encode_range("QsQhJcTc", 4)
    player.flop.ranges.append(filtered)

    player.apply_board_change(Board.from_string("2c3d5hAs"))

    assert player.active_street is player.turn
    assert _masks(player.turn.ranges[-1]) == _masks(filtered)
    assert player.flop.ranges[-1] is filtered


def test_unrelated_turn_discards_flop_and_rebuilds_from_preflop(
    player: PlayerRangeState, base_range: EncodedRange
) -> None:
    player.apply_board_change(Board.from_string("2c3d5h"))
    player.flop.ranges.append(encode_range("QsQhJcTc", 4))

    # This turn isn't an extension of the saved flop.
    player.apply_board_change(Board.from_string("Ks7c8h6c"))

    assert player.flop is None
    assert player.active_street is player.turn
    assert _masks(player.turn.ranges[-1]) == {
        int(base_range.masks[i]) for i in (0, 2, 3)
    }


@pytest.mark.parametrize("parent", ["turn", "flop", "preflop"])
def test_river_uses_nearest_compatible_street(
    player: PlayerRangeState, base_range: EncodedRange, parent: str
) -> None:
    player.apply_board_change(Board.from_string("2c3d5h"))
    player.flop.ranges.append(encode_range("KsKhJdTd,QsQhJcTc", 4))
    player.apply_board_change(Board.from_string("2c3d5hAs"))
    player.turn.ranges.append(encode_range("QsQhJcTc", 4))

    if parent == "turn":
        board_text = "2c3d5hAs6c"
        expected = {int(base_range.masks[2])}
    elif parent == "flop":
        board_text = "2c3d5hKs6c"
        expected = {int(base_range.masks[2])}
    else:
        board_text = "Ks7c8h6cTc"
        expected = {int(base_range.masks[i]) for i in (0, 3)}

    player.apply_board_change(Board.from_string(board_text))

    assert player.active_street is player.river
    assert _masks(player.river.ranges[-1]) == expected
    if parent == "flop":
        assert player.turn is None
        assert player.flop is not None
    elif parent == "preflop":
        assert player.turn is player.flop is None


def test_clear_board_restores_preflop(player: PlayerRangeState) -> None:
    player.apply_board_change(Board.from_string("2c3d5hAsKs"))
    original_preflop = player.preflop

    player.clear_board()

    assert player.active_street is original_preflop
    assert player.flop is player.turn is player.river is None


@pytest.mark.parametrize("with_board", [False, True])
def test_clone_has_independent_street_containers_without_copying_ranges(
    player: PlayerRangeState, with_board: bool
) -> None:
    if with_board:
        player.apply_board_change(Board.from_string("2c3d5hAs"))

    cloned = player.clone()

    assert cloned is not player
    for name in ("preflop", "flop", "turn", "river"):
        original = getattr(player, name)
        copied = getattr(cloned, name)
        if original is None:
            assert copied is None
        else:
            assert copied is not original
            assert copied.ranges is not original.ranges
            assert copied.ranges[0] is original.ranges[0]
            assert copied.board is original.board
            if player.active_street is original:
                assert cloned.active_street is copied

    old_length = len(player.active_street.ranges)
    cloned.active_street.ranges.append(cloned.active_street.ranges[-1])
    assert len(player.active_street.ranges) == old_length


def test_blocked_entire_range_raises() -> None:
    player = PlayerRangeState(encode_range("AsAhKdQc", 4), id=0, mode=4)

    with pytest.raises(ValueError, match="No combinations left"):
        player.apply_board_change(Board.from_string("As2c3d"))
