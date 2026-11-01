"""Tests for the range manager's player slots, board updates, and equity delegation."""

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from plorangelab.encoder import encode_range
from plorangelab.services import range_manager as manager_module
from plorangelab.services.range_manager import RangeManager

BASE_RANGE = "AsAhKdQc,KsKhJdTd,QsQhJcTc,9s9h8d8c"


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> RangeManager:
    # Keep real range encoding and street logic, but skip rating-table loading.
    monkeypatch.setattr(
        manager_module,
        "encode_input",
        lambda text, mode, _rating: encode_range(text, mode),
    )
    resources = SimpleNamespace(
        ratings={4: object(), 5: object()},
        rank_lookup=object(),
    )
    return RangeManager(resources)


def test_add_replace_and_remove_range(manager: RangeManager) -> None:
    assert manager.players == [None] * 6
    assert manager.board is None

    manager.add_range(BASE_RANGE, mode=4, player_index=2)
    first_player = manager.players[2]
    assert first_player is not None
    assert first_player.id == 2
    assert first_player.mode == 4
    assert first_player.active_street is first_player.preflop

    manager.add_range("JsJhTdTc", mode=4, player_index=2)
    assert manager.players[2] is not first_player
    assert len(manager.players[2].preflop.ranges[0].masks) == 1
    assert all(player is None for i, player in enumerate(manager.players) if i != 2)

    manager.remove_range(2)
    assert manager.players == [None] * 6


@pytest.mark.parametrize("index", [-1, 6])
def test_invalid_player_index_raises(manager: RangeManager, index: int) -> None:
    with pytest.raises(ValueError, match="Out of range player index"):
        manager.add_range(BASE_RANGE, mode=4, player_index=index)

    with pytest.raises(ValueError, match="Out of range player index"):
        manager.remove_range(index)


def test_change_board_updates_players_without_mutating_old_states(
    manager: RangeManager,
) -> None:
    manager.add_range(BASE_RANGE, mode=4, player_index=0)
    manager.add_range("JsJhTdTc", mode=4, player_index=5)
    previous_players = manager.players.copy()

    manager.change_board("2c3d5h")

    assert manager.board is not None
    assert manager.players[1:5] == [None] * 4
    for index in (0, 5):
        previous = previous_players[index]
        current = manager.players[index]
        assert current is not previous
        assert previous.flop is None
        assert current.active_street is current.flop
        assert current.flop.board is manager.board


def test_change_board_failure_preserves_all_player_states(
    manager: RangeManager,
) -> None:
    manager.add_range(BASE_RANGE, mode=4, player_index=0)
    manager.add_range("AsAhKdQc", mode=4, player_index=3)
    manager.change_board("2c3d5h")
    previous_board = manager.board
    previous_players = manager.players.copy()

    # The first player's range survives, but the second player's range does not.
    with pytest.raises(ValueError, match="No combinations left"):
        manager.change_board("2c3dAs")

    assert manager.board is previous_board
    assert all(
        current is previous
        for current, previous in zip(manager.players, previous_players, strict=True)
    )


def test_add_range_respects_current_board_and_failure_keeps_slot(
    manager: RangeManager,
) -> None:
    manager.change_board("2c3d5h")
    manager.add_range(BASE_RANGE, mode=4, player_index=2)
    previous_player = manager.players[2]

    assert previous_player.active_street is previous_player.flop
    assert previous_player.flop.board is manager.board

    with pytest.raises(ValueError, match="No combinations left"):
        manager.add_range("2cAsAhKd", mode=4, player_index=2)

    assert manager.players[2] is previous_player


def test_clear_board_returns_players_to_preflop(
    manager: RangeManager,
) -> None:
    manager.add_range(BASE_RANGE, mode=4, player_index=1)
    manager.change_board("2c3d5h")
    previous_player = manager.players[1]

    manager.clear_board()

    current = manager.players[1]
    assert manager.board is None
    assert current is not previous_player
    assert current.active_street is current.preflop
    assert current.flop is current.turn is current.river is None
    assert current.preflop.ranges[-1] is previous_player.preflop.ranges[-1]
    assert previous_player.flop is not None


@pytest.mark.parametrize("board_text", [None, "2c3d5h"])
def test_calculate_equity_forwards_active_ranges_and_returns_slot_ids(
    manager: RangeManager,
    monkeypatch: pytest.MonkeyPatch,
    board_text: str | None,
) -> None:
    manager.add_range(BASE_RANGE, mode=4, player_index=5)
    manager.add_range("JsJhTdTc", mode=4, player_index=1)
    if board_text is not None:
        manager.change_board(board_text)

    expected_result = object()
    compute_mock = Mock(return_value=expected_result)
    monkeypatch.setattr(manager_module, "compute_equity", compute_mock)
    on_progress = Mock()
    cancel_event = Event()

    ids, result = manager.calculate_equity(
        mode=4,
        on_progress=on_progress,
        cancel_event=cancel_event,
        target_moe=0.002,
    )

    assert ids == [1, 5]  # Slot order, not insertion order.
    assert result is expected_result
    compute_mock.assert_called_once()
    kwargs = compute_mock.call_args.kwargs
    assert len(kwargs["players"]) == 2
    assert all(
        actual is manager.players[index].active_street.ranges[-1]
        for actual, index in zip(kwargs["players"], ids, strict=True)
    )
    assert kwargs["mode"] == 4
    assert kwargs["board"] is (manager.board.cards if manager.board else None)
    assert kwargs["on_progress"] is on_progress
    assert kwargs["cancel_event"] is cancel_event
    assert kwargs["target_moe"] == 0.002
    assert kwargs["rank_lookup"] is manager.rank_lookup



def test_empty_board_resets_players(manager):
    manager.add_range("AsAhKdQc", mode=4, player_index=0)
    manager.change_board("2c3d4h")

    assert manager.players[0].active_street is manager.players[0].flop

    manager.change_board("")

    player = manager.players[0]

    assert manager.board is None
    assert player.active_street is player.preflop
    assert player.flop is None
    assert player.turn is None
    assert player.river is None