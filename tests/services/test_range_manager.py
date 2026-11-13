"""Tests for the range manager's player slots, board updates, and equity delegation."""

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from plorangelab.encoder import encode_range
from plorangelab.services import range_manager as manager_module
from plorangelab.services.filter_manager import FilterAddress
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
    assert len(manager.players[2].preflop.filter_manager.base_range.masks) == 1
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

    snapshots = manager.change_board("2c3d5h")

    assert len(snapshots) == 6
    assert snapshots[1:5] == [None] * 4

    assert snapshots[0].base_hands == 4
    assert snapshots[0].filtered_hands == 4
    assert snapshots[0].filtered_coverage == 1.0
    assert snapshots[0].filters == {}

    assert snapshots[5].base_hands == 1
    assert snapshots[5].filtered_hands == 1

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
    assert current.preflop.current_range is previous_player.preflop.current_range
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
        actual is manager.players[index].current_range
        for actual, index in zip(kwargs["players"], ids, strict=True)
    )
    assert kwargs["mode"] == 4
    assert kwargs["board"] is (manager.board.cards if manager.board else None)
    assert kwargs["on_progress"] is on_progress
    assert kwargs["cancel_event"] is cancel_event
    assert kwargs["target_moe"] == 0.002
    assert kwargs["rank_lookup"] is manager.rank_lookup


def test_none_board_resets_players(manager: RangeManager):
    manager.add_range("AsAhKdQc", mode=4, player_index=0)
    manager.change_board("2c3d4h")

    assert manager.players[0].active_street is manager.players[0].flop

    snapshots = manager.change_board(None)

    assert len(snapshots) == 6
    assert snapshots[0].base_hands == 1
    assert snapshots[0].filtered_hands == 1
    assert snapshots[0].filters == {}
    assert snapshots[1:] == [None] * 5

    player = manager.players[0]

    assert manager.board is None
    assert player.active_street is player.preflop
    assert player.flop is None
    assert player.turn is None
    assert player.river is None


def test_filter_updates_are_copy_on_write(manager: RangeManager) -> None:
    manager.add_range(BASE_RANGE, mode=4, player_index=0)
    original = manager.players[0]
    snapshot = manager.add_filter("AA", player_index=0, name="aces")
    assert snapshot.base_hands == 4
    assert snapshot.filtered_hands == 4
    assert snapshot.filtered_coverage == 1.0
    assert snapshot.filters["aces"]["coverage"] == pytest.approx(0.25)
    assert snapshot.filters["aces"]["active"] is False
    after_save = manager.players[0]
    assert after_save is not original
    assert original.current_range is original.preflop.filter_manager.base_range
    assert after_save.current_range is after_save.preflop.filter_manager.base_range

    snapshot = manager.set_filter_active("aces", player_index=0, active=True)
    assert snapshot.base_hands == 4
    assert snapshot.filtered_hands == 1
    assert snapshot.filtered_coverage == pytest.approx(0.25)
    assert snapshot.filters["aces"]["active"] is True
    activated = manager.players[0]
    assert activated is not after_save
    assert len(activated.current_range.masks) == 1
    assert not after_save.preflop.filter_manager.get_node(FilterAddress("aces")).active

    with pytest.raises(ValueError):
        manager.add_filter("A|", player_index=0)
    assert manager.players[0] is activated

    snapshot = manager.clear_filters(player_index=0)
    assert snapshot.filtered_hands == 4
    assert snapshot.filtered_coverage == 1.0
    assert "aces" in snapshot.filters
    assert snapshot.filters["aces"]["active"] is False
    assert manager.players[0].current_range is (
        manager.players[0].preflop.filter_manager.base_range
    )
    assert (
        not manager.players[0]
        .preflop.filter_manager.get_node(FilterAddress("aces"))
        .active
    )


def test_nested_group_management_via_range_manager(manager: RangeManager) -> None:
    manager.add_range(BASE_RANGE, mode=4, player_index=2)
    snapshot = manager.add_group("premium", player_index=2)

    assert snapshot.filtered_coverage == 1.0
    assert snapshot.filters["premium"]["type"] == "group"
    assert snapshot.filters["premium"]["filters"] == {}

    manager.add_filter("AA", player_index=2, path=("premium",))
    manager.add_filter("KK", player_index=2, path=("premium",))
    snapshot = manager.set_filter_active("premium", player_index=2, active=True)
    assert snapshot.filtered_coverage == pytest.approx(0.5)
    assert snapshot.filtered_hands == 2
    assert snapshot.filters["premium"]["active"] is True
    assert len(manager.players[2].current_range.masks) == 2

    snapshot = manager.set_filter_active(
        "KK", player_index=2, active=False, path=("premium",)
    )
    assert snapshot.filtered_coverage == pytest.approx(0.25)
    assert snapshot.filtered_hands == 1
    assert snapshot.filters["premium"]["filters"]["KK"]["active"] is False
    assert len(manager.players[2].current_range.masks) == 1

    snapshot = manager.delete_filter("AA", player_index=2, path=("premium",))
    assert snapshot.filtered_coverage == 1.0
    assert snapshot.filtered_hands == 4
    assert "AA" not in snapshot.filters["premium"]["filters"]
    assert manager.players[2].current_range is (
        manager.players[2].active_street.filter_manager.base_range
    )


def test_filter_commands_reject_invalid_and_empty_slots(manager: RangeManager) -> None:
    with pytest.raises(ValueError, match="empty"):
        manager.add_filter("A", player_index=0)
    with pytest.raises(ValueError, match="empty"):
        manager.add_group("premium", player_index=1)
    for index in (-1, 6):
        with pytest.raises(ValueError, match="index"):
            manager.add_filter("A", player_index=index)
        with pytest.raises(ValueError, match="index"):
            manager.clear_filters(player_index=index)


def test_equity_receives_filtered_range_after_activation(
    manager: RangeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager.add_range(BASE_RANGE, mode=4, player_index=0)
    manager.add_filter("QQ", player_index=0)
    manager.set_filter_active("QQ", player_index=0, active=True)

    compute_mock = Mock(return_value=object())
    monkeypatch.setattr(manager_module, "compute_equity", compute_mock)
    manager.calculate_equity(mode=4)

    passed_range = compute_mock.call_args.kwargs["players"][0]
    assert passed_range is manager.players[0].current_range
    assert len(passed_range.masks) == 1


def test_filtered_preflop_range_carries_to_board(
    manager: RangeManager,
) -> None:
    manager.add_range(BASE_RANGE, mode=4, player_index=0)
    manager.add_filter("AA", player_index=0)
    manager.set_filter_active("AA", player_index=0, active=True)
    old_player = manager.players[0]

    snapshots = manager.change_board("2c3d5h")
    snapshot = snapshots[0]

    assert snapshot.filtered_hands == 1
    assert snapshot.filtered_coverage == pytest.approx(1.0)
    updated_player = manager.players[0]
    assert updated_player is not old_player
    assert len(updated_player.current_range.masks) == 1
    assert updated_player.flop.filter_manager.base_range is not (
        updated_player.preflop.filter_manager.base_range
    )
    assert updated_player.preflop.filter_manager.get_node(FilterAddress("AA")).active


def test_filter_retrieval_delegates_to_correct_player(
    manager: RangeManager,
) -> None:
    manager.add_range(BASE_RANGE, mode=4, player_index=2)
    manager.add_range(BASE_RANGE, mode=4, player_index=4)

    manager.add_group("premium", player_index=2)
    manager.add_filter(
        "AA",
        player_index=2,
        name="aces",
        path=("premium",),
    )
    manager.add_filter("KK", player_index=4, name="kings")

    player = manager.players[2]

    tree = manager.get_filters(player_index=2)

    assert set(tree) == {"premium"}
    assert (
        manager.get_filter(
            "aces",
            player_index=2,
            path=("premium",),
        )
        == tree["premium"]["filters"]["aces"]
    )

    assert set(manager.get_filters(player_index=4)) == {"kings"}

    # Read operations must not clone or replace player state.
    assert manager.players[2] is player


@pytest.mark.parametrize("board", [None, "2c3d5h"])
def test_change_board_without_players_returns_empty_snapshots(
    manager: RangeManager,
    board: str | None,
) -> None:
    snapshots = manager.change_board(board)

    assert snapshots == [None] * 6


def test_change_board_restores_cached_filter_snapshot(
    manager: RangeManager,
) -> None:
    manager.add_range(BASE_RANGE, mode=4, player_index=2)
    manager.change_board("2c3d5h")

    manager.add_filter("AA", player_index=2, name="aces")
    expected = manager.set_filter_active("aces", player_index=2, active=True)

    manager.change_board("2c3d5h7s")
    snapshots = manager.change_board("2c3d5h")

    assert snapshots[2] == expected
