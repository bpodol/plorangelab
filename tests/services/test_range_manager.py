"""Integration tests for RangeManager V2's street-centered range state.

Keep real encoding, filtering, and board sanitization. Use a tiny deterministic
auto-filter tree so these tests verify integration and street state, not generator
rules (which have their own unit tests). Mock expensive equity computation.
"""

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from plorangelab.encoder import encode_range
from plorangelab.filters.filter_generator import FilterNode, GroupNode
from plorangelab.services import player_range as player_range_module
from plorangelab.services import range_manager as manager_module
from plorangelab.services.player_range import FilterAddress, PlayerRange
from plorangelab.services.range_manager import RangeManager, Street

BASE_RANGE = "AsAhKdQc,KsKhJdTd,QsQhJcTc,9s9h8d8c"
FLOP = "2c3d5h"
TURN = "2c3d5h7s"
RIVER = "2c3d5h7s9d"
AUTO_GROUP = "__auto_generated__"
AUTO_LEAF = "aces"


def assert_auto_filters(player: PlayerRange) -> None:
    """Generated definitions exist without automatically restricting the range."""
    tree = player.get_tree()
    assert AUTO_GROUP in tree
    assert tree[AUTO_GROUP]["type"] == "group"
    assert not tree[AUTO_GROUP]["active"]
    assert tree[AUTO_GROUP]["filters"][AUTO_LEAF]["type"] == "filter"
    assert not tree[AUTO_GROUP]["filters"][AUTO_LEAF]["active"]
    assert player.current_range is player.base_range
    assert player.filtered_coverage == pytest.approx(1.0)


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> RangeManager:
    # Use actual encoding/filtering rules but avoid loading rating tables.
    monkeypatch.setattr(
        manager_module,
        "encode_input",
        lambda text, mode, _rating: encode_range(text, mode),
    )
    # Exercise PlayerRange.autogenerate_filters() and its group/leaf wiring,
    # without depending on the contents of the real generator.
    monkeypatch.setattr(
        player_range_module,
        "generate_filters",
        lambda _board: [
            GroupNode(AUTO_GROUP, [FilterNode(AUTO_LEAF, "AA")]),
        ],
    )
    resources = SimpleNamespace(
        ratings={4: object(), 5: object()},
        rank_lookup=object(),
    )
    return RangeManager(resources)


def player_at(manager: RangeManager, street: Street, slot: int) -> PlayerRange:
    """Access the manager for a slot on a particular existing street."""
    player = manager._streets[street].ranges[slot]
    assert player is not None
    return player


def test_add_replace_and_remove_range(manager: RangeManager) -> None:
    assert manager.mode == 4
    assert manager._active_street is Street.PREFLOP
    assert set(manager._streets) == {Street.PREFLOP}
    assert manager._current_state.board is None
    assert manager._current_state.ranges == [None] * 6

    snapshot = manager.add_range(BASE_RANGE, 2)
    assert snapshot.base_hands == 4
    assert snapshot.filtered_hands == 4
    assert snapshot.filtered_coverage == pytest.approx(1.0)
    assert snapshot.filters == {}  # Never generate filters preflop.
    first = player_at(manager, Street.PREFLOP, 2)
    assert first.mode == 4
    assert len(first.base_range.hands) == 4
    assert first.current_range is first.base_range
    assert all(
        player is None
        for slot, player in enumerate(manager._current_state.ranges)
        if slot != 2
    )

    manager.add_range("JsJhTdTc", 2)
    replacement = player_at(manager, Street.PREFLOP, 2)
    assert replacement is not first
    assert len(replacement.base_range.masks) == 1

    manager.remove_range(2)
    assert manager._current_state.ranges == [None] * 6


@pytest.mark.parametrize("slot", [-1, 6])
def test_invalid_slot_rejected(manager: RangeManager, slot: int) -> None:
    with pytest.raises(ValueError, match="Out of range slot index"):
        manager.add_range(BASE_RANGE, slot)
    with pytest.raises(ValueError, match="Out of range slot index"):
        manager.remove_range(slot)
    with pytest.raises(ValueError, match="Out of range slot index"):
        manager.add_filter("AA", slot)
    with pytest.raises(ValueError, match="Out of range slot index"):
        manager.clear_filters(slot)


def test_filter_commands_reject_empty_slots(manager: RangeManager) -> None:
    with pytest.raises(ValueError, match="empty"):
        manager.add_filter("AA", 0)
    with pytest.raises(ValueError, match="empty"):
        manager.add_group("premium", 1)
    with pytest.raises(ValueError, match="empty"):
        manager.clear_filters(2)


def test_change_board_creates_shared_street_without_mutating_preflop(
    manager: RangeManager,
) -> None:
    manager.add_range(BASE_RANGE, 0)
    manager.add_range("JsJhTdTc", 5)
    preflop = manager._streets[Street.PREFLOP]
    old_player_0 = player_at(manager, Street.PREFLOP, 0)
    old_player_5 = player_at(manager, Street.PREFLOP, 5)

    snapshots = manager.change_board(FLOP)

    assert manager._active_street is Street.FLOP
    assert set(manager._streets) == {Street.PREFLOP, Street.FLOP}
    assert manager._streets[Street.PREFLOP] is preflop
    assert player_at(manager, Street.PREFLOP, 0) is old_player_0
    assert player_at(manager, Street.PREFLOP, 5) is old_player_5
    assert manager._current_state.board is not None
    assert not manager._current_state.board.cards.flags.writeable
    assert manager._current_state.ranges[1:5] == [None] * 4
    assert player_at(manager, Street.FLOP, 0) is not old_player_0
    assert player_at(manager, Street.FLOP, 5) is not old_player_5

    assert len(snapshots) == 6
    assert snapshots[1:5] == [None] * 4
    assert snapshots[0].base_hands == 4
    assert snapshots[0].filtered_hands == 4
    assert snapshots[0].filtered_coverage == pytest.approx(1.0)
    assert snapshots[0].filters == player_at(manager, Street.FLOP, 0).get_tree()
    assert_auto_filters(player_at(manager, Street.FLOP, 0))
    assert_auto_filters(player_at(manager, Street.FLOP, 5))
    assert snapshots[5].base_hands == 1
    assert snapshots[5].filtered_hands == 1


def test_change_board_failure_is_atomic_for_all_slots(manager: RangeManager) -> None:
    manager.add_range(BASE_RANGE, 0)
    manager.add_range("AsAhKdQc", 3)
    manager.change_board(FLOP)
    original_streets = manager._streets
    original_street = manager._active_street
    original_flop = manager._current_state

    # One player's range survives, but slot 3 has no remaining hands.
    with pytest.raises(ValueError, match="No combinations left"):
        manager.change_board("2c3dAs")

    assert manager._streets is original_streets
    assert manager._active_street is original_street
    assert manager._current_state is original_flop


def test_invalid_board_does_not_change_state(manager: RangeManager) -> None:
    manager.add_range(BASE_RANGE, 0)
    manager.change_board(FLOP)
    original_streets = manager._streets

    with pytest.raises(ValueError):
        manager.change_board("2c2c5h")

    assert manager._streets is original_streets
    assert manager._active_street is Street.FLOP


def test_add_range_on_active_board_and_failed_replacement_is_atomic(
    manager: RangeManager,
) -> None:
    manager.change_board(FLOP)
    snapshot = manager.add_range(BASE_RANGE, 2)
    first_flop = player_at(manager, Street.FLOP, 2)
    assert snapshot.filters == first_flop.get_tree()
    assert snapshot.base_hands == 4
    assert snapshot.filtered_hands == 4
    assert_auto_filters(first_flop)
    assert player_at(manager, Street.PREFLOP, 2).get_tree() == {}
    first_preflop = player_at(manager, Street.PREFLOP, 2)
    previous_streets = manager._streets

    assert len(first_flop.base_range.hands) == 4
    assert first_flop is not first_preflop

    with pytest.raises(ValueError, match="No combinations left"):
        manager.add_range("2cAsAhKd", 2)

    assert manager._streets is previous_streets
    assert player_at(manager, Street.FLOP, 2) is first_flop
    assert player_at(manager, Street.PREFLOP, 2) is first_preflop


def test_clearing_board_preserves_preflop_filters(manager: RangeManager) -> None:
    manager.add_range(BASE_RANGE, 1)
    manager.add_filter("AA", 1)
    preflop_snapshot = manager.set_filter_active("AA", 1, True)
    preflop = manager._streets[Street.PREFLOP]
    manager.change_board(FLOP)
    old_flop = manager._streets[Street.FLOP]

    snapshots = manager.change_board(None)

    assert manager._active_street is Street.PREFLOP
    assert set(manager._streets) == {Street.PREFLOP}
    assert manager._current_state is preflop
    assert manager._current_state.board is None
    assert snapshots[1] == preflop_snapshot
    assert snapshots[:1] + snapshots[2:] == [None] * 5
    assert old_flop.board is not None  # Previously captured state unchanged.


@pytest.mark.parametrize("board", [None, FLOP])
def test_equity_delegation_preserves_slot_order_and_arguments(
    manager: RangeManager,
    monkeypatch: pytest.MonkeyPatch,
    board: str | None,
) -> None:
    manager.add_range(BASE_RANGE, 5)
    manager.add_range("JsJhTdTc", 1)
    if board is not None:
        manager.change_board(board)

    expected_result = object()
    compute_mock = Mock(return_value=expected_result)
    monkeypatch.setattr(manager_module, "compute_equity", compute_mock)
    on_progress = Mock()
    cancel_event = Event()

    ids, result = manager.calculate_equity(
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
        current is player_at(manager, manager._active_street, slot).current_range
        for current, slot in zip(kwargs["players"], ids, strict=True)
    )
    assert kwargs["mode"] == manager.mode
    assert kwargs["board"] is (
        manager._current_state.board.cards if board is not None else None
    )
    assert kwargs["on_progress"] is on_progress
    assert kwargs["cancel_event"] is cancel_event
    assert kwargs["target_moe"] == 0.002
    assert kwargs["rank_lookup"] is manager.resources.rank_lookup


def test_filter_edits_are_copy_on_write_and_return_snapshots(
    manager: RangeManager,
) -> None:
    manager.add_range(BASE_RANGE, 0)
    original = player_at(manager, Street.PREFLOP, 0)

    saved = manager.add_filter("AA", 0, name="aces")
    after_save = player_at(manager, Street.PREFLOP, 0)
    assert after_save is not original
    assert original.current_range is original.base_range
    assert after_save.current_range is after_save.base_range
    assert saved.base_hands == 4
    assert saved.filtered_hands == 4
    assert saved.filtered_coverage == pytest.approx(1.0)
    assert saved.filters["aces"]["coverage"] == pytest.approx(0.25)
    assert saved.filters["aces"]["active"] is False

    activated_snapshot = manager.set_filter_active("aces", 0, True)
    activated = player_at(manager, Street.PREFLOP, 0)
    assert activated is not after_save
    assert len(activated.current_range.masks) == 1
    assert activated_snapshot.filtered_hands == 1
    assert activated_snapshot.filtered_coverage == pytest.approx(0.25)
    assert activated_snapshot.filters["aces"]["active"] is True
    assert not after_save.get_node(FilterAddress("aces")).active
    assert saved.filters["aces"]["active"] is False  # Snapshot is detached.

    with pytest.raises(ValueError):
        manager.add_filter("A|", 0)
    assert player_at(manager, Street.PREFLOP, 0) is activated

    cleared = manager.clear_filters(0)
    current = player_at(manager, Street.PREFLOP, 0)
    assert cleared.filtered_hands == 4
    assert cleared.filtered_coverage == pytest.approx(1.0)
    assert cleared.filters["aces"]["active"] is False
    assert current.current_range is current.base_range
    assert not current.get_node(FilterAddress("aces")).active


def test_snapshot_creation_failure_does_not_publish_mutation(
    manager: RangeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager.add_range(BASE_RANGE, 0)
    original = player_at(manager, Street.PREFLOP, 0)

    def broken_tree(_self: PlayerRange) -> dict[str, object]:
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(PlayerRange, "get_tree", broken_tree)
    with pytest.raises(RuntimeError, match="snapshot failed"):
        manager.add_group("premium", 0)

    assert player_at(manager, Street.PREFLOP, 0) is original
    assert "premium" not in original.filters


def test_nested_groups_through_manager(manager: RangeManager) -> None:
    manager.add_range(BASE_RANGE, 2)
    created = manager.add_group("premium", 2)
    assert created.filters["premium"]["type"] == "group"
    assert created.filters["premium"]["filters"] == {}

    manager.add_filter("AA", 2, path=("premium",))
    manager.add_filter("KK", 2, path=("premium",))
    grouped = manager.set_filter_active("premium", 2, True)
    assert grouped.filtered_coverage == pytest.approx(0.5)
    assert grouped.filtered_hands == 2
    assert grouped.filters["premium"]["active"] is True

    without_kings = manager.set_filter_active("KK", 2, False, path=("premium",))
    assert without_kings.filtered_coverage == pytest.approx(0.25)
    assert without_kings.filtered_hands == 1
    assert without_kings.filters["premium"]["filters"]["KK"]["active"] is False

    deleted = manager.delete_filter("AA", 2, path=("premium",))
    assert deleted.filtered_coverage == pytest.approx(1.0)
    assert deleted.filtered_hands == 4
    assert "AA" not in deleted.filters["premium"]["filters"]
    player = player_at(manager, Street.PREFLOP, 2)
    assert player.current_range is player.base_range


def test_equity_receives_active_filtered_range(
    manager: RangeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager.add_range(BASE_RANGE, 0)
    manager.add_filter("QQ", 0)
    manager.set_filter_active("QQ", 0, True)

    compute_mock = Mock(return_value=object())
    monkeypatch.setattr(manager_module, "compute_equity", compute_mock)
    manager.calculate_equity()

    passed_range = compute_mock.call_args.kwargs["players"][0]
    assert passed_range is player_at(manager, Street.PREFLOP, 0).current_range
    assert len(passed_range.masks) == 1


def test_filtered_preflop_range_carries_to_flop(manager: RangeManager) -> None:
    manager.add_range(BASE_RANGE, 0)
    manager.add_filter("AA", 0)
    manager.set_filter_active("AA", 0, True)
    preflop = player_at(manager, Street.PREFLOP, 0)

    snapshots = manager.change_board(FLOP)
    flop = player_at(manager, Street.FLOP, 0)

    assert snapshots[0].filtered_hands == 1
    assert snapshots[0].filtered_coverage == pytest.approx(1.0)
    assert len(flop.base_range.hands) == 1
    assert flop.base_range is not preflop.base_range
    assert_auto_filters(flop)
    assert preflop.get_node(FilterAddress("AA")).active


def test_board_street_progression_carries_filtered_parent_ranges(
    manager: RangeManager,
) -> None:
    manager.add_range(BASE_RANGE, 0)
    manager.change_board(FLOP)
    manager.add_filter("KK", 0)
    manager.set_filter_active("KK", 0, True)
    flop = player_at(manager, Street.FLOP, 0)

    turn_snapshots = manager.change_board(TURN)
    turn = player_at(manager, Street.TURN, 0)
    assert turn_snapshots[0].base_hands == 1
    assert_auto_filters(turn)
    assert len(turn.current_range.masks) == 1

    manager.add_group("turn-only", 0)
    updated_turn = player_at(manager, Street.TURN, 0)
    assert updated_turn is not turn  # Filter edits replace their manager.
    river_snapshots = manager.change_board(RIVER)
    river = player_at(manager, Street.RIVER, 0)
    assert river_snapshots[0].base_hands == 1
    assert_auto_filters(river)
    assert "turn-only" not in river.get_tree()
    assert "turn-only" in updated_turn.get_tree()
    assert flop.get_node(FilterAddress("KK")).active


def test_reordered_same_board_reuses_street_and_filters(manager: RangeManager) -> None:
    manager.add_range(BASE_RANGE, 2)
    manager.change_board(FLOP)
    manager.add_filter("AA", 2, name="aces")
    expected = manager.set_filter_active("aces", 2, True)
    previous = manager._current_state

    snapshots = manager.change_board("5h2c3d")

    assert manager._current_state is previous
    assert snapshots[2] == expected
    assert manager._current_state.board is previous.board


def test_revisiting_flop_discards_later_streets(manager: RangeManager) -> None:
    manager.add_range(BASE_RANGE, 2)
    manager.change_board(FLOP)
    manager.add_filter("AA", 2, name="aces")
    expected = manager.set_filter_active("aces", 2, True)
    flop = manager._streets[Street.FLOP]
    manager.change_board(TURN)
    manager.change_board(RIVER)

    snapshots = manager.change_board(FLOP)

    assert snapshots[2] == expected
    assert manager._current_state is flop
    assert set(manager._streets) == {Street.PREFLOP, Street.FLOP}


def test_changing_turn_card_preserves_flop_but_rebuilds_turn_and_river(
    manager: RangeManager,
) -> None:
    manager.add_range(BASE_RANGE, 0)
    manager.change_board(FLOP)
    flop = manager._streets[Street.FLOP]
    manager.change_board(TURN)
    old_turn = manager._streets[Street.TURN]
    manager.add_group("old-turn-filter", 0)
    manager.change_board(RIVER)

    manager.change_board("2c3d5h8s")

    assert manager._current_state is manager._streets[Street.TURN]
    assert manager._streets[Street.FLOP] is flop
    assert manager._streets[Street.TURN] is not old_turn
    rebuilt = player_at(manager, Street.TURN, 0)
    assert_auto_filters(rebuilt)
    assert "old-turn-filter" not in rebuilt.get_tree()
    assert Street.RIVER not in manager._streets


def test_changing_flop_invalidates_incompatible_following_streets(
    manager: RangeManager,
) -> None:
    manager.add_range(BASE_RANGE, 0)
    manager.change_board(FLOP)
    old_flop = manager._streets[Street.FLOP]
    manager.change_board(TURN)
    manager.change_board(RIVER)
    original_preflop = manager._streets[Street.PREFLOP]

    manager.change_board("2c3d6h")

    assert manager._active_street is Street.FLOP
    assert set(manager._streets) == {Street.PREFLOP, Street.FLOP}
    assert manager._streets[Street.PREFLOP] is original_preflop
    assert manager._streets[Street.FLOP] is not old_flop


def test_skip_street_uses_closest_compatible_parent(manager: RangeManager) -> None:
    manager.add_range(BASE_RANGE, 0)
    manager.change_board(FLOP)
    manager.add_filter("QQ", 0)
    manager.set_filter_active("QQ", 0, True)

    snapshots = manager.change_board(RIVER)

    assert set(manager._streets) == {Street.PREFLOP, Street.FLOP, Street.RIVER}
    assert snapshots[0].base_hands == 1
    assert manager._active_street is Street.RIVER
    assert_auto_filters(player_at(manager, Street.RIVER, 0))


def test_replace_player_updates_all_cached_streets_only_for_that_slot(
    manager: RangeManager,
) -> None:
    manager.add_range(BASE_RANGE, 0)
    manager.add_range(BASE_RANGE, 5)
    manager.change_board(FLOP)
    manager.add_filter("AA", 0)
    manager.change_board(TURN)
    manager.add_group("turn-group", 0)
    old_other = {street: player_at(manager, street, 5) for street in manager._streets}
    old_slot = {street: player_at(manager, street, 0) for street in manager._streets}

    manager.add_range("JsJhTdTc", 0)

    for street in manager._streets:
        replacement = player_at(manager, street, 0)
        assert replacement is not old_slot[street]
        assert len(replacement.base_range.hands) == 1
        if street is Street.TURN:  # Only the active street generates filters.
            assert_auto_filters(replacement)
        else:
            assert replacement.get_tree() == {}
        assert player_at(manager, street, 5) is old_other[street]


def test_remove_player_removes_it_from_every_street(manager: RangeManager) -> None:
    manager.add_range(BASE_RANGE, 0)
    manager.add_range(BASE_RANGE, 5)
    manager.change_board(FLOP)
    manager.change_board(TURN)
    before_other = {
        street: player_at(manager, street, 5) for street in manager._streets
    }

    manager.remove_range(0)

    assert all(state.ranges[0] is None for state in manager._streets.values())
    assert all(
        player_at(manager, street, 5) is other for street, other in before_other.items()
    )


@pytest.mark.parametrize("board", [None, FLOP])
def test_change_board_without_players_returns_empty_snapshots(
    manager: RangeManager, board: str | None
) -> None:
    assert manager.change_board(board) == [None] * 6


def test_change_mode_is_no_op_for_same_mode(manager: RangeManager) -> None:
    manager.add_range(BASE_RANGE, 0)
    manager.change_board(FLOP)
    original_streets = manager._streets

    manager.change_mode(4)

    assert manager._streets is original_streets
    assert manager._active_street is Street.FLOP


def test_change_mode_rejects_active_ranges_without_changing_state(
    manager: RangeManager,
) -> None:
    manager.add_range(BASE_RANGE, 0)
    manager.change_board(FLOP)
    manager.add_filter("AA", 0)
    manager.set_filter_active("AA", 0, True)
    original_streets = manager._streets
    original_street = manager._active_street
    original_state = manager._current_state

    with pytest.raises(
        RuntimeError, match="Cannot change mode with active player ranges"
    ):
        manager.change_mode(5)

    assert manager.mode == 4
    assert manager._streets is original_streets
    assert manager._active_street is original_street
    assert manager._current_state is original_state
    assert player_at(manager, Street.FLOP, 0).get_node(FilterAddress("AA")).active


def test_change_mode_succeeds_without_active_ranges(manager: RangeManager) -> None:
    manager.change_mode(5)

    assert manager.mode == 5
    assert manager._active_street is Street.PREFLOP
    assert set(manager._streets) == {Street.PREFLOP}
    assert manager._current_state.ranges == [None] * 6
    assert manager._current_state.board is None

    manager.add_range("AsAhKdQcJs", 1)
    assert player_at(manager, Street.PREFLOP, 1).mode == 5
    assert player_at(manager, Street.PREFLOP, 1).base_range.hands.shape[1] == 5


def test_invalid_mode_does_not_reset_state(manager: RangeManager) -> None:
    manager.add_range(BASE_RANGE, 0)
    original_streets = manager._streets
    with pytest.raises(ValueError, match="Invalid Mode"):
        manager.change_mode(6)
    assert manager.mode == 4
    assert manager._streets is original_streets


def test_add_range_generation_failure_does_not_publish_partial_state(
    manager: RangeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: range insertion and autogeneration must commit together."""
    manager.add_range(BASE_RANGE, 0)
    manager.change_board(FLOP)
    previous_streets = manager._streets
    previous_flop = player_at(manager, Street.FLOP, 0)
    previous_preflop = player_at(manager, Street.PREFLOP, 0)

    def fail_generation(_self: PlayerRange, _board: object) -> None:
        raise ValueError("synthetic generator failure")

    monkeypatch.setattr(PlayerRange, "autogenerate_filters", fail_generation)

    with pytest.raises(ValueError, match="synthetic generator failure"):
        manager.add_range("JsJhTdTc", 0)

    assert manager._streets is previous_streets
    assert player_at(manager, Street.FLOP, 0) is previous_flop
    assert player_at(manager, Street.PREFLOP, 0) is previous_preflop


def test_replacing_player_generates_filters_only_for_current_street(
    manager: RangeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacing ranges must not pay to generate filters for inactive streets."""
    manager.add_range(BASE_RANGE, 0)
    for board in (FLOP, TURN, RIVER):
        manager.change_board(board)

    calls: list[object] = []
    original_generate = PlayerRange.autogenerate_filters

    def track_generation(self: PlayerRange, board) -> None:
        calls.append(board)
        original_generate(self, board)

    monkeypatch.setattr(PlayerRange, "autogenerate_filters", track_generation)
    snapshot = manager.add_range("JsJhTdTc", 0)

    assert len(calls) == 1  # River only, not existing flop/turn.
    assert calls[0] is manager._current_state.board.cards
    assert snapshot.base_hands == 1
    assert snapshot.filtered_hands == 1
    assert snapshot.filters == player_at(manager, Street.RIVER, 0).get_tree()
    assert_auto_filters(player_at(manager, Street.RIVER, 0))
    assert player_at(manager, Street.PREFLOP, 0).get_tree() == {}
    assert player_at(manager, Street.FLOP, 0).get_tree() == {}
    assert player_at(manager, Street.TURN, 0).get_tree() == {}


def test_revisiting_compatible_street_reuses_existing_filters(
    manager: RangeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not regenerate existing filters when the board state is reused."""
    manager.add_range(BASE_RANGE, 0)
    manager.change_board(FLOP)
    manager.add_filter("KK", 0, name="manual-kings")
    expected = manager.set_filter_active("manual-kings", 0, True)
    saved_flop = manager._streets[Street.FLOP]
    manager.change_board(TURN)

    def fail_if_regenerated(_self: PlayerRange, _board: object) -> None:
        raise AssertionError("compatible street should reuse generated filters")

    monkeypatch.setattr(PlayerRange, "autogenerate_filters", fail_if_regenerated)
    snapshots = manager.change_board("5h2c3d")

    assert manager._current_state is saved_flop
    assert snapshots[0] == expected
    assert "manual-kings" in player_at(manager, Street.FLOP, 0).get_tree()
