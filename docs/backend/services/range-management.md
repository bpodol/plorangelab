# Range management

## Overview

The range-management services maintain the player ranges used in a PLO4/PLO5 equity calculation. They coordinate up to six player slots, a shared community board, street-specific filter state, and each player's currently selected weighted range.

The design separates three responsibilities:

1. **`RangeManager`** coordinates players, the shared board, range input, filter commands, and equity requests.
2. **`PlayerRangeState`** owns one player's preflop, flop, turn, and river states and manages transitions between them.
3. **`FilterManager`** owns a single street's hierarchical filter definitions and derives the selected range from their activation states.

The services reuse the stateless [filter expression engine](../filter-engine.md) to evaluate predicates, the [range encoder](../encoding.md) to parse inputs, and the [equity engine](../equity.md) to calculate equities. They do not implement those algorithms themselves.

## Source files and ownership

| File | Responsibility |
| --- | --- |
| `src/plorangelab/services/range_manager.py` | Six player slots, shared board, coordinated updates, and equity-engine integration. |
| `src/plorangelab/services/player_range_state.py` | Board representation, per-street range states, filter carryover, and street invalidation. |
| `src/plorangelab/services/filter_manager.py` | Filter trees, activation, coverage, effective selections, and cloning. |

```text
RangeManager
├── board: Board | None
├── players[0..5]: PlayerRangeState | None
│   ├── preflop: StreetState ── FilterManager
│   ├── flop:    StreetState | None ── FilterManager
│   ├── turn:    StreetState | None ── FilterManager
│   ├── river:   StreetState | None ── FilterManager
│   └── active_street ── one of the states above
├── ratings: PLO4/PLO5 starting-hand orderings
└── rank_lookup: five-card ranking tables
```

Each occupied player slot has its own `PlayerRangeState`. A `StreetState` has its own `FilterManager`; it does not share a mutable filter tree with other streets or players.

## Player slots and range input

`RangeManager` starts with six empty slots, indexed `0` through `5`. `add_range(input_range, mode, player_index)` parses explicit weighted combinations or percentage notation using `encode_input()` and the appropriate precomputed starting-hand ordering. The result initializes a new `PlayerRangeState` with a preflop `StreetState`.

If a board is already selected, the new player is advanced to that board **before** the slot is replaced. Replacing a slot discards that player's previous range and filter history; `remove_range()` clears the slot.

Player range data uses `EncodedRange`, with index-aligned arrays:

- `hands`: encoded PLO4/PLO5 hole cards (`uint8`).
- `masks`: one 52-card bitmask per hand (`uint64`).
- `weights`: positive per-hand weights (`float32`).

The manager uses the same `EngineResources` instance's ratings and ranking tables across requests. It does not load or regenerate those resources itself.

## Community board and street state

### Board representation

`Board.from_string()` converts a three-, four-, or five-card board string into a `Board` containing:

- `cards`: an encoded, read-only `uint8` array, preserving input card order.
- `mask`: a `uint64` set of occupied cards, independent of input order.

The mask is used to recognize previously visited boards and test whether a newer board extends an earlier board. Board strings are validated by the shared encoder. The services assume direct callers of `PlayerRangeState.apply_board_change()` supply a valid `Board`.

### Street-local ranges

Each `StreetState` owns a `FilterManager` whose `base_range` is fixed when the state is created. Its `current_range` and `filtered_coverage` are derived by the manager, not separately maintained by `PlayerRangeState`.

When advancing to a newly created street, the new base range is constructed from the **current filtered range of the latest compatible earlier street**, after removing hands blocked by the new community board:

```text
Preflop current range
    │ remove flop blockers
    ▼
Flop base range → flop filters → flop current range
    │ remove turn blockers
    ▼
Turn base range → turn filters → turn current range
    │ remove river blockers
    ▼
River base range → river filters → river current range
```

Only the **hands and weights** carry forward; the previous street's filter definitions are **not copied**. New street states begin with empty filter trees. This means a filter's effect can persist in the next street's starting range even though the filter itself is not present there.

### Reusing and invalidating street states

`apply_board_change()` compares board masks and preserves an existing street state when its board is still applicable. Incompatible states are replaced or discarded.

| Operation | Existing state behavior |
| --- | --- |
| Select the same flop | Reuses the flop state; discards turn and river states. |
| Select a new flop | Rebuilds the flop from the current preflop range; discards turn and river states. |
| Select the same turn | Reuses the turn state and its retained flop state; discards river. |
| Select a different turn | Retains the flop only if the new board contains all of its cards; rebuilds turn. |
| Select the same river | Reuses the river and retains its existing earlier states. |
| Select a different river | Retains compatible turn and/or flop states; rebuilds river from the latest retained state, otherwise from preflop. |
| Clear the board | Returns to preflop, preserving preflop filters and discarding all postflop states. |

Compatibility is based on **card membership**, not the original order of board-string characters. A state that has been discarded is not retained in a separate history for later restoration.

The filtering baseline is **street-relative**: the new street's `base_range` may already be narrower than the preflop range due to earlier filtering and card removal.

## Filter trees

Each `FilterManager` owns a tree of two node kinds:

- **`Filter`**: a saved expression, a read-only Boolean selection aligned with the street's `base_range`, weighted membership coverage, and an `active` flag.
- **`FilterGroup`**: a named collection of filters and/or nested groups, with its own membership coverage and `active` flag.

Nodes are addressed using `FilterAddress(name, path)`. The `path` is a tuple of **parent group names**, starting at the root; the target node's own name is held separately.

For example, `FilterAddress(name="flushes", path=("draws", "flop"))` identifies `draws → flop → flushes`. Names must be nonempty and unique among siblings; different groups may contain children with the same name.

### Creating and deleting nodes

`create_filter()` evaluates its expression once against **every hand mask in the street's base range**, stores the resulting selection, and calculates coverage. A new root-level filter starts inactive. A filter created inside an active parent group starts active when it matches at least one hand.

A zero-match filter can be saved, but direct activation of a zero-membership node is rejected. New groups start empty and inactive, including when created inside active groups.

Deleting a filter removes its definition. Deleting a group also removes its entire subtree. Membership coverage and the current effective selection are refreshed where necessary.

### Activation rules

Activation determines which saved selections contribute to the street's current range:

- **Root-level contributions are ORed together.** Filter groups are organizational gates, not intersections of sibling selections.
- **Inactive group = inactive contribution.** No descendant of an inactive group contributes, regardless of the stored descendant flags.
- Activating a **leaf** also activates all ancestor groups, but does not activate its siblings.
- Activating a **group** recursively activates all of its descendants; deactivating a group recursively deactivates them.
- Asking for the same activation state again is a no-op. In particular, reactivating an already-active group does not reset individually disabled children.
- `clear_all_filters()` deactivates every node while preserving definitions and groups.

An activation request for a node with no membership is rejected. A nuance of the current implementation is that activating a *nonempty group* recursively sets all descendants active, including any zero-match leaves it contains; such leaves still contribute no hands.

### Membership versus effective selection

These concepts are intentionally different:

**Membership selection** is the union of **all** saved leaf selections below a node, including inactive leaves. It determines a filter or group's full coverage.

**Effective selection** includes only currently active leaves whose parent chain is also active. It determines which hands contribute to the current filtered range.

`get_selection(address, active_only=False)` returns the membership selection; `active_only=True` returns the effective selection with ancestor gating. Both produce new, read-only Boolean arrays.

### Weighted coverage

Coverage is calculated against the **original base range for that street**, using hand weights rather than hand counts:

```text
coverage = sum(weights of selected base-range hands)
           / sum(weights of all base-range hands)
```

The selection is Boolean, so overlapping filters do not double-count a hand's weight. A group's membership coverage includes its inactive descendants; `get_coverage(..., active_only=True)` measures only active contribution.

Two related values have different meanings:

- **Node active coverage:** zero when the addressed node contributes no active hands.
- **`filtered_coverage`:** `1.0` when no active filter contributes, because the manager treats that as **unfiltered** and returns the complete street base range.

Consequently, a new turn can begin at `filtered_coverage == 1.0` even though its base range has already been narrowed by filters applied on the flop. The denominator is the turn's base range, not the player's original preflop range.

### Derived state and read operations

`FilterManager` maintains three derived values:

| Property | Meaning |
| --- | --- |
| `active_selection` | Read-only Boolean mask aligned with `base_range`; `None` when no filtering applies. |
| `current_range` | Subset of the base range selected by active filters, or the original base range when unfiltered. |
| `filtered_coverage` | Weighted share of the street's base range retained by the current selection. |

`_refresh()` recalculates these caches after relevant edits; it also updates group membership coverages when leaves are created or deleted. `current_range` is constructed by applying the active Boolean selection to the aligned `hands`, `masks`, and `weights` arrays.

`get_tree()` and `get_node_info()` return nested dictionaries with node names, kinds, expressions, coverage, and activation state. They do not expose the underlying Boolean selection arrays. These snapshots are suitable for API serialization; filter selection itself is not serialized here.

The `filters` property exposes a read-only **root mapping**, and `get_node()` returns node objects for inspection. Neither prevents mutation of the nested `Filter` / `FilterGroup` objects; callers are expected to treat those objects as read-only and use manager methods for changes.

### Player filter snapshots

`RangeManager` produces a `PlayerFilterSnapshot` representing an occupied player's current-street state.

The snapshot is a frozen dataclass containing:

| Field | Description |
| --- | --- |
| `base_hands` | Number of combinations in the current street's base range. |
| `filtered_hands` | Number of combinations remaining after applying the current street's active filters. |
| `filtered_coverage` | Weighted fraction of the current street's base range retained by filtering. |
| `filters` | Complete serialized filter tree for the current street, including inactive nodes. |

Hand counts represent combinations, while coverage accounts for their weights.

The base range may already incorporate filtering from earlier streets and community-card exclusions. Consequently, `filtered_coverage == 1.0` does not necessarily mean that the player retains their entire original preflop range.

### Snapshot-producing operations

Filter and group mutations return the updated `PlayerFilterSnapshot` for the affected player after the change succeeds.

`change_board()` returns a list of six elements in player-slot order:

- Occupied slots contain their updated current-street snapshots.
- Empty slots contain `None`.

This also applies to `change_board(None)`, which returns the resulting preflop snapshots after clearing the board.

Snapshots are produced from the published player state. A failed mutation does not publish a partially modified state.

### Snapshot immutability

`PlayerFilterSnapshot` is a frozen dataclass. Its filter tree is serialized separately from the mutable `Filter` and `FilterGroup` objects.

However, freezing the dataclass does not make its nested dictionaries deeply immutable. Consumers should treat snapshots as read-only data.

## Update consistency and cloning

### Player edits

`RangeManager._edit_player()` performs a **clone–modify–publish** operation:

```text
1. Find the existing player.
2. Clone the player and its street-local filter managers.
3. Apply the requested edit to the clone.
4. Replace the player slot only after the edit succeeds.
```

A failed filter operation leaves the original player in the slot unchanged. The manager uses this mechanism for adding/deleting filters, creating groups, toggling activation, and clearing filters.

### Shared-board edits

`change_board()` and `clear_board()` build a new player list and update clones of **all** occupied players before publishing the replacement list and board. If any player fails board-card filtering, the previous published players and board remain in place.

This gives the operations **failure atomicity** for ordinary exceptions during preparation. It is **not** a thread-safety or transaction-isolation mechanism: concurrent writes to the same `RangeManager` still require coordination outside these classes.

### Copying strategy

Cloning copies mutable player, street, and filter-tree configuration while reusing encoded range arrays, board objects, and read-only leaf-selection masks. Filter and group objects are reconstructed so changes to their activation flags in a clone do not mutate the original manager's tree.

The design relies on callers treating shared encoded arrays and other exposed mutable-looking data as immutable by convention. The public state objects are not a comprehensive defensive-copy boundary.

## Equity calculation integration

`RangeManager.calculate_equity()` gathers the `current_range` of every occupied slot, preserving ascending slot order. It passes those ranges, the shared board, the selected PLO mode, and the shared `rank_lookup` to `compute_equity()`.

The method forwards optional Monte Carlo progress callbacks, a cancellation event, and a target margin of error. It delegates sampling, exact enumeration, input validation, and statistics to the [equity engine](../equity.md).

Its return value is a pair:

```python
(player_indices, equity_result)
```

`player_indices` maps the returned equity values back to the original six slots. For example, if slots `0`, `2`, and `5` are occupied, the result indices are `[0, 2, 5]`, and the three equity values use that same order.

The shared calculation mode must be compatible with all occupied ranges. A mixed PLO4/PLO5 configuration is not normalized by `RangeManager`; the delegated equity engine rejects incompatible hand dimensions. The engine also enforces its own player-count requirements.

## Boundaries and current constraints

- **No persistence in this layer.** These classes maintain in-memory state; they do not save range/filter state to a database.
- **No API or session lifecycle.** HTTP/WebSocket endpoints, session ownership, and job scheduling are handled by other components.
- **No new filter algorithm.** Hand predicates are evaluated by `plorangelab.filters`; this layer combines and manages the returned selections.
- **One shared board per manager.** All occupied players transition against the same community cards.
- **Filter carryover is by range, not by definition.** Later streets start from the parent street's currently selected hands, with no copied filter tree.
- **State restoration is limited.** Compatible previously held streets may be reused, but invalidated street states are discarded rather than kept in an undo/history store.
- **No built-in concurrent-mutation protection.** Clone-based publishing protects against failed edits, not races between simultaneous requests.

## Related documentation

- [Encoding and starting-hand orderings](../encoding.md)
- [Card filter expression engine](../filter-engine.md)
- [Equity engine](../equity.md)
- [Five-card hand ranking](../hand-ranking.md)

[← Services overview](README.md) · [Session and equity jobs](session-and-jobs.md)