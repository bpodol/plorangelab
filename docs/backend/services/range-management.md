# Range management

## Overview

The range-management services maintain the player ranges used in a PLO4/PLO5 equity calculation. They coordinate up to six player slots, a shared community board, street-specific filter state, and each player's currently selected weighted range.

The design separates three responsibilities:

1. **`RangeManager`** owns the shared PLO mode, all existing street states, the active street selection, board transitions, and equity calculation coordination.
2. **`StreetState`** represents a single street, holding its community board and up to six independent player ranges.
3. **`PlayerRange`** owns one player's range and filter configuration for a single street, including hierarchical filters, activation states, and derived weighted selections.

The services reuse the stateless [filter expression engine and generator](../filter-engine.md) to evaluate predicates and generate board-dependent definitions, the [range encoder](../encoding.md) to parse inputs, and the [equity engine](../equity.md) to calculate equities. They do not implement those algorithms themselves.

## Source files and ownership

| File | Responsibility |
| --- | --- |
| `src/plorangelab/services/range_manager.py` | Shared PLO mode, street states, board transitions, player range updates, and equity-engine integration. |
| `src/plorangelab/services/player_range.py` | `PlayerRange`, filter trees, activation, coverage, effective selections, and cloning. |

```text
RangeManager
├── mode: PLO4 | PLO5
├── _active_street: Street
├── _streets: dict[Street, StreetState]
│   ├── PREFLOP ── StreetState
│   │   ├── board: None
│   │   └── ranges[0..5]: PlayerRange | None
│   ├── FLOP ──── StreetState (optional)
│   │   ├── board: Board
│   │   └── ranges[0..5]: PlayerRange | None
│   ├── TURN ──── StreetState (optional)
│   │   ├── board: Board
│   │   └── ranges[0..5]: PlayerRange | None
│   └── RIVER ─── StreetState (optional)
│       ├── board: Board
│       └── ranges[0..5]: PlayerRange | None
└── resources: EngineResources
```

`RangeManager` owns all existing street states. The preflop state always exists, while postflop states are created on demand.

Each `StreetState` owns a shared community board and six player slots. Each occupied slot contains an independent `PlayerRange` with its own filter tree and selected weighted range.

`_active_street` identifies the street currently used for filter operations and equity calculations.

## Player slots and range input


`RangeManager` maintains six player slots, indexed `0` through `5`, for each existing street. All players share a single PLO mode (4 or 5), which is managed at the `RangeManager` level.

`add_range(range_input, slot_index)` parses explicit weighted combinations or percentage notation using `encode_input()` and the precomputed starting-hand ordering for the current mode.

The encoded range initializes a new preflop `PlayerRange`. The manager then propagates this range through all existing postflop states, removing hands blocked by each street's community board. When the active street has a board, the new active-street `PlayerRange` receives automatically generated filters.

Adding or replacing a range discards the previous range and filter configurations for that slot across all streets. Automatic filters are generated **only for the active postflop street**, rather than eagerly for the other recreated cached streets. `remove_range(slot_index)` clears the slot across all streets. Other players' ranges and filters remain unchanged.

`add_range()` returns the current player's `PlayerFilterSnapshot`. Its `filters` field contains the active street's generated definitions when applicable.

Updates are atomic: replacement states are prepared before being published, so a failed operation leaves the existing state unchanged.

Player range data uses `EncodedRange`, with index-aligned arrays:

- `hands`: encoded PLO4/PLO5 hole cards (`uint8`).
- `masks`: one 52-card bitmask per hand (`uint64`).
- `weights`: positive per-hand weights (`float32`).

The manager uses the same `EngineResources` instance's ratings and ranking tables across requests. It does not load or regenerate those resources itself.

## Shared PLO mode

`RangeManager` owns a single PLO mode shared by all players and streets. New managers default to PLO4 (`mode = 4`).

`change_mode(mode)` switches between PLO4 and PLO5. A mode change is rejected while any player slot isoccupied Player-slot occupancy is consistent across all existing streets, so checking the active street is sufficient Requesting the already active mode is a no-op, even when ranges exist.

Changing the mode does not clear the community board or street states. New player ranges are encoded using the manager's current mode.

The mode is not specified separately when adding player ranges or starting equity calculations.

## Community board and street state

### Board representation

`Board.from_string()` converts a three-, four-, or five-card board string into a `Board` containing:

- `cards`: an encoded, read-only `uint8` array, preserving input card order.
- `mask`: a `uint64` bitmask representing the occupied cards, independent of input order.

`RangeManager.change_board()` validates the input by creating a `Board` and uses its mask to determine board identity and compatibility.

Boards containing the same cards in different orders are considered identical. When transitioning between streets, a previously created board is compatible if all its cards are present in the new board.

Each `StreetState` holds its own `Board` reference. The preflop state has `board = None`.

### Street-local ranges

Each `StreetState` contains a shared `Board | None` and six player slots. Each occupied slot holds an independent `PlayerRange` for that street.

A `PlayerRange` owns its `base_range`, filter tree, current filtered range, and weighted coverage. Its base range is established when the street state or player range is created.

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

Only the **hands and weights** carry forward; the previous street's filter definitions are **not copied**. A newly created postflop state instead generates its own board-dependent filter tree for each occupied player. This means an earlier filter's effect can persist in the next street's starting range even though that specific filter definition was not carried forward.

The generated filter definitions are initially inactive. Generation populates membership and coverage without narrowing the new street's effective range.

### Reusing and invalidating street states

`RangeManager.change_board()` selects the target street using the number of community cards and determines state compatibility using board masks.

When the target street already exists with an identical board mask, its state and already-created filter trees are reused **without regenerating filters**. Later streets are discarded.

For a new board, the manager preserves compatible earlier street states (including their filter trees), creates the target state from the nearest surviving parent, generates the target street's filters for occupied players, and discards incompatible states. All players follow the same street transition.

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

Each `PlayerRange` owns a tree of two node kinds:

- **`Filter`**: a saved expression, a read-only Boolean selection aligned with the street's `base_range`, weighted membership coverage, and an `active` flag.
- **`FilterGroup`**: a named collection of filters and/or nested groups, with its own membership coverage and `active` flag.

Nodes are addressed using `FilterAddress(name, path)`. The `path` is a tuple of **parent group names**, starting at the root; the target node's own name is held separately.

For example, `FilterAddress(name="flushes", path=("draws", "flop"))` identifies `draws → flop → flushes`. Names must be nonempty and unique among siblings; different groups may contain children with the same name.

### Automatic filter initialization

`PlayerRange.autogenerate_filters(board.cards)` calls `generate_filters(board)` from the `filters` package and recursively installs the resulting definitions. `GroupNode` definitions become `FilterGroup` nodes via `create_group()`; `FilterNode` definitions become evaluated `Filter` nodes via `create_filter()`. A generated leaf is evaluated against that player's street `base_range`, so coverage reflects the player's actual weighted range.

Automatic filters start inactive, including descendants of generated groups. They are ordinary saved nodes afterward: the same activation, deletion, clearing, and nesting rules apply as for user-created filters. The generator does not classify the player's hands during definition construction; `create_filter()` computes their membership.

**Lazy lifecycle:** Creating a new board state initializes filters for its occupied players once; revisiting an identical compatible board reuses the existing street state without regeneration. Replacing one player's range rebuilds that player's ranges across cached streets but runs generation only for the currently active postflop street. This avoids evaluating potentially expensive generated definitions on every cached street during one replacement.

### Creating and deleting nodes

`create_filter()` evaluates its expression once against **every hand mask in the street's base range**, stores the resulting selection, and calculates coverage. A new root-level filter starts inactive. A filter created inside an active parent group starts active when it matches at least one hand.

A zero-match filter can be saved, but direct activation of a zero-membership node is rejected. New groups start empty and inactive, including when created inside active groups.

Deleting a filter removes its definition. Deleting a group also removes its entire subtree. If the deletion leaves an ancestor without any active descendant filters, that ancestor is automatically deactivated. Ancestors with active filters in other branches remain active.

Membership coverage and the current effective selection are refreshed
after deletion.

### Activation rules

Activation determines which saved selections contribute to the street's current range:

- **Root-level contributions are ORed together.** Filter groups are organizational gates, not intersections of sibling selections.
- **Inactive group = inactive contribution.** No descendant of an inactive group contributes, regardless of the stored descendant flags.
- Activating a **leaf** also activates all ancestor groups, but does not activate its siblings.
- Activating a **group** recursively activates descendants that match at least one hand. Zero-selection filters remain inactive, as do groups without any active descendant filters. Deactivating a group recursively deactivates its entire subtree.
- Asking for the same activation state again is a no-op. In particular, reactivating an already-active group does not reset individually disabled children.
- `clear_all_filters()` deactivates every node while preserving definitions and groups.
- Deactivating a filter or group automatically deactivates its ancestors when they no longer contain any active descendant filters. Ancestors with active filters in other branches remain active.

Direct activation of a node with no membership is rejected. Recursive group activation applies the same nonempty-selection requirement to descendant filters, keeping zero-match leaves and empty groups inactive.

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

`PlayerRange` maintains three derived values:

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

`add_range()` and filter/group mutations return the updated `PlayerFilterSnapshot` for the affected player after the change succeeds. A postflop range-add snapshot includes automatically generated filter definitions for the active street.

`change_board()` returns a list of six elements in player-slot order:

- Occupied slots contain their updated current-street snapshots.
- Empty slots contain `None`.

This also applies to `change_board(None)`, which returns the resulting preflop snapshots after clearing the board.

For filter mutations, snapshots are constructed from the modified clone before it is published. If mutation or snapshot construction fails, the original player state remains unchanged.

### Snapshot immutability

`PlayerFilterSnapshot` is a frozen dataclass. Its filter tree is serialized separately from the mutable `Filter` and `FilterGroup` objects.

However, freezing the dataclass does not make its nested dictionaries deeply immutable. Consumers should treat snapshots as read-only data.

## Update consistency and cloning

### Player edits

`RangeManager._edit_player()` performs a **clone-modify-publish** operation for the affected player's active-street range:

1. Retrieve the `PlayerRange` from the active `StreetState`.
2. Clone that `PlayerRange`, including its mutable filter-tree configuration.
3. Apply the requested mutation and construct a `PlayerFilterSnapshot` from the clone.
4. Publish the clone by replacing only the affected slot in the active street.

If mutation or snapshot construction fails, the original `PlayerRange` remains unchanged.

Other players and inactive street states are not cloned when editing a filter. The same mechanism handles filter creation, deletion, activation, group creation, and clearing.

### Shared-board edits

`change_board()` manages street transitions centrally.

If the requested board matches an existing state, the manager reuses that state and discards later streets. Otherwise, it constructs a new target `StreetState` and derives each occupied player's range from the nearest compatible parent state.

The replacement street mapping is published only after the new state has been constructed successfully. Failed preparation leaves the previous street mapping and active street unchanged.

`change_board(None)` returns to preflop and discards postflop states while preserving the preflop ranges and filters.

Adding or replacing a player range also uses prepared replacement street states. It rebuilds that player's range across all existing streets, generates filters only in the new active-street `PlayerRange` when a board exists, and constructs the response snapshot **before** publishing the updated mapping.

These operations provide failure atomicity for ordinary exceptions during preparation, not general transaction isolation or automatic thread safety.

### Copying strategy

Filter mutations clone only the affected `PlayerRange` for the active street. The clone reconstructs mutable filter and group objects while sharing immutable-by-convention encoded range arrays and read-only Boolean selection masks.

Board transitions reuse compatible `StreetState` objects. Newly created streets receive new `PlayerRange` objects derived from their parent ranges.

Adding or removing a player constructs replacement `StreetState` objects with copied six-slot lists, preserving other players' existing `PlayerRange` objects.

The implementation relies on callers treating shared encoded arrays and exposed data as immutable by convention. It is not a comprehensive defensive-copy boundary.

## Equity calculation integration

`RangeManager.calculate_equity()` gathers the `current_range` of every occupied slot in the active `StreetState` into a dictionary keyed by the original zero-based player slot ID. It iterates slots in ascending order, but player identity is represented by the dictionary keys, not by a separate positional index list.

For example, when slots `0`, `2`, and `5` are occupied, the input to the engine has this shape:

```python
players = {0: range_0, 2: range_2, 5: range_5}
```

The manager passes that mapping, the active street's board, the session-wide PLO mode, and shared ranking resources to `compute_equity()`.

The method forwards optional Monte Carlo progress callbacks, a cancellation event, and a target margin of error. It delegates sampling, exact enumeration, input validation, and statistics to the [equity engine](../equity.md).

It returns a single `EquityResult`, not an `(indices, result)` tuple. The result's `equities` dictionary, and its `margins_of_error` dictionary when present, use the same original slot IDs as keys. For the example above, `set(result.equities) == {0, 2, 5}`. No separate player-index list is returned.

The calculation mode is owned by `RangeManager`. All player ranges are encoded using that shared mode, so the manager does not support mixed PLO4/PLO5 player configurations. The equity engine independently validates its input requirements, including the number of participating players.

## Boundaries and current constraints

- **No persistence in this layer.** These classes maintain in-memory state; they do not save range/filter state to a database.
- **No API or session lifecycle.** HTTP/SSE endpoints, session ownership, and job scheduling are handled by other components.
- **Definition generation is delegated.** Board-dependent definitions come from `plorangelab.filters.generate_filters`; `PlayerRange` installs them and evaluates their expressions, while the filtering service manages selections.
- **Shared board per street.** Each `StreetState` stores one board shared by all players on that street. `RangeManager` owns all existing street states and selects one active street.
- **Filter carryover is by range, not by definition.** Later streets start from the parent street's currently selected hands and create their own generated filter trees instead of copying parent definitions.
- **State restoration is limited.** Compatible previously held streets may be reused, but invalidated street states are discarded rather than kept in an undo/history store.
- **No built-in concurrent-mutation protection.** Clone-based publishing protects against failed edits, not races between simultaneous requests.

## Related documentation

- [Encoding and starting-hand orderings](../encoding.md)
- [Card filter expression engine](../filter-engine.md)
- [Equity engine](../equity.md)
- [Five-card hand ranking](../hand-ranking.md)

[← Services overview](README.md) · [Session and equity jobs](session-and-jobs.md)