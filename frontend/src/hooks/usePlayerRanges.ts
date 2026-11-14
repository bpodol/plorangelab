import { useState } from 'react';
import { addRange, deleteRange } from '../api/players.ts';
import {
  createFilter,
  createGroup,
  setFilterActive,
  deleteFilter,
  clearFilters,
  type FilterResource,
} from '../api/filters.ts';
import type { Mode, FilterTree, FilterNode } from '../types.ts';
import type { PlayerFilterSnapshot } from '../types.ts';

/**
 * Represent the frontend state of one player slot.
 *
 * Tracks the editable range expression, effective hand count,
 * current-street filter snapshot, applied-range status,
 * and loading/error states for range and filter operations.
 *
 * The editable range expression may differ from the range currently
 * stored on the backend until the user applies it.
 */
export interface PlayerState {
  range: string;
  numberOfHands: number;
  /** Current-street filter state returned by the backend, or null if unavailable. */
  filterSnapshot: PlayerFilterSnapshot | null;
  hasAppliedRange: boolean;
  isLoading: boolean;
  error: string | null;
}

/**
 * Manage frontend state and REST operations for six player slots.
 *
 * Maintains editable range expressions, hand counts, filter snapshots,
 * applied-range flags, loading indicators, and per-player errors.
 *
 * Provides operations for creating, replacing, and deleting player ranges,
 * as well as adding, removing, activating, and clearing filters and groups.
 * Synchronizes hand counts and filter snapshots after board changes.
 *
 * @param sessionId - Active backend session ID, or null when absent.
 * @param mode - PLO variant used when submitting player ranges.
 *
 * @remarks
 * The backend is the source of truth for filter state. Successful filter
 * mutations replace the player's snapshot with the backend response.
 * Filter mutations are skipped when the session is unavailable or the
 * player is loading.
 *
 * The hook does not automatically fetch player state from the backend
 * or reset its state when sessionId changes.
 */
export function usePlayerRanges(sessionId: string | null, mode: Mode) {
  const [players, setPlayers] = useState<PlayerState[]>(
    Array.from({ length: 6 }, () => ({
      range: '',
      numberOfHands: 0,
      filterSnapshot: null,
      hasAppliedRange: false,
      isLoading: false,
      error: null,
    })),
  );

  function updatePlayer(playerId: number, changes: Partial<PlayerState>) {
    setPlayers((previousPlayers) =>
      previousPlayers.map((player, index) =>
        index === playerId ? { ...player, ...changes } : player,
      ),
    );
  }

  /**
   * Submit a player's range expression to the backend.
   *
   * Creates or replaces the range in the selected player slot.
   * On success, updates the reported hand count, marks the range as applied,
   * and clears the previous filter
   * snapshot because the backend initializes a new player range state.
   *
   * Loading and error states are maintained for the target slot.
   *
   * @param playerId - Zero-based player slot index (0-5).
   *
   * @remarks
   * Returns immediately when no backend session exists.
   * Errors are stored in the player's state rather than rethrown.
   */
  async function handleAddRange(playerId: number) {
    if (sessionId === null) return;

    const player = players[playerId];

    updatePlayer(playerId, { isLoading: true, error: null });

    try {
      const hands = await addRange(sessionId, playerId, player.range, mode);

      updatePlayer(playerId, {
        numberOfHands: hands.hands,
        filterSnapshot: null,
        hasAppliedRange: true,
      });
    } catch (err) {
      updatePlayer(playerId, { error: err instanceof Error ? err.message : 'Unknown error' });
    } finally {
      updatePlayer(playerId, { isLoading: false });
    }
  }

  /**
   * Remove a player's range from the backend.
   *
   * On success, resets the player's editable range, hand count,
   * and applied-range flag.
   *
   * Maintains per-player loading and error states throughout
   * the operation.
   *
   * @param playerId - Zero-based player slot index (0-5).
   *
   * @remarks
   * Returns immediately when no backend session exists.
   * Failed deletions preserve the existing range state.
   */
  async function handleDeleteRange(playerId: number) {
    if (sessionId === null) return;

    updatePlayer(playerId, { isLoading: true, error: null });

    try {
      await deleteRange(sessionId, playerId);

      updatePlayer(playerId, {
        range: '',
        numberOfHands: 0,
        filterSnapshot: null,
        hasAppliedRange: false,
      });
    } catch (err) {
      updatePlayer(playerId, { error: err instanceof Error ? err.message : 'Unknown error' });
    } finally {
      updatePlayer(playerId, { isLoading: false });
    }
  }

  /**
   * Execute a filter mutation and synchronize the player's frontend state.
   *
   * Handles loading, errors, and updates the player's hand count and
   * filter snapshot using the backend response.
   *
   * Skips execution if no session exists or the player is loading.
   * Errors are stored in player state rather than rethrown.
   *
   * @param playerId - Zero-based player slot index (0-5).
   * @param operation - Async mutation that receives the session ID
   *                    and returns an updated player filter snapshot.
   */
  async function runFilterMutation(
    playerId: number,
    operation: (sessionId: string) => Promise<PlayerFilterSnapshot>,
  ): Promise<boolean> {
    if (sessionId === null) return false;
    if (players[playerId].isLoading) return false;

    updatePlayer(playerId, { isLoading: true, error: null });

    try {
      const filterSnapshot = await operation(sessionId);

      updatePlayer(playerId, {
        numberOfHands: filterSnapshot.filtered_hands,
        filterSnapshot,
      });

      return true;
    } catch (err) {
      updatePlayer(playerId, {
        error: err instanceof Error ? err.message : 'Unknown error',
      });
      return false;
    } finally {
      updatePlayer(playerId, { isLoading: false });
    }
  }

  /**
   * Retrieve a filter or group from a nested filter tree.
   *
   * @param tree - Root filter tree.
   * @param name - Name of the target filter or group.
   * @param path - Names of parent groups, ordered from root to parent.
   * @returns The matching node, or undefined if the path or node is invalid.
   */
  function getFilterNode(
    tree: FilterTree,
    name: string,
    path: string[] = [],
  ): FilterNode | undefined {
    let currentTree = tree;

    for (const groupName of path) {
      const group = currentTree[groupName];

      if (group?.type !== 'group') {
        return undefined;
      }

      currentTree = group.filters;
    }

    return currentTree[name];
  }

  /**
   * Add a filter to the specified player's filter tree.
   *
   * Sends the filter definition and path to the backend and updates the player's
   * hand count and filter snapshot on success.
   *
   * Skips the operation if no session exists or the player is loading.
   * API errors are stored in the player's state rather than rethrown.
   *
   * @param playerId - Zero-based player slot index (0-5).
   * @param expression - Filter expression to evaluate.
   * @param name - Optional custom name for the filter.
   * @param path - Optional path to the parent filter group.
   */
  function addFilter(
    playerId: number,
    expression: string,
    name?: string,
    path?: string[],
  ): Promise<boolean> {
    return runFilterMutation(playerId, (sessionId) =>
      createFilter(sessionId, playerId, { expression, name, path }),
    );
  }

  /**
   * Create a filter group in the specified player's filter tree.
   *
   * Sends the group definition to the backend and updates the player's
   * hand count and filter snapshot on success.
   *
   * Skips the operation if no session exists or the player is loading.
   * API errors are stored in the player's state rather than rethrown.
   *
   * @param playerId - Zero-based player slot index (0-5).
   * @param name - Name of the new filter group.
   * @param path - Optional path to the parent group.
   */
  function addGroup(playerId: number, name: string, path?: string[]): Promise<boolean> {
    return runFilterMutation(playerId, (sessionId) =>
      createGroup(sessionId, playerId, { name, path }),
    );
  }

  /**
   * Toggle the activation state of a filter or group.
   *
   * Reads the current activation state from the player's filter tree
   * and requests the opposite state from the backend.
   *
   * Skips execution if the target node does not exist.
   * Loading, errors, and snapshot updates are handled by runFilterMutation.
   *
   * @param playerId - Zero-based player slot index (0-5).
   * @param resource - Resource type (filter or group).
   * @param name - Name of the target node.
   * @param path - Optional path to its parent group.
   */
  function toggleFilterActive(
    playerId: number,
    resource: FilterResource,
    name: string,
    path?: string[],
  ): Promise<boolean> {
    const tree = players[playerId]?.filterSnapshot?.filters;

    if (!tree) return Promise.resolve(false);

    const node = getFilterNode(tree, name, path);

    if (!node) return Promise.resolve(false);

    return runFilterMutation(playerId, (sessionId) =>
      setFilterActive(sessionId, playerId, resource, name, !node.active, path),
    );
  }

  /**
   * Remove a filter or group from the specified player's filter tree.
   *
   * Sends the deletion request to the backend and synchronizes the
   * player's filter snapshot and hand count on success.
   *
   * Loading, errors, and state updates are handled by runFilterMutation.
   *
   * @param playerId - Zero-based player slot index (0-5).
   * @param resource - Type of resource to remove (filter or group).
   * @param name - Name of the target filter or group.
   * @param path - Optional path to the parent group.
   */
  function removeFilter(
    playerId: number,
    resource: FilterResource,
    name: string,
    path?: string[],
  ): Promise<boolean> {
    return runFilterMutation(playerId, (sessionId) =>
      deleteFilter(sessionId, playerId, resource, name, path),
    );
  }

  /**
   * Deactivate all filters and groups for the specified player.
   *
   * Preserves filter definitions and hierarchy while restoring the
   * effective hand count to the unfiltered base range.
   *
   * Loading, errors, and snapshot synchronization are handled
   * by runFilterMutation.
   *
   * @param playerId - Zero-based player slot index (0-5).
   */
  function clearAllFilters(playerId: number): Promise<boolean> {
    return runFilterMutation(playerId, (sessionId) => clearFilters(sessionId, playerId));
  }

  /**
   * Synchronize player hand counts after a community board change.
   *
   * Updates each slot's count using the backend response while
   * preserving its range expression and other frontend state.
   *
   * Missing count entries default to zero.
   *
   * @param counts - Hand counts in player slot order.
   */
  function updateHandCounts(counts: number[]) {
    setPlayers((previousPlayers) =>
      previousPlayers.map((player, index) => ({
        ...player,
        numberOfHands: counts[index] ?? 0,
      })),
    );
  }

  /**
   * Apply the player snapshots returned after a board change.
   *
   * Updates effective hand counts and filter states together, preserving
   * player slot order. Empty slots receive a zero count and null snapshot.
   *
   * @param snapshots - Six slot-aligned snapshots returned by the backend.
   */
  function applyBoardSnapshots(snapshots: (PlayerFilterSnapshot | null)[]) {
    setPlayers((previousPlayers) =>
      previousPlayers.map((player, index) => {
        const snapshot = snapshots[index] ?? null;

        return {
          ...player,
          numberOfHands: snapshot?.filtered_hands ?? 0,
          filterSnapshot: snapshot,
        };
      }),
    );
  }

  return {
    players,
    updateHandCounts,
    applyBoardSnapshots,
    updatePlayer,
    handleAddRange,
    handleDeleteRange,
    addFilter,
    addGroup,
    toggleFilterActive,
    removeFilter,
    clearAllFilters,
  };
}
