import { useState } from 'react';
import { addRange, deleteRange } from '../api/players.ts';
import type { Mode } from '../types.ts';

/**
 * Represent the frontend state of one player slot.
 *
 * Tracks the editable range expression, the last reported
 * hand count, whether a range has been applied successfully,
 * and the loading/error state of range operations.
 *
 * The range expression may differ from the range currently
 * stored on the backend until the user applies it.
 */
export interface PlayerState {
  range: string;
  numberOfHands: number;
  hasAppliedRange: boolean;
  isLoading: boolean;
  error: string | null;
}

/**
 * Manage frontend state and REST operations for six player slots.
 *
 * Maintains editable range expressions, hand counts, applied-range
 * flags, loading indicators, and per-player errors.
 *
 * Provides handlers for adding, replacing, and deleting ranges,
 * along with utilities for updating individual player state
 * and synchronizing hand counts after board changes.
 *
 * @param sessionId - Active backend session ID, or null when absent.
 * @param mode - PLO variant used when submitting player ranges.
 *
 * @remarks
 * Range operations are sent to the backend through the players
 * API client. The hook does not automatically fetch player state
 * from the backend or reset its state when sessionId changes.
 */
export function usePlayerRanges(sessionId: string | null, mode: Mode) {
  const [players, setPlayers] = useState<PlayerState[]>(
    Array.from({ length: 6 }, () => ({
      range: '',
      numberOfHands: 0,
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
   * On success, updates the reported hand count and marks the
   * range as applied.
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

      updatePlayer(playerId, { numberOfHands: hands.hands, hasAppliedRange: true });
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

      updatePlayer(playerId, { range: '', numberOfHands: 0, hasAppliedRange: false });
    } catch (err) {
      updatePlayer(playerId, { error: err instanceof Error ? err.message : 'Unknown error' });
    } finally {
      updatePlayer(playerId, { isLoading: false });
    }
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

  return {
    players,
    updateHandCounts,
    updatePlayer,
    handleAddRange,
    handleDeleteRange,
  };
}
