import { useState } from 'react';
import { addRange, deleteRange } from '../api/players.ts';
import type { Mode } from '../types.ts';

export interface PlayerState {
  range: string;
  numberOfHands: number;
  hasAppliedRange: boolean;
  isLoading: boolean;
  error: string | null;
}

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

  async function handleAddRange(playerId: number) {
    if (sessionId === null) return;

    const player = players[playerId];

    updatePlayer(playerId, { isLoading: true, error: null });

    try {
      const hands = await addRange(playerId, player.range, mode, sessionId);

      updatePlayer(playerId, { numberOfHands: hands.hands, hasAppliedRange: true });
    } catch (err) {
      updatePlayer(playerId, { error: err instanceof Error ? err.message : 'Unknown error' });
    } finally {
      updatePlayer(playerId, { isLoading: false });
    }
  }

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
