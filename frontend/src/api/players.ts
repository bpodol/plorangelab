import { apiRequest } from './client.ts';
import type { Mode } from '../types.ts';

export interface RangeResponse {
  hands: number;
}

/**
 * Create or replace the range assigned to a player slot.
 *
 * The backend parses the range, replaces the player's existing
 * state, and removes hands conflicting with the current board.
 * Replacing a range also discards its previous filter states.
 *
 * @param sessionId - Identifier of the target session.
 * @param playerId - Zero-based player slot index (0-5).
 * @param hands - Range expression accepted by the backend encoder.
 * @param mode - PLO variant (4 or 5).
 * @returns Number of hands in the resulting range.
 */
export function addRange(
  sessionId: string,
  playerId: number,
  hands: string,
  mode: Mode,
): Promise<RangeResponse> {
  return apiRequest<RangeResponse>(`/api/sessions/${sessionId}/players/${playerId}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      hands,
      mode,
    }),
  });
}

/**
 * Remove the range assigned to a player slot.
 *
 * Clears the player's range and associated street and filter
 * states without affecting other players or the shared board.
 * Removing an already empty slot is allowed.
 *
 * @param sessionId - Identifier of the target session.
 * @param playerId - Zero-based player slot index (0-5).
 * @returns A promise that resolves without a value on success.
 */
export function deleteRange(sessionId: string, playerId: number): Promise<void> {
  return apiRequest<void>(`/api/sessions/${sessionId}/players/${playerId}`, {
    method: 'DELETE',
  });
}
