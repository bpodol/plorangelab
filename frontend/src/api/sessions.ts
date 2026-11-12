import { apiRequest } from './client.ts';

export interface SessionResponse {
  session_id: string;
}

export interface BoardResponse {
  /** Current hand counts for all six player slots, in slot order. */
  players: number[];
}

/**
 * Create a new backend session with six empty player slots.
 *
 * @returns The generated session identifier.
 */
export function createSession(): Promise<SessionResponse> {
  return apiRequest<SessionResponse>('/api/sessions', {
    method: 'POST',
  });
}

/**
 * Update or clear the community board for a session.
 *
 * The backend recalculates the affected player ranges and
 * preserves compatible street states where possible.
 *
 * @param sessionId - Identifier of the target session.
 * @param board - Concatenated card notation, or null to clear the board.
 * @returns Current hand counts for all six player slots.
 */
export function changeBoard(sessionId: string, board: string | null): Promise<BoardResponse> {
  return apiRequest<BoardResponse>(`/api/sessions/${sessionId}/board`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ board }),
  });
}
