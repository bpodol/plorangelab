
import { apiRequest } from './client.ts';

export interface SessionResponse {
  session_id: string;
}

export type Mode = 4 | 5;

export interface RangeResponse {
  hands: number;
}

export function createSession(): Promise<SessionResponse> {
  return apiRequest<SessionResponse>('/api/sessions', {
    method: 'POST',
  });
}

export async function changeBoard(
  sessionId: string,
  board: string | null
): Promise<void> {
  await apiRequest<unknown>(`/api/sessions/${sessionId}/board`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ board }),
  });
}

export function addRange(
  playerId: number,
  hands: string,
  mode: Mode,
  sessionId: string
): Promise<RangeResponse> {
  return apiRequest<RangeResponse>(`/api/sessions/${sessionId}/range`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      player_id: playerId,
      hands,
      mode,
    }),
  });
}

export function deleteRange(
  sessionId: string,
  playerId: number
): Promise<void> {
  return apiRequest<void>(
    `/api/sessions/${sessionId}/range?player_id=${playerId}`,
    { method: 'DELETE' }
  );
}
