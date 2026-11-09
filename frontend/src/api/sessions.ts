import { apiRequest } from './client.ts';

export interface SessionResponse {
  session_id: string;
}

export type Mode = 4 | 5;

export interface RangeResponse {
  hands: number;
}

export interface BoardResponse {
  players: number[];
}

export function createSession(): Promise<SessionResponse> {
  return apiRequest<SessionResponse>('/api/sessions', {
    method: 'POST',
  });
}

export function changeBoard(sessionId: string, board: string | null): Promise<BoardResponse> {
  return apiRequest<BoardResponse>(`/api/sessions/${sessionId}/board`, {
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
  sessionId: string,
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

export function deleteRange(sessionId: string, playerId: number): Promise<void> {
  return apiRequest<void>(`/api/sessions/${sessionId}/players/${playerId}`, {
    method: 'DELETE',
  });
}
