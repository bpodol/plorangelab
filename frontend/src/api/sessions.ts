import { apiRequest } from './client.ts';

export interface SessionResponse {
  session_id: string;
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
