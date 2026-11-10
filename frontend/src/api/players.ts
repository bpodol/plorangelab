import { apiRequest } from './client.ts';
import type { Mode } from '../types.ts';

export interface RangeResponse {
  hands: number;
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
