import { addRange, deleteRange } from './players';
import { beforeEach, expect, test, vi } from 'vitest';
import { apiRequest } from './client';
import type { PlayerFilterSnapshot } from '../types';

// Replace the real apiRequest with a mock.
vi.mock('./client', () => ({
  apiRequest: vi.fn(),
}));

beforeEach(() => {
  vi.resetAllMocks();
});

test('adds a player range using PUT and returns the full snapshot', async () => {
  const snapshot: PlayerFilterSnapshot = {
    base_hands: 1,
    filtered_hands: 1,
    filtered_coverage: 1,
    filters: {
      auto_ranks: {
        type: 'group',
        name: 'Auto-ranked hands',
        coverage: 1,
        active: false,
        filters: {
          aces: {
            type: 'filter',
            name: 'Aces',
            expression: 'AA',
            coverage: 1,
            active: false,
          },
        },
      },
    },
  };
  vi.mocked(apiRequest).mockResolvedValue(snapshot);

  const result = await addRange('session-123', 2, 'AsAhQcJd');

  expect(apiRequest).toHaveBeenCalledWith('/api/sessions/session-123/players/2', {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      hands: 'AsAhQcJd',
    }),
  });

  expect(result).toBe(snapshot);
});

test('deletes a player range using DELETE', async () => {
  vi.mocked(apiRequest).mockResolvedValue(undefined);

  const result = await deleteRange('session-123', 2);

  expect(apiRequest).toHaveBeenCalledWith('/api/sessions/session-123/players/2', {
    method: 'DELETE',
  });

  expect(result).toBeUndefined();
});

test.each([
  ['addRange', () => addRange('session-123', 0, 'AsAhQcJd')],
  ['deleteRange', () => deleteRange('session-123', 0)],
])('%s propagates API errors', async (_name, operation) => {
  const error = new Error('API request failed: 422 Unprocessable Entity');

  vi.mocked(apiRequest).mockRejectedValue(error);

  await expect(operation()).rejects.toBe(error);
});
