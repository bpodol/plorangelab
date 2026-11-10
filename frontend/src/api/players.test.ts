import { addRange, deleteRange } from './players';
import { beforeEach, expect, test, vi } from 'vitest';
import { apiRequest } from './client';

// Replace the real apiRequest with a mock.
vi.mock('./client', () => ({
  apiRequest: vi.fn(),
}));

beforeEach(() => {
  vi.resetAllMocks();
});

test('adds a player range using PUT', async () => {
  vi.mocked(apiRequest).mockResolvedValue({ hands: 1 });

  const result = await addRange('session-123', 2, 'AsAhQcJd', 4);

  expect(apiRequest).toHaveBeenCalledWith('/api/sessions/session-123/players/2', {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      hands: 'AsAhQcJd',
      mode: 4,
    }),
  });

  expect(result).toEqual({ hands: 1 });
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
  ['addRange', () => addRange('session-123', 0, 'AsAhQcJd', 4)],
  ['deleteRange', () => deleteRange('session-123', 0)],
])('%s propagates API errors', async (_name, operation) => {
  const error = new Error('API request failed: 422 Unprocessable Entity');

  vi.mocked(apiRequest).mockRejectedValue(error);

  await expect(operation()).rejects.toBe(error);
});
