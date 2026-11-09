import { beforeEach, expect, test, vi } from 'vitest';
import { apiRequest } from './client';
import { createSession, changeBoard, addRange, deleteRange } from './sessions';

// Replace the real apiRequest with a mock.
vi.mock('./client', () => ({
  apiRequest: vi.fn(),
}));

beforeEach(() => {
  vi.resetAllMocks();
});

test('creates a session using POST', async () => {
  const response = { session_id: 'session-123' };
  vi.mocked(apiRequest).mockResolvedValue(response);

  const result = await createSession();

  expect(apiRequest).toHaveBeenCalledWith('/api/sessions', {
    method: 'POST',
  });

  expect(result).toEqual(response);
});

test.each(['AsKd2c', null] as const)('updates the board with value %s', async (board) => {
  const response = { players: [100, 50, 0, 0, 0, 0] };
  vi.mocked(apiRequest).mockResolvedValue(response);

  const result = await changeBoard('session-123', board);

  expect(apiRequest).toHaveBeenCalledWith('/api/sessions/session-123/board', {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ board }),
  });

  expect(result).toEqual(response);
});

test('adds a player range using PUT', async () => {
  vi.mocked(apiRequest).mockResolvedValue({ hands: 1 });

  const result = await addRange(2, 'AsAhQcJd', 4, 'session-123');

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
  ['createSession', () => createSession()],
  ['changeBoard', () => changeBoard('session-123', 'AsKd2c')],
  ['addRange', () => addRange(0, 'AsAhQcJd', 4, 'session-123')],
  ['deleteRange', () => deleteRange('session-123', 0)],
])('%s propagates API errors', async (_name, operation) => {
  const error = new Error('API request failed: 422 Unprocessable Entity');

  vi.mocked(apiRequest).mockRejectedValue(error);

  await expect(operation()).rejects.toBe(error);
});
