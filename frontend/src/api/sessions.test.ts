import { beforeEach, expect, test, vi } from 'vitest';
import { apiRequest } from './client';
import { createSession, changeBoard } from './sessions';

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

test.each([
  ['createSession', () => createSession()],
  ['changeBoard', () => changeBoard('session-123', 'AsKd2c')],
])('%s propagates API errors', async (_name, operation) => {
  const error = new Error('API request failed: 422 Unprocessable Entity');

  vi.mocked(apiRequest).mockRejectedValue(error);

  await expect(operation()).rejects.toBe(error);
});
