import { afterEach, expect, test, vi } from 'vitest';
import { apiRequest } from './client';

afterEach(() => {
  vi.unstubAllGlobals();
});

test('returns parsed JSON for a successful response', async () => {
  const mockResponse = {
    ok: true,
    status: 200,
    json: vi.fn().mockResolvedValue({ hands: 100 }),
  };

  const mockFetch = vi.fn().mockResolvedValue(mockResponse);

  vi.stubGlobal('fetch', mockFetch);

  const result = await apiRequest<{ hands: number }>('/api/test');

  expect(result).toEqual({ hands: 100 });
  expect(mockFetch).toHaveBeenCalledWith('/api/test', {});
  expect(mockResponse.json).toHaveBeenCalledOnce();
});

test.each([
  ['string detail', { detail: 'Board contains duplicate cards' }, 'Board contains duplicate cards'],
  [
    'validation errors',
    {
      detail: [{ msg: 'Field required' }, { msg: 'Invalid player ID' }],
    },
    'Field required; Invalid player ID',
  ],
  [
    'missing detail',
    { error: 'Something went wrong' },
    'API request failed: 422 Unprocessable Entity',
  ],
  ['empty validation errors', { detail: [] }, 'API request failed: 422 Unprocessable Entity'],
])('handles %s', async (_name, body, expectedError) => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      statusText: 'Unprocessable Entity',
      json: vi.fn().mockResolvedValue(body),
    }),
  );

  await expect(apiRequest('/api/test')).rejects.toThrow(expectedError);
});

test('falls back to HTTP status when error response is not valid JSON', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      json: vi.fn().mockRejectedValue(new SyntaxError('Invalid JSON')),
    }),
  );

  await expect(apiRequest('/api/test')).rejects.toThrow(
    'API request failed: 500 Internal Server Error',
  );
});

test.each([
  [404, 'Not Found'],
  [422, 'Unprocessable Entity'],
])('throws an error for HTTP %i without error details', async (status, statusText) => {
  const mockResponse = {
    ok: false,
    status,
    statusText,
    json: vi.fn().mockResolvedValue({}),
  };

  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(mockResponse));

  await expect(apiRequest('/api/test')).rejects.toThrow(
    `API request failed: ${status} ${statusText}`,
  );
});

test('returns undefined for 204 without parsing JSON', async () => {
  const mockResponse = {
    ok: true,
    status: 204,
    json: vi.fn(),
  };

  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(mockResponse));

  const result = await apiRequest<void>('/api/test', {
    method: 'DELETE',
  });

  expect(result).toBeUndefined();
  expect(mockResponse.json).not.toHaveBeenCalled();
});
