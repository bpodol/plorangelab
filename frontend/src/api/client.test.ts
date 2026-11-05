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
  [404, 'Not Found'],
  [422, 'Unprocessable Entity'],
])('throws an error for HTTP %i', async (status, statusText) => {
  const mockResponse = {
    ok: false,
    status,
    statusText,
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
