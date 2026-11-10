import { beforeEach, expect, test, vi } from 'vitest';
import { apiRequest } from './client';
import {
  getFilters,
  getFilter,
  createFilter,
  createGroup,
  setFilterActive,
  deleteFilter,
  clearFilters,
} from './filters';

vi.mock('./client', () => ({
  apiRequest: vi.fn(),
}));

beforeEach(() => {
  vi.resetAllMocks();
});

const base = '/api/sessions/session-123/players/2';

test('retrieves the complete filter tree', async () => {
  const tree = {
    Draws: {
      type: 'group',
      name: 'Draws',
      coverage: 0.5,
      active: true,
      filters: {},
    },
  };

  vi.mocked(apiRequest).mockResolvedValue(tree);

  expect(await getFilters('session-123', 2)).toEqual(tree);
  expect(apiRequest).toHaveBeenCalledWith(`${base}/filters`);
});

test('retrieves nested nodes with encoded paths', async () => {
  const node = {
    type: 'filter',
    name: 'Nut FD',
    expression: 'FD',
    coverage: 0.3,
    active: true,
  };

  vi.mocked(apiRequest).mockResolvedValue(node);

  const result = await getFilter('session-123', 2, 'filters', 'Nut FD', ['Flop', 'Strong draws']);

  expect(result).toEqual(node);
  expect(apiRequest).toHaveBeenCalledWith(`${base}/filters/Nut%20FD?path=Flop&path=Strong+draws`);
});

test('creates filters and groups', async () => {
  vi.mocked(apiRequest)
    .mockResolvedValueOnce({ coverage: 0.3 })
    .mockResolvedValueOnce({ name: 'Draws', path: [] });

  await createFilter('session-123', 2, {
    expression: 'FD',
    name: 'Flush Draw',
    path: ['Draws'],
  });

  expect(apiRequest).toHaveBeenNthCalledWith(1, `${base}/filters`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      expression: 'FD',
      name: 'Flush Draw',
      path: ['Draws'],
    }),
  });

  await createGroup('session-123', 2, { name: 'Draws' });

  expect(apiRequest).toHaveBeenNthCalledWith(2, `${base}/groups`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: 'Draws' }),
  });
});

test.each(['filters', 'groups'] as const)('activates and deletes %s', async (resource) => {
  vi.mocked(apiRequest).mockResolvedValue({ coverage: 0.4 });

  const url = `${base}/${resource}/FD?path=Draws`;

  await setFilterActive('session-123', 2, resource, 'FD', true, ['Draws']);

  expect(apiRequest).toHaveBeenNthCalledWith(1, url, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ active: true }),
  });

  await deleteFilter('session-123', 2, resource, 'FD', ['Draws']);

  expect(apiRequest).toHaveBeenNthCalledWith(2, url, {
    method: 'DELETE',
  });
});

test('clears filters', async () => {
  vi.mocked(apiRequest).mockResolvedValue({ coverage: 1 });

  const result = await clearFilters('session-123', 2);

  expect(apiRequest).toHaveBeenCalledWith(`${base}/filters/clear`, { method: 'POST' });

  expect(result).toEqual({ coverage: 1 });
});

test('propagates API errors', async () => {
  const error = new Error('API request failed');

  vi.mocked(apiRequest).mockRejectedValue(error);

  await expect(getFilters('session-123', 2)).rejects.toBe(error);
});
