import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, test, vi } from 'vitest';

import {
  clearFilters,
  createFilter,
  createGroup,
  deleteFilter,
  setFilterActive,
} from '../api/filters';
import type { FilterTree, PlayerFilterSnapshot } from '../types';
import { usePlayerRanges } from './usePlayerRanges';

vi.mock('../api/filters', () => ({
  clearFilters: vi.fn(),
  createFilter: vi.fn(),
  createGroup: vi.fn(),
  deleteFilter: vi.fn(),
  setFilterActive: vi.fn(),
}));

beforeEach(() => {
  vi.resetAllMocks();
});

const SESSION_ID = 'session-123';

function makeSnapshot(filteredHands: number, filters: FilterTree = {}): PlayerFilterSnapshot {
  return {
    base_hands: 100,
    filtered_hands: filteredHands,
    filtered_coverage: filteredHands / 100,
    filters,
  };
}

function makeNestedTree(): FilterTree {
  return {
    top_pair: {
      type: 'filter',
      name: 'top_pair',
      expression: 'A',
      coverage: 0.4,
      active: true,
    },
    draws: {
      type: 'group',
      name: 'draws',
      coverage: 0.2,
      active: true,
      filters: {
        nut_draws: {
          type: 'group',
          name: 'nut_draws',
          coverage: 0.15,
          active: true,
          filters: {
            nut_fd: {
              type: 'filter',
              name: 'nut_fd',
              expression: 'Ass',
              coverage: 0.15,
              active: false,
            },
          },
        },
      },
    },
  };
}

describe('usePlayerRanges filter operations', () => {
  test('addFilter sends its definition and replaces only the target player snapshot', async () => {
    const updated = makeSnapshot(35, makeNestedTree());
    vi.mocked(createFilter).mockResolvedValue(updated);

    const { result } = renderHook(() => usePlayerRanges(SESSION_ID));

    act(() => {
      result.current.updatePlayer(2, {
        range: 'AsAhQcJd',
        hasAppliedRange: true,
        numberOfHands: 90,
      });
    });

    await act(async () => {
      await result.current.addFilter(2, 'Ass', 'nut_fd', ['draws', 'nut_draws']);
    });

    expect(createFilter).toHaveBeenCalledWith(SESSION_ID, 2, {
      expression: 'Ass',
      name: 'nut_fd',
      path: ['draws', 'nut_draws'],
    });
    expect(result.current.players[2]).toMatchObject({
      range: 'AsAhQcJd',
      hasAppliedRange: true,
      numberOfHands: 35,
      filterSnapshot: updated,
      isLoading: false,
      error: null,
    });
    expect(result.current.players[1].filterSnapshot).toBeNull();
    expect(result.current.players[1].numberOfHands).toBe(0);
  });

  test('addGroup forwards the parent path and synchronizes the returned snapshot', async () => {
    const updated = makeSnapshot(60, makeNestedTree());
    vi.mocked(createGroup).mockResolvedValue(updated);

    const { result } = renderHook(() => usePlayerRanges(SESSION_ID));

    await act(async () => {
      await result.current.addGroup(0, 'nut_draws', ['draws']);
    });

    expect(createGroup).toHaveBeenCalledWith(SESSION_ID, 0, {
      name: 'nut_draws',
      path: ['draws'],
    });
    expect(result.current.players[0]).toMatchObject({
      filterSnapshot: updated,
      numberOfHands: 60,
      isLoading: false,
      error: null,
    });
  });

  test('toggleFilterActive activates a deeply nested inactive filter', async () => {
    const previous = makeSnapshot(40, makeNestedTree());
    const updated = makeSnapshot(55, makeNestedTree());
    vi.mocked(setFilterActive).mockResolvedValue(updated);

    const { result } = renderHook(() => usePlayerRanges(SESSION_ID));
    act(() => {
      result.current.updatePlayer(1, { filterSnapshot: previous, numberOfHands: 40 });
    });

    await act(async () => {
      await result.current.toggleFilterActive(1, 'filters', 'nut_fd', ['draws', 'nut_draws']);
    });

    expect(setFilterActive).toHaveBeenCalledWith(SESSION_ID, 1, 'filters', 'nut_fd', true, [
      'draws',
      'nut_draws',
    ]);
    expect(result.current.players[1].filterSnapshot).toBe(updated);
    expect(result.current.players[1].numberOfHands).toBe(55);
  });

  test('toggleFilterActive deactivates a nested active group', async () => {
    const previous = makeSnapshot(40, makeNestedTree());
    const updated = makeSnapshot(80, makeNestedTree());
    vi.mocked(setFilterActive).mockResolvedValue(updated);

    const { result } = renderHook(() => usePlayerRanges(SESSION_ID));
    act(() => {
      result.current.updatePlayer(1, { filterSnapshot: previous });
    });

    await act(async () => {
      await result.current.toggleFilterActive(1, 'groups', 'nut_draws', ['draws']);
    });

    expect(setFilterActive).toHaveBeenCalledWith(SESSION_ID, 1, 'groups', 'nut_draws', false, [
      'draws',
    ]);
    expect(result.current.players[1].filterSnapshot).toBe(updated);
  });

  test('toggleFilterActive handles a root-level filter without a path', async () => {
    const previous = makeSnapshot(40, makeNestedTree());
    vi.mocked(setFilterActive).mockResolvedValue(makeSnapshot(100));

    const { result } = renderHook(() => usePlayerRanges(SESSION_ID));
    act(() => {
      result.current.updatePlayer(0, { filterSnapshot: previous });
    });

    await act(async () => {
      await result.current.toggleFilterActive(0, 'filters', 'top_pair');
    });

    expect(setFilterActive).toHaveBeenCalledWith(
      SESSION_ID,
      0,
      'filters',
      'top_pair',
      false,
      undefined,
    );
  });

  test('toggleFilterActive does not request nonexistent nodes or invalid paths', async () => {
    const { result } = renderHook(() => usePlayerRanges(SESSION_ID));

    // No snapshot exists yet.
    await act(async () => {
      await result.current.toggleFilterActive(0, 'filters', 'top_pair');
    });

    act(() => {
      result.current.updatePlayer(0, {
        filterSnapshot: makeSnapshot(40, makeNestedTree()),
      });
    });

    await act(async () => {
      await result.current.toggleFilterActive(0, 'filters', 'missing');
      await result.current.toggleFilterActive(0, 'filters', 'nut_fd', ['missing']);
      // A filter is not a group, so it cannot be traversed.
      await result.current.toggleFilterActive(0, 'filters', 'nut_fd', ['top_pair']);
    });

    expect(setFilterActive).not.toHaveBeenCalled();
  });

  test.each([
    { resource: 'filters' as const, name: 'top_pair', path: undefined },
    { resource: 'groups' as const, name: 'nut_draws', path: ['draws'] },
  ])('removeFilter deletes the addressed $resource node', async ({ resource, name, path }) => {
    const updated = makeSnapshot(75, makeNestedTree());
    vi.mocked(deleteFilter).mockResolvedValue(updated);

    const { result } = renderHook(() => usePlayerRanges(SESSION_ID));

    await act(async () => {
      await result.current.removeFilter(3, resource, name, path);
    });

    expect(deleteFilter).toHaveBeenCalledWith(SESSION_ID, 3, resource, name, path);
    expect(result.current.players[3]).toMatchObject({
      filterSnapshot: updated,
      numberOfHands: 75,
      isLoading: false,
      error: null,
    });
  });

  test('clearAllFilters applies the backend snapshot without discarding filter definitions', async () => {
    const tree = makeNestedTree();
    // The backend clear operation deactivates nodes; it does not delete them.
    tree.top_pair.active = false;
    tree.draws.active = false;
    if (tree.draws.type === 'group') {
      tree.draws.filters.nut_draws.active = false;
    }

    const updated = makeSnapshot(100, tree);
    vi.mocked(clearFilters).mockResolvedValue(updated);

    const { result } = renderHook(() => usePlayerRanges(SESSION_ID));
    act(() => {
      result.current.updatePlayer(0, {
        numberOfHands: 40,
        filterSnapshot: makeSnapshot(40, makeNestedTree()),
      });
    });

    await act(async () => {
      await result.current.clearAllFilters(0);
    });

    expect(clearFilters).toHaveBeenCalledWith(SESSION_ID, 0);
    expect(result.current.players[0]).toMatchObject({
      filterSnapshot: updated,
      numberOfHands: 100,
      isLoading: false,
      error: null,
    });
    expect(result.current.players[0].filterSnapshot?.filters).toHaveProperty('top_pair');
    expect(result.current.players[0].filterSnapshot?.filters).toHaveProperty('draws');
    expect(result.current.players[0].filterSnapshot?.filters.top_pair.active).toBe(false);
  });

  test('failed filter mutations preserve the prior snapshot and store the error', async () => {
    const previous = makeSnapshot(30, makeNestedTree());
    vi.mocked(createFilter).mockRejectedValue(new Error('Invalid filter expression'));

    const { result } = renderHook(() => usePlayerRanges(SESSION_ID));
    act(() => {
      result.current.updatePlayer(4, {
        filterSnapshot: previous,
        numberOfHands: 30,
      });
    });

    await act(async () => {
      await result.current.addFilter(4, 'invalid');
    });

    expect(result.current.players[4]).toMatchObject({
      filterSnapshot: previous,
      numberOfHands: 30,
      isLoading: false,
      error: 'Invalid filter expression',
    });

    // A subsequent successful mutation clears the previous error.
    const updated = makeSnapshot(20);
    vi.mocked(createFilter).mockResolvedValue(updated);
    await act(async () => {
      await result.current.addFilter(4, 'A');
    });
    expect(result.current.players[4]).toMatchObject({
      filterSnapshot: updated,
      numberOfHands: 20,
      error: null,
    });
  });

  test('filter mutations are not sent when there is no session', async () => {
    const { result } = renderHook(() => usePlayerRanges(null));
    act(() => {
      result.current.updatePlayer(0, {
        filterSnapshot: makeSnapshot(30, makeNestedTree()),
      });
    });

    await act(async () => {
      await result.current.addFilter(0, 'A');
      await result.current.addGroup(0, 'draws');
      await result.current.toggleFilterActive(0, 'filters', 'top_pair');
      await result.current.removeFilter(0, 'filters', 'top_pair');
      await result.current.clearAllFilters(0);
    });

    expect(createFilter).not.toHaveBeenCalled();
    expect(createGroup).not.toHaveBeenCalled();
    expect(setFilterActive).not.toHaveBeenCalled();
    expect(deleteFilter).not.toHaveBeenCalled();
    expect(clearFilters).not.toHaveBeenCalled();
  });

  test('skips another filter mutation while the same player is loading', async () => {
    let resolveRequest!: (snapshot: PlayerFilterSnapshot) => void;
    const pendingRequest = new Promise<PlayerFilterSnapshot>((resolve) => {
      resolveRequest = resolve;
    });
    vi.mocked(createFilter).mockReturnValue(pendingRequest);

    const { result } = renderHook(() => usePlayerRanges(SESSION_ID));
    let operation!: Promise<boolean>;

    act(() => {
      operation = result.current.addFilter(2, 'A');
    });

    expect(result.current.players[2].isLoading).toBe(true);
    expect(result.current.players[1].isLoading).toBe(false);

    await act(async () => {
      await result.current.addGroup(2, 'blocked');
    });

    expect(createFilter).toHaveBeenCalledTimes(1);
    expect(createGroup).not.toHaveBeenCalled();

    await act(async () => {
      resolveRequest(makeSnapshot(45));
      await operation;
    });

    expect(result.current.players[2]).toMatchObject({
      numberOfHands: 45,
      isLoading: false,
      error: null,
    });
  });
});
