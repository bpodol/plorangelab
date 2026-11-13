import { act, renderHook } from '@testing-library/react';
import { beforeEach, expect, test, vi } from 'vitest';
import { addRange, deleteRange } from '../api/players';
import { usePlayerRanges } from './usePlayerRanges';
import type { PlayerFilterSnapshot } from '../types';

vi.mock('../api/players', () => ({
  addRange: vi.fn(),
  deleteRange: vi.fn(),
}));

beforeEach(() => {
  vi.resetAllMocks();
});

test('initializes six empty players', () => {
  const { result } = renderHook(() => usePlayerRanges('test-session', 4));

  expect(result.current.players).toHaveLength(6);

  for (const player of result.current.players) {
    expect(player.range).toBe('');
    expect(player.numberOfHands).toBe(0);
    expect(player.filterSnapshot).toBe(null);
    expect(player.hasAppliedRange).toBe(false);
    expect(player.isLoading).toBe(false);
    expect(player.error).toBeNull();
  }
});

test('updates only the selected player', () => {
  const { result } = renderHook(() => usePlayerRanges('session-123', 4));

  act(() => {
    result.current.updatePlayer(2, { range: 'AsAhQcJd' });
  });

  expect(result.current.players[2].range).toBe('AsAhQcJd');

  for (let i = 0; i < 6; i++) {
    if (i !== 2) {
      expect(result.current.players[i].range).toBe('');
    }
  }
});

test('updates hand counts after a board change', () => {
  const { result } = renderHook(() => usePlayerRanges('session-123', 4));

  act(() => {
    result.current.updatePlayer(0, { hasAppliedRange: true });
    result.current.updateHandCounts([100, 0, 50, 0, 0, 0]);
  });

  expect(result.current.players.map((player) => player.numberOfHands)).toEqual([
    100, 0, 50, 0, 0, 0,
  ]);

  expect(result.current.players[0].hasAppliedRange).toBe(true);
});

test('adds a range and updates player state', async () => {
  vi.mocked(addRange).mockResolvedValue({ hands: 150 });

  const { result } = renderHook(() => usePlayerRanges('session-123', 4));

  act(() => {
    result.current.updatePlayer(2, { range: 'AsAhQcJd' });
  });

  await act(async () => {
    await result.current.handleAddRange(2);
  });

  expect(addRange).toHaveBeenCalledWith('session-123', 2, 'AsAhQcJd', 4);

  expect(result.current.players[2]).toMatchObject({
    range: 'AsAhQcJd',
    numberOfHands: 150,
    hasAppliedRange: true,
    isLoading: false,
    error: null,
  });

  expect(result.current.players[0].numberOfHands).toBe(0);
});

test('stores an error when adding a range fails', async () => {
  vi.mocked(addRange).mockRejectedValue(new Error('Invalid range'));

  const { result } = renderHook(() => usePlayerRanges('session-123', 4));

  act(() => {
    result.current.updatePlayer(1, { range: 'invalid' });
  });

  await act(async () => {
    await result.current.handleAddRange(1);
  });

  expect(result.current.players[1]).toMatchObject({
    range: 'invalid',
    numberOfHands: 0,
    hasAppliedRange: false,
    isLoading: false,
    error: 'Invalid range',
  });
});

test('deletes a range and resets player state', async () => {
  vi.mocked(deleteRange).mockResolvedValue(undefined);

  const { result } = renderHook(() => usePlayerRanges('session-123', 4));

  act(() => {
    result.current.updatePlayer(1, {
      range: 'AsAhQcJd',
      numberOfHands: 1,
      hasAppliedRange: true,
    });
  });

  await act(async () => {
    await result.current.handleDeleteRange(1);
  });

  expect(deleteRange).toHaveBeenCalledWith('session-123', 1);

  expect(result.current.players[1]).toMatchObject({
    range: '',
    numberOfHands: 0,
    filterSnapshot: null,
    hasAppliedRange: false,
    isLoading: false,
    error: null,
  });
});

test('preserves the range when deletion fails', async () => {
  vi.mocked(deleteRange).mockRejectedValue(new Error('API request failed: 409 Conflict'));

  const { result } = renderHook(() => usePlayerRanges('session-123', 4));

  act(() => {
    result.current.updatePlayer(1, {
      range: 'AsAhQcJd',
      numberOfHands: 1,
      hasAppliedRange: true,
    });
  });

  await act(async () => {
    await result.current.handleDeleteRange(1);
  });

  expect(result.current.players[1]).toMatchObject({
    range: 'AsAhQcJd',
    numberOfHands: 1,
    hasAppliedRange: true,
    isLoading: false,
    error: 'API request failed: 409 Conflict',
  });
});

test('sets isLoading while adding a range', async () => {
  let resolveRequest!: (value: { hands: number }) => void;

  const pendingRequest = new Promise<{ hands: number }>((resolve) => {
    resolveRequest = resolve;
  });

  vi.mocked(addRange).mockReturnValue(pendingRequest);

  const { result } = renderHook(() => usePlayerRanges('session-123', 4));

  // Start the request without waiting for it to finish.
  let operation!: Promise<void>;

  act(() => {
    operation = result.current.handleAddRange(2);
  });

  // The request is still pending.
  expect(result.current.players[2].isLoading).toBe(true);
  expect(result.current.players[0].isLoading).toBe(false);

  // Simulate the backend completing the request.
  await act(async () => {
    resolveRequest({ hands: 150 });
    await operation;
  });

  expect(result.current.players[2].isLoading).toBe(false);
  expect(result.current.players[2].numberOfHands).toBe(150);
});

test('applies board snapshots to the correct player slots', () => {
  const { result } = renderHook(() => usePlayerRanges('session-123', 4));

  const snapshot: PlayerFilterSnapshot = {
    base_hands: 100,
    filtered_hands: 25,
    filtered_coverage: 0.25,
    filters: {
      aces: {
        type: 'filter',
        name: 'aces',
        expression: 'AA',
        coverage: 0.25,
        active: true,
      },
    },
  };

  act(() => {
    result.current.applyBoardSnapshots([
      snapshot,
      null,
      {
        base_hands: 50,
        filtered_hands: 50,
        filtered_coverage: 1,
        filters: {},
      },
      null,
      null,
      null,
    ]);
  });

  expect(result.current.players[0]).toMatchObject({
    numberOfHands: 25,
    filterSnapshot: snapshot,
  });

  expect(result.current.players[1]).toMatchObject({
    numberOfHands: 0,
    filterSnapshot: null,
  });

  expect(result.current.players[2]).toMatchObject({
    numberOfHands: 50,
    filterSnapshot: {
      base_hands: 50,
      filtered_coverage: 1,
      filters: {},
    },
  });
});

test('replacing a range clears the previous filter snapshot', async () => {
  vi.mocked(addRange).mockResolvedValue({ hands: 3 });

  const { result } = renderHook(() => usePlayerRanges('session-123', 4));

  act(() => {
    result.current.updatePlayer(0, {
      range: 'AsAhKdQc',
      numberOfHands: 1,
      hasAppliedRange: true,
      filterSnapshot: {
        base_hands: 4,
        filtered_hands: 1,
        filtered_coverage: 0.25,
        filters: {},
      },
    });
  });

  await act(async () => {
    await result.current.handleAddRange(0);
  });

  expect(addRange).toHaveBeenCalledWith('session-123', 0, 'AsAhKdQc', 4);

  expect(result.current.players[0]).toMatchObject({
    numberOfHands: 3,
    hasAppliedRange: true,
    filterSnapshot: null,
    isLoading: false,
    error: null,
  });
});
