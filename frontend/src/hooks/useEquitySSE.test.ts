import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import {
  cancelEquityCalculation,
  startEquityStream,
  type EquityStreamEvent,
  type JobStatus,
} from '../api/equity';
import { useEquitySSE } from './useEquitySSE';

// The API module tests EventSource and HTTP. Here we exercise only hook behavior.
vi.mock('../api/equity', () => ({
  startEquityStream: vi.fn(),
  cancelEquityCalculation: vi.fn(),
}));

class FakeSource {
  readyState = 0;
  close = vi.fn(() => {
    this.readyState = 2;
  });

  private readonly listeners = new Map<string, Array<() => void>>();

  addEventListener(type: string, listener: () => void) {
    const callbacks = this.listeners.get(type) ?? [];
    callbacks.push(listener);
    this.listeners.set(type, callbacks);
  }

  emitOpen() {
    this.readyState = 1;
    this.listeners.get('open')?.forEach((listener) => listener());
  }
}

type OnUpdate = Parameters<typeof startEquityStream>[2];
type OnStreamError = Parameters<typeof startEquityStream>[3];

interface MockJob {
  jobId: string;
  source: FakeSource;
  onUpdate: OnUpdate;
  onStreamError: OnStreamError;
}

let jobs: MockJob[];
let nextJobId: number;

function setupStream() {
  vi.mocked(startEquityStream).mockImplementation(
    async (_sessionId, _moe, onUpdate, onStreamError) => {
      const job: MockJob = {
        jobId: `job-${++nextJobId}`,
        source: new FakeSource(),
        onUpdate,
        onStreamError,
      };
      jobs.push(job);
      return { jobId: job.jobId, source: job.source as unknown as EventSource };
    },
  );
}

function makeEvent(
  status: JobStatus,
  overrides: Partial<EquityStreamEvent> = {},
): EquityStreamEvent {
  return {
    job_id: 'job-1',
    status,
    error: null,
    details: null,
    ...overrides,
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.stubGlobal('EventSource', { OPEN: 1 });
  jobs = [];
  nextJobId = 0;
  setupStream();
  vi.mocked(cancelEquityCalculation).mockResolvedValue({
    status: 'cancelling',
    job_id: 'job-1',
  });
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

test('does not submit or cancel when no session is available', async () => {
  const { result } = renderHook(() => useEquitySSE(null));

  await act(async () => {
    await result.current.startEquity(0.0015);
    await result.current.cancelEquity();
  });

  expect(result.current.equityStatus).toBe('idle');
  expect(startEquityStream).not.toHaveBeenCalled();
  expect(cancelEquityCalculation).not.toHaveBeenCalled();
});

test('starts the job, receives progress, and releases it on completion', async () => {
  const { result } = renderHook(() => useEquitySSE('session-1'));

  await act(async () => {
    await result.current.startEquity(0.0015);
  });

  expect(startEquityStream).toHaveBeenCalledOnce();
  expect(startEquityStream).toHaveBeenCalledWith(
    'session-1',
    0.0015,
    expect.any(Function),
    expect.any(Function),
  );
  expect(result.current.equityStatus).toBe('starting');

  const first = jobs[0];
  act(() => first.source.emitOpen());
  expect(result.current.equityStatus).toBe('running');

  const progress = makeEvent('running', {
    details: {
      equities: { '0': 0.6, '2': 0.4 },
      method: 'monte-carlo',
      margins_of_error: { '0': 0.003, '2': 0.003 },
      simulations: 1000,
      attempts: 1100,
      stop_reason: null,
      sampler: 'rejection',
      effective_samples: null,
    },
  });
  act(() => first.onUpdate(progress));
  expect(result.current.equityStatus).toBe('running');
  expect(result.current.equityMessage).toEqual(progress);

  const completed = makeEvent('completed', { details: progress.details });
  act(() => first.onUpdate(completed));
  expect(result.current.equityStatus).toBe('completed');
  expect(result.current.equityMessage).toEqual(completed);
  expect(result.current.equityError).toBeNull();

  // The terminal event releases the job, allowing another submission.
  await act(async () => {
    await result.current.startEquity(0.002);
  });
  expect(startEquityStream).toHaveBeenCalledTimes(2);
});

test('reports submission failures and permits another attempt', async () => {
  vi.mocked(startEquityStream).mockRejectedValueOnce(new Error('Submission rejected'));
  const { result } = renderHook(() => useEquitySSE('session-1'));

  await act(async () => {
    await result.current.startEquity(0.0015);
  });

  expect(result.current.equityStatus).toBe('failed');
  expect(result.current.equityError).toBe('Submission rejected');
  expect(jobs).toHaveLength(0);

  await act(async () => {
    await result.current.startEquity(0.0015);
  });

  expect(startEquityStream).toHaveBeenCalledTimes(2);
  expect(result.current.equityStatus).toBe('starting');
  expect(result.current.equityError).toBeNull();
});

test('prevents duplicate submissions while POST is pending and while a job is active', async () => {
  const pending = deferred<{ jobId: string; source: EventSource }>();
  const source = new FakeSource();
  vi.mocked(startEquityStream).mockReturnValue(pending.promise);
  const { result } = renderHook(() => useEquitySSE('session-1'));

  act(() => {
    void result.current.startEquity(0.0015);
    void result.current.startEquity(0.002);
  });

  expect(result.current.equityStatus).toBe('starting');
  expect(startEquityStream).toHaveBeenCalledOnce();

  await act(async () => {
    pending.resolve({ jobId: 'job-1', source: source as unknown as EventSource });
    await pending.promise;
  });

  await act(async () => {
    await result.current.startEquity(0.002);
  });
  expect(startEquityStream).toHaveBeenCalledOnce();
});

test.each([
  { status: 'cancelled' as const, error: null },
  { status: 'failed' as const, error: 'Sampling exhausted' },
])(
  'handles backend $status events and makes the job eligible for retry',
  async ({ status, error }) => {
    const { result } = renderHook(() => useEquitySSE('session-1'));
    await act(async () => {
      await result.current.startEquity(0.0015);
    });

    act(() => jobs[0].onUpdate(makeEvent(status, { error })));

    expect(result.current.equityStatus).toBe(status);
    expect(result.current.equityError).toBe(error);
    expect(cancelEquityCalculation).not.toHaveBeenCalled();

    await act(async () => {
      await result.current.startEquity(0.0015);
    });
    expect(startEquityStream).toHaveBeenCalledTimes(2);
  },
);

test('requests cancellation once and waits for the terminal SSE event', async () => {
  const pendingCancel = deferred<{ status: 'cancelling'; job_id: string }>();
  vi.mocked(cancelEquityCalculation).mockReturnValue(pendingCancel.promise);
  const { result } = renderHook(() => useEquitySSE('session-1'));
  await act(async () => {
    await result.current.startEquity(0.0015);
  });
  act(() => jobs[0].source.emitOpen());

  act(() => {
    void result.current.cancelEquity();
    void result.current.cancelEquity();
  });

  expect(cancelEquityCalculation).toHaveBeenCalledExactlyOnceWith('session-1', 'job-1');
  expect(result.current.equityStatus).toBe('running');
  expect(jobs[0].source.close).not.toHaveBeenCalled();

  await act(async () => {
    pendingCancel.resolve({ status: 'cancelling', job_id: 'job-1' });
    await pendingCancel.promise;
  });
  expect(result.current.equityStatus).toBe('cancelling');

  // Progress must not overwrite a successfully requested cancellation.
  act(() => jobs[0].onUpdate(makeEvent('running')));
  expect(result.current.equityStatus).toBe('cancelling');

  act(() => jobs[0].onUpdate(makeEvent('cancelled')));
  expect(result.current.equityStatus).toBe('cancelled');
  expect(result.current.equityError).toBeNull();
});

test('shows cancellation HTTP errors without losing the still-running calculation', async () => {
  vi.mocked(cancelEquityCalculation).mockRejectedValue(new Error('Cancellation rejected'));
  const { result } = renderHook(() => useEquitySSE('session-1'));
  await act(async () => {
    await result.current.startEquity(0.0015);
  });
  act(() => jobs[0].source.emitOpen());

  await act(async () => {
    await result.current.cancelEquity();
  });

  expect(result.current.equityStatus).toBe('running');
  expect(result.current.equityError).toBe('Cancellation rejected');

  act(() => jobs[0].onUpdate(makeEvent('completed')));
  expect(result.current.equityStatus).toBe('completed');
  expect(result.current.equityError).toBeNull();
});

test('ignores late cancellation responses after the job finishes', async () => {
  const pendingCancel = deferred<{ status: 'cancelling'; job_id: string }>();
  vi.mocked(cancelEquityCalculation).mockReturnValue(pendingCancel.promise);
  const { result } = renderHook(() => useEquitySSE('session-1'));
  await act(async () => {
    await result.current.startEquity(0.0015);
  });

  act(() => {
    void result.current.cancelEquity();
    jobs[0].onUpdate(makeEvent('completed'));
  });

  await act(async () => {
    pendingCancel.resolve({ status: 'cancelling', job_id: 'job-1' });
    await pendingCancel.promise;
  });

  expect(result.current.equityStatus).toBe('completed');
  expect(result.current.equityError).toBeNull();
});

test('cancels an unusable stream once, then clears the job for a new attempt', async () => {
  const pendingCancel = deferred<{ status: 'cancelling'; job_id: string }>();
  vi.mocked(cancelEquityCalculation).mockReturnValueOnce(pendingCancel.promise);
  const { result } = renderHook(() => useEquitySSE('session-1'));
  await act(async () => {
    await result.current.startEquity(0.0015);
  });
  const first = jobs[0];

  act(() => {
    first.onStreamError(
      new Error('Invalid equity SSE message received'),
      first.jobId,
      first.source as unknown as EventSource,
    );
    // Two reports for the same broken connection must not request cancellation twice.
    first.onStreamError(
      new Error('Another stream error'),
      first.jobId,
      first.source as unknown as EventSource,
    );
  });

  expect(first.source.close).toHaveBeenCalledOnce();
  expect(cancelEquityCalculation).toHaveBeenCalledExactlyOnceWith('session-1', 'job-1');
  expect(result.current.equityStatus).toBe('cancelling');

  await act(async () => {
    await result.current.startEquity(0.002);
  });
  expect(startEquityStream).toHaveBeenCalledOnce();

  await act(async () => {
    pendingCancel.resolve({ status: 'cancelling', job_id: 'job-1' });
    await pendingCancel.promise;
  });

  expect(result.current.equityStatus).toBe('failed');
  expect(result.current.equityError).toBe(
    'Invalid equity SSE message received. Cancellation requested.',
  );

  await act(async () => {
    await result.current.startEquity(0.002);
  });
  expect(startEquityStream).toHaveBeenCalledTimes(2);
});

test('releases a failed stream even when the backend cancellation request fails', async () => {
  const logError = vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.mocked(cancelEquityCalculation).mockRejectedValue(new Error('Network unavailable'));
  const { result } = renderHook(() => useEquitySSE('session-1'));
  await act(async () => {
    await result.current.startEquity(0.0015);
  });
  const first = jobs[0];

  act(() => {
    first.onStreamError(
      new Error('Connection closed'),
      first.jobId,
      first.source as unknown as EventSource,
    );
  });

  await waitFor(() => expect(result.current.equityStatus).toBe('failed'));
  expect(logError).toHaveBeenCalledOnce();
  expect(result.current.equityError).toBe('Connection closed. Could not request cancellation.');

  await act(async () => {
    await result.current.startEquity(0.002);
  });
  expect(startEquityStream).toHaveBeenCalledTimes(2);
});

test('ignores obsolete submissions and closes the stream after a session change', async () => {
  const pendingStart = deferred<{ jobId: string; source: EventSource }>();
  vi.mocked(startEquityStream).mockReturnValueOnce(pendingStart.promise);
  const { result, rerender } = renderHook(
    ({ sessionId }: { sessionId: string }) => useEquitySSE(sessionId),
    { initialProps: { sessionId: 'session-1' } },
  );

  act(() => {
    void result.current.startEquity(0.0015);
  });
  expect(result.current.equityStatus).toBe('starting');

  rerender({ sessionId: 'session-2' });
  expect(result.current.equityStatus).toBe('idle');

  const oldSource = new FakeSource();
  await act(async () => {
    pendingStart.resolve({ jobId: 'old-job', source: oldSource as unknown as EventSource });
    await pendingStart.promise;
  });

  expect(oldSource.close).toHaveBeenCalledOnce();
  expect(result.current.equityStatus).toBe('idle');
  expect(result.current.equityMessage).toBeNull();

  await act(async () => {
    await result.current.startEquity(0.002);
  });
  expect(startEquityStream).toHaveBeenLastCalledWith(
    'session-2',
    0.002,
    expect.any(Function),
    expect.any(Function),
  );
});

test('closes the active stream and ignores old updates after a session change', async () => {
  const { result, rerender } = renderHook(
    ({ sessionId }: { sessionId: string }) => useEquitySSE(sessionId),
    { initialProps: { sessionId: 'session-1' } },
  );
  await act(async () => {
    await result.current.startEquity(0.0015);
  });
  const oldJob = jobs[0];
  act(() => oldJob.onUpdate(makeEvent('running')));

  rerender({ sessionId: 'session-2' });
  expect(oldJob.source.close).toHaveBeenCalledOnce();
  expect(result.current.equityStatus).toBe('idle');
  expect(result.current.equityMessage).toBeNull();

  act(() => oldJob.onUpdate(makeEvent('completed')));
  expect(result.current.equityStatus).toBe('idle');
});

test('ignores stale cancellation responses after changing sessions', async () => {
  const pendingCancel = deferred<{ status: 'cancelling'; job_id: string }>();
  vi.mocked(cancelEquityCalculation).mockReturnValue(pendingCancel.promise);
  const { result, rerender } = renderHook(
    ({ sessionId }: { sessionId: string }) => useEquitySSE(sessionId),
    { initialProps: { sessionId: 'session-1' } },
  );
  await act(async () => {
    await result.current.startEquity(0.0015);
  });

  act(() => {
    void result.current.cancelEquity();
  });
  rerender({ sessionId: 'session-2' });

  await act(async () => {
    pendingCancel.resolve({ status: 'cancelling', job_id: 'job-1' });
    await pendingCancel.promise;
  });

  expect(result.current.equityStatus).toBe('idle');
  expect(result.current.equityError).toBeNull();
});

test('closes the active SSE connection when unmounted', async () => {
  const { result, unmount } = renderHook(() => useEquitySSE('session-1'));
  await act(async () => {
    await result.current.startEquity(0.0015);
  });
  const job = jobs[0];

  unmount();

  expect(job.source.close).toHaveBeenCalledOnce();
  act(() => job.onUpdate(makeEvent('completed')));
  expect(cancelEquityCalculation).not.toHaveBeenCalled();
});
