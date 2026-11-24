import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { apiRequest } from './client';
import {
  cancelEquityCalculation,
  startEquityStream,
  submitEquityCalculation,
  type EquityStreamEvent,
} from './equity';

vi.mock('./client', () => ({ apiRequest: vi.fn() }));

class FakeEventSource {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSED = 2;

  static instances: FakeEventSource[] = [];

  readonly url: string;
  readyState = FakeEventSource.CONNECTING;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  onerror: (() => void) | null = null;

  close = vi.fn(() => {
    this.readyState = FakeEventSource.CLOSED;
  });

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  emitMessage(value: unknown) {
    this.emitRaw(JSON.stringify(value));
  }

  emitRaw(data: string) {
    this.onmessage?.(new MessageEvent('message', { data }));
  }

  emitError(state: number) {
    this.readyState = state;
    this.onerror?.();
  }
}

const progress: EquityStreamEvent = {
  job_id: 'job-123',
  status: 'running',
  error: null,
  details: {
    equities: { '0': 0.6, '3': 0.4 },
    method: 'monte-carlo',
    margins_of_error: { '0': 0.002, '3': null },
    simulations: 1000,
    attempts: 1200,
    stop_reason: null,
    sampler: 'rejection',
    effective_samples: null,
  },
};

beforeEach(() => {
  vi.resetAllMocks();
  FakeEventSource.instances = [];
  vi.stubGlobal('EventSource', FakeEventSource);
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

test('submits the target MOE using the backend snake_case request contract', async () => {
  const response = { status: 'starting' as const, job_id: 'job-123' };
  vi.mocked(apiRequest).mockResolvedValue(response);

  await expect(submitEquityCalculation('session-1', 0.0015)).resolves.toEqual(response);

  expect(apiRequest).toHaveBeenCalledExactlyOnceWith('/api/sessions/session-1/equity', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ target_moe: 0.0015 }),
  });
});

test('requests cooperative cancellation through a separate HTTP POST', async () => {
  const response = { status: 'cancelling' as const, job_id: 'job-123' };
  vi.mocked(apiRequest).mockResolvedValue(response);

  await expect(cancelEquityCalculation('session-1', 'job-123')).resolves.toEqual(response);

  expect(apiRequest).toHaveBeenCalledExactlyOnceWith(
    '/api/sessions/session-1/equity/job-123/cancel',
    { method: 'POST' },
  );
});

test('opens the job-specific SSE stream only after submission succeeds', async () => {
  let resolveSubmit!: (value: { status: 'starting'; job_id: string }) => void;
  vi.mocked(apiRequest).mockReturnValue(
    new Promise((resolve) => {
      resolveSubmit = resolve;
    }),
  );

  const onUpdate = vi.fn();
  const onStreamError = vi.fn();
  const starting = startEquityStream('session-1', 0.0015, onUpdate, onStreamError);

  expect(FakeEventSource.instances).toHaveLength(0);

  resolveSubmit({ status: 'starting', job_id: 'job-123' });
  const { jobId, source } = await starting;

  expect(jobId).toBe('job-123');
  expect(source).toBe(FakeEventSource.instances[0]);
  expect(FakeEventSource.instances[0].url).toBe('/api/sessions/session-1/equity/job-123/events');
  expect(onUpdate).not.toHaveBeenCalled();
});

test('does not open a stream if the submission request fails', async () => {
  vi.mocked(apiRequest).mockRejectedValue(new Error('Submission rejected'));

  await expect(startEquityStream('session-1', 0.0015, vi.fn(), vi.fn())).rejects.toThrow(
    'Submission rejected',
  );
  expect(FakeEventSource.instances).toHaveLength(0);
});

test('forwards parsed progress snapshots without closing the stream', async () => {
  vi.mocked(apiRequest).mockResolvedValue({ status: 'starting', job_id: 'job-123' });
  const onUpdate = vi.fn();
  const onStreamError = vi.fn();
  await startEquityStream('session-1', 0.0015, onUpdate, onStreamError);
  const source = FakeEventSource.instances[0];

  source.emitMessage(progress);

  expect(onUpdate).toHaveBeenCalledExactlyOnceWith(progress);
  expect(source.close).not.toHaveBeenCalled();
  expect(onStreamError).not.toHaveBeenCalled();
});

test.each(['completed', 'cancelled', 'failed'] as const)(
  'closes the stream after the %s terminal event',
  async (status) => {
    vi.mocked(apiRequest).mockResolvedValue({ status: 'starting', job_id: 'job-123' });
    const onUpdate = vi.fn();
    const onStreamError = vi.fn();
    await startEquityStream('session-1', 0.0015, onUpdate, onStreamError);
    const source = FakeEventSource.instances[0];
    const terminal: EquityStreamEvent = {
      job_id: 'job-123',
      status,
      error: status === 'failed' ? 'Calculation failed' : null,
      details: status === 'completed' ? progress.details : null,
    };

    source.emitMessage(terminal);

    expect(onUpdate).toHaveBeenCalledExactlyOnceWith(terminal);
    expect(source.close).toHaveBeenCalledOnce();
    expect(onStreamError).not.toHaveBeenCalled();
  },
);

test('closes the stream and reports malformed SSE JSON to its consumer', async () => {
  vi.mocked(apiRequest).mockResolvedValue({ status: 'starting', job_id: 'job-123' });
  const onUpdate = vi.fn();
  const onStreamError = vi.fn();
  await startEquityStream('session-1', 0.0015, onUpdate, onStreamError);
  const source = FakeEventSource.instances[0];

  source.emitRaw('{invalid-json');

  expect(onUpdate).not.toHaveBeenCalled();
  expect(source.close).toHaveBeenCalledOnce();
  expect(onStreamError).toHaveBeenCalledExactlyOnceWith(
    expect.objectContaining({ message: 'Invalid equity SSE message received' }),
    'job-123',
    source,
  );
});

test('allows EventSource to reconnect after a temporary connection failure', async () => {
  vi.mocked(apiRequest).mockResolvedValue({ status: 'starting', job_id: 'job-123' });
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  const onStreamError = vi.fn();
  await startEquityStream('session-1', 0.0015, vi.fn(), onStreamError);
  const source = FakeEventSource.instances[0];

  source.emitError(FakeEventSource.CONNECTING);

  expect(onStreamError).not.toHaveBeenCalled();
  expect(source.close).not.toHaveBeenCalled();
});

test('reports permanently closed SSE connections for cancellation and cleanup', async () => {
  vi.mocked(apiRequest).mockResolvedValue({ status: 'starting', job_id: 'job-123' });
  const onStreamError = vi.fn();
  await startEquityStream('session-1', 0.0015, vi.fn(), onStreamError);
  const source = FakeEventSource.instances[0];

  source.emitError(FakeEventSource.CLOSED);

  expect(onStreamError).toHaveBeenCalledExactlyOnceWith(
    expect.objectContaining({ message: 'Equity SSE connection closed unexpectedly' }),
    'job-123',
    source,
  );
});
