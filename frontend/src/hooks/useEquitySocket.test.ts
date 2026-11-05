import { act, renderHook } from '@testing-library/react';
import { beforeEach, expect, test, vi } from 'vitest';
import { connectEquityWebSocket, type EquityMessage } from '../api/equity';
import { useEquitySocket } from './useEquitySocket';

vi.mock('../api/equity', () => ({
  connectEquityWebSocket: vi.fn(),
}));

beforeEach(() => {
  vi.resetAllMocks();
});

class FakeWebSocket {
  readyState: number = WebSocket.CONNECTING;

  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  onerror: (() => void) | null = null;

  send = vi.fn();
  close = vi.fn();

  open() {
    this.readyState = WebSocket.OPEN;
    this.onopen?.();
  }

  emitMessage(message: EquityMessage) {
    this.onmessage?.(
      new MessageEvent('message', {
        data: JSON.stringify(message),
      }),
    );
  }

  disconnect() {
    this.readyState = WebSocket.CLOSED;
    this.onclose?.();
  }
}

function setupSocket() {
  const socket = new FakeWebSocket();

  vi.mocked(connectEquityWebSocket).mockReturnValue(socket as unknown as WebSocket);

  return socket;
}

test('connects when a session exists', () => {
  const socket = setupSocket();

  const { result } = renderHook(() => useEquitySocket('session-123'));

  expect(connectEquityWebSocket).toHaveBeenCalledWith('session-123');

  expect(result.current.isConnected).toBe(false);

  act(() => {
    socket.open();
  });

  expect(result.current.isConnected).toBe(true);

  act(() => {
    socket.disconnect();
  });

  expect(result.current.isConnected).toBe(false);
});

test('does not connect without a session', () => {
  const { result } = renderHook(() => useEquitySocket(null));

  expect(connectEquityWebSocket).not.toHaveBeenCalled();
  expect(result.current.isConnected).toBe(false);
  expect(result.current.equityStatus).toBe('idle');
});

test('sends a start command only when connected', () => {
  const socket = setupSocket();

  const { result } = renderHook(() => useEquitySocket('session-123'));

  act(() => {
    result.current.startEquity(4, 0.0015);
  });

  expect(socket.send).not.toHaveBeenCalled();

  act(() => {
    socket.open();
  });

  act(() => {
    result.current.startEquity(5, 0.0015);
  });

  expect(socket.send).toHaveBeenCalledOnce();

  expect(socket.send).toHaveBeenCalledWith(
    JSON.stringify({
      type: 'start',
      mode: 5,
      target_moe: 0.0015,
    }),
  );
});

test('updates state when receiving equity messages', () => {
  const socket = setupSocket();

  const { result } = renderHook(() => useEquitySocket('session-123'));

  const started: EquityMessage = {
    type: 'started',
    job_id: 'job-1',
  };

  act(() => {
    socket.emitMessage(started);
  });

  expect(result.current.equityStatus).toBe('running');
  expect(result.current.equityMessage).toEqual(started);

  const progress: EquityMessage = {
    type: 'progress',
    job_id: 'job-1',
    simulations: 1000,
    margins_of_error: [0.01, 0.02],
  };

  act(() => {
    socket.emitMessage(progress);
  });

  expect(result.current.equityStatus).toBe('running');
  expect(result.current.equityMessage).toEqual(progress);

  const completed: EquityMessage = {
    type: 'result',
    job_id: 'job-1',
    equities: {
      '0': 0.65,
      '1': 0.35,
    },
    details: {
      method: 'monte-carlo',
      margins_of_error: [0.002, 0.002],
      simulations: 10000,
      attempts: 11000,
      stop_reason: 'target_reached',
      sampler: 'rejection',
      effective_samples: null,
    },
  };

  act(() => {
    socket.emitMessage(completed);
  });

  expect(result.current.equityStatus).toBe('completed');
  expect(result.current.equityMessage).toEqual(completed);
});

test('closes the WebSocket when the hook unmounts', () => {
  const socket = setupSocket();

  const { unmount } = renderHook(() => useEquitySocket('session-123'));

  unmount();

  expect(socket.close).toHaveBeenCalledOnce();
});

test('resets connection and equity state when the session changes', () => {
  const firstSocket = new FakeWebSocket();
  const secondSocket = new FakeWebSocket();
  interface HookProps {
    sessionId: string | null;
  }
  const initialProps: HookProps = { sessionId: 'sessionId' };

  vi.mocked(connectEquityWebSocket)
    .mockReturnValueOnce(firstSocket as unknown as WebSocket)
    .mockReturnValueOnce(secondSocket as unknown as WebSocket);

  const { result, rerender } = renderHook(
    ({ sessionId }: HookProps) => useEquitySocket(sessionId),
    { initialProps },
  );

  // First session connects and starts a calculation.
  act(() => {
    firstSocket.open();
    firstSocket.emitMessage({
      type: 'started',
      job_id: 'job-1',
    });
  });

  expect(result.current.isConnected).toBe(true);
  expect(result.current.equityStatus).toBe('running');

  // Switch to another session.
  rerender({ sessionId: 'session-2' });

  expect(firstSocket.close).toHaveBeenCalledOnce();
  expect(connectEquityWebSocket).toHaveBeenCalledWith('session-2');

  // The second socket has not connected yet.
  expect(result.current.isConnected).toBe(false);
  expect(result.current.equityStatus).toBe('idle');
  expect(result.current.equityMessage).toBeNull();

  // Now the second connection opens.
  act(() => {
    secondSocket.open();
  });

  act(() => {
    firstSocket.emitMessage({
      type: 'started',
      job_id: 'old-job',
    });
  });
  expect(result.current.equityStatus).toBe('idle');
  expect(result.current.equityMessage).toBeNull();
  expect(result.current.isConnected).toBe(true);
  expect(result.current.isConnected).toBe(true);

  rerender({ sessionId: null });

  expect(secondSocket.close).toHaveBeenCalledOnce();
  expect(result.current.isConnected).toBe(false);
  expect(result.current.equityStatus).toBe('idle');
});
