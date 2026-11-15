import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, expect, test, vi } from 'vitest';

import App from './App';
import { createSession, changeBoard } from './api/sessions';
import { addRange, deleteRange } from './api/players';
import { connectEquityWebSocket } from './api/equity';

vi.mock('./api/sessions', () => ({
  createSession: vi.fn(),
  changeBoard: vi.fn(),
}));

vi.mock('./api/players', () => ({
  addRange: vi.fn(),
  deleteRange: vi.fn(),
}));

vi.mock('./api/equity', () => ({
  connectEquityWebSocket: vi.fn(),
}));

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
}

let socket: FakeWebSocket;

beforeEach(() => {
  vi.resetAllMocks();

  socket = new FakeWebSocket();

  vi.mocked(connectEquityWebSocket).mockReturnValue(socket as unknown as WebSocket);
});

test('automatically creates a session on mount', async () => {
  let resolveSession!: (value: { session_id: string }) => void;

  vi.mocked(createSession).mockImplementation(
    () =>
      new Promise<{ session_id: string }>((resolve) => {
        resolveSession = resolve;
      }),
  );

  render(<App />);

  expect(screen.getByRole('heading', { name: 'PLO RangeLab' })).toBeInTheDocument();
  expect(screen.getAllByRole('textbox', { name: /^Player [1-6]$/ })).toHaveLength(6);
  expect(createSession).toHaveBeenCalledOnce();

  // Session-dependent actions must remain disabled until initialization finishes.
  expect(screen.getByRole('button', { name: /apply board/i })).toBeDisabled();
  expect(screen.getByRole('button', { name: /calculate equity/i })).toBeDisabled();
  expect(screen.queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument();

  await act(async () => {
    resolveSession({ session_id: 'session-123' });
  });

  await waitFor(() => {
    expect(connectEquityWebSocket).toHaveBeenCalledWith('session-123');
  });

  expect(screen.getByRole('button', { name: /apply board/i })).toBeEnabled();
  expect(screen.queryByText(/Session:/i)).not.toBeInTheDocument();
});

test('starts an equity calculation after automatic session creation', async () => {
  vi.mocked(createSession).mockResolvedValue({
    session_id: 'session-123',
  });

  const user = userEvent.setup();

  render(<App />);

  await waitFor(() => {
    expect(connectEquityWebSocket).toHaveBeenCalledWith('session-123');
  });

  expect(createSession).toHaveBeenCalledOnce();

  // Simulate the WebSocket connection opening.
  act(() => {
    socket.open();
  });

  await user.click(screen.getByRole('button', { name: /calculate equity/i }));

  expect(socket.send).toHaveBeenCalledWith(
    JSON.stringify({
      type: 'start',
      mode: 4,
      target_moe: 0.0015,
    }),
  );
});

test('updates hand counts and shows loading state while applying the board', async () => {
  vi.mocked(createSession).mockResolvedValue({
    session_id: 'session-123',
  });

  const boardResponse: Awaited<ReturnType<typeof changeBoard>> = {
    players: [
      {
        base_hands: 100,
        filtered_hands: 100,
        filtered_coverage: 1.0,
        filters: {},
      },
      null,
      {
        base_hands: 100,
        filtered_hands: 50,
        filtered_coverage: 0.5,
        filters: {
          aces: {
            type: 'filter',
            name: 'aces',
            expression: 'AA',
            coverage: 0.5,
            active: true,
          },
        },
      },
      null,
      null,
      null,
    ],
  };

  let resolveBoard!: (value: Awaited<ReturnType<typeof changeBoard>>) => void;

  vi.mocked(changeBoard).mockImplementation(
    () =>
      new Promise((resolve) => {
        resolveBoard = resolve;
      }),
  );

  const user = userEvent.setup();

  render(<App />);

  await waitFor(() => {
    expect(connectEquityWebSocket).toHaveBeenCalledWith('session-123');
  });

  await user.type(screen.getByRole('textbox', { name: 'Board' }), 'AsKd2c');

  await user.click(screen.getByRole('button', { name: 'Apply Board' }));

  expect(changeBoard).toHaveBeenCalledWith('session-123', 'AsKd2c');
  expect(screen.getByRole('button', { name: 'Applying...' })).toBeDisabled();

  await act(async () => {
    resolveBoard(boardResponse);
  });

  expect(screen.getByText('Hands: 100')).toBeInTheDocument();
  expect(screen.getByText('Hands: 50')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Apply Board' })).toBeEnabled();
});

test('adds and deletes a player range through the UI', async () => {
  vi.mocked(createSession).mockResolvedValue({
    session_id: 'session-123',
  });

  vi.mocked(addRange).mockResolvedValue({ hands: 1 });
  vi.mocked(deleteRange).mockResolvedValue(undefined);

  const user = userEvent.setup();

  render(<App />);

  await waitFor(() => {
    expect(connectEquityWebSocket).toHaveBeenCalledWith('session-123');
  });

  const input = screen.getByRole('textbox', { name: 'Player 1' });
  const modeSelector = screen.getByRole('combobox', { name: 'Game mode' });

  await user.type(input, 'AsAhQcJd');

  await user.click(screen.getAllByRole('button', { name: /add range/i })[0]);

  expect(addRange).toHaveBeenCalledWith('session-123', 0, 'AsAhQcJd', 4);

  expect(await screen.findByText('Hands: 1')).toBeInTheDocument();
  expect(modeSelector).toBeDisabled();

  await user.click(screen.getAllByRole('button', { name: /delete range/i })[0]);

  await waitFor(() => {
    expect(input).toHaveValue('');
    expect(modeSelector).toBeEnabled();
  });

  expect(deleteRange).toHaveBeenCalledWith('session-123', 0);
});

test('renders filter panels only for applied players without renumbering them', async () => {
  vi.mocked(createSession).mockResolvedValue({ session_id: 'session-123' });
  vi.mocked(addRange).mockResolvedValue({ hands: 1 });

  const user = userEvent.setup();
  render(<App />);

  expect(screen.queryByRole('region', { name: /Player [1-6] filters/ })).not.toBeInTheDocument();

  await waitFor(() => {
    expect(connectEquityWebSocket).toHaveBeenCalledWith('session-123');
  });

  for (const [playerId, range] of [
    [1, 'AsAhQcJd'],
    [5, 'KsKhQdJc'],
  ] as const) {
    await user.type(screen.getByRole('textbox', { name: `Player ${playerId + 1}` }), range);
    await user.click(screen.getAllByRole('button', { name: /add range/i })[playerId]);
    expect(addRange).toHaveBeenCalledWith('session-123', playerId, range, 4);
    await screen.findByRole('region', { name: `Player ${playerId + 1} filters` });
  }

  const panels = screen.getAllByRole('region', { name: /Player [1-6] filters/ });
  expect(panels).toHaveLength(2);
  expect(panels[0]).toHaveAttribute('aria-label', 'Player 2 filters');
  expect(panels[1]).toHaveAttribute('aria-label', 'Player 6 filters');
  expect(screen.getAllByRole('textbox', { name: /^Player [1-6]$/ })).toHaveLength(6);
});

test('displays an error and retry button when session creation fails', async () => {
  vi.mocked(createSession).mockRejectedValue(new Error('Server unavailable'));

  render(<App />);

  expect(await screen.findByRole('alert')).toHaveTextContent('Server unavailable');
  expect(screen.getByRole('button', { name: 'Retry' })).toBeEnabled();
  expect(createSession).toHaveBeenCalledOnce();
  expect(connectEquityWebSocket).not.toHaveBeenCalled();
  expect(screen.getByRole('button', { name: /apply board/i })).toBeDisabled();
});

test('retries session creation and clears the error after success', async () => {
  vi.mocked(createSession)
    .mockRejectedValueOnce(new Error('Server unavailable'))
    .mockResolvedValueOnce({ session_id: 'session-123' });

  const user = userEvent.setup();
  render(<App />);

  expect(await screen.findByRole('alert')).toHaveTextContent('Server unavailable');

  await user.click(screen.getByRole('button', { name: 'Retry' }));

  await waitFor(() => {
    expect(connectEquityWebSocket).toHaveBeenCalledWith('session-123');
  });

  expect(createSession).toHaveBeenCalledTimes(2);
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: /apply board/i })).toBeEnabled();
});

test('displays an error when board update fails', async () => {
  vi.mocked(createSession).mockResolvedValue({
    session_id: 'session-123',
  });

  vi.mocked(changeBoard).mockRejectedValue(
    new Error('API request failed: 422 Unprocessable Entity'),
  );

  const user = userEvent.setup();
  render(<App />);

  await waitFor(() => {
    expect(connectEquityWebSocket).toHaveBeenCalledWith('session-123');
  });

  await user.type(screen.getByRole('textbox', { name: 'Board' }), 'invalid');

  await user.click(screen.getByRole('button', { name: /apply board/i }));

  expect(await screen.findByRole('alert')).toHaveTextContent(
    'API request failed: 422 Unprocessable Entity',
  );

  expect(screen.queryByText('Board updated successfully')).not.toBeInTheDocument();

  expect(screen.getByRole('button', { name: /apply board/i })).toBeEnabled();
});

test('enables cancellation only while connected and running', async () => {
  vi.mocked(createSession).mockResolvedValue({
    session_id: 'session-123',
  });

  const user = userEvent.setup();
  render(<App />);

  const cancelButton = screen.getByRole('button', {
    name: /cancel equity/i,
  });

  // No session or running job.
  expect(cancelButton).toBeDisabled();

  await waitFor(() => {
    expect(connectEquityWebSocket).toHaveBeenCalledWith('session-123');
  });

  act(() => {
    socket.open();
  });

  // Connected, but no job running.
  expect(cancelButton).toBeDisabled();

  await user.click(screen.getByRole('button', { name: /calculate equity/i }));

  // Backend confirms that the job started.
  act(() => {
    socket.onmessage?.(
      new MessageEvent('message', {
        data: JSON.stringify({
          type: 'started',
          job_id: 'job-123',
        }),
      }),
    );
  });

  expect(cancelButton).toBeEnabled();

  await user.click(cancelButton);

  expect(socket.send).toHaveBeenLastCalledWith(JSON.stringify({ type: 'cancel' }));

  // Requesting cancellation should not immediately mark it cancelled.
  expect(screen.getByText('Status: running')).toBeInTheDocument();

  // Connection loss makes cancellation unavailable.
  act(() => {
    socket.readyState = WebSocket.CLOSED;
    socket.onclose?.();
  });

  expect(cancelButton).toBeDisabled();
});

test('submits the complete large range instead of its preview', async () => {
  vi.mocked(createSession).mockResolvedValue({
    session_id: 'session-123',
  });
  vi.mocked(addRange).mockResolvedValue({ hands: 150 });

  const user = userEvent.setup();
  render(<App />);

  await waitFor(() => {
    expect(connectEquityWebSocket).toHaveBeenCalledWith('session-123');
  });

  const fullRange = 'AsAhKdQc,'.repeat(150);
  const input = screen.getByRole('textbox', { name: 'Player 1' });

  await user.click(input);
  await user.paste(fullRange);

  expect(input).toHaveValue(fullRange.slice(0, 300));

  await user.click(screen.getAllByRole('button', { name: /add range/i })[0]);

  expect(addRange).toHaveBeenCalledWith('session-123', 0, fullRange, 4);

  await waitFor(() => {
    expect(screen.queryByRole('button', { name: /clear input/i })).not.toBeInTheDocument();
  });
});
