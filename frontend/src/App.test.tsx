import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, expect, test, vi } from 'vitest';

import App from './App';
import { createSession, changeBoard, addRange, deleteRange } from './api/sessions';
import { connectEquityWebSocket } from './api/equity';

vi.mock('./api/sessions', () => ({
  createSession: vi.fn(),
  changeBoard: vi.fn(),
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

test('renders six players and disables actions before session creation', () => {
  render(<App />);

  expect(screen.getByRole('heading', { name: 'PLO RangeLab' })).toBeInTheDocument();

  expect(screen.getAllByRole('textbox', { name: /^Player [1-6]$/ })).toHaveLength(6);

  expect(screen.getByRole('button', { name: /apply board/i })).toBeDisabled();

  expect(screen.getByRole('button', { name: /calculate equity/i })).toBeDisabled();

  expect(createSession).not.toHaveBeenCalled();
});

test('creates a session and starts an equity calculation', async () => {
  vi.mocked(createSession).mockResolvedValue({
    session_id: 'session-123',
  });

  const user = userEvent.setup();

  render(<App />);

  await user.click(screen.getByRole('button', { name: /create session/i }));

  expect(await screen.findByText('Session: session-123')).toBeInTheDocument();

  expect(createSession).toHaveBeenCalledOnce();

  expect(connectEquityWebSocket).toHaveBeenCalledWith('session-123');

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

test('updates displayed hand counts after changing the board', async () => {
  vi.mocked(createSession).mockResolvedValue({
    session_id: 'session-123',
  });

  vi.mocked(changeBoard).mockResolvedValue({
    players: [100, 0, 50, 0, 0, 0],
  });

  const user = userEvent.setup();

  render(<App />);

  await user.click(screen.getByRole('button', { name: /create session/i }));

  await screen.findByText('Session: session-123');

  await user.type(screen.getByRole('textbox', { name: 'Board' }), 'AsKd2c');

  await user.click(screen.getByRole('button', { name: /apply board/i }));

  expect(changeBoard).toHaveBeenCalledWith('session-123', 'AsKd2c');

  expect(await screen.findByText('Hands: 100')).toBeInTheDocument();
  expect(screen.getByText('Hands: 50')).toBeInTheDocument();
});

test('adds and deletes a player range through the UI', async () => {
  vi.mocked(createSession).mockResolvedValue({
    session_id: 'session-123',
  });

  vi.mocked(addRange).mockResolvedValue({ hands: 1 });
  vi.mocked(deleteRange).mockResolvedValue(undefined);

  const user = userEvent.setup();

  render(<App />);

  await user.click(screen.getByRole('button', { name: /create session/i }));
  await screen.findByText('Session: session-123');

  const input = screen.getByRole('textbox', { name: 'Player 1' });
  const modeSelector = screen.getByRole('combobox', { name: 'Game mode' });

  await user.type(input, 'AsAhQcJd');

  await user.click(screen.getAllByRole('button', { name: /add range/i })[0]);

  expect(addRange).toHaveBeenCalledWith(0, 'AsAhQcJd', 4, 'session-123');

  expect(await screen.findByText('Hands: 1')).toBeInTheDocument();
  expect(modeSelector).toBeDisabled();

  await user.click(screen.getAllByRole('button', { name: /delete range/i })[0]);

  await waitFor(() => {
    expect(input).toHaveValue('');
    expect(modeSelector).toBeEnabled();
  });

  expect(deleteRange).toHaveBeenCalledWith('session-123', 0);
});

test('displays an error when session creation fails', async () => {
  vi.mocked(createSession).mockRejectedValue(new Error('Server unavailable'));

  const user = userEvent.setup();
  render(<App />);

  await user.click(screen.getByRole('button', { name: /create session/i }));

  expect(await screen.findByRole('alert')).toHaveTextContent('Server unavailable');

  expect(screen.getByText('Session: Not created')).toBeInTheDocument();

  expect(screen.getByRole('button', { name: /create session/i })).toBeEnabled();

  expect(connectEquityWebSocket).not.toHaveBeenCalled();
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

  await user.click(screen.getByRole('button', { name: /create session/i }));
  await screen.findByText('Session: session-123');

  await user.type(screen.getByRole('textbox', { name: 'Board' }), 'invalid');

  await user.click(screen.getByRole('button', { name: /apply board/i }));

  expect(await screen.findByRole('alert')).toHaveTextContent(
    'API request failed: 422 Unprocessable Entity',
  );

  expect(screen.queryByText('Board updated successfully')).not.toBeInTheDocument();

  expect(screen.getByRole('button', { name: /apply board/i })).toBeEnabled();
});

test.each([
  ['default value', null, 0.0015],
  ['custom value', '0.003', 0.003],
])('sends the correct target MOE for %s', async (_name, inputValue, expectedMoe) => {
  vi.mocked(createSession).mockResolvedValue({
    session_id: 'session-123',
  });

  const user = userEvent.setup();
  render(<App />);

  await user.click(screen.getByRole('button', { name: /create session/i }));
  await screen.findByText('Session: session-123');

  act(() => {
    socket.open();
  });

  const input = screen.getByLabelText(/target moe/i);

  if (inputValue !== null) {
    await user.clear(input);

    if (inputValue !== '') {
      await user.type(input, inputValue);
    }
  }

  await user.click(screen.getByRole('button', { name: /calculate equity/i }));

  expect(socket.send).toHaveBeenCalledWith(
    JSON.stringify({
      type: 'start',
      mode: 4,
      target_moe: expectedMoe,
    }),
  );
});

test.each(['0', '0.006', '-0.001'])(
  'disables calculation for invalid target MOE %s',
  async (value) => {
    vi.mocked(createSession).mockResolvedValue({
      session_id: 'session-123',
    });

    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole('button', { name: /create session/i }));
    await screen.findByText('Session: session-123');

    act(() => {
      socket.open();
    });

    const input = screen.getByLabelText(/target moe/i);

    await user.clear(input);
    await user.type(input, value);

    expect(screen.getByRole('button', { name: /calculate equity/i })).toBeDisabled();

    expect(socket.send).not.toHaveBeenCalled();
  },
);

test('disables calculation when target MOE is empty', async () => {
  vi.mocked(createSession).mockResolvedValue({
    session_id: 'session-123',
  });

  const user = userEvent.setup();
  render(<App />);

  await user.click(screen.getByRole('button', { name: /create session/i }));
  await screen.findByText('Session: session-123');

  act(() => {
    socket.open();
  });

  const input = screen.getByRole('spinbutton', {
    name: /target moe/i,
  });

  await user.clear(input);

  const calculateButton = screen.getByRole('button', {
    name: /calculate equity/i,
  });

  expect(calculateButton).toBeDisabled();
  expect(socket.send).not.toHaveBeenCalled();
});
