import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, expect, test, vi } from 'vitest';

import App from './App';
import type { PlayerFilterSnapshot } from './types';
import { createSession, changeBoard, changeMode } from './api/sessions';
import { addRange, deleteRange } from './api/players';
import { useEquitySSE } from './hooks/useEquitySSE';

vi.mock('./api/sessions', () => ({
  createSession: vi.fn(),
  changeBoard: vi.fn(),
  changeMode: vi.fn(),
}));

vi.mock('./api/players', () => ({
  addRange: vi.fn(),
  deleteRange: vi.fn(),
}));

vi.mock('./hooks/useEquitySSE', () => ({
  useEquitySSE: vi.fn(),
}));

// App tests verify wiring to the hook, not HTTP requests or EventSource behavior.
const startEquity = vi.fn(async (_targetMoe: number): Promise<void> => {});
const cancelEquity = vi.fn(async (): Promise<void> => {});

beforeEach(() => {
  vi.resetAllMocks();
  startEquity.mockResolvedValue(undefined);
  cancelEquity.mockResolvedValue(undefined);

  vi.mocked(useEquitySSE).mockReturnValue({
    equityStatus: 'idle',
    equityMessage: null,
    equityError: null,
    startEquity,
    cancelEquity,
  });
});

function makeRangeSnapshot(
  baseHands: number,
  filters: PlayerFilterSnapshot['filters'] = {},
): PlayerFilterSnapshot {
  return {
    base_hands: baseHands,
    filtered_hands: baseHands,
    filtered_coverage: 1,
    filters,
  };
}

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

  const modeSelector = screen.getByRole('combobox', { name: 'Game mode' });
  expect(modeSelector).toHaveValue('4');
  expect(modeSelector).toBeDisabled();
  expect(changeMode).not.toHaveBeenCalled();

  // Session-dependent actions must remain disabled until initialization finishes.
  expect(screen.getByRole('button', { name: /apply board/i })).toBeDisabled();
  expect(screen.getByRole('button', { name: /calculate equity/i })).toBeDisabled();
  expect(screen.queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument();

  await act(async () => {
    resolveSession({ session_id: 'session-123' });
  });

  await waitFor(() => {
    expect(screen.getByRole('button', { name: /apply board/i })).toBeEnabled();
  });

  expect(screen.getByRole('button', { name: /apply board/i })).toBeEnabled();
  expect(modeSelector).toBeEnabled();
  expect(screen.queryByText(/Session:/i)).not.toBeInTheDocument();
});

test('passes the session to the equity hook and delegates calculation requests', async () => {
  vi.mocked(createSession).mockResolvedValue({ session_id: 'session-123' });
  const user = userEvent.setup();

  render(<App />);

  await waitFor(() => {
    expect(useEquitySSE).toHaveBeenLastCalledWith('session-123');
    expect(screen.getByRole('button', { name: /calculate equity/i })).toBeEnabled();
  });

  expect(createSession).toHaveBeenCalledOnce();
  await user.click(screen.getByRole('button', { name: /calculate equity/i }));
  expect(startEquity).toHaveBeenCalledExactlyOnceWith(0.0015);
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
    expect(screen.getByRole('button', { name: /apply board/i })).toBeEnabled();
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

  vi.mocked(addRange).mockResolvedValue(makeRangeSnapshot(1));
  vi.mocked(deleteRange).mockResolvedValue(undefined);

  const user = userEvent.setup();

  render(<App />);

  await waitFor(() => {
    expect(screen.getByRole('button', { name: /apply board/i })).toBeEnabled();
  });

  const input = screen.getByRole('textbox', { name: 'Player 1' });
  const modeSelector = screen.getByRole('combobox', { name: 'Game mode' });

  await user.type(input, 'AsAhQcJd');

  await user.click(screen.getAllByRole('button', { name: /add range/i })[0]);

  expect(addRange).toHaveBeenCalledWith('session-123', 0, 'AsAhQcJd');

  expect(await screen.findByText('Hands: 1')).toBeInTheDocument();
  expect(modeSelector).toBeDisabled();

  await user.click(screen.getAllByRole('button', { name: /delete range/i })[0]);

  await waitFor(() => {
    expect(input).toHaveValue('');
    expect(modeSelector).toBeEnabled();
  });

  expect(deleteRange).toHaveBeenCalledWith('session-123', 0);
});

test('shows auto-generated filter groups when adding a range on an active board', async () => {
  vi.mocked(createSession).mockResolvedValue({ session_id: 'session-123' });
  vi.mocked(changeBoard).mockResolvedValue({ players: [null, null, null, null, null, null] });
  vi.mocked(addRange).mockResolvedValue(
    makeRangeSnapshot(2, {
      auto_ranks: {
        type: 'group',
        name: 'Auto-ranked hands',
        coverage: 0.5,
        active: false,
        filters: {
          aces: {
            type: 'filter',
            name: 'Aces',
            expression: 'AA',
            coverage: 0.5,
            active: false,
          },
        },
      },
    }),
  );

  const user = userEvent.setup();
  render(<App />);
  await waitFor(() => {
    expect(screen.getByRole('button', { name: /apply board/i })).toBeEnabled();
  });

  await user.type(screen.getByRole('textbox', { name: 'Board' }), 'AsKd2c');
  await user.click(screen.getByRole('button', { name: 'Apply Board' }));
  await waitFor(() => {
    expect(changeBoard).toHaveBeenCalledWith('session-123', 'AsKd2c');
  });

  await user.type(screen.getByRole('textbox', { name: 'Player 1' }), 'QsQhJcTc,9s9h8d8c');
  await user.click(screen.getAllByRole('button', { name: /add range/i })[0]);

  expect(addRange).toHaveBeenCalledWith('session-123', 0, 'QsQhJcTc,9s9h8d8c');
  expect(await screen.findByText('Hands: 2')).toBeInTheDocument();
  expect(screen.getByRole('region', { name: 'Player 1 filters' })).toBeInTheDocument();
  expect(await screen.findByText(/Auto-ranked hands/)).toBeInTheDocument();
});

test('renders filter panels only for applied players without renumbering them', async () => {
  vi.mocked(createSession).mockResolvedValue({ session_id: 'session-123' });
  vi.mocked(addRange).mockResolvedValue(makeRangeSnapshot(1));

  const user = userEvent.setup();
  render(<App />);

  expect(screen.queryByRole('region', { name: /Player [1-6] filters/ })).not.toBeInTheDocument();

  await waitFor(() => {
    expect(screen.getByRole('button', { name: /apply board/i })).toBeEnabled();
  });

  for (const [playerId, range] of [
    [1, 'AsAhQcJd'],
    [5, 'KsKhQdJc'],
  ] as const) {
    await user.type(screen.getByRole('textbox', { name: `Player ${playerId + 1}` }), range);
    await user.click(screen.getAllByRole('button', { name: /add range/i })[playerId]);
    expect(addRange).toHaveBeenCalledWith('session-123', playerId, range);
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
  expect(screen.getByRole('button', { name: /calculate equity/i })).toBeDisabled();
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
    expect(screen.getByRole('button', { name: /apply board/i })).toBeEnabled();
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
    expect(screen.getByRole('button', { name: /apply board/i })).toBeEnabled();
  });

  await user.type(screen.getByRole('textbox', { name: 'Board' }), 'invalid');

  await user.click(screen.getByRole('button', { name: /apply board/i }));

  expect(await screen.findByRole('alert')).toHaveTextContent(
    'API request failed: 422 Unprocessable Entity',
  );

  expect(screen.queryByText('Board updated successfully')).not.toBeInTheDocument();

  expect(screen.getByRole('button', { name: /apply board/i })).toBeEnabled();
});

test('delegates cancellation when the equity hook reports a running job', async () => {
  vi.mocked(createSession).mockResolvedValue({ session_id: 'session-123' });
  vi.mocked(useEquitySSE).mockReturnValue({
    equityStatus: 'running',
    equityMessage: null,
    equityError: null,
    startEquity,
    cancelEquity,
  });

  const user = userEvent.setup();
  render(<App />);
  const cancelButton = screen.getByRole('button', { name: /cancel equity/i });

  // App must not enable session-dependent actions before session creation.
  expect(cancelButton).toBeDisabled();
  await waitFor(() => expect(cancelButton).toBeEnabled());

  await user.click(cancelButton);
  expect(cancelEquity).toHaveBeenCalledOnce();
  expect(startEquity).not.toHaveBeenCalled();
});

test('submits the complete large range instead of its preview', async () => {
  vi.mocked(createSession).mockResolvedValue({
    session_id: 'session-123',
  });
  vi.mocked(addRange).mockResolvedValue(makeRangeSnapshot(150));

  const user = userEvent.setup();
  render(<App />);

  await waitFor(() => {
    expect(screen.getByRole('button', { name: /apply board/i })).toBeEnabled();
  });

  const fullRange = 'AsAhKdQc,'.repeat(150);
  const input = screen.getByRole('textbox', { name: 'Player 1' });

  await user.click(input);
  await user.paste(fullRange);

  expect(input).toHaveValue(fullRange.slice(0, 300));

  await user.click(screen.getAllByRole('button', { name: /add range/i })[0]);

  expect(addRange).toHaveBeenCalledWith('session-123', 0, fullRange);

  await waitFor(() => {
    expect(screen.queryByRole('button', { name: /clear input/i })).not.toBeInTheDocument();
  });
});

test('changes mode only after backend confirmation', async () => {
  vi.mocked(createSession).mockResolvedValue({ session_id: 'session-123' });

  let resolveMode!: () => void;
  vi.mocked(changeMode).mockImplementation(
    () =>
      new Promise<void>((resolve) => {
        resolveMode = resolve;
      }),
  );

  const user = userEvent.setup();
  render(<App />);

  const modeSelector = screen.getByRole('combobox', { name: 'Game mode' });
  await waitFor(() => {
    expect(modeSelector).toBeEnabled();
  });

  await user.selectOptions(modeSelector, '5');

  expect(changeMode).toHaveBeenCalledOnce();
  expect(changeMode).toHaveBeenCalledWith('session-123', 5);
  expect(modeSelector).toBeDisabled();
  // The controlled value must not change until the backend accepts the mode.
  expect(modeSelector).toHaveValue('4');

  await act(async () => {
    resolveMode();
  });

  expect(modeSelector).toBeEnabled();
  expect(modeSelector).toHaveValue('5');
});

test('retains the previous mode and displays an error when mode change fails', async () => {
  vi.mocked(createSession).mockResolvedValue({ session_id: 'session-123' });
  vi.mocked(changeMode).mockRejectedValue(new Error('Mode change rejected'));

  const user = userEvent.setup();
  render(<App />);

  const modeSelector = screen.getByRole('combobox', { name: 'Game mode' });
  await waitFor(() => expect(modeSelector).toBeEnabled());

  await user.selectOptions(modeSelector, '5');

  expect(changeMode).toHaveBeenCalledWith('session-123', 5);
  expect(await screen.findByRole('alert')).toHaveTextContent('Mode change rejected');
  expect(modeSelector).toHaveValue('4');
  expect(modeSelector).toBeEnabled();
});

test('disables mode changes while a player range is being added', async () => {
  vi.mocked(createSession).mockResolvedValue({ session_id: 'session-123' });

  let resolveRange!: (value: PlayerFilterSnapshot) => void;
  vi.mocked(addRange).mockImplementation(
    () =>
      new Promise<PlayerFilterSnapshot>((resolve) => {
        resolveRange = resolve;
      }),
  );

  const user = userEvent.setup();
  render(<App />);

  const modeSelector = screen.getByRole('combobox', { name: 'Game mode' });
  await waitFor(() => expect(modeSelector).toBeEnabled());

  await user.type(screen.getByRole('textbox', { name: 'Player 1' }), 'AsAhQcJd');
  await user.click(screen.getAllByRole('button', { name: /add range/i })[0]);

  expect(addRange).toHaveBeenCalledWith('session-123', 0, 'AsAhQcJd');
  expect(modeSelector).toBeDisabled();
  expect(changeMode).not.toHaveBeenCalled();

  await act(async () => {
    resolveRange(makeRangeSnapshot(1));
  });

  expect(await screen.findByText('Hands: 1')).toBeInTheDocument();
  // An applied range also prevents further mode changes.
  expect(modeSelector).toBeDisabled();
});
