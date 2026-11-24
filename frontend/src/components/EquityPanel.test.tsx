import type { ComponentProps } from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, test, vi } from 'vitest';
import type { EquityStreamEvent } from '../api/equity';
import { EquityPanel } from './EquityPanel';

type PanelProps = ComponentProps<typeof EquityPanel>;

function renderPanel(overrides: Partial<PanelProps> = {}) {
  const props: PanelProps = {
    hasSession: false,
    status: 'idle',
    message: null,
    error: null,
    onCalculate: vi.fn(),
    onCancel: vi.fn(),
    ...overrides,
  };

  render(<EquityPanel {...props} />);
  return { user: userEvent.setup(), props };
}

// These fixtures represent the panel's input props. Transport and parsing
// behavior belongs in the equity API and hook tests, not component tests.
const progressMessage = {
  job_id: 'job-123',
  status: 'running',
  error: null,
  details: {
    equities: { '0': 0.653, '2': 0.347 },
    method: 'monte-carlo',
    margins_of_error: { '0': 0.0018, '2': 0.0023 },
    simulations: 125000,
    attempts: 130000,
    stop_reason: null,
    sampler: 'rejection',
    effective_samples: null,
  },
} satisfies EquityStreamEvent;

const resultMessage = {
  job_id: 'job-123',
  status: 'completed',
  error: null,
  details: {
    equities: { '2': 0.347, '0': 0.653 },
    method: 'monte-carlo',
    margins_of_error: { '2': 0.0023, '0': 0.0012 },
    simulations: 100000,
    attempts: 110000,
    stop_reason: 'target_reached',
    sampler: 'rejection',
    effective_samples: null,
  },
} satisfies EquityStreamEvent;

test('shows the default input and disables actions without a session', () => {
  renderPanel();

  expect(screen.getByText('Status: idle')).toBeInTheDocument();
  expect(screen.getByRole('spinbutton', { name: /target moe/i })).toHaveValue(0.0015);
  expect(screen.getByRole('button', { name: /calculate equity/i })).toBeDisabled();
  expect(screen.getByRole('button', { name: /cancel equity/i })).toBeDisabled();
  expect(screen.queryByRole('heading', { name: 'Results' })).not.toBeInTheDocument();
});

test.each([
  { label: 'default', input: null, expected: 0.0015 },
  { label: 'custom', input: '0.003', expected: 0.003 },
  { label: 'maximum allowed', input: '0.005', expected: 0.005 },
])('sends the $label target MOE when calculating', async ({ input, expected }) => {
  const { user, props } = renderPanel({ hasSession: true });
  const moeInput = screen.getByRole('spinbutton', { name: /target moe/i });

  if (input !== null) {
    await user.clear(moeInput);
    await user.type(moeInput, input);
  }

  await user.click(screen.getByRole('button', { name: /calculate equity/i }));

  expect(props.onCalculate).toHaveBeenCalledExactlyOnceWith(expected);
});

test.each(['', '0', '0.006'])('disables calculation for invalid MOE %j', async (input) => {
  const { user, props } = renderPanel({ hasSession: true });
  const moeInput = screen.getByRole('spinbutton', { name: /target moe/i });

  await user.clear(moeInput);
  if (input !== '') await user.type(moeInput, input);

  expect(screen.getByRole('button', { name: /calculate equity/i })).toBeDisabled();
  expect(props.onCalculate).not.toHaveBeenCalled();
});

test.each(['starting', 'running', 'cancelling'] as const)(
  'disables calculation while status is %s',
  (status) => {
    renderPanel({ hasSession: true, status });
    expect(screen.getByRole('button', { name: /calculate equity/i })).toBeDisabled();
  },
);

test('allows cancellation for a running calculation', async () => {
  const { user, props } = renderPanel({ hasSession: true, status: 'running' });

  expect(screen.getByRole('button', { name: /calculate equity/i })).toBeDisabled();

  const cancelButton = screen.getByRole('button', { name: /cancel equity/i });
  expect(cancelButton).toBeEnabled();
  await user.click(cancelButton);

  expect(props.onCancel).toHaveBeenCalledOnce();
  expect(props.onCalculate).not.toHaveBeenCalled();
});

test.each(['idle', 'starting', 'cancelling', 'completed', 'cancelled', 'failed'] as const)(
  'disables cancellation while status is %s',
  (status) => {
    renderPanel({ hasSession: true, status });
    expect(screen.getByRole('button', { name: /cancel equity/i })).toBeDisabled();
  },
);

test('displays running equities, simulations, and the highest non-null progress MOE', () => {
  renderPanel({ status: 'running', message: progressMessage });
  const formattedCount = (125000).toLocaleString().replace(/\s/g, ' ');

  expect(screen.getByText(`Simulations: ${formattedCount}`)).toBeInTheDocument();
  expect(screen.getByText('Highest MOE: ±0.230 pp')).toBeInTheDocument();
  expect(screen.getByText(/Player 1: 65\.30%/)).toBeInTheDocument();
  expect(screen.getByText(/Player 3: 34\.70%/)).toBeInTheDocument();
});

const missingMoeCases: { label: string; moes: Record<string, number | null> | null }[] = [
  { label: 'all-null', moes: { '0': null, '2': null } },
  { label: 'empty', moes: {} },
  { label: 'null', moes: null },
];

test.each(missingMoeCases)('hides the highest progress MOE for $label values', ({ moes }) => {
  renderPanel({
    status: 'running',
    message: {
      ...progressMessage,
      details: { ...progressMessage.details, margins_of_error: moes },
    },
  });

  expect(screen.queryByText(/Highest MOE:/i)).not.toBeInTheDocument();
});

test('renders player equity, corresponding MOEs, and simulation count', () => {
  renderPanel({ status: 'completed', message: resultMessage });

  const players = screen.getAllByRole('listitem');
  expect(players).toHaveLength(2);
  expect(players[0]).toHaveTextContent('Player 1: 65.30% ±0.120 pp');
  expect(players[1]).toHaveTextContent('Player 3: 34.70% ±0.230 pp');
  const formattedCount = (100000).toLocaleString().replace(/\s/g, ' ');

  expect(screen.getByRole('heading', { name: 'Results' })).toBeInTheDocument();
  expect(screen.getByText(`Simulations: ${formattedCount}`)).toBeInTheDocument();
});

test('renders results even without MOEs or a simulation count', () => {
  renderPanel({
    status: 'completed',
    message: {
      ...resultMessage,
      details: { ...resultMessage.details, margins_of_error: null, simulations: null },
    },
  });

  expect(screen.getByText(/Player 1: 65\.30%/)).toBeInTheDocument();
  expect(screen.getByText(/Player 3: 34\.70%/)).toBeInTheDocument();
  expect(screen.queryByText(/Simulations:/i)).not.toBeInTheDocument();
  expect(screen.queryByText(/±/)).not.toBeInTheDocument();
});

test('renders partial results reported by a cancelled calculation', () => {
  renderPanel({
    status: 'cancelled',
    message: { ...resultMessage, status: 'cancelled' },
  });

  expect(screen.getByRole('heading', { name: 'Results' })).toBeInTheDocument();
  expect(screen.getByText(/Player 1: 65\.30%/)).toBeInTheDocument();
});

test('displays equity errors from the hook as alerts', () => {
  renderPanel({ status: 'failed', error: 'Calculation failed' });
  expect(screen.getByRole('alert')).toHaveTextContent('Calculation failed');
});
