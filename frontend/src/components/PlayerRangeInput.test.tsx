import type { ComponentProps } from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, test, vi } from 'vitest';

import { PlayerRangeInput } from './PlayerRangeInput';

// Helper to avoid repeating the same props in every test.
function renderPlayer(overrides: Partial<ComponentProps<typeof PlayerRangeInput>> = {}) {
  const props: ComponentProps<typeof PlayerRangeInput> = {
    playerId: 0,
    range: '',
    numberOfHands: 0,
    onRangeChange: vi.fn(),
    onAdd: vi.fn(),
    onDelete: vi.fn(),
    disabled: false,
    isLoading: false,
    error: null,
    ...overrides,
  };

  render(<PlayerRangeInput {...props} />);

  return props;
}

test('renders the player number, range and hand count', () => {
  renderPlayer({
    playerId: 2,
    range: 'AsAhQcJd',
    numberOfHands: 150,
  });

  expect(screen.getByRole('textbox', { name: 'Player 3' })).toHaveValue('AsAhQcJd');

  expect(screen.getByText('Hands: 150')).toBeInTheDocument();
});

test('calls onRangeChange when the input changes', () => {
  const onRangeChange = vi.fn();

  renderPlayer({ onRangeChange });

  fireEvent.change(screen.getByRole('textbox'), {
    target: { value: 'AsAhQcJd' },
  });

  expect(onRangeChange).toHaveBeenCalledWith('AsAhQcJd');
});

test('calls onAdd and onDelete when buttons are clicked', async () => {
  const user = userEvent.setup();
  const onAdd = vi.fn();
  const onDelete = vi.fn();

  renderPlayer({ onAdd, onDelete });

  await user.click(screen.getByRole('button', { name: /add range/i }));
  await user.click(screen.getByRole('button', { name: /delete range/i }));

  expect(onAdd).toHaveBeenCalledOnce();
  expect(onDelete).toHaveBeenCalledOnce();
});

test('disables both buttons when disabled is true', () => {
  renderPlayer({ disabled: true });

  expect(screen.getByRole('button', { name: /add range/i })).toBeDisabled();
  expect(screen.getByRole('button', { name: /delete range/i })).toBeDisabled();
});

test('displays loading text', () => {
  renderPlayer({ isLoading: true, disabled: true });

  expect(screen.getByRole('button', { name: /adding/i })).toBeDisabled();
  expect(screen.getByRole('button', { name: /deleting/i })).toBeDisabled();
});

test('displays an error when provided', () => {
  renderPlayer({ error: 'Invalid range' });

  expect(screen.getByRole('alert')).toHaveTextContent('Invalid range');
});

test('does not display an error when error is null', () => {
  renderPlayer({ error: null });

  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});
