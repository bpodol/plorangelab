import type { ComponentProps } from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, test, vi } from 'vitest';

import { PlayerRangeInput } from './PlayerRangeInput';

type RangeProps = ComponentProps<typeof PlayerRangeInput>;

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
    hasAppliedRange: false,
    ...overrides,
  };

  const view = render(<PlayerRangeInput {...props} />);
  return {
    props,
    user: userEvent.setup(),
    rerenderRange: (changes: Partial<RangeProps>) =>
      view.rerender(<PlayerRangeInput {...props} {...changes} />),
  };
}

function pasteText(input: HTMLElement, text: string) {
  fireEvent.paste(input, {
    clipboardData: {
      getData: (format: string) => (format === 'text' ? text : ''),
    },
  });
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

test('large paste preserves the complete range but displays only a preview', () => {
  const fullRange = 'AsAhKdQc,'.repeat(150);
  const { props, rerenderRange } = renderPlayer();

  const input = screen.getByRole('textbox', {
    name: 'Player 1',
  });

  pasteText(input, fullRange);

  expect(props.onRangeChange).toHaveBeenCalledExactlyOnceWith(fullRange);

  rerenderRange({ range: fullRange });

  expect(input).toHaveValue(fullRange.slice(0, 300));
  expect(input).toHaveProperty('readOnly', true);
});

test('large paste replaces only the selected part of a small range', () => {
  const original = 'AsAhKdQc';
  const pasted = 'Q'.repeat(1200);
  const { props } = renderPlayer({ range: original });
  const input = screen.getByRole('textbox', { name: 'Player 1' }) as HTMLInputElement;

  input.setSelectionRange(2, 6);
  pasteText(input, pasted);

  expect(props.onRangeChange).toHaveBeenCalledExactlyOnceWith(
    original.slice(0, 2) + pasted + original.slice(6),
  );
});

test('does not accept further pastes into a read-only preview', () => {
  const { props } = renderPlayer({ range: 'A'.repeat(1200) });
  const input = screen.getByRole('textbox', { name: 'Player 1' });

  pasteText(input, 'B'.repeat(1100));

  expect(props.onRangeChange).not.toHaveBeenCalled();
});

test('clear input disappears once a large range has been applied', async () => {
  const { props, user, rerenderRange } = renderPlayer({ range: 'A'.repeat(1200) });
  const clearButton = screen.getByRole('button', { name: /clear input/i });

  await user.click(clearButton);
  expect(props.onRangeChange).toHaveBeenCalledExactlyOnceWith('');

  rerenderRange({ hasAppliedRange: true });
  expect(screen.queryByRole('button', { name: /clear input/i })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: /delete range/i })).toBeInTheDocument();
});

test('disables clear input while a large range is loading', () => {
  renderPlayer({ range: 'A'.repeat(1200), isLoading: true });
  expect(screen.getByRole('button', { name: /clear input/i })).toBeDisabled();
});
