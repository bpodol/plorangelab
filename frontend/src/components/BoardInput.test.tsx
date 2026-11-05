import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, test, vi } from 'vitest';
import { BoardInput } from './BoardInput';

test('renders the board value', () => {
  render(
    <BoardInput
      board="AsKd2c"
      onBoardChange={vi.fn()}
      onApply={vi.fn()}
      disabled={false}
      isLoading={false}
    />,
  );

  const input = screen.getByRole('textbox', { name: /board/i });

  expect(input).toHaveValue('AsKd2c');
});

test('calls onBoardChange when the user edits the board', () => {
  const onBoardChange = vi.fn();

  render(
    <BoardInput
      board=""
      onBoardChange={onBoardChange}
      onApply={vi.fn()}
      disabled={false}
      isLoading={false}
    />,
  );

  const input = screen.getByRole('textbox', { name: /board/i });

  fireEvent.change(input, {
    target: { value: 'AsKd2c' },
  });

  expect(onBoardChange).toHaveBeenCalledWith('AsKd2c');
});

test('calls onApply when the Apply button is clicked', async () => {
  const onApply = vi.fn();
  const user = userEvent.setup();

  render(
    <BoardInput
      board="AsKd2c"
      onBoardChange={vi.fn()}
      onApply={onApply}
      disabled={false}
      isLoading={false}
    />,
  );

  await user.click(screen.getByRole('button', { name: /apply board/i }));

  expect(onApply).toHaveBeenCalledOnce();
});

test('disables the Apply button during loading', () => {
  render(
    <BoardInput
      board="AsKd2c"
      onBoardChange={vi.fn()}
      onApply={vi.fn()}
      disabled={true}
      isLoading={true}
    />,
  );

  const button = screen.getByRole('button', { name: /applying/i });

  expect(button).toBeDisabled();
});
