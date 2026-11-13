interface BoardInputProps {
  board: string;
  onBoardChange: (value: string) => void;
  onApply: () => void;
  disabled: boolean;
  isLoading: boolean;
}

/**
 * Render a controlled community board input and apply button.
 *
 * Displays the board expression provided by the parent and
 * reports input changes through onBoardChange.
 *
 * The onApply callback delegates board submission to the parent.
 * The component does not validate or submit the board itself.
 *
 * @remarks
 * The disabled prop controls the apply button, while the text
 * input remains editable.
 */
export function BoardInput({
  board,
  onBoardChange,
  onApply,
  disabled,
  isLoading,
}: BoardInputProps) {
  return (
    <section>
      <label htmlFor="board">Board</label>

      <input
        id="board"
        type="text"
        value={board}
        onChange={(event) => onBoardChange(event.target.value)}
        placeholder="AsKd2c"
      />

      <p>Current input: {board || 'None'}</p>

      <button onClick={onApply} disabled={disabled}>
        {isLoading ? 'Applying...' : 'Apply Board'}
      </button>
    </section>
  );
}
