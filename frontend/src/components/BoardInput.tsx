interface BoardInputProps {
  board: string;
  onBoardChange: (value: string) => void;
  onApply: () => void;
  disabled: boolean;
  isLoading: boolean;
}

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
