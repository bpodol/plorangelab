interface PlayerRangeProps {
  playerId: number;
  range: string;
  numberOfHands: number;
  onRangeChange: (value: string) => void;
  onAdd: () => void;
  onDelete: () => void;
  disabled: boolean;
  isLoading: boolean;
  error: string | null;
}

export function PlayerRangeInput({
  playerId,
  range,
  numberOfHands,
  onRangeChange,
  onAdd,
  onDelete,
  disabled,
  isLoading,
  error,
}: PlayerRangeProps) {
  return (
    <section>
      <label htmlFor={`range-${playerId}`}>Player {playerId + 1}</label>
      <input
        id={`range-${playerId}`}
        type="text"
        value={range}
        onChange={(event) => onRangeChange(event.target.value)}
      />

      <button onClick={onAdd} disabled={disabled}>
        {isLoading ? 'adding...' : 'add range'}
      </button>

      <button onClick={onDelete} disabled={disabled}>
        {isLoading ? 'deleting...' : 'delete range'}
      </button>

      <p>Hands: {numberOfHands}</p>
      {error && <p role="alert">{error}</p>}
    </section>
  );
}
