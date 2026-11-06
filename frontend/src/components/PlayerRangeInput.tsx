import type { ClipboardEvent } from 'react';

const MAX_EDITABLE_LENGTH = 1000;
const PREVIEW_LENGTH = 300;

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
  hasAppliedRange: boolean;
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
  hasAppliedRange,
}: PlayerRangeProps) {
  const isLargeRange = range.length > MAX_EDITABLE_LENGTH;

  function handlePaste(event: ClipboardEvent<HTMLInputElement>) {
    if (isLargeRange) return;
    const pastedText = event.clipboardData.getData('text');
    const input = event.currentTarget;

    const start = input.selectionStart ?? range.length;
    const end = input.selectionEnd ?? start;

    const newLength = range.length - (end - start) + pastedText.length;

    if (newLength <= MAX_EDITABLE_LENGTH) return;

    event.preventDefault();

    const newRange = range.slice(0, start) + pastedText + range.slice(end);

    onRangeChange(newRange);
  }

  return (
    <section>
      <label htmlFor={`range-${playerId}`}>Player {playerId + 1}</label>

      <input
        id={`range-${playerId}`}
        type="text"
        value={isLargeRange ? range.slice(0, PREVIEW_LENGTH) : range}
        onChange={(event) => onRangeChange(event.target.value)}
        onPaste={handlePaste}
        readOnly={isLargeRange}
      />

      {isLargeRange && !hasAppliedRange && (
        <button type="button" onClick={() => onRangeChange('')} disabled={isLoading}>
          Clear Input
        </button>
      )}

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
