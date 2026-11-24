import { useState } from 'react';
import type { EquityStreamEvent } from '../api/equity';
import type { EquityStatus } from '../hooks/useEquitySSE';

interface EquityPanelProps {
  hasSession: boolean;
  status: EquityStatus;
  message: EquityStreamEvent | null;
  error: string | null;
  onCalculate: (targetMoe: number) => void;
  onCancel: () => void;
}

/**
 * Render equity calculation controls, progress, and results.
 *
 * The parent owns the HTTP/SSE lifecycle. This component validates the
 * target MOE, displays streamed snapshots, and delegates user actions.
 * An SSE connection is established only after submission, so it must not
 * be required to start a calculation or request cancellation.
 */
export function EquityPanel({
  hasSession,
  status,
  message,
  error,
  onCalculate,
  onCancel,
}: EquityPanelProps) {
  const [targetMoeInput, setTargetMoeInput] = useState('0.0015');

  // Keep the input as a string to support intermediate editing
  // states; convert it to a number separately for validation.
  const targetMoe = Number(targetMoeInput);

  const isValidTargetMoe =
    targetMoeInput.trim() !== '' &&
    Number.isFinite(targetMoe) &&
    targetMoe > 0 &&
    targetMoe <= 0.005;

  const isBusy = status === 'starting' || status === 'running' || status === 'cancelling';
  const details = message?.details;
  const isProgress = message?.status === 'pending' || message?.status === 'running';
  const isResult = message?.status === 'completed' || message?.status === 'cancelled';

  // Show the highest available player MOE as the current precision
  // indicator; null entries represent unavailable estimates.
  const validMoes =
    isProgress && details
      ? Object.values(details.margins_of_error ?? {}).filter((moe) => moe !== null)
      : [];

  const highestMoe =
    validMoes.length > 0 && validMoes.every(Number.isFinite) ? Math.max(...validMoes) : null;

  return (
    <section className="equity-panel" aria-label="Equity calculation">
      <h2>Equity Calculation</h2>

      <label htmlFor="target-moe">Target MOE</label>
      <input
        id="target-moe"
        type="number"
        min="0"
        max="0.005"
        step="any"
        value={targetMoeInput}
        onChange={(event) => setTargetMoeInput(event.target.value)}
      />

      <button
        onClick={() => onCalculate(targetMoe)}
        disabled={!hasSession || isBusy || !isValidTargetMoe}
      >
        Calculate Equity
      </button>

      <button onClick={onCancel} disabled={!hasSession || status !== 'running'}>
        Cancel Equity
      </button>

      <p>Status: {status}</p>

      {(isProgress || isResult) && details && (
        <div>
          {isResult && <h3>Results</h3>}

          {details.simulations !== null && (
            <p>Simulations: {details.simulations.toLocaleString()}</p>
          )}

          <ul>
            {Object.entries(details.equities)
              .sort(([a], [b]) => Number(a) - Number(b))
              .map(([playerId, equity]) => {
                const moe = details.margins_of_error?.[playerId];

                return (
                  <li key={playerId}>
                    Player {Number(playerId) + 1}: {(equity * 100).toFixed(2)}%
                    {moe != null && Number.isFinite(moe) && (
                      <span> ±{(moe * 100).toFixed(3)} pp</span>
                    )}
                  </li>
                );
              })}
          </ul>

          {isProgress && highestMoe !== null && (
            <p>Highest MOE: ±{(highestMoe * 100).toFixed(3)} pp</p>
          )}
        </div>
      )}

      {error !== null && <p role="alert">{error}</p>}
    </section>
  );
}
