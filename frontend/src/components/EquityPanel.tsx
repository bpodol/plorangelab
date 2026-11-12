import { useState } from 'react';
import type { EquityMessage } from '../api/equity';
import type { Mode } from '../types';
import type { EquityStatus } from '../hooks/useEquitySocket';

interface EquityPanelProps {
  mode: Mode;
  status: EquityStatus;
  message: EquityMessage | null;
  isConnected: boolean;
  onCalculate: (mode: Mode, targetMoe: number) => void;
  onCancel: () => void;
}

/**
 * Render equity calculation controls, progress, and results.
 *
 * Maintains the target margin of error input locally and validates
 * it before allowing a calculation to start.
 *
 * Receives connection status, calculation status, and WebSocket
 * messages from the parent. Calculation and cancellation requests
 * are delegated through callbacks.
 *
 * Displays simulation progress, the highest reported margin of
 * error, and per-player equity results when available.
 *
 * @remarks
 * This component does not manage the WebSocket connection or
 * perform equity calculations.
 */
export function EquityPanel({
  mode,
  status,
  message,
  isConnected,
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

  // Show the highest available player MOE as the current precision
  // indicator; null entries represent unavailable estimates.
  const validMoes =
    message?.type === 'progress' ? message.margins_of_error.filter((moe) => moe !== null) : [];

  const highestMoe =
    validMoes.length > 0 && validMoes.every(Number.isFinite) ? Math.max(...validMoes) : null;

  return (
    <section aria-label="Equity calculation">
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
        onClick={() => onCalculate(mode, targetMoe)}
        disabled={!isConnected || status === 'running' || !isValidTargetMoe}
      >
        Calculate Equity
      </button>

      <button onClick={onCancel} disabled={!isConnected || status !== 'running'}>
        Cancel Equity
      </button>

      <p>Status: {status}</p>

      {message?.type === 'progress' && (
        <div>
          <p>Simulations: {message.simulations.toLocaleString()}</p>

          {highestMoe !== null && <p>Highest MOE: ±{(highestMoe * 100).toFixed(3)} pp</p>}
        </div>
      )}

      {message?.type === 'result' && (
        <div>
          <h3>Results</h3>

          <ul>
            {Object.entries(message.equities)
              .sort(([a], [b]) => Number(a) - Number(b))
              .map(([playerId, equity]) => {
                const moe = message.details.margins_of_error?.[Number(playerId)];

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
          {message.details.simulations !== null && (
            <p>Simulations: {message.details.simulations.toLocaleString()}</p>
          )}
        </div>
      )}

      {message?.type === 'error' && <p role="alert">{message.detail}</p>}
    </section>
  );
}
