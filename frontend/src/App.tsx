import { useState } from 'react';
import { createSession, changeBoard } from './api/sessions';
import { BoardInput } from './components/BoardInput';
import { PlayerRangeInput } from './components/PlayerRangeInput';
import { EquityPanel } from './components/EquityPanel';
import { useEquitySocket } from './hooks/useEquitySocket';
import { usePlayerRanges } from './hooks/usePlayerRanges';

/**
 * Coordinate the frontend's session, player, board, and equity workflows.
 *
 * Owns application-level state for the active session, community
 * board input, PLO mode, loading indicators, and feedback messages.
 *
 * Delegates player range management to usePlayerRanges() and equity
 * calculation state to useEquitySocket().
 *
 * Connects these hooks to presentational components and coordinates
 * operations that affect multiple parts of the application.
 *
 * @remarks
 * Session creation and board changes are handled directly through
 * REST API clients. Individual components receive state and actions
 * through props rather than communicating with the backend.
 */
function App() {
  const [board, setBoard] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [boardMessage, setBoardMessage] = useState<string | null>(null);
  const [mode, setMode] = useState<4 | 5>(4);
  const { equityStatus, isConnected, equityMessage, startEquity, cancelEquity } =
    useEquitySocket(sessionId);
  const { players, applyBoardSnapshots, updatePlayer, handleAddRange, handleDeleteRange } =
    usePlayerRanges(sessionId, mode);

  /**
   * Create a backend session and store its identifier.
   *
   * Updates the application loading and error states while the
   * request is pending.
   *
   * A successful response updates sessionId, allowing the
   * session-dependent hooks to establish backend communication.
   *
   * Errors are displayed through application-level state.
   */
  async function handleCreateSession() {
    setIsLoading(true);
    setError(null);

    try {
      const session = await createSession();
      setSessionId(session.session_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsLoading(false);
    }
  }

  /**
   * Apply the current board input to the backend session.
   *
   * Trims the entered board expression and converts an empty
   * string to null, which requests clearing the board.
   *
   * On success, synchronizes player hand counts with the backend
   * response and displays confirmation feedback.
   *
   * On failure, preserves the existing hand counts and displays
   * the returned error.
   *
   * @remarks
   * The input is not validated locally. Board notation validation
   * and street transitions are handled by the backend.
   */
  async function handleApplyBoard() {
    if (sessionId === null) return;

    setIsLoading(true);
    setError(null);
    setBoardMessage(null);

    const boardValue = board.trim() || null;

    try {
      const result = await changeBoard(sessionId, boardValue);
      setBoardMessage(boardValue === null ? 'Board cleared' : 'Board updated successfully');
      applyBoardSnapshots(result.players);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main>
      <h1>PLO RangeLab</h1>

      <button onClick={handleCreateSession} disabled={isLoading || sessionId !== null}>
        {isLoading && sessionId === null ? 'Creating...' : 'Create Session'}
      </button>

      <p>Session: {sessionId ?? 'Not created'}</p>

      <label htmlFor="mode">Game mode</label>
      {/* Prevent switching PLO modes while player ranges exist
          or a player range operation is in progress. */}
      <select
        id="mode"
        value={mode}
        onChange={(event) => setMode(Number(event.target.value) as 4 | 5)}
        disabled={players.some((player) => player.hasAppliedRange || player.isLoading)}
      >
        <option value={4}>PLO4</option>
        <option value={5}>PLO5</option>
      </select>

      <BoardInput
        board={board}
        onBoardChange={setBoard}
        onApply={handleApplyBoard}
        disabled={isLoading || sessionId === null}
        isLoading={isLoading}
      />

      {error && <p role="alert">{error}</p>}
      {boardMessage && <p>{boardMessage}</p>}

      <section>
        <h2>Players</h2>

        {players.map((player, index) => (
          <PlayerRangeInput
            key={index}
            playerId={index}
            range={player.range}
            numberOfHands={player.numberOfHands}
            onRangeChange={(value) => updatePlayer(index, { range: value })}
            onAdd={() => handleAddRange(index)}
            onDelete={() => handleDeleteRange(index)}
            disabled={sessionId === null || player.isLoading}
            isLoading={player.isLoading}
            error={player.error}
            hasAppliedRange={player.hasAppliedRange}
          />
        ))}
      </section>

      <EquityPanel
        mode={mode}
        status={equityStatus}
        message={equityMessage}
        isConnected={isConnected}
        onCalculate={startEquity}
        onCancel={cancelEquity}
      />
    </main>
  );
}

export default App;
