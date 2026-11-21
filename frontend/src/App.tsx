import { useState, useEffect, useCallback, useRef } from 'react';
import { createSession, changeBoard, changeMode } from './api/sessions';
import { BoardInput } from './components/BoardInput';
import { PlayerRangeInput } from './components/PlayerRangeInput';
import { EquityPanel } from './components/EquityPanel';
import { useEquitySocket } from './hooks/useEquitySocket';
import { usePlayerRanges } from './hooks/usePlayerRanges';
import { FilterPanel } from './components/FilterPanel';

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
 * Automatically creates a backend session on mount and allows
 * manual retries if initialization fails.
 *
 * Session creation and board application use independent loading
 * states and communicate through REST API clients.
 *
 * Child components receive state and callbacks through props
 * rather than communicating directly with the backend.
 */
function App() {
  const [board, setBoard] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isCreatingSession, setIsCreatingSession] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [boardMessage, setBoardMessage] = useState<string | null>(null);
  const [isApplyingBoard, setIsApplyingBoard] = useState(false);
  const [mode, setMode] = useState<4 | 5>(4);
  const [isChangingMode, setIsChangingMode] = useState(false);
  const { equityStatus, isConnected, equityMessage, startEquity, cancelEquity } =
    useEquitySocket(sessionId);
  const {
    players,
    applyBoardSnapshots,
    updatePlayer,
    handleAddRange,
    handleDeleteRange,
    addFilter,
    addGroup,
    toggleFilterActive,
    removeFilter,
    clearAllFilters,
  } = usePlayerRanges(sessionId);

  /**
   * Create a backend session on application startup or manual retry.
   *
   * Tracks session creation independently of board updates.
   * On success, stores the session ID to enable backend communication.
   * On failure, stores the error for display with a Retry button.
   */
  const handleCreateSession = useCallback(async () => {
    setIsCreatingSession(true);
    setError(null);

    try {
      const session = await createSession();
      setSessionId(session.session_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsCreatingSession(false);
    }
  }, []);

  /**
   * Automatically create a session when the app mounts.
   *
   * The ref prevents duplicate requests caused by React Strict Mode
   * re-running effects in development. Failed requests can still
   * be retried manually through handleCreateSession().
   */
  const sessionRequested = useRef(false);

  useEffect(() => {
    if (sessionRequested.current) return;

    sessionRequested.current = true;
    void handleCreateSession();
  }, [handleCreateSession]);

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

    setIsApplyingBoard(true);
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
      setIsApplyingBoard(false);
    }
  }

  async function handleChangeMode(nextMode: 4 | 5) {
    if (sessionId === null) return;

    setIsChangingMode(true);
    setError(null);

    try {
      await changeMode(sessionId, nextMode);
      setMode(nextMode);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsChangingMode(false);
    }
  }

  return (
    <main>
      <h1>PLO RangeLab</h1>
      {sessionId === null && !isCreatingSession && error && (
        <section className="session-panel">
          <p role="alert">{error}</p>

          <button type="button" onClick={() => void handleCreateSession()}>
            Retry
          </button>
        </section>
      )}
      <section>
        <label htmlFor="mode">Game mode</label>
        {/* Prevent switching PLO modes while player ranges exist
            or a player range operation is in progress. */}
        <select
          id="mode"
          value={mode}
          onChange={(event) => handleChangeMode(Number(event.target.value) as 4 | 5)}
          disabled={
            players.some((player) => player.hasAppliedRange || player.isLoading) ||
            sessionId === null ||
            isChangingMode
          }
        >
          <option value={4}>PLO4</option>
          <option value={5}>PLO5</option>
        </select>
      </section>
      <BoardInput
        board={board}
        onBoardChange={setBoard}
        onApply={handleApplyBoard}
        disabled={isApplyingBoard || sessionId === null}
        isLoading={isApplyingBoard}
      />

      {sessionId !== null && error && <p role="alert">{error}</p>}
      {boardMessage && <p>{boardMessage}</p>}

      <EquityPanel
        mode={mode}
        status={equityStatus}
        message={equityMessage}
        isConnected={isConnected}
        onCalculate={startEquity}
        onCancel={cancelEquity}
      />

      <section className="players-layout">
        {/* Left column: all player range inputs */}
        <div className="players-layout__ranges">
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
              disabled={sessionId === null || player.isLoading || isChangingMode}
              isLoading={player.isLoading}
              error={player.error}
              hasAppliedRange={player.hasAppliedRange}
            />
          ))}
        </div>

        {/* Right column: only players with applied ranges */}
        <div className="players-layout__filters">
          <h2>Filters</h2>

          {players.map((player, index) =>
            player.hasAppliedRange ? (
              <FilterPanel
                key={index}
                playerId={index}
                filterSnapshot={player.filterSnapshot}
                disabled={sessionId === null || player.isLoading || isApplyingBoard}
                onAddFilter={addFilter}
                onAddGroup={addGroup}
                onToggleFilterActive={toggleFilterActive}
                onRemoveFilter={removeFilter}
                onClearAllFilters={clearAllFilters}
              />
            ) : null,
          )}
        </div>
      </section>
    </main>
  );
}

export default App;
