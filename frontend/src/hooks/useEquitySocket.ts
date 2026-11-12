import { useEffect, useRef, useState } from 'react';
import { connectEquityWebSocket, type EquityMessage } from '../api/equity';
import type { Mode } from '../types';

export type EquityStatus = 'idle' | 'running' | 'completed' | 'error' | 'cancelling' | 'cancelled';

/**
 * Manage the WebSocket connection and equity calculation state.
 *
 * Opens a connection for the active session, processes equity
 * messages, and exposes the latest message, calculation status,
 * connection state, and start/cancel actions.
 *
 * Automatically reconnects after connection loss using capped
 * exponential backoff. When a known job is still tracked, the
 * hook requests its resumption after reconnecting.
 *
 * @param sessionId - Active backend session ID, or null when absent.
 *
 * @remarks
 * The WebSocket connection is recreated when sessionId changes.
 * Cleanup closes the connection and cancels pending retries,
 * but does not cancel the backend equity calculation.
 *
 * Job identifiers are stored in a ref and do not survive a
 * component remount or page reload.
 */
export function useEquitySocket(sessionId: string | null) {
  const socketRef = useRef<WebSocket | null>(null);

  const [equityStatus, setEquityStatus] = useState<EquityStatus>('idle');
  const [equityMessage, setEquityMessage] = useState<EquityMessage | null>(null);

  const [isConnected, setIsConnected] = useState(false);

  // Track the active backend job across renders and reconnections
  // without triggering a React rerender when its ID changes.
  const jobRef = useRef<string | null>(null);

  useEffect(() => {
    jobRef.current = null;
    setIsConnected(false);
    setEquityStatus('idle');
    setEquityMessage(null);

    if (sessionId === null) return;

    const activeSessionId = sessionId;

    let disposed = false;
    let retryCount = 0;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    /**
     * Open a WebSocket and register its lifecycle handlers.
     *
     * On connection, resumes the tracked job when one exists.
     * Incoming messages update the hook's equity state.
     *
     * Unexpected disconnections schedule another connection attempt.
     * Handlers ignore sockets that have been replaced or disposed.
     */
    function connect() {
      if (disposed) return;
      const socket = connectEquityWebSocket(activeSessionId);
      socketRef.current = socket;

      socket.onopen = () => {
        // Ignore callbacks from an effect that has been cleaned up or
        // from a socket that is no longer the current connection.
        if (disposed || socketRef.current !== socket) return;
        retryCount = 0;
        setIsConnected(true);
        if (jobRef.current !== null) {
          socket.send(
            JSON.stringify({
              type: 'resume',
              job_id: jobRef.current,
            }),
          );
        }
      };

      socket.onmessage = (event) => {
        if (disposed || socketRef.current !== socket) return;

        try {
          const raw: unknown = JSON.parse(event.data);

          if (
            typeof raw !== 'object' ||
            raw === null ||
            !('type' in raw) ||
            typeof raw.type !== 'string'
          ) {
            throw new Error('Invalid WebSocket message');
          }

          const message = raw as EquityMessage;

          setEquityMessage(message);

          switch (message.type) {
            case 'started':
              setEquityStatus('running');
              jobRef.current = message.job_id;
              console.log('Job started:', message.job_id);
              break;

            case 'resumed':
              setEquityStatus('running');
              console.log('Job resumed:', message.job_id);
              console.log('Current status:', message.status);
              break;

            case 'progress':
              setEquityStatus('running');
              console.log('Simulations:', message.simulations);
              console.log('Margins of error:', message.margins_of_error);
              break;

            case 'result':
              setEquityStatus(
                message.details.stop_reason === 'cancelled' ? 'cancelled' : 'completed',
              );
              jobRef.current = null;
              console.log('Equities:', message.equities);
              console.log('Details:', message.details);
              break;

            case 'cancel_requested':
              setEquityStatus('cancelling');
              console.log('Cancellation requested');
              break;

            case 'cancelled':
              setEquityStatus('cancelled');
              jobRef.current = null;
              break;

            case 'error':
              setEquityStatus('error');
              jobRef.current = null;
              console.error(message.detail);
              break;
          }
        } catch (err) {
          console.error('Failed to process WebSocket message:', err);
        }
      };

      socket.onerror = () => {
        console.error('WebSocket error');
      };

      socket.onclose = () => {
        if (disposed || socketRef.current !== socket) return;

        socketRef.current = null;
        setIsConnected(false);

        // Retry with exponential backoff (1s, 2s, 4s, ...), capped at
        // 10 seconds. A successful connection resets the retry counter.
        const delay = Math.min(1000 * 2 ** retryCount, 10000);
        retryCount++;

        retryTimer = setTimeout(connect, delay);
      };
    }

    connect();

    return () => {
      disposed = true;

      if (retryTimer !== null) {
        clearTimeout(retryTimer);
      }

      const socket = socketRef.current;
      socketRef.current = null;
      socket?.close();
    };
  }, [sessionId]);

  /**
   * Request a new equity calculation through the active WebSocket.
   *
   * Sends the selected PLO mode and target margin of error.
   * Does nothing when the socket is not open.
   *
   * The calculation status and job ID are updated when the
   * backend acknowledges the request with a "started" message.
   *
   * @param mode - PLO variant (4 or 5).
   * @param targetMoe - Requested Monte Carlo margin of error.
   */
  function startEquity(mode: Mode, targetMoe: number) {
    const socket = socketRef.current;

    if (socket?.readyState !== WebSocket.OPEN) return;

    socket.send(
      JSON.stringify({
        type: 'start',
        mode,
        target_moe: targetMoe,
      }),
    );
  }

  /**
   * Request cancellation of the current equity calculation.
   *
   * Sends a cancellation command when the WebSocket is open.
   * Does nothing when disconnected.
   *
   * Cancellation is cooperative: the backend may continue
   * calculating until it observes the cancellation request.
   *
   * This function does not immediately change the local
   * calculation status.
   */
  function cancelEquity() {
    const socket = socketRef.current;
    if (socket?.readyState !== WebSocket.OPEN) return;

    socket.send(
      JSON.stringify({
        type: 'cancel',
      }),
    );
  }

  return {
    equityStatus,
    equityMessage,
    isConnected,
    startEquity,
    cancelEquity,
  };
}
