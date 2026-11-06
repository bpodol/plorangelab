import { useEffect, useRef, useState } from 'react';
import { connectEquityWebSocket, type EquityMessage } from '../api/equity';
import type { Mode } from '../api/sessions';

export type EquityStatus = 'idle' | 'running' | 'completed' | 'error' | 'cancelling' | 'cancelled';

export function useEquitySocket(sessionId: string | null) {
  const socketRef = useRef<WebSocket | null>(null);

  const [equityStatus, setEquityStatus] = useState<EquityStatus>('idle');
  const [equityMessage, setEquityMessage] = useState<EquityMessage | null>(null);

  const [isConnected, setIsConnected] = useState(false);

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

    function connect() {
      if (disposed) return;
      const socket = connectEquityWebSocket(activeSessionId);
      socketRef.current = socket;

      socket.onopen = () => {
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
              setEquityStatus('completed');
              jobRef.current = null;
              console.log('Equities:', message.equities);
              console.log('Details:', message.details);
              break;

            case 'cancel_requested':
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
