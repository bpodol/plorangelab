import { useEffect, useRef, useState } from 'react';
import { connectEquityWebSocket, type EquityMessage } from '../api/equity';
import type { Mode } from '../api/sessions';

type EquityStatus = 'idle' | 'running' | 'completed' | 'error' | 'cancelled';

export function useEquitySocket(sessionId: string | null) {
  const socketRef = useRef<WebSocket | null>(null);

  const [equityStatus, setEquityStatus] = useState<EquityStatus>('idle');
  const [equityMessage, setEquityMessage] = useState<EquityMessage | null>(null);

  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    setIsConnected(false);
    setEquityStatus('idle');
    setEquityMessage(null);

    if (sessionId === null) return;

    const socket = connectEquityWebSocket(sessionId);
    socketRef.current = socket;

    socket.onopen = () => {
      if (socketRef.current !== socket) return;

      setIsConnected(true);
      console.log('WebSocket connected');
    };

    socket.onmessage = (event) => {
      if (socketRef.current !== socket) return;

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
            console.log('Job started:', message.job_id);
            break;

          case 'resumed':
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
            console.log('Equities:', message.equities);
            console.log('Details:', message.details);
            break;

          case 'cancel_requested':
            console.log('Cancellation requested');
            break;

          case 'cancelled':
            setEquityStatus('cancelled');
            break;

          case 'error':
            setEquityStatus('error');
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
      if (socketRef.current === socket) {
        socketRef.current = null;
        setIsConnected(false);
      }
    };

    return () => {
      if (socketRef.current === socket) {
        socketRef.current = null;
      }
      socket.close();
    };
  }, [sessionId]);

  function startEquity(mode: Mode) {
    const socket = socketRef.current;

    if (socket?.readyState !== WebSocket.OPEN) return;

    socket.send(
      JSON.stringify({
        type: 'start',
        mode,
        target_moe: 0.002,
      }),
    );
  }

  return {
    equityStatus,
    equityMessage,
    isConnected,
    startEquity,
  };
}
