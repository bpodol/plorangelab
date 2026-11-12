export type JobStatus = 'pending' | 'running' | 'completed' | 'cancelled' | 'failed';

/**
 * Describe the calculation method and diagnostics of an equity result.
 *
 * Includes Monte Carlo statistics, sampling information, and the
 * reason execution stopped. Fields that do not apply to the
 * calculation method are represented by null.
 */
export interface EquityDetails {
  method: 'exact' | 'monte-carlo';
  margins_of_error: (number | null)[] | null;
  simulations: number | null;
  attempts: number | null;
  stop_reason: 'target_reached' | 'max_simulations' | 'cancelled' | 'attempt_limit' | null;
  sampler: 'rejection' | 'importance' | null;
  effective_samples: number | null;
}

/**
 * Represent messages received from the equity WebSocket.
 *
 * The "type" field discriminates between job acknowledgements,
 * progress updates, terminal results, cancellation, and errors.
 *
 * A "result" message may contain a stop_reason of "cancelled"
 * when the calculation stops cooperatively.
 */
export type EquityMessage =
  | {
      type: 'started';
      job_id: string;
    }
  | {
      type: 'resumed';
      job_id: string;
      status: JobStatus;
    }
  | {
      type: 'progress';
      job_id: string;
      simulations: number;
      margins_of_error: (number | null)[];
    }
  | {
      type: 'result';
      job_id: string;
      /** Player slot indices serialized as JSON object keys. */
      equities: Record<string, number>;
      details: EquityDetails;
    }
  | {
      type: 'cancel_requested';
    }
  | {
      type: 'cancelled';
      job_id: string;
    }
  | {
      type: 'error';
      detail: string;
    };

/**
 * Open a WebSocket connection for a session's equity calculations.
 *
 * Uses the current page's hostname and port, selecting ws://
 * or wss:// according to the page protocol.
 *
 * @param sessionId - Identifier of the target backend session.
 * @returns The newly created WebSocket instance.
 *
 * @remarks
 * This function only creates the connection. Reconnection,
 * job resumption, and message handling are managed by
 * useEquitySocket().
 *
 * The backend WebSocket endpoint must be reachable through
 * the same host as the frontend.
 */
export function connectEquityWebSocket(sessionId: string): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

  const url = `${protocol}//${window.location.host}/api/sessions/${sessionId}/equity/ws`;

  return new WebSocket(url);
}
