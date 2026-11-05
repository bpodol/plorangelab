export type JobStatus = 'pending' | 'running' | 'completed' | 'cancelled' | 'failed';

export interface EquityDetails {
  method: 'exact' | 'monte-carlo';
  margins_of_error: (number | null)[] | null;
  simulations: number | null;
  attempts: number | null;
  stop_reason: 'target_reached' | 'max_simulations' | 'cancelled' | 'attempt_limit' | null;
  sampler: 'rejection' | 'importance' | null;
  effective_samples: number | null;
}

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

export function connectEquityWebSocket(sessionId: string): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

  const url = `${protocol}//${window.location.host}/api/sessions/${sessionId}/equity/ws`;

  return new WebSocket(url);
}
