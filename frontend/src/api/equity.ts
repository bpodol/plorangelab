import { apiRequest } from './client';

export type JobStatus = 'pending' | 'running' | 'completed' | 'cancelled' | 'failed';

export interface EquityResponse {
  status: 'starting' | 'cancelling';
  job_id: string;
}

export interface EquityDetails {
  equities: Record<string, number>;
  method: 'exact' | 'monte-carlo';
  margins_of_error: Record<string, number | null> | null;
  simulations: number | null;
  attempts: number | null;
  stop_reason: 'target_reached' | 'max_simulations' | 'cancelled' | 'attempt_limit' | null;
  sampler: 'rejection' | 'importance' | null;
  effective_samples: number | null;
}

export interface EquityStreamEvent {
  job_id: string;
  status: JobStatus;
  error: string | null;
  details: EquityDetails | null;
}

export function submitEquityCalculation(
  sessionId: string,
  targetMOE: number,
): Promise<EquityResponse> {
  return apiRequest<EquityResponse>(`/api/sessions/${sessionId}/equity`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      target_moe: targetMOE,
    }),
  });
}

export function cancelEquityCalculation(sessionId: string, jobId: string): Promise<EquityResponse> {
  return apiRequest<EquityResponse>(`/api/sessions/${sessionId}/equity/${jobId}/cancel`, {
    method: 'POST',
  });
}

/**
 * Submit an equity calculation and subscribe to its SSE stream.
 *
 * The backend executes the calculation independently of the connection.
 * EventSource automatically reconnects after temporary disconnections.
 *
 * The stream closes when a terminal result is received.
 *
 * @param sessionId - Active backend session ID.
 * @param targetMOE - Requested margin of error.
 * @param onUpdate - Callback invoked for each progress or terminal event.
 * @param onStreamError - Reports unrecoverable connection or message parsing errors.
 * @returns The job ID and EventSource instance for lifecycle management.
 */
export async function startEquityStream(
  sessionId: string,
  targetMOE: number,
  onUpdate: (message: EquityStreamEvent) => void,
  onStreamError: (error: Error, jobId: string, source: EventSource) => void,
): Promise<{ jobId: string; source: EventSource }> {
  // Submit the calculation. The backend responds with HTTP 202
  // and a job ID without waiting for the calculation to finish.
  const response = await submitEquityCalculation(sessionId, targetMOE);
  const jobId = response.job_id;

  // Establish an SSE connection to the existing calculation.
  const source = new EventSource(`/api/sessions/${sessionId}/equity/${jobId}/events`);

  source.onmessage = (event: MessageEvent<string>) => {
    let message: EquityStreamEvent;

    try {
      message = JSON.parse(event.data) as EquityStreamEvent;
    } catch {
      source.close();
      onStreamError(new Error('Invalid equity SSE message received'), jobId, source);
      return;
    }

    // Forward the event to the consumer (typically a React hook).
    onUpdate(message);

    // The backend closes the stream after sending a terminal event,
    // but EventSource would otherwise attempt to reconnect.
    if (
      message.status === 'completed' ||
      message.status === 'cancelled' ||
      message.status === 'failed'
    ) {
      source.close();
    }
  };

  source.onerror = () => {
    if (source.readyState === EventSource.CONNECTING) {
      console.warn('Equity SSE connection lost. Reconnecting...');
    } else if (source.readyState === EventSource.CLOSED) {
      onStreamError(new Error('Equity SSE connection closed unexpectedly'), jobId, source);
    }
  };

  return { jobId, source };
}
