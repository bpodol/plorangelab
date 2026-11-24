import { useCallback, useEffect, useRef, useState } from 'react';
import { startEquityStream, cancelEquityCalculation, type EquityStreamEvent } from '../api/equity';

export type EquityStatus =
  'idle' | 'starting' | 'running' | 'cancelling' | 'completed' | 'cancelled' | 'failed';

/**
 * Manage the lifecycle of an equity calculation and its SSE connection.
 *
 * Submits calculations through HTTP, receives progress and terminal
 * results through SSE, and requests cancellation through HTTP.
 *
 * EventSource handles temporary connection loss and reconnection.
 * Closing the stream does not cancel the backend calculation.
 *
 * Changing sessions or unmounting closes the current connection
 * and prevents outstanding asynchronous operations from updating
 * obsolete React state.
 */
export function useEquitySSE(sessionId: string | null) {
  const [equityStatus, setEquityStatus] = useState<EquityStatus>('idle');
  const [equityMessage, setEquityMessage] = useState<EquityStreamEvent | null>(null);
  const [equityError, setEquityError] = useState<string | null>(null);

  const sourceRef = useRef<EventSource | null>(null);
  const jobIdRef = useRef<string | null>(null);

  // Guards against duplicate submissions and cancellation requests.
  const submittingRef = useRef(false);
  const cancellingRef = useRef(false);
  const cancellationRequestedRef = useRef(false);

  // Invalidate callbacks and requests belonging to old sessions.
  const generationRef = useRef(0);

  useEffect(() => {
    generationRef.current += 1;

    sourceRef.current?.close();
    sourceRef.current = null;
    jobIdRef.current = null;

    submittingRef.current = false;
    cancellingRef.current = false;
    cancellationRequestedRef.current = false;

    setEquityStatus('idle');
    setEquityMessage(null);
    setEquityError(null);

    return () => {
      generationRef.current += 1;

      sourceRef.current?.close();
      sourceRef.current = null;
      jobIdRef.current = null;

      submittingRef.current = false;
      cancellingRef.current = false;
      cancellationRequestedRef.current = false;
    };
  }, [sessionId]);

  /**
   * Submit a new calculation and subscribe to its SSE updates.
   */
  const startEquity = useCallback(
    async (targetMOE: number): Promise<void> => {
      if (sessionId === null || submittingRef.current || jobIdRef.current !== null) {
        return;
      }

      submittingRef.current = true;

      const generation = generationRef.current;
      let terminalReceived = false;
      let handlingStreamFailure = false;

      // Close an unusable stream, request cancellation, then release the job.
      // Both malformed messages and permanently closed connections use this path.
      const handleStreamFailure = async (
        failedJobId: string,
        failedSource: EventSource,
        reason: string,
      ): Promise<void> => {
        if (
          handlingStreamFailure ||
          generationRef.current !== generation ||
          jobIdRef.current !== failedJobId ||
          sourceRef.current !== failedSource
        ) {
          return;
        }

        handlingStreamFailure = true;
        failedSource.close();
        setEquityStatus('cancelling');
        setEquityError(null);

        let cancellationAccepted = false;
        try {
          await cancelEquityCalculation(sessionId, failedJobId);
          cancellationAccepted = true;
        } catch (err) {
          console.error('Failed to cancel equity job after SSE failure:', err);
        }

        // Ignore cancellation responses from an obsolete session or job.
        if (
          generationRef.current !== generation ||
          jobIdRef.current !== failedJobId ||
          sourceRef.current !== failedSource
        ) {
          return;
        }

        sourceRef.current = null;
        jobIdRef.current = null;
        cancellationRequestedRef.current = false;

        setEquityStatus('failed');
        setEquityError(
          cancellationAccepted
            ? `${reason}. Cancellation requested.`
            : `${reason}. Could not request cancellation.`,
        );
      };

      setEquityStatus('starting');
      setEquityMessage(null);
      setEquityError(null);

      cancellationRequestedRef.current = false;

      try {
        const { jobId: newJobId, source } = await startEquityStream(
          sessionId,
          targetMOE,
          (message) => {
            if (generationRef.current !== generation) return;

            setEquityMessage(message);

            switch (message.status) {
              case 'pending':
              case 'running':
                if (!cancellationRequestedRef.current) {
                  setEquityStatus('running');
                }
                break;

              case 'completed':
              case 'cancelled':
              case 'failed':
                terminalReceived = true;

                setEquityStatus(message.status);

                if (message.status === 'failed') {
                  setEquityError(message.error ?? 'Equity calculation failed');
                } else {
                  setEquityError(null);
                }

                jobIdRef.current = null;
                sourceRef.current = null;
                cancellationRequestedRef.current = false;
                break;
            }
          },
          (error, failedJobId, failedSource) => {
            void handleStreamFailure(failedJobId, failedSource, error.message);
          },
        );

        // The session may have changed while POST was pending.
        if (generationRef.current !== generation) {
          source.close();
          return;
        }

        // A very fast job may finish before this promise resolves.
        if (terminalReceived) {
          source.close();
          return;
        }

        jobIdRef.current = newJobId;
        sourceRef.current = source;

        source.addEventListener('open', () => {
          if (generationRef.current !== generation || sourceRef.current !== source) {
            return;
          }

          if (!cancellationRequestedRef.current) {
            setEquityStatus('running');
          }
        });

        // The connection might already be open.
        if (source.readyState === EventSource.OPEN) {
          setEquityStatus('running');
        }
      } catch (err) {
        if (generationRef.current !== generation) return;

        setEquityStatus('failed');
        setEquityError(err instanceof Error ? err.message : 'Failed to submit equity calculation');
      } finally {
        if (generationRef.current === generation) {
          submittingRef.current = false;
        }
      }
    },
    [sessionId],
  );

  /**
   * Request cooperative cancellation.
   *
   * The SSE connection remains open until the worker reports
   * a terminal status. HTTP 202 only confirms the request.
   */
  const cancelEquity = useCallback(async (): Promise<void> => {
    const activeJobId = jobIdRef.current;

    if (sessionId === null || activeJobId === null || cancellingRef.current) {
      return;
    }

    cancellingRef.current = true;
    const generation = generationRef.current;

    try {
      await cancelEquityCalculation(sessionId, activeJobId);

      if (generationRef.current !== generation || jobIdRef.current !== activeJobId) {
        return;
      }

      cancellationRequestedRef.current = true;
      setEquityStatus('cancelling');
      setEquityError(null);
    } catch (err) {
      if (generationRef.current === generation && jobIdRef.current === activeJobId) {
        setEquityError(err instanceof Error ? err.message : 'Failed to request cancellation');
      }
    } finally {
      if (generationRef.current === generation) {
        cancellingRef.current = false;
      }
    }
  }, [sessionId]);

  return {
    equityStatus,
    equityMessage,
    equityError,
    startEquity,
    cancelEquity,
  };
}
