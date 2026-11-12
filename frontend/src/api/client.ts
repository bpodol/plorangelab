/**
 * Execute an HTTP request and parse its JSON response.
 *
 * Converts FastAPI error responses into Error messages, using
 * string details or combined Pydantic validation messages when
 * available. Falls back to the HTTP status for other errors.
 *
 * Returns undefined for HTTP 204 responses.
 *
 * @remarks
 * The generic type T describes the expected response shape but
 * does not validate the received JSON at runtime.
 */
export async function apiRequest<T>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(url, options);

  if (!response.ok) {
    const fallback = `API request failed: ${response.status} ${response.statusText}`;

    let errorBody: unknown;

    try {
      errorBody = await response.json();
    } catch {
      throw new Error(fallback);
    }

    if (typeof errorBody === 'object' && errorBody !== null && 'detail' in errorBody) {
      const detail = errorBody.detail;

      // FastAPI HTTPException: {"detail": "Error message"}
      if (typeof detail === 'string') {
        throw new Error(detail);
      }

      // FastAPI validation errors: {"detail": [{...}, {...}]}
      if (Array.isArray(detail)) {
        const messages = detail
          .filter(
            (item) =>
              typeof item === 'object' &&
              item !== null &&
              'msg' in item &&
              typeof item.msg === 'string',
          )
          .map((item) => item.msg as string);

        if (messages.length > 0) {
          throw new Error(messages.join('; '));
        }
      }
    }

    throw new Error(fallback);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}
