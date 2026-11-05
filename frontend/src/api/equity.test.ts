import { afterEach, expect, test, vi } from 'vitest';
import { connectEquityWebSocket } from './equity';

class MockWebSocket {
  readonly url: string;

  constructor(url: string) {
    this.url = url;
  }
}

afterEach(() => {
  vi.unstubAllGlobals();
});

test.each([
  ['http:', 'localhost:5173', 'ws:'],
  ['https:', 'plorangelab.com', 'wss:'],
])('creates a WebSocket for %s', (pageProtocol, host, socketProtocol) => {
  vi.stubGlobal('window', {
    location: {
      protocol: pageProtocol,
      host,
    },
  });

  vi.stubGlobal('WebSocket', MockWebSocket);

  const socket = connectEquityWebSocket('session-123');

  expect(socket).toBeInstanceOf(MockWebSocket);

  expect(socket.url).toBe(`${socketProtocol}//${host}/api/sessions/session-123/equity/ws`);
});
