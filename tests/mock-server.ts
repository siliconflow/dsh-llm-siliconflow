import { createServer } from 'node:http'
import type { IncomingMessage, Server, ServerResponse } from 'node:http'

/** One scripted behavior for the next request the mock server receives. */
export type Behavior =
  | { kind: 'sse'; events: string[]; delayMs?: number }
  | { kind: 'http-error'; status: number; body: string; contentType?: string; headers?: Record<string, string> }
  | { kind: 'close-early'; events: string[] }

/** One scripted reply for the mock model-listing server (GET /models). */
export interface ModelsBehavior {
  status?: number
  /** Complete reply body; sent with a matching `content-length` (a declared length). */
  body?: string
  /** Chunked reply; sent without a declared length, as a streamed body arrives. */
  chunks?: string[]
  /** Keep a chunked reply open this long after the last chunk, so a caller's abort lands mid-read. */
  holdOpenMs?: number
  contentType?: string
  headers?: Record<string, string>
}

export interface MockServer {
  url: string
  /** Bodies of received requests, in order. */
  requests: unknown[]
  /** Header bags of received requests, in order (parallel to `requests`). */
  headers: IncomingMessage['headers'][]
  script: Behavior[]
  close(): Promise<void>
}

export interface MockModelsServer {
  url: string
  /** Requested URLs (path + query), in order. */
  paths: string[]
  /** Header bags of received requests, in order (parallel to `paths`). */
  headers: IncomingMessage['headers'][]
  close(): Promise<void>
}

const servers: Server[] = []

/** Close every server opened since the last call; run from each spec's afterEach. */
export async function closeMockServers(): Promise<void> {
  await Promise.all(servers.splice(0).map(server => new Promise(resolve => server.close(resolve))))
}

/** A minimal complete text generation, reused by request-shape assertions. */
export const textEvents = [
  '{"choices":[{"delta":{"role":"assistant","content":null,"reasoning_content":""}}]}',
  '{"choices":[{"delta":{"content":"hello"}}]}',
  '{"choices":[{"delta":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}',
  '[DONE]',
]

/** Local chat-completions stand-in: replays scripted behaviors per request. */
export async function mockServer(script: Behavior[]): Promise<MockServer> {
  const requests: unknown[] = []
  const headers: IncomingMessage['headers'][] = []
  const server = createServer((request: IncomingMessage, response: ServerResponse) => {
    let body = ''
    request.on('data', (chunk: Buffer) => { body += chunk.toString('utf8') })
    request.on('end', () => {
      requests.push(JSON.parse(body))
      headers.push(request.headers)
      const behavior = script.shift()
      if (!behavior) {
        response.writeHead(500).end('mock script exhausted')
        return
      }
      if (behavior.kind === 'http-error') {
        response.writeHead(behavior.status, {
          'content-type': behavior.contentType ?? 'application/json',
          ...behavior.headers,
        })
        response.end(behavior.body)
        return
      }
      response.writeHead(200, { 'content-type': 'text/event-stream' })
      const write = (index: number): void => {
        if (index >= behavior.events.length) {
          if (behavior.kind === 'sse') response.end()
          else response.destroy() // close-early: drop the socket mid-stream
          return
        }
        response.write(`data: ${behavior.events[index]}\n\n`)
        setTimeout(() => { write(index + 1) }, behavior.kind === 'sse' ? behavior.delayMs ?? 0 : 5)
      }
      write(0)
    })
  })
  servers.push(server)
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('no port')
  return {
    url: `http://127.0.0.1:${address.port}`,
    requests,
    headers,
    script,
    close: () => new Promise(resolve => server.close(() => { resolve() })),
  }
}

/** Local `GET /models` stand-in: replies to every request with the next scripted behavior. */
export async function mockModelsServer(script: ModelsBehavior[]): Promise<MockModelsServer> {
  const paths: string[] = []
  const headers: IncomingMessage['headers'][] = []
  const server = createServer((request: IncomingMessage, response: ServerResponse) => {
    paths.push(request.url ?? '')
    headers.push(request.headers)
    request.resume()
    request.on('end', () => {
      const behavior = script.shift()
      if (!behavior) {
        response.writeHead(500).end('mock script exhausted')
        return
      }
      if (behavior.chunks !== undefined) {
        // No declared length: the byte ceiling holds on what is actually read.
        response.writeHead(behavior.status ?? 200, {
          'content-type': behavior.contentType ?? 'application/json',
          ...behavior.headers,
        })
        for (const chunk of behavior.chunks) response.write(chunk)
        if (behavior.holdOpenMs === undefined) { response.end(); return }
        setTimeout(() => { response.end() }, behavior.holdOpenMs)
        return
      }
      const body = behavior.body ?? '{}'
      response.writeHead(behavior.status ?? 200, {
        'content-type': behavior.contentType ?? 'application/json',
        'content-length': String(Buffer.byteLength(body)),
        ...behavior.headers,
      })
      response.end(body)
    })
  })
  servers.push(server)
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('no port')
  return {
    url: `http://127.0.0.1:${address.port}`,
    paths,
    headers,
    close: () => new Promise(resolve => server.close(() => { resolve() })),
  }
}
