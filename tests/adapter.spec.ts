import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import { createLaunchEnvironmentSnapshot } from '@deepseek-ai/dsh-launch-environment'
import LlmRuntime, { createUserMessage,
  CONTEXT_WINDOW_EXCEEDED_CODE,
  ProviderRequestId,
  QUOTA_EXCEEDED_CODE,
  ReasoningEffortId,
  userAgent,
} from '@deepseek-ai/dsh-llm'
import { MAX_TIMER_DELAY_MS } from '@deepseek-ai/dsh-timeout'
import { getOrCreateAnonymousUserId, type AnonymousUserId } from '@deepseek-ai/dsh-anonymous-user-id'
import { SessionId } from '@deepseek-ai/dsh-session'
import * as LlmSiliconFlow from '../src/index.ts'
import { DISCOVERY_TTL_MS, SiliconFlowAdapter, resolveAdapterOptions } from '../src/index.ts'
import { httpErrorCode } from '../src/adapter.ts'
import { assemble } from './assemble.ts'
import { closeMockServers, mockModelsServer, mockServer, textEvents } from './mock-server.ts'
import type { Behavior } from './mock-server.ts'

const TEST_USER_ID = '00000000-0000-4000-8000-000000000001' as AnonymousUserId
const MODEL = 'deepseek-ai/DeepSeek-V4-Flash'
let testHome: string

beforeEach(() => {
  testHome = mkdtempSync(join(tmpdir(), 'dsh-llm-siliconflow-'))
  vi.stubEnv('DSH_HOME', testHome)
})

afterEach(async () => {
  await closeMockServers()
  vi.unstubAllEnvs()
  vi.useRealTimers()
  rmSync(testHome, { recursive: true, force: true })
})

async function harness(baseURL: string, config: object = {}) {
  // Configuration carries only the reference; the key comes from the
  // environment, which is the whole credential plane without a mounted seam.
  vi.stubEnv('SILICONFLOW_API_KEY', 'test-key')
  const ctx = new Context()
  await ctx.plugin(LlmRuntime)
  await ctx.plugin(LlmSiliconFlow, { baseURL, ...config })
  return ctx
}

/** Direct adapter over the plugin's real resolve step, with a static key. */
function adapterOf(config: Partial<LlmSiliconFlow.Config> & { apiKey?: string } = {}): SiliconFlowAdapter {
  const { apiKey, ...rest } = config
  return new SiliconFlowAdapter({
    options: () => resolveAdapterOptions(rest),
    resolveApiKey: () => Promise.resolve(apiKey ?? 'k'),
    resolveUserId: () => TEST_USER_ID,
  })
}

describe('SiliconFlowAdapter against a mock server', () => {
  it('streams a text generation end to end through the assembler', async () => {
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const ctx = await harness(server.url)

    const result = await assemble(ctx, {
      model: MODEL,
      messages: [createUserMessage({
        content: [{ type: 'text', text: 'hi' }],
        source: { kind: 'plugin', plugin: 'test' },
      })],
    })
    expect(result.message.content).toEqual([{ type: 'text', text: 'hello' }])
    expect(result.finish).toEqual({ kind: 'stop' })
    expect(result.usage).toEqual({ inputTokens: 3, outputTokens: 1 })

    // The wire request carried the auth header contents we configured, and no
    // thinking/effort fields SiliconFlow does not accept.
    expect(server.requests[0]).toMatchObject({
      model: MODEL,
      max_tokens: 8_192,
      stream: true,
      stream_options: { include_usage: true },
    })
    expect(server.requests[0]).not.toHaveProperty('thinking')
    expect(server.requests[0]).not.toHaveProperty('reasoning_effort')
    // App attribution and request identity are independent wire facts.
    expect(server.headers[0]?.['user-agent']).toBe(userAgent())
    expect(server.headers[0]?.['x-siliconflow-harness-user-id']).toBe(getOrCreateAnonymousUserId())
    expect(server.headers[0]).not.toHaveProperty('x-siliconflow-harness-session-id')
    expect(server.headers[0]).not.toHaveProperty('x-siliconflow-harness-compact')
  })

  it('streams raw chunks through ctx.llm.stream', async () => {
    const server = await mockServer([{ kind: 'sse', events: textEvents, delayMs: 2 }])
    const ctx = await harness(server.url)

    const kinds: string[] = []
    for await (const chunk of ctx.llm.stream({
      provider: 'siliconflow',
      model: MODEL,
      messages: [createUserMessage({
        content: [{ type: 'text', text: 'hi' }],
        source: { kind: 'plugin', plugin: 'test' },
      })],
    })) {
      kinds.push(chunk.type)
    }
    expect(kinds).toEqual(['block-start', 'text-delta', 'block-end', 'usage', 'finish'])
  })

  it('forwards the harness user and session ids for host-side trajectory routing', async () => {
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const ctx = await harness(server.url)

    await assemble(ctx, {
      model: MODEL,
      messages: [createUserMessage({
        content: [{ type: 'text', text: 'hi' }],
        source: { kind: 'plugin', plugin: 'test' },
      })],
      sessionId: SessionId('child-session'),
    })

    expect(server.headers[0]?.['x-siliconflow-harness-session-id']).toBe('child-session')
    expect(server.headers[0]?.['x-siliconflow-harness-user-id']).toBe(getOrCreateAnonymousUserId())
  })

  it('marks the auxiliary compaction call on the wire', async () => {
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const ctx = await harness(server.url)

    await assemble(ctx, {
      model: MODEL,
      messages: [createUserMessage({
        content: [{ type: 'text', text: 'hi' }],
        source: { kind: 'plugin', plugin: 'test' },
      })],
      purpose: 'compaction',
    })

    expect(server.headers[0]?.['x-siliconflow-harness-compact']).toBe('1')
  })

  it('uses the configured maxTokens default and preserves an explicit request cap', async () => {
    const server = await mockServer([
      { kind: 'sse', events: textEvents },
      { kind: 'sse', events: textEvents },
    ])
    const ctx = await harness(server.url, { maxTokens: 32_000 })

    await assemble(ctx, { model: MODEL, messages: [] })
    await assemble(ctx, { model: MODEL, messages: [], maxTokens: 8_192 })

    expect(server.requests[0]).toMatchObject({ max_tokens: 32_000 })
    expect(server.requests[1]).toMatchObject({ max_tokens: 8_192 })
  })

  it.each([
    [401, 'AUTH'],
    [403, 'AUTH'],
    [429, 'RATE_LIMIT'],
    [400, 'INVALID_REQUEST'],
    [500, 'SERVER'],
    [503, 'SERVER'],
  ])('maps HTTP %d to failure code %s with the body message', async (status, code) => {
    const behavior: Behavior = {
      kind: 'http-error',
      status,
      body: JSON.stringify({ error: { message: `failed with ${status}`, type: 't', code: 'c' } }),
    }
    const server = await mockServer([behavior])
    const ctx = await harness(server.url)
    const result = await assemble(ctx, { model: MODEL, messages: [] })
    expect(result.finish).toEqual({
      kind: 'error',
      failure: { message: `failed with ${status}`, code, status },
    })
  })

  it('classifies an HTTP context-window failure with the canonical code', async () => {
    const server = await mockServer([{
      kind: 'http-error',
      status: 400,
      body: JSON.stringify({
        error: {
          message: 'This model maximum context length is 128000 tokens; your input exceeds that limit.',
          type: 'invalid_request_error',
          code: 'context_length_exceeded',
        },
      }),
    }])
    const ctx = await harness(server.url)
    const result = await assemble(ctx, { model: MODEL, messages: [] })
    expect(result.finish).toMatchObject({
      kind: 'error',
      failure: { code: CONTEXT_WINDOW_EXCEEDED_CODE },
    })
  })

  it('retains status, Retry-After seconds, and provider request id as structured facts', async () => {
    const server = await mockServer([{
      kind: 'http-error',
      status: 429,
      body: JSON.stringify({ error: { message: 'slow down' } }),
      headers: { 'retry-after': '2', 'x-request-id': 'req-429' },
    }])
    const ctx = await harness(server.url)
    const result = await assemble(ctx, { model: MODEL, messages: [] })
    expect(result.finish).toEqual({
      kind: 'error',
      failure: {
        message: 'slow down',
        code: 'RATE_LIMIT',
        status: 429,
        providerRetryAfterMs: 2_000,
        requestId: ProviderRequestId('req-429'),
      },
    })
  })

  it('parses a future Retry-After HTTP date', async () => {
    const now = 1_800_000_000_000
    const dateNow = vi.spyOn(Date, 'now').mockReturnValue(now)
    try {
      const server = await mockServer([{
        kind: 'http-error',
        status: 503,
        body: JSON.stringify({ error: { message: 'come back later' } }),
        headers: { 'retry-after': new Date(now + 3_000).toUTCString() },
      }])
      const ctx = await harness(server.url)
      const result = await assemble(ctx, { model: MODEL, messages: [] })
      expect(result.finish).toEqual({
        kind: 'error',
        failure: {
          message: 'come back later',
          code: 'SERVER',
          status: 503,
          providerRetryAfterMs: 3_000,
        },
      })
    } finally {
      dateNow.mockRestore()
    }
  })

  it('omits zero, non-finite, invalid, and past Retry-After values', async () => {
    const values = [
      '0',
      '9'.repeat(400),
      'not-a-date',
      new Date(0).toUTCString(),
    ]
    for (const value of values) {
      const server = await mockServer([{
        kind: 'http-error',
        status: 429,
        body: JSON.stringify({ error: { message: 'retry later' } }),
        headers: { 'retry-after': value },
      }])
      const ctx = await harness(server.url)
      const result = await assemble(ctx, { model: MODEL, messages: [] })
      expect(result.finish).toEqual({
        kind: 'error',
        failure: { message: 'retry later', code: 'RATE_LIMIT', status: 429 },
      })
    }
  })

  it('classifies only context-capacity HTTP 400 details as context overflow', () => {
    expect(httpErrorCode(400, { message: 'request too large for model context' }))
      .toBe(CONTEXT_WINDOW_EXCEEDED_CODE)
    expect(httpErrorCode(400, { message: 'invalid input: temperature exceeds maximum allowed value' }))
      .toBe('INVALID_REQUEST')
    expect(httpErrorCode(413, { code: 'context_length_exceeded' })).toBe('HTTP_413')
  })

  it('distinguishes terminal quota exhaustion from transient HTTP 429 throttling', () => {
    expect(httpErrorCode(429, { code: 'insufficient_quota', message: 'account credits exhausted' }))
      .toBe(QUOTA_EXCEEDED_CODE)
    expect(httpErrorCode(429, { message: 'request rate limit exceeded' })).toBe('RATE_LIMIT')
  })

  it('keeps the status-line message for JSON error bodies without a message', async () => {
    const server = await mockServer([{ kind: 'http-error', status: 500, body: '{"error":{"type":"x"}}' }])
    const ctx = await harness(server.url)
    const result = await assemble(ctx, { model: MODEL, messages: [] })
    expect(result.finish.kind).toBe('error')
    if (result.finish.kind !== 'error') throw new Error('expected an error finish')
    expect(result.finish.failure.code).toBe('SERVER')
    expect(result.finish.failure.message).toMatch(/HTTP 500/)
  })

  it('keeps the status-line message for non-JSON error bodies', async () => {
    const server = await mockServer([{ kind: 'http-error', status: 502, body: 'Bad Gateway', contentType: 'text/plain' }])
    const ctx = await harness(server.url)
    const result = await assemble(ctx, { model: MODEL, messages: [] })
    expect(result.finish.kind).toBe('error')
    if (result.finish.kind !== 'error') throw new Error('expected an error finish')
    expect(result.finish.failure.code).toBe('SERVER')
    expect(result.finish.failure.message).toMatch(/HTTP 502/)
  })

  it('maps unusual statuses to HTTP_<status>', () => {
    expect(httpErrorCode(418)).toBe('HTTP_418')
  })

  it('reports a transport failure with the endpoint in the message', async () => {
    // Port 1 is reserved/unbound, so the service normalizes the fetch failure.
    const ctx = await harness('http://127.0.0.1:1')
    const result = await assemble(ctx, { model: MODEL, messages: [] })
    expect(result.finish).toMatchObject({
      kind: 'error',
      failure: {
        code: 'TRANSPORT',
        message: 'SiliconFlow API request to http://127.0.0.1:1 failed',
      },
    })
  })

  it('classifies an aborted request as an aborted finish', async () => {
    const controller = new AbortController()
    controller.abort()
    const ctx = await harness('http://127.0.0.1:1')
    const result = await assemble(ctx, {
      model: MODEL,
      messages: [],
      signal: controller.signal,
    })
    expect(result.finish).toMatchObject({ kind: 'aborted', failure: { code: 'ABORTED' } })
  })

  it('throws EMPTY_RESPONSE when the response has no body', async () => {
    const adapter = adapterOf({ baseURL: 'http://127.0.0.1:1' })
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(null, { status: 200 }),
    )
    try {
      const iterate = async (): Promise<void> => {
        for await (const _chunk of adapter.stream({ provider: 'siliconflow', model: 'm', messages: [] })) { /* drain */ }
      }
      await expect(iterate()).rejects.toThrow(/no response body/)
    } finally {
      fetchSpy.mockRestore()
    }
  })

  it('classifies an abrupt body close as TRANSPORT', async () => {
    const server = await mockServer([{
      kind: 'close-early',
      events: ['{"choices":[{"delta":{"content":"par"}}]}'],
    }])
    const ctx = await harness(server.url)
    const result = await assemble(ctx, { model: MODEL, messages: [] })
    expect(result.finish.kind).toBe('error')
    if (result.finish.kind !== 'error') throw new Error('expected an error finish')
    expect(result.finish.failure.code).toBe('TRANSPORT')
    expect(result.finish.failure.message).toMatch(/^SiliconFlow API stream from .* failed$/)
  })

  it('aborts mid-stream via the request signal', async () => {
    const server = await mockServer([{ kind: 'sse', events: textEvents, delayMs: 50 }])
    const ctx = await harness(server.url)
    const controller = new AbortController()

    const pending = (async () => {
      const chunks = []
      for await (const chunk of ctx.llm.stream({
        provider: 'siliconflow',
        model: MODEL,
        messages: [],
        signal: controller.signal,
      })) {
        chunks.push(chunk)
      }
      return chunks
    })()

    setTimeout(() => { controller.abort() }, 30)
    const chunks = await pending
    expect(chunks).toHaveLength(1)
    expect(chunks[0]?.type).toBe('finish')
    if (chunks[0]?.type !== 'finish') throw new Error('expected a finish chunk')
    expect(chunks[0].reason.kind).toBe('aborted')
    if (chunks[0].reason.kind !== 'aborted') throw new Error('expected an aborted finish')
    expect(chunks[0].reason.failure.code).toBe('ABORTED')
  })

  it('maps connection failures to TRANSPORT without losing the cause', async () => {
    const cause = new TypeError('connection refused')
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockRejectedValue(cause)
    const adapter = adapterOf({ baseURL: 'https://example.invalid' })
    try {
      const drain = async (): Promise<void> => {
        for await (const _chunk of adapter.stream({ provider: 'siliconflow', model: 'm', messages: [] })) { /* drain */ }
      }
      await expect(drain()).rejects.toMatchObject({ code: 'TRANSPORT', cause })
    } finally {
      fetchSpy.mockRestore()
    }
  })

  it('renders a non-Error transport rejection without losing its cause', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(() => {
      const failed = Promise.withResolvers<Response>()
      failed.reject('offline')
      return failed.promise
    })
    const adapter = adapterOf({ baseURL: 'https://example.invalid' })
    try {
      const drain = async (): Promise<void> => {
        for await (const _chunk of adapter.stream({ provider: 'siliconflow', model: 'm', messages: [] })) { /* drain */ }
      }
      await expect(drain()).rejects.toMatchObject({
        message: 'SiliconFlow API request to https://example.invalid failed',
        code: 'TRANSPORT',
        cause: 'offline',
      })
    } finally {
      fetchSpy.mockRestore()
    }
  })

  it('aborts the underlying body when the stream stays idle past its watchdog', async () => {
    vi.useFakeTimers()
    let stopped = false
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation((_input, init) => {
      const signal = init?.signal
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          signal?.addEventListener('abort', () => {
            stopped = true
            controller.error(signal.reason)
          }, { once: true })
        },
      })
      return Promise.resolve(new Response(body, { status: 200 }))
    })
    const adapter = adapterOf({ baseURL: 'https://example.invalid', streamIdleTimeoutMs: 100 })
    try {
      const drain = (async () => {
        for await (const _chunk of adapter.stream({ provider: 'siliconflow', model: 'm', messages: [] })) { /* drain */ }
      })()
      const rejected = expect(drain).rejects.toMatchObject({ code: 'TIMEOUT' })
      await vi.advanceTimersByTimeAsync(0)
      await vi.advanceTimersByTimeAsync(100)
      await rejected
      expect(stopped).toBe(true)
    } finally {
      fetchSpy.mockRestore()
    }
  })

  it('keeps an idle provider read alive through SSE comments', async () => {
    vi.useFakeTimers()
    const encoder = new TextEncoder()
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(() => {
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          setTimeout(() => { controller.enqueue(encoder.encode(': keep-alive\n\n')) }, 75)
          setTimeout(() => { controller.enqueue(encoder.encode(': keep-alive\n\n')) }, 150)
          setTimeout(() => {
            controller.enqueue(encoder.encode(textEvents.map(event => `data: ${event}\n\n`).join('')))
            controller.close()
          }, 225)
        },
      })
      return Promise.resolve(new Response(body, { status: 200 }))
    })
    const adapter = adapterOf({ baseURL: 'https://example.invalid', streamIdleTimeoutMs: 100 })
    try {
      const chunks: string[] = []
      const drain = (async () => {
        for await (const chunk of adapter.stream({ provider: 'siliconflow', model: 'm', messages: [] })) {
          chunks.push(chunk.type)
        }
      })()
      await vi.advanceTimersByTimeAsync(75)
      await vi.advanceTimersByTimeAsync(75)
      await vi.advanceTimersByTimeAsync(75)
      await expect(drain).resolves.toBeUndefined()
      expect(chunks).toEqual(['block-start', 'text-delta', 'block-end', 'usage', 'finish'])
    } finally {
      fetchSpy.mockRestore()
    }
  })
})

describe('plugin registration and config', () => {
  it('keeps wire helpers off the package root', () => {
    for (const helper of [
      'httpErrorCode',
      'serializeMessages',
      'serializeRequest',
      'DONE',
      'parseSse',
      'mapFinishReason',
      'mapUsage',
      'translate',
    ]) expect(LlmSiliconFlow).not.toHaveProperty(helper)
  })

  it('registers the siliconflow provider and unregisters on dispose (HMR safety)', async () => {
    const server = await mockServer([])
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    const fiber = await ctx.plugin(LlmSiliconFlow, {
      baseURL: server.url,
    })
    expect(ctx.llm.listProviders()).toEqual([{ id: 'siliconflow', name: 'SiliconFlow' }])
    expect(ctx.llm.listConfigurableProviders()).toEqual([{
      provider: 'siliconflow',
      displayName: 'SiliconFlow',
      settingsNs: 'llm-siliconflow',
      settingsPath: [],
    }])
    await fiber.dispose()
    expect(ctx.llm.listProviders()).toEqual([])
    expect(ctx.llm.listConfigurableProviders()).toEqual([])
  })

  it('registers retryPolicy from the provider config', async () => {
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmSiliconFlow, {
      baseURL: 'http://127.0.0.1:1',
      retryPolicy: {
        mode: 'always',
        backoff: { initialDelayMs: 25, maxDelayMs: 100, jitterRatio: 0.2 },
      },
    })

    expect(ctx.llm.providerRetryPolicy('siliconflow')).toEqual({
      mode: 'always',
      initialDelayMs: 25,
      maxDelayMs: 100,
      jitterRatio: 0.2,
    })
  })

  it('owns the siliconflow provider and advertises the default models', async () => {
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmSiliconFlow, { baseURL: 'http://127.0.0.1:1' })
    expect(ctx.llm.listProviders()).toEqual([{ id: 'siliconflow', name: 'SiliconFlow' }])
    // Without a key, discovery cannot run, so the picker falls back to the
    // configured default catalog. VLM entries declare image input modality.
    await expect(ctx.llm.listModels('siliconflow')).resolves.toEqual([
      { provider: 'siliconflow', id: 'zai-org/GLM-5.2', name: 'zai-org/GLM-5.2', inputModalities: ['text'] },
      { provider: 'siliconflow', id: 'deepseek-ai/DeepSeek-V4-Pro', name: 'deepseek-ai/DeepSeek-V4-Pro', inputModalities: ['text'] },
      { provider: 'siliconflow', id: 'deepseek-ai/DeepSeek-V4-Flash', name: 'deepseek-ai/DeepSeek-V4-Flash', inputModalities: ['text'] },
      { provider: 'siliconflow', id: 'Pro/zai-org/GLM-5.1', name: 'Pro/zai-org/GLM-5.1', inputModalities: ['text'] },
      { provider: 'siliconflow', id: 'moonshotai/Kimi-K2.7-Code', name: 'moonshotai/Kimi-K2.7-Code', inputModalities: ['text', 'image'] },
      { provider: 'siliconflow', id: 'Pro/moonshotai/Kimi-K2.6', name: 'Pro/moonshotai/Kimi-K2.6', inputModalities: ['text', 'image'] },
      { provider: 'siliconflow', id: 'Qwen/Qwen3.5-397B-A17B', name: 'Qwen/Qwen3.5-397B-A17B', inputModalities: ['text', 'image'] },
      { provider: 'siliconflow', id: 'zai-org/GLM-4.5V', name: 'zai-org/GLM-4.5V', inputModalities: ['text', 'image'] },
      { provider: 'siliconflow', id: 'Qwen/Qwen3-VL-32B-Instruct', name: 'Qwen/Qwen3-VL-32B-Instruct', inputModalities: ['text', 'image'] },
      { provider: 'siliconflow', id: 'Qwen/Qwen3-VL-8B-Instruct', name: 'Qwen/Qwen3-VL-8B-Instruct', inputModalities: ['text', 'image'] },
      { provider: 'siliconflow', id: 'Qwen/Qwen3-VL-32B-Thinking', name: 'Qwen/Qwen3-VL-32B-Thinking', inputModalities: ['text', 'image'] },
      { provider: 'siliconflow', id: 'Qwen/Qwen3-VL-8B-Thinking', name: 'Qwen/Qwen3-VL-8B-Thinking', inputModalities: ['text', 'image'] },
      { provider: 'siliconflow', id: 'deepseek-ai/DeepSeek-OCR', name: 'deepseek-ai/DeepSeek-OCR', inputModalities: ['text', 'image'] },
    ])
    await expect(ctx.llm.resolveModelInfo('siliconflow', MODEL))
      .resolves.toMatchObject({
        provider: 'siliconflow',
        id: MODEL,
        name: MODEL,
        context: { contextWindow: 1_000_000 },
        defaultMaxTokens: 8_192,
      })
  })

  it('resolves no reasoning efforts (SiliconFlow has no effort knobs)', async () => {
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmSiliconFlow, { baseURL: 'http://127.0.0.1:1' })
    const info = await ctx.llm.resolveModelInfo('siliconflow', MODEL)
    expect(info.reasoning).toBeUndefined()
    // An explicit reasoning effort is therefore refused before any I/O.
    const result = await assemble(ctx, {
      model: MODEL,
      reasoningEffort: ReasoningEffortId('high'),
      messages: [],
    })
    expect(result.finish).toMatchObject({
      kind: 'error',
      failure: { code: 'UNSUPPORTED_REASONING_EFFORT' },
    })
  })

  it('infers VLM modality from authoritative model set for uncatalogued models', async () => {
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmSiliconFlow, {
      baseURL: 'http://127.0.0.1:1',
      models: [{ id: 'Qwen/Qwen3-VL-32B-Instruct', contextWindow: 131_072 }],
    })
    // A catalogued VLM (explicit inputModalities)
    await expect(ctx.llm.resolveModelInfo('siliconflow', 'Qwen/Qwen3-VL-32B-Instruct'))
      .resolves.toMatchObject({ inputModalities: ['text', 'image'] })
    // An uncatalogued VLM from the authoritative set
    await expect(ctx.llm.resolveModelInfo('siliconflow', 'moonshotai/Kimi-K2.7-Code'))
      .resolves.toMatchObject({ inputModalities: ['text', 'image'] })
    await expect(ctx.llm.resolveModelInfo('siliconflow', 'nex-agi/Nex-N2-Pro'))
      .resolves.toMatchObject({ inputModalities: ['text', 'image'] })
    // An uncatalogued non-VLM
    await expect(ctx.llm.resolveModelInfo('siliconflow', 'deepseek-ai/DeepSeek-V4-Flash'))
      .resolves.toMatchObject({ inputModalities: ['text'] })
    // GLM-5.2 is NOT a VLM despite matching GLM naming patterns
    await expect(ctx.llm.resolveModelInfo('siliconflow', 'zai-org/GLM-5.2'))
      .resolves.toMatchObject({ inputModalities: ['text'] })
  })

  it('advertises configured models without restricting arbitrary request ids', async () => {
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmSiliconFlow, {
      baseURL: 'http://127.0.0.1:1',
      models: [
        { id: 'org/private-fast', contextWindow: 32_000 },
        {
          id: 'org/private-reasoner',
          name: 'Private Reasoner',
          description: 'Higher reasoning budget',
          contextWindow: 64_000,
        },
      ],
    })
    await expect(ctx.llm.listModels('siliconflow')).resolves.toEqual([
      { provider: 'siliconflow', id: 'org/private-fast', name: 'org/private-fast', inputModalities: ['text'] },
      { provider: 'siliconflow', id: 'org/private-reasoner', name: 'Private Reasoner', description: 'Higher reasoning budget', inputModalities: ['text'] },
    ])
    await expect(ctx.llm.resolveModelInfo('siliconflow', 'org/private-fast'))
      .resolves.toMatchObject({ context: { contextWindow: 32_000 } })
    await expect(ctx.llm.resolveModelInfo('siliconflow', 'org/private-reasoner'))
      .resolves.toMatchObject({
        name: 'Private Reasoner',
        description: 'Higher reasoning budget',
      })
    await expect(ctx.llm.resolveModelInfo('siliconflow', 'arbitrary-unlisted'))
      .resolves.toMatchObject({
        context: { contextWindow: 32_768 },
        defaultMaxTokens: 8_192,
      })
  })

  it('uses exact model capacity before the adapter-wide default', async () => {
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmSiliconFlow, {
      baseURL: 'http://127.0.0.1:1',
      defaultContextWindow: 256_000,
      models: [
        { id: 'inherits-default' },
        { id: 'exact-override', contextWindow: 64_000 },
      ],
    })

    await expect(ctx.llm.resolveModelInfo('siliconflow', 'inherits-default'))
      .resolves.toMatchObject({ context: { contextWindow: 256_000 } })
    await expect(ctx.llm.resolveModelInfo('siliconflow', 'exact-override'))
      .resolves.toMatchObject({ context: { contextWindow: 64_000 } })
    await expect(ctx.llm.resolveModelInfo('siliconflow', 'unlisted-pass-through'))
      .resolves.toMatchObject({ context: { contextWindow: 256_000 } })
  })

  it('allows an explicit empty model catalog', async () => {
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmSiliconFlow, {
      baseURL: 'http://127.0.0.1:1',
      models: [],
    })
    await expect(ctx.llm.listModels('siliconflow')).resolves.toEqual([])
  })

  it.each([
    [[{ id: '' }], /ids must be non-empty/],
    [[{ id: 'm', name: '' }], /empty name/],
    [[{ id: 'm', contextWindow: 0 }], /contextWindow/],
    [[{ id: 'm', contextWindow: 1.5 }], /contextWindow/],
    [[{ id: 'm' }, { id: 'm' }], /duplicate catalog model/],
  ] as const)('rejects invalid advisory model config', async (models, message) => {
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await expect(ctx.plugin(LlmSiliconFlow, {
      baseURL: 'http://127.0.0.1:1',
      models: [...models],
    })).rejects.toThrow(message)
    expect(ctx.llm.listProviders()).toEqual([])
  })

  it.each([0, 1.5])('rejects a per-model output cap of %s', (maxTokens) => {
    expect(() => resolveAdapterOptions({ models: [{ id: 'bad-cap', maxTokens }] }))
      .toThrow(/maxTokens must be a positive integer/)
  })

  it('prefers a model\'s own output cap over the profile default', async () => {
    const adapter = adapterOf({ maxTokens: 4096, models: [
      { id: 'capped', maxTokens: 512 },
      { id: 'uncapped' },
    ] })
    await expect(adapter.resolveModel('siliconflow', 'capped'))
      .resolves.toMatchObject({ defaultMaxTokens: 512 })
    await expect(adapter.resolveModel('siliconflow', 'uncapped'))
      .resolves.toMatchObject({ defaultMaxTokens: 4096 })
    await expect(adapter.resolveModel('siliconflow', 'not-in-catalog'))
      .resolves.toMatchObject({ defaultMaxTokens: 4096 })
  })

  it('rejects invalid context capacity when apply is called directly', async () => {
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    expect(() => {
      LlmSiliconFlow.apply(ctx, {
        baseURL: 'http://127.0.0.1:1',
        models: [{ id: 'invalid-context', contextWindow: 0 }],
      })
    }).toThrow(/contextWindow must be a positive integer/)
    expect(ctx.llm.listProviders()).toEqual([])
  })

  it.each([0, 1.5])(
    'rejects invalid adapter-wide default context capacity %s',
    async (defaultContextWindow) => {
      expect(() => resolveAdapterOptions({ defaultContextWindow }))
        .toThrow(/defaultContextWindow must be a positive integer/)

      const ctx = new Context()
      await ctx.plugin(LlmRuntime)
      await expect(ctx.plugin(LlmSiliconFlow, {
        baseURL: 'http://127.0.0.1:1',
        defaultContextWindow,
      })).rejects.toThrow(/defaultContextWindow/)
      expect(ctx.llm.listProviders()).toEqual([])
    },
  )

  it.each([0, 1.5, Number.MAX_SAFE_INTEGER + 1])(
    'rejects invalid adapter-wide maxTokens %s',
    async (maxTokens) => {
      expect(() => resolveAdapterOptions({ maxTokens }))
        .toThrow(/maxTokens must be a positive safe integer/)

      const ctx = new Context()
      await ctx.plugin(LlmRuntime)
      await expect(ctx.plugin(LlmSiliconFlow, {
        baseURL: 'http://127.0.0.1:1',
        maxTokens,
      })).rejects.toThrow(/maxTokens/)
      expect(ctx.llm.listProviders()).toEqual([])
    },
  )

  it('falls back to SILICONFLOW_API_KEY and SILICONFLOW_BASE_URL env vars', async () => {
    vi.stubEnv('SILICONFLOW_API_KEY', 'env-key')
    vi.stubEnv('SILICONFLOW_BASE_URL', 'http://127.0.0.1:1')
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmSiliconFlow, {})
    expect(ctx.llm.listProviders()).toEqual([{ id: 'siliconflow', name: 'SiliconFlow' }])
  })

  it('loads keyless, keeps the catalog browsable, and fails the request actionably', async () => {
    vi.stubEnv('SILICONFLOW_API_KEY', '')
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmSiliconFlow, { baseURL: 'http://127.0.0.1:1' })
    // First-boot onboarding: the route registers so models stay discoverable;
    // only the request itself needs a key.
    expect(ctx.llm.listProviders()).toEqual([{ id: 'siliconflow', name: 'SiliconFlow' }])
    await expect(ctx.llm.listModels('siliconflow')).resolves.toHaveLength(13)
    const first = await assemble(ctx, { model: MODEL, messages: [] })
    expect(first.finish).toMatchObject({ kind: 'error', failure: { code: 'MISSING_CREDENTIAL' } })
    const second = await assemble(ctx, { model: MODEL, messages: [] })
    expect(second.finish.kind).toBe('error')
    if (second.finish.kind !== 'error') throw new Error('expected an error finish')
    expect(second.finish.failure.message)
      .toMatch(/store SILICONFLOW_API_KEY through the credentials service.*export SILICONFLOW_API_KEY/s)
  })

  it('reads the ambient variable when no credentials seam is mounted', async () => {
    vi.stubEnv('SILICONFLOW_API_KEY', 'ambient-key')
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmSiliconFlow, { baseURL: server.url })
    await assemble(ctx, { model: MODEL, messages: [] })
    expect(server.headers[0]?.authorization).toBe('Bearer ambient-key')
  })

  it('treats an empty ambient variable as no key when no credentials seam is mounted', async () => {
    vi.stubEnv('SILICONFLOW_API_KEY', '')
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmSiliconFlow, { baseURL: 'http://127.0.0.1:1' })
    const result = await assemble(ctx, { model: MODEL, messages: [] })
    expect(result.finish).toMatchObject({ kind: 'error', failure: { code: 'MISSING_CREDENTIAL' } })
  })

  it('prefers explicit config over env for key and base URL', async () => {
    vi.stubEnv('SILICONFLOW_API_KEY', 'env-key')
    vi.stubEnv('SILICONFLOW_BASE_URL', 'http://env-host:1')
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const ctx = await harness(server.url) // harness passes explicit config
    await assemble(ctx, { model: MODEL, messages: [] })
    expect(server.requests).toHaveLength(1) // hit the explicit URL, not env
  })

  it('uses SILICONFLOW_BASE_URL when config omits baseURL', async () => {
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    vi.stubEnv('SILICONFLOW_BASE_URL', server.url)
    vi.stubEnv('SILICONFLOW_API_KEY', 'test-key')
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmSiliconFlow, {})
    await assemble(ctx, { model: MODEL, messages: [] })
    expect(server.requests).toHaveLength(1)
  })

  it('takes SILICONFLOW_BASE_URL from any environment layer, with explicit config still on top', () => {
    const trusted = createLaunchEnvironmentSnapshot([
      { source: 'user-env', path: '/home/.dsh/.env', values: { SILICONFLOW_BASE_URL: 'https://user.example' } },
    ])
    expect(resolveAdapterOptions({}, trusted).baseURL).toBe('https://user.example')
    const project = createLaunchEnvironmentSnapshot([
      { source: 'project-env', path: '/work/.env', values: { SILICONFLOW_BASE_URL: 'https://project.example' } },
    ])
    expect(resolveAdapterOptions({}, project).baseURL).toBe('https://project.example')
    const shell = createLaunchEnvironmentSnapshot([
      { source: 'process', values: { SILICONFLOW_BASE_URL: 'https://stale.example' } },
    ])
    expect(resolveAdapterOptions({ baseURL: 'https://gateway.internal' }, shell).baseURL).toBe('https://gateway.internal')
  })

  it('defaults to the public base URL without config or env', async () => {
    vi.stubEnv('SILICONFLOW_API_KEY', 'k')
    vi.stubEnv('SILICONFLOW_BASE_URL', undefined)
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    // Registration succeeds; no call is made (would hit api.siliconflow.cn).
    await ctx.plugin(LlmSiliconFlow, {})
    expect(ctx.llm.listProviders()).toEqual([{ id: 'siliconflow', name: 'SiliconFlow' }])
  })

  it('adapter is constructible directly for embedding over the shared resolver', async () => {
    const adapter = adapterOf()
    expect(adapter).toBeInstanceOf(SiliconFlowAdapter)
    // Direct embedding shares the plugin's one resolve step, so it advertises
    // the same default catalog instead of a divergent empty one.
    await expect(adapter.listModels('siliconflow')).resolves.toHaveLength(13)
  })

  it('resolves connection facts and the credential exactly once per stream call', async () => {
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const options = vi.fn(() => resolveAdapterOptions({ baseURL: server.url }))
    const resolveApiKey = vi.fn(() => Promise.resolve('per-request-key'))
    const resolveUserId = vi.fn(() => TEST_USER_ID)
    const adapter = new SiliconFlowAdapter({ options, resolveApiKey, resolveUserId })

    for await (const _chunk of adapter.stream({ provider: 'siliconflow', model: 'm', messages: [] })) { /* drain */ }

    expect(options).toHaveBeenCalledTimes(1)
    expect(resolveApiKey).toHaveBeenCalledTimes(1)
    expect(resolveUserId).toHaveBeenCalledTimes(1)
    expect(server.headers[0]?.authorization).toBe('Bearer per-request-key')
  })

  it('rejects invalid idle watchdog bounds for direct and plugin composition', async () => {
    expect(() => resolveAdapterOptions({ streamIdleTimeoutMs: Number.POSITIVE_INFINITY }))
      .toThrow(/streamIdleTimeoutMs.*positive finite/)
    expect(() => resolveAdapterOptions({ streamIdleTimeoutMs: MAX_TIMER_DELAY_MS + 1 }))
      .toThrow(/streamIdleTimeoutMs.*no greater/)

    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await expect(ctx.plugin(LlmSiliconFlow, {
      baseURL: 'http://127.0.0.1:1',
      streamIdleTimeoutMs: 0,
    })).rejects.toThrow(/streamIdleTimeoutMs/)
    await expect(ctx.plugin(LlmSiliconFlow, {
      baseURL: 'http://127.0.0.1:1',
      streamIdleTimeoutMs: MAX_TIMER_DELAY_MS + 1,
    })).rejects.toThrow(/streamIdleTimeoutMs/)
  })

  it('rejects invalid nested retryPolicy before registering the provider', async () => {
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)

    await expect(ctx.plugin(LlmSiliconFlow, {
      baseURL: 'http://127.0.0.1:1',
      retryPolicy: { mode: 'normal', maxRetries: -1 },
    })).rejects.toThrow(/retryPolicy/)
    expect(ctx.llm.listProviders()).toEqual([])
  })
})

describe('advisory catalog discovery', () => {
  it('lists the live chat models in endpoint order when a key is present', async () => {
    const server = await mockModelsServer([{ body: JSON.stringify({ data: [
      { id: 'zai-org/GLM-5.2' },
      { id: 'moonshotai/Kimi-K2.7-Code', context_window: 131_072 },
    ] }) }])
    const ctx = await harness(server.url)

    await expect(ctx.llm.listModels('siliconflow')).resolves.toEqual([
      { provider: 'siliconflow', id: 'zai-org/GLM-5.2', name: 'zai-org/GLM-5.2', inputModalities: ['text'] },
      { provider: 'siliconflow', id: 'moonshotai/Kimi-K2.7-Code', name: 'moonshotai/Kimi-K2.7-Code', inputModalities: ['text', 'image'] },
    ])
    expect(server.paths).toEqual(['/models?sub_type=chat'])
    expect(server.headers[0]?.authorization).toBe('Bearer test-key')
  })

  it('falls back to the configured catalog when discovery fails', async () => {
    const server = await mockModelsServer([{ status: 500, body: '{}' }])
    const ctx = await harness(server.url)

    await expect(ctx.llm.listModels('siliconflow')).resolves.toHaveLength(13)
  })

  it('serves the configured catalog without a key, making no network call', async () => {
    vi.stubEnv('SILICONFLOW_API_KEY', '')
    const server = await mockModelsServer([{ body: JSON.stringify({ data: [{ id: 'unreachable' }] }) }])
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmSiliconFlow, { baseURL: server.url })

    await expect(ctx.llm.listModels('siliconflow')).resolves.toHaveLength(13)
    expect(server.paths).toEqual([])
  })

  it('caches a successful listing until the TTL expires', async () => {
    const now = vi.spyOn(Date, 'now').mockReturnValue(1_800_000_000_000)
    const server = await mockModelsServer([
      { body: JSON.stringify({ data: [{ id: 'first-list' }] }) },
      { body: JSON.stringify({ data: [{ id: 'second-list' }] }) },
    ])
    const ctx = await harness(server.url)

    await expect(ctx.llm.listModels('siliconflow')).resolves.toEqual([
      { provider: 'siliconflow', id: 'first-list', name: 'first-list', inputModalities: ['text'] },
    ])
    // A second read before the TTL is served from cache: no second request.
    await expect(ctx.llm.listModels('siliconflow')).resolves.toHaveLength(1)
    expect(server.paths).toHaveLength(1)

    // Advance past the TTL and the next read re-interrogates.
    now.mockReturnValue(1_800_000_000_000 + DISCOVERY_TTL_MS + 1)
    await expect(ctx.llm.listModels('siliconflow')).resolves.toEqual([
      { provider: 'siliconflow', id: 'second-list', name: 'second-list', inputModalities: ['text'] },
    ])
    expect(server.paths).toHaveLength(2)
    now.mockRestore()
  })

  it('resolves a discovered model\'s capacity from the warm cache', async () => {
    const server = await mockModelsServer([{ body: JSON.stringify({ data: [
      { id: 'live-model', context_window: 200_000, max_output_tokens: 16_384 },
    ] }) }])
    const ctx = await harness(server.url)

    await ctx.llm.listModels('siliconflow') // warm the cache
    await expect(ctx.llm.resolveModelInfo('siliconflow', 'live-model')).resolves.toMatchObject({
      id: 'live-model',
      name: 'live-model',
      context: { contextWindow: 200_000 },
      defaultMaxTokens: 16_384,
    })
  })
})

describe('config-surface model discovery', () => {
  it('interrogates a draft endpoint with the draft key', async () => {
    const server = await mockModelsServer([{ body: JSON.stringify({ data: [{ id: 'draft' }] }) }])
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmSiliconFlow, { baseURL: 'http://127.0.0.1:1' })

    const models = await ctx.llm.discoverModels('llm-siliconflow', { baseURL: server.url, apiKey: 'draft-key' })
    expect(models).toEqual([{ id: 'draft' }])
    expect(server.headers[0]?.authorization).toBe('Bearer draft-key')
  })

  it('falls back to the configured baseURL and ambient key when the draft omits them', async () => {
    vi.stubEnv('SILICONFLOW_API_KEY', 'ambient-key')
    const server = await mockModelsServer([{ body: JSON.stringify({ data: [{ id: 'ambient' }] }) }])
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmSiliconFlow, { baseURL: server.url })

    const models = await ctx.llm.discoverModels('llm-siliconflow', { provider: 'siliconflow' })
    expect(models).toEqual([{ id: 'ambient' }])
    expect(server.headers[0]?.authorization).toBe('Bearer ambient-key')
  })

  it.each(['', undefined] as const)('probes unauthenticated when the ambient key is %s', async (value) => {
    vi.stubEnv('SILICONFLOW_API_KEY', value)
    const server = await mockModelsServer([{ body: JSON.stringify({ data: [{ id: 'anon' }] }) }])
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmSiliconFlow, { baseURL: server.url })

    const models = await ctx.llm.discoverModels('llm-siliconflow', { provider: 'siliconflow' })
    expect(models).toEqual([{ id: 'anon' }])
    expect(server.headers[0]).not.toHaveProperty('authorization')
  })
})
