import { afterEach, describe, expect, it, vi } from 'vitest'
import { userAgent } from '@deepseek-ai/dsh-llm'
import { discoverChatModels, listingUrl, readListing } from '../src/discovery.ts'
import { closeMockServers, mockModelsServer } from './mock-server.ts'

afterEach(async () => {
  await closeMockServers()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

const listing = (models: object[]): string => JSON.stringify({ object: 'list', data: models })

describe('listingUrl', () => {
  it('appends the chat-filtered listing path', () => {
    expect(listingUrl('https://api.siliconflow.cn/v1')).toBe('https://api.siliconflow.cn/v1/models?sub_type=chat')
  })

  it('strips trailing slashes so the base is a prefix', () => {
    expect(listingUrl('https://gateway.example/openai/v1/')).toBe('https://gateway.example/openai/v1/models?sub_type=chat')
  })
})

describe('readListing', () => {
  it('preserves endpoint order and maps disclosed fields', () => {
    const entries = readListing({ data: [
      { id: 'a', display_name: 'A', context_window: 10, max_output_tokens: 5 },
      { id: 'b', name: 'B', context_length: 20 },
      { id: 'c' },
    ] })
    expect(entries).toEqual([
      { id: 'a', name: 'A', contextWindow: 10, maxTokens: 5 },
      { id: 'b', name: 'B', contextWindow: 20 },
      { id: 'c' },
    ])
  })

  it('skips rows without a usable id rather than failing the whole listing', () => {
    expect(readListing({ data: [{ name: 'nameless' }, { id: 'real' }, null, { id: '' }] }))
      .toEqual([{ id: 'real' }])
  })

  it('rejects a reply with no data array', () => {
    expect(() => readListing(null)).toThrow(expect.objectContaining({ code: 'DISCOVERY_FAILED' }))
    expect(() => readListing({ models: [] })).toThrow(expect.objectContaining({ code: 'DISCOVERY_FAILED' }))
  })
})

describe('discoverChatModels', () => {
  it('interrogates the chat-filtered listing with bearer and attribution headers, in endpoint order', async () => {
    const server = await mockModelsServer([{ body: listing([
      { id: 'zai-org/GLM-5.2' },
      { id: 'moonshotai/Kimi-K2.7-Code', context_window: 131_072 },
    ]) }])

    const models = await discoverChatModels(server.url, 'sk-test')

    expect(models).toEqual([
      { id: 'zai-org/GLM-5.2' },
      { id: 'moonshotai/Kimi-K2.7-Code', contextWindow: 131_072 },
    ])
    expect(server.paths).toEqual(['/models?sub_type=chat'])
    expect(server.headers[0]?.authorization).toBe('Bearer sk-test')
    expect(server.headers[0]?.['user-agent']).toBe(userAgent())
  })

  it('probes unauthenticated when no key is supplied', async () => {
    const server = await mockModelsServer([{ body: listing([{ id: 'm' }]) }])
    await discoverChatModels(server.url, undefined)
    expect(server.headers[0]).not.toHaveProperty('authorization')
  })

  it('rejects a non-2xx reply naming the endpoint', async () => {
    const server = await mockModelsServer([{ status: 401, body: '{"error":"nope"}' }])
    await expect(discoverChatModels(server.url, 'bad')).rejects.toThrow(/answered 401/)
  })

  it('reports an unreachable endpoint', async () => {
    await expect(discoverChatModels('http://127.0.0.1:9/v1', undefined))
      .rejects.toMatchObject({ code: 'DISCOVERY_FAILED' })
  })

  it('rejects a reply that is not JSON', async () => {
    const server = await mockModelsServer([{ body: '<html>nope</html>', contentType: 'text/html' }])
    await expect(discoverChatModels(server.url, undefined)).rejects.toThrow(/did not answer with JSON/)
  })

  it('rejects a JSON reply with no data array', async () => {
    const server = await mockModelsServer([{ body: '{"models":[]}' }])
    await expect(discoverChatModels(server.url, undefined)).rejects.toThrow(/no "data" array/)
  })

  it('honors a signal that is already aborted', async () => {
    await expect(discoverChatModels('http://127.0.0.1:9/v1', undefined, AbortSignal.abort('stop')))
      .rejects.toMatchObject({ code: 'ABORTED' })
  })

  it('reports cancellation during the body read as an abort, not a raw reason', async () => {
    const controller = new AbortController()
    const bodyRead = Promise.withResolvers<undefined>()
    vi.stubGlobal('fetch', async (_url: string | URL, init?: RequestInit) => {
      const signal = init?.signal
      if (signal === undefined || signal === null) throw new Error('expected a discovery signal')
      return new Response(new ReadableStream<Uint8Array>({
        pull(stream) {
          bodyRead.resolve(undefined)
          return new Promise<void>((resolve) => {
            signal.addEventListener('abort', () => {
              stream.error(signal.reason)
              resolve()
            }, { once: true })
          })
        },
      }))
    })
    const probe = discoverChatModels('https://slow.example/v1', undefined, controller.signal)
    await bodyRead.promise
    controller.abort('test cancellation')

    await expect(probe).rejects.toMatchObject({ code: 'ABORTED' })
  })

  it('refuses an oversized reply whether its length is declared or streamed', async () => {
    const oversized = listing([{ id: 'm', pad: 'x'.repeat(4 * 1024 * 1024) }])

    const declared = await mockModelsServer([{ body: oversized }])
    await expect(discoverChatModels(declared.url, undefined)).rejects.toThrow(/answered with more than 4194304 bytes/)

    const streamed = await mockModelsServer([{
      chunks: ['{"data":[{"id":"m","pad":"', 'x'.repeat(4 * 1024 * 1024), '"}]}'],
    }])
    await expect(discoverChatModels(streamed.url, undefined)).rejects.toThrow(/answered with more than 4194304 bytes/)
  })
})
