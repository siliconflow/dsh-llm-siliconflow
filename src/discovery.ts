/**
 * Interrogate the SiliconFlow (OpenAI-compatible) `GET /models` listing for
 * the chat models an endpoint serves, filtered with `sub_type=chat` and kept
 * in the endpoint's own order — SiliconFlow's listing is already ordered by
 * its own preference, so the adapter must not re-sort it.
 *
 * This module is transport-only: it takes the endpoint and bearer token for
 * one interrogation and returns the entries it read. The registering plugin
 * owns credential policy, and the adapter owns caching and the fallback to its
 * configured catalog.
 *
 * @module dsh-llm-siliconflow/discovery
 */

import { attributionHeaders, LlmError } from '@deepseek-ai/dsh-llm'

/** One chat model the listing endpoint reports. Structural subset of the adapter's catalog entry. */
export interface SiliconFlowListingEntry {
  /** Model id the endpoint accepts. */
  id: string
  /** Human-readable name when the endpoint supplies one; selectors default to the id. */
  name?: string
  /** Maximum combined request/response context in tokens, when disclosed. */
  contextWindow?: number
  /** Maximum output tokens, when disclosed. */
  maxTokens?: number
}

/**
 * Endpoint replies larger than this are refused. The bound holds on the bytes
 * actually read, not the length the server claims, so a streaming or
 * under-declaring reply cannot exhaust memory before it is turned away.
 */
const MAX_RESPONSE_BYTES = 4 * 1024 * 1024

/** One entry of the `data` array; every field is `unknown` until validated. */
interface ListingEntry {
  id?: unknown
  name?: unknown
  display_name?: unknown
  context_window?: unknown
  context_length?: unknown
  max_tokens?: unknown
  max_output_tokens?: unknown
}

/** A positive integer field, or `undefined` when absent or unusable. */
function capacity(...candidates: readonly unknown[]): number | undefined {
  for (const candidate of candidates) {
    if (typeof candidate === 'number' && Number.isInteger(candidate) && candidate > 0) return candidate
  }
  return undefined
}

/** A non-empty string field, or `undefined`. */
function label(...candidates: readonly unknown[]): string | undefined {
  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.length > 0) return candidate
  }
  return undefined
}

/**
 * Join the endpoint base with the chat-filtered listing path. The base is a
 * prefix, not a URL to resolve against, so a gateway path such as
 * `https://gateway.example/openai/v1` keeps its segments.
 * @param baseURL - the chat-completions base.
 * @returns the `GET /models?sub_type=chat` URL.
 */
export function listingUrl(baseURL: string): string {
  return `${baseURL.replace(/\/+$/, '')}/models?sub_type=chat`
}

/**
 * Read a reply body, refusing one that outgrows {@link MAX_RESPONSE_BYTES}.
 * @param response - the settled listing response.
 * @param url - the endpoint, for the oversize diagnostic.
 * @returns the decoded body text.
 */
async function readBounded(response: Response, url: string): Promise<string> {
  const oversized = (): LlmError =>
    new LlmError(`${url} answered with more than ${MAX_RESPONSE_BYTES} bytes`, 'DISCOVERY_FAILED')
  const declared = Number(response.headers.get('content-length') ?? Number.NaN)
  if (Number.isFinite(declared) && declared > MAX_RESPONSE_BYTES) {
    await response.body?.cancel()
    throw oversized()
  }
  /* v8 ignore next -- fetch always exposes a body stream on a 2xx Response; the null guard is defensive. */
  if (response.body === null) return ''
  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let total = 0
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      total += value.byteLength
      if (total > MAX_RESPONSE_BYTES) throw oversized()
      chunks.push(value)
    }
  } finally {
    /* v8 ignore next 4 -- cancel() after a completed or abandoned read settles without rejecting; unobserved best-effort cleanup. */
    await reader.cancel().catch(() => {
      // Cancel after a drained read, or after this function walked away from an
      // oversized one, is cleanup; the reply is already decided either way.
    })
  }
  const body = new Uint8Array(total)
  let offset = 0
  for (const chunk of chunks) {
    body.set(chunk, offset)
    offset += chunk.byteLength
  }
  return new TextDecoder().decode(body)
}

/**
 * Map one listing reply into entries, preserving endpoint order. A row without
 * a usable id is skipped rather than failing the whole interrogation: a single
 * malformed row should not hide the rest of a working endpoint's catalog.
 * @param body - the parsed reply body.
 * @returns the entries in arrival order.
 */
export function readListing(body: unknown): SiliconFlowListingEntry[] {
  const data = (body as { data?: unknown } | null)?.data
  if (!Array.isArray(data)) {
    throw new LlmError('the endpoint\'s model listing has no "data" array', 'DISCOVERY_FAILED')
  }
  const entries: SiliconFlowListingEntry[] = []
  for (const raw of data) {
    const entry = raw as ListingEntry | null
    const id = label(entry?.id)
    if (id === undefined) continue
    const name = label(entry?.name, entry?.display_name)
    const contextWindow = capacity(entry?.context_window, entry?.context_length)
    const maxTokens = capacity(entry?.max_output_tokens, entry?.max_tokens)
    entries.push({
      id,
      ...name === undefined ? {} : { name },
      ...contextWindow === undefined ? {} : { contextWindow },
      ...maxTokens === undefined ? {} : { maxTokens },
    })
  }
  return entries
}

/**
 * Interrogate one endpoint for the chat models it advertises.
 * @param baseURL - the chat-completions base; `/models?sub_type=chat` is appended.
 * @param apiKey - bearer token, or `undefined` to probe unauthenticated.
 * @param signal - caller cancellation; the fetch and body read honor it.
 * @returns the advertised models in endpoint order.
 * @throws LlmError when the endpoint is unreachable, refuses the request, or
 *   the reply is not a readable listing.
 */
export async function discoverChatModels(
  baseURL: string,
  apiKey: string | undefined,
  signal?: AbortSignal,
): Promise<readonly SiliconFlowListingEntry[]> {
  const url = listingUrl(baseURL)
  let response: Response
  try {
    response = await fetch(url, {
      method: 'GET',
      headers: {
        accept: 'application/json',
        ...apiKey === undefined ? {} : { authorization: `Bearer ${apiKey}` },
        ...attributionHeaders(),
      },
      ...signal === undefined ? {} : { signal },
    })
  } catch (error: unknown) {
    if (signal?.aborted) {
      throw new LlmError('model discovery aborted by caller', 'ABORTED', { cause: error })
    }
    throw new LlmError(`could not reach ${url}`, 'DISCOVERY_FAILED', { cause: error })
  }
  if (!response.ok) {
    throw new LlmError(`${url} answered ${response.status}`, 'DISCOVERY_FAILED')
  }
  let text: string
  try {
    text = await readBounded(response, url)
  } catch (error: unknown) {
    if (signal?.aborted) {
      throw new LlmError('model discovery aborted by caller', 'ABORTED', { cause: error })
    }
    throw error
  }
  let body: unknown
  try {
    body = JSON.parse(text)
  } catch (error: unknown) {
    throw new LlmError(`${url} did not answer with JSON`, 'DISCOVERY_FAILED', { cause: error })
  }
  return readListing(body)
}
