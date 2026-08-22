/**
 * `SiliconFlowAdapter`: fetch + SSE against a SiliconFlow (OpenAI-compatible)
 * chat-completions endpoint, emitting harness StreamChunks. The adapter is
 * transport-only: connection facts arrive through a thunk resolved once per
 * operation and the bearer token through a per-request resolver, so the
 * registering plugin owns validation, layering, and credential policy.
 *
 * @module dsh-llm-siliconflow/adapter
 */

import { attributionHeaders, CONTEXT_WINDOW_EXCEEDED_CODE, isContextWindowExceededError, isQuotaExceededError, LlmAdapter, LlmError, ProviderRequestId, QUOTA_EXCEEDED_CODE } from '@deepseek-ai/dsh-llm'
import type {
  GenerateOptions,
  LlmModelInfo,
  LlmProviderInfo,
  LlmResolvedModelInfo,
  ModelModality,
  ResolvedRetryPolicy,
  StreamChunk,
} from '@deepseek-ai/dsh-llm'
import type { CredentialRef } from '@deepseek-ai/dsh-credentials'
import type { ImageAttachmentRef } from '@deepseek-ai/dsh-attachment'
import { idleWatchdog, timeoutOf } from '@deepseek-ai/dsh-timeout'
import type { AnonymousUserId } from '@deepseek-ai/dsh-anonymous-user-id'
import { discoverChatModels } from './discovery.ts'
import { serializeRequest } from './serialize.ts'
import { parseSse } from './sse.ts'
import { translate } from './translate.ts'
import type { WireError } from './types.ts'

/** One optional model entry advertised by the direct-fetch adapter. */
export interface SiliconFlowCatalogModel {
  /** Wire model id accepted by the configured endpoint. */
  id: string
  /** Selector label; defaults to {@link id}. */
  name?: string
  /** Optional selector detail for deployments with similar model variants. */
  description?: string
  /** Known combined request/response context capacity; omitted when deployment metadata is unavailable. */
  contextWindow?: number
  /** Per-request output cap for this model; omission falls back to the profile's {@link SiliconFlowConnectionOptions.maxTokens}. */
  maxTokens?: number
  /** Accepted input modalities; omitted infers from the built-in VLM model set. */
  inputModalities?: readonly ModelModality[]
}

/**
 * Validated connection facts for one operation. The plugin's
 * `resolveAdapterOptions` is the one explicit resolve step producing this
 * shape; the adapter trusts it and re-reads it per operation, which is what
 * makes a configuration change reach the next request without re-registration.
 */
export interface SiliconFlowConnectionOptions {
  /** Endpoint base; `/chat/completions` is appended. */
  baseURL: string
  /**
   * Credential reference of this same resolution, resolved per request.
   * Travelling with the endpoint is the point: a request can never pair one
   * generation's URL with another generation's secret. Configuration carries
   * only this name — a literal key is not a configuration value.
   */
  apiKeyEnv: CredentialRef
  /** Default per-request output cap; explicit request values win. */
  maxTokens: number
  /** Positive context capacity used when the selected model has no exact value. */
  defaultContextWindow: number
  /** Advisory models exposed to discovery consumers; requests remain unrestricted. */
  models: readonly SiliconFlowCatalogModel[]
  /** Maximum provider idle time while one stream read is outstanding. */
  streamIdleTimeoutMs: number
  /** Provider-owned model-request retry policy, already resolved. */
  retryPolicy: ResolvedRetryPolicy
}

/** Constructor options for {@link SiliconFlowAdapter}: the operation-local resolution hooks the plugin owns. */
export interface SiliconFlowAdapterOptions {
  /** Current validated connection facts; called once per operation. */
  options: () => SiliconFlowConnectionOptions
  /**
   * Resolve the bearer token for the connection facts of one request. The
   * snapshot is passed in — never re-read — so the key can only ever come
   * from the same resolution as the endpoint it is sent to. Throws `LlmError`
   * `MISSING_CREDENTIAL` when no key is available anywhere.
   */
  resolveApiKey: (connection: SiliconFlowConnectionOptions) => Promise<string>
  /** Resolve the harness-home anonymous id shared with telemetry and feedback. */
  resolveUserId: () => AnonymousUserId
  /**
   * Resolve one image attachment to a base64 data URL for wire serialization.
   * Returns `undefined` when the attachment store is unavailable or the read
   * fails; the serializer substitutes the `OFFLOADED_IMAGE_TEXT` sentinel.
   */
  resolveImage?: (ref: ImageAttachmentRef) => Promise<string | undefined>
}

/** Default maximum idle interval while an adapter stream read is outstanding. */
export const DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000
/** Default combined request/response context capacity. */
export const DEFAULT_CONTEXT_WINDOW = 32_768
/** Default per-request output-token cap. */
export const DEFAULT_MAX_TOKENS = 8_192
/** How long a cached model-listing discovery stays fresh before the next `listModels` re-interrogates. */
export const DISCOVERY_TTL_MS = 5 * 60_000
const STREAM_IDLE_TIMEOUT_CODE = 'LLM_STREAM_IDLE_TIMEOUT'

/**
 * Authoritative set of SiliconFlow model ids that accept image input.
 *
 * **Static data — see maintenance note below.**
 *
 * SiliconFlow's model marketplace (siliconflow.cn/models) tags every chat model
 * with a `vlm` boolean. The OpenAI-compatible `GET /models` API does not expose
 * this attribute, so the adapter ships a set of known VLM model ids extracted
 * from the marketplace's SSR data. Catalog entries can still declare explicit
 * `inputModalities` to override this set.
 *
 * **Maintenance**: this set is hand-maintained and will drift from the live
 * marketplace as new VLM models are added or existing ones are deprecated. It is
 * refreshed periodically. Once the OpenAI-compatible API exposes model
 * capabilities (e.g. a `vision` sub_type or a `vlm` field in the model listing),
 * this static set will be replaced by a runtime query.
 *
 * Last updated: 2026-08-22 (23 VLM models out of 88 chat models).
 */
const KNOWN_VLM_MODELS: ReadonlySet<string> = new Set([
  'PaddlePaddle/PaddleOCR-VL-1.5',
  'Pro/moonshotai/Kimi-K2.6',
  'Qwen/Qwen3-Omni-30B-A3B-Captioner',
  'Qwen/Qwen3-Omni-30B-A3B-Instruct',
  'Qwen/Qwen3-Omni-30B-A3B-Thinking',
  'Qwen/Qwen3-VL-30B-A3B-Instruct',
  'Qwen/Qwen3-VL-30B-A3B-Thinking',
  'Qwen/Qwen3-VL-32B-Instruct',
  'Qwen/Qwen3-VL-32B-Thinking',
  'Qwen/Qwen3-VL-8B-Instruct',
  'Qwen/Qwen3-VL-8B-Thinking',
  'Qwen/Qwen3.5-122B-A10B',
  'Qwen/Qwen3.5-27B',
  'Qwen/Qwen3.5-35B-A3B',
  'Qwen/Qwen3.5-397B-A17B',
  'Qwen/Qwen3.5-4B',
  'Qwen/Qwen3.5-9B',
  'Qwen/Qwen3.6-27B',
  'Qwen/Qwen3.6-35B-A3B',
  'deepseek-ai/DeepSeek-OCR',
  'moonshotai/Kimi-K2.7-Code',
  'nex-agi/Nex-N2-Pro',
  'zai-org/GLM-4.5V',
])

/**
 * Determine the input modalities for a SiliconFlow model id.
 *
 * Uses the authoritative VLM model set from the SiliconFlow marketplace rather
 * than naming-pattern heuristics. Catalog entries can declare explicit
 * `inputModalities` to override this lookup.
 * @param id - the wire model id (e.g. `Qwen/Qwen3-VL-8B-Instruct`).
 * @returns `['text', 'image']` when the id is a known VLM, `['text']` otherwise.
 */
export function inferInputModalities(id: string): readonly ModelModality[] {
  return KNOWN_VLM_MODELS.has(id) ? ['text', 'image'] : ['text']
}

function modelInfo(provider: string, model: SiliconFlowCatalogModel): LlmModelInfo {
  const explicit = model.inputModalities
  const modalities = explicit !== undefined && explicit.length > 0 ? explicit : inferInputModalities(model.id)
  return {
    provider,
    id: model.id,
    name: model.name ?? model.id,
    ...model.description === undefined ? {} : { description: model.description },
    inputModalities: modalities,
  }
}

function providerRetryAfterMs(value: string | null): number | undefined {
  if (value === null) return undefined
  if (/^\d+$/.test(value)) {
    const delay = Number(value) * 1_000
    return Number.isFinite(delay) && delay > 0 ? delay : undefined
  }
  const delay = Date.parse(value) - Date.now()
  return Number.isFinite(delay) && delay > 0 ? delay : undefined
}

function requestId(headers: Headers): ReturnType<typeof ProviderRequestId> | undefined {
  const value = headers.get('x-request-id')
  return value === null || value.length === 0 ? undefined : ProviderRequestId(value)
}

/**
 * Map an HTTP status to a stable LlmError code.
 * @param status - status of a non-2xx provider response.
 * @param error - parsed provider error body, when available.
 * @returns the normalized harness error code.
 */
export function httpErrorCode(status: number, error?: WireError['error']): string {
  if (status === 401 || status === 403) return 'AUTH'
  const detail = [error?.code, error?.type, error?.message].filter(Boolean).join(' ')
  if (isQuotaExceededError(detail)) return QUOTA_EXCEEDED_CODE
  if (status === 429) return 'RATE_LIMIT'
  if (status === 400) {
    if (isContextWindowExceededError(detail)) return CONTEXT_WINDOW_EXCEEDED_CODE
    return 'INVALID_REQUEST'
  }
  if (status >= 500) return 'SERVER'
  return `HTTP_${status}`
}

/**
 * A direct-fetch `LlmAdapter` for SiliconFlow's OpenAI-compatible
 * chat-completions endpoint. One instance serves every model name it was
 * registered under (the harness model name IS the wire model name).
 *
 * One stable signal reaches both initial fetch and body reads. Caller aborts
 * map to `ABORTED`; the configured per-read idle watchdog maps to `TIMEOUT`.
 */
export class SiliconFlowAdapter extends LlmAdapter {
  /** Cached listing result, keyed by the baseURL it was read from. */
  private discovery: { baseURL: string; models: readonly SiliconFlowCatalogModel[]; fetchedAt: number } | undefined

  constructor(private readonly config: SiliconFlowAdapterOptions) {
    super()
  }

  override providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: 'SiliconFlow' }
  }

  override providerRetryPolicy(_provider: string): ResolvedRetryPolicy {
    return this.config.options().retryPolicy
  }

  /** The cached listing for `connection` when it is still fresh; never re-interrogates. */
  private freshDiscovery(connection: SiliconFlowConnectionOptions): readonly SiliconFlowCatalogModel[] | undefined {
    if (this.discovery?.baseURL !== connection.baseURL) return undefined
    if (Date.now() - this.discovery.fetchedAt > DISCOVERY_TTL_MS) return undefined
    return this.discovery.models
  }

  /**
   * The models this adapter currently advertises: the live chat listing when a
   * key and a reachable endpoint can supply one, else the configured catalog.
   * Discovery is advisory and best-effort — a missing key or any interrogation
   * failure falls back to the configured `models` rather than breaking the
   * picker, because an absent catalog would hide the provider entirely.
   */
  private async discover(connection: SiliconFlowConnectionOptions): Promise<readonly SiliconFlowCatalogModel[]> {
    const cached = this.freshDiscovery(connection)
    if (cached !== undefined) return cached
    try {
      const apiKey = await this.config.resolveApiKey(connection)
      const models = await discoverChatModels(connection.baseURL, apiKey)
      this.discovery = { baseURL: connection.baseURL, models: [...models], fetchedAt: Date.now() }
      return this.discovery.models
    } catch {
      // Only discovery's own failures are swallowed: a missing credential, an
      // unreachable endpoint, or an unreadable reply all mean "no live list
      // this round" — the configured catalog serves instead, and the next
      // `listModels` (or a baseURL change) tries again.
      this.discovery = undefined
      return connection.models
    }
  }

  override async listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    const connection = this.config.options()
    const models = await this.discover(connection)
    return models.map(model => modelInfo(provider, model))
  }

  override resolveModel(
    provider: string,
    model: string,
    _signal?: AbortSignal,
  ): Promise<LlmResolvedModelInfo> {
    const connection = this.config.options()
    const configured = connection.models.find(entry => entry.id === model)
    // A listing discovered by `listModels` supplies exact capacity for models
    // the static catalog does not; this read never triggers a fetch, so the
    // request path stays prompt.
    const discovered = this.freshDiscovery(connection)?.find(entry => entry.id === model)
    const entry = configured ?? discovered
    return Promise.resolve({
      // Uncatalogued models look up modalities from the built-in VLM model set:
      // a VLM id like `Qwen/Qwen3-VL-8B-Instruct` declares image input so the
      // host accepts and persists images the serializer then sends as
      // `image_url` content parts. A non-VLM id keeps `['text']` — "unknown"
      // here would let the host accept images the serializer must then reject.
      ...entry === undefined
        ? { provider, id: model, name: model, inputModalities: inferInputModalities(model) }
        : modelInfo(provider, entry),
      context: { contextWindow: entry?.contextWindow ?? connection.defaultContextWindow },
      defaultMaxTokens: entry?.maxTokens ?? connection.maxTokens,
    })
  }

  async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    // One resolution per stream call: connection facts and the credential
    // freeze here and hold for this whole request, so an in-flight stream
    // never observes a configuration change and the next call re-resolves.
    // The key resolves *from this snapshot*, so an endpoint and the secret
    // sent to it can never come from different configuration generations.
    const connection = this.config.options()
    const apiKey = await this.config.resolveApiKey(connection)
    const userId = this.config.resolveUserId()
    const consumer = new AbortController()
    const upstream = options.signal === undefined
      ? consumer.signal
      : AbortSignal.any([options.signal, consumer.signal])
    using watchdog = idleWatchdog(upstream, connection.streamIdleTimeoutMs, STREAM_IDLE_TIMEOUT_CODE)
    const iterator = this.request(
      options,
      watchdog.signal,
      connection,
      apiKey,
      userId,
      () => { watchdog.pulse() },
    )[Symbol.asyncIterator]()
    let exhausted = false
    try {
      while (true) {
        const result = await watchdog.next(iterator)
        if (result.done) {
          exhausted = true
          return
        }
        yield result.value
      }
    } catch (error: unknown) {
      if (timeoutOf(watchdog.signal, STREAM_IDLE_TIMEOUT_CODE) !== undefined) {
        throw new LlmError(
          `SiliconFlow stream idle timeout after ${connection.streamIdleTimeoutMs}ms`,
          'TIMEOUT',
          { cause: error },
        )
      }
      if (options.signal?.aborted) {
        throw new LlmError('SiliconFlow request aborted by caller', 'ABORTED', { cause: error })
      }
      if (error instanceof LlmError) throw error
      throw new LlmError(`SiliconFlow API stream from ${connection.baseURL} failed`, 'TRANSPORT', { cause: error })
    } finally {
      consumer.abort('SiliconFlow stream consumer stopped')
      if (!exhausted && iterator.return !== undefined) {
        try {
          await iterator.return()
        } catch (_abortedTransportTeardown) {
          // The consumer controller already owns termination; a return-time abort cannot add a second outcome.
        }
      }
    }
  }

  private async * request(
    options: GenerateOptions,
    signal: AbortSignal,
    connection: SiliconFlowConnectionOptions,
    apiKey: string,
    userId: AnonymousUserId,
    onComment: () => void,
  ): AsyncIterable<StreamChunk> {
    const body = await serializeRequest(options, this.config.resolveImage)
    // Prepared outside the try so the TRANSPORT label below covers exactly the
    // transport boundary, never a serialization failure.
    const payload = JSON.stringify(body)
    const headers = {
      'authorization': `Bearer ${apiKey}`,
      'content-type': 'application/json',
      'accept': 'text/event-stream',
      ...attributionHeaders(),
      'x-siliconflow-harness-user-id': String(userId),
      ...options.sessionId !== undefined
        ? { 'x-siliconflow-harness-session-id': String(options.sessionId) }
        : {},
      ...options.purpose === 'compaction'
        ? { 'x-siliconflow-harness-compact': '1' }
        : {},
    }

    // TODO(http): adopt the Cordis HTTP service when shared transport configuration
    // outweighs its additional runtime dependencies.
    let response: Response
    try {
      response = await fetch(`${connection.baseURL}/chat/completions`, {
        method: 'POST',
        headers,
        body: payload,
        signal,
      })
    } catch (error: unknown) {
      // The outer stream distinguishes caller cancellation and watchdog expiry.
      if (signal.aborted) throw error
      // fetch wraps every transport failure (DNS, refused connection, TLS,
      // proxy) in a bare `TypeError: fetch failed` whose actionable detail
      // lives on `cause`. Wrapping with the endpoint and chaining the cause
      // lets `errorChain` render the full diagnosis at every reporting boundary.
      throw new LlmError(
        `SiliconFlow API request to ${connection.baseURL} failed`,
        'TRANSPORT',
        { cause: error },
      )
    }

    if (!response.ok) {
      let message = `SiliconFlow API error (HTTP ${response.status})`
      let providerError: WireError['error']
      try {
        const parsed = await response.json() as WireError
        providerError = parsed.error
        if (providerError?.message) message = providerError.message
      } catch {
        // Only swallow error-body parsing: the HTTP status still identifies the
        // failure, so malformed gateway JSON must not mask it.
      }
      const delay = providerRetryAfterMs(response.headers.get('retry-after'))
      const id = requestId(response.headers)
      throw new LlmError(message, httpErrorCode(response.status, providerError), {
        status: response.status,
        ...delay === undefined ? {} : { providerRetryAfterMs: delay },
        ...id === undefined ? {} : { requestId: id },
      })
    }
    if (!response.body) {
      throw new LlmError('SiliconFlow API returned no response body', 'EMPTY_RESPONSE')
    }

    yield* translate(parseSse(response.body, onComment))
  }
}
