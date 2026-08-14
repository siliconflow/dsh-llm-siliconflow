# @siliconflow/dsh-llm-siliconflow

English | [中文](README.zh.md)

SiliconFlow chat-completions adapter plugin for the [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) LLM seam: direct `fetch` + SSE (framed by `eventsource-parser`) translating SiliconFlow's OpenAI-compatible wire format into the `StreamChunk` protocol. SiliconFlow hosts a broad catalog of open models, including reasoning models whose deltas carry `reasoning_content` (DeepSeek-R1, QwQ, Kimi-K2-Thinking) — the adapter translates that channel into harness reasoning blocks and passes it back on tool-call turns as those models require.

The package owns the `siliconflow` provider route, so a deployment only has to supply a SiliconFlow API key to use it. Its model picker fills from the live `GET /models?sub_type=chat` listing in endpoint order; the configured `models` list is the fallback shown while no key is available or discovery fails. It is a plain OpenAI-compatible endpoint with no `thinking`/`reasoning_effort` toggles, so the adapter exposes no reasoning-effort metadata and serializes none: a reasoning model is selected by its catalog id, and an explicit `reasoningEffort` on a request is refused with `UNSUPPORTED_REASONING_EFFORT` before network I/O. Registering another adapter for `siliconflow` throws `LlmError('DUPLICATE_ADAPTER')`.

The package root exposes the Cordis plugin contract and `SiliconFlowAdapter`; wire serialization, SSE parsing, chunk translation, and discovery helpers are not part of that root contract.

## Install

The `@deepseek-ai/dsh-*` dependencies are not yet published, so this package builds and tests inside a pinned DeepSeek Harness checkout — the CI workflow checks it out into `packages/llm/llm-siliconflow` and runs the harness's gates. Once those dependencies publish, install it into a profile with `dsh plugin --profile <name> add @siliconflow/dsh-llm-siliconflow`, or mount it directly in a `cordis.patch.yml` layer:

```yaml
- insert:
    - id: llm-siliconflow
      name: '@siliconflow/dsh-llm-siliconflow'
```

This code is derived from the MIT-licensed DeepSeek Harness `llm-deepseek` adapter; see [LICENSE](LICENSE).

## Config

```yaml
- id: llm-siliconflow
  name: '@siliconflow/dsh-llm-siliconflow'
  config:
    apiKeyEnv: SILICONFLOW_API_KEY # default; resolved per request via ctx.credentials, then the environment
    baseURL: https://api.siliconflow.cn/v1 # optional; $SILICONFLOW_BASE_URL then the public API when omitted
    maxTokens: 8192         # optional positive per-request output cap; this is the default
    streamIdleTimeoutMs: 300000 # optional; positive finite Node timer delay; five-minute default
    retryPolicy:            # optional; omission uses bounded normal defaults
      mode: normal          # normal | always
      backoff:
        initialDelayMs: 500
        maxDelayMs: 10000
        jitterRatio: 0.1
    defaultContextWindow: 32768 # optional positive-integer fallback; this is the default
    models:                 # optional; the fallback catalog shown when discovery cannot run
      - id: zai-org/GLM-5.2
      - id: deepseek-ai/DeepSeek-V4-Flash
```

The plugin registers the single provider route `siliconflow` together with its resolved `retryPolicy`. A request selects it with `provider: siliconflow`; its `model` is passed through as the wire `model` string, so changing SiliconFlow models does not require lifecycle-time registration. The wire model id is SiliconFlow's `org/model` spelling (`deepseek-ai/DeepSeek-V4-Flash`), never a short alias. Omitting `models` keeps a small fallback catalog of six currently hosted chat models; an explicit list replaces those defaults, while `models: []` advertises none. Catalog entries are exposed through `ctx.llm.listModels('siliconflow')` for clients such as ACP editors and the Web selector, but remain advisory: unlisted model ids still pass through unchanged. An omitted entry name defaults to its id.

`contextWindow` is optional per configured model. `ctx.llm.resolveModelInfo('siliconflow', model).context` returns an exact value first — from the configured entry or a warm discovery cache — then `defaultContextWindow` for a model nothing sized. The adapter default is 32,768; SiliconFlow's catalog spans roughly 8k to 128k+ contexts, so a discovered listing that discloses a context window is the authoritative value, and the fallback is what remains when nothing disclosed one. Pressure-sensitive plugins get deployment-owned capacity without treating the model selector as authoritative.

`maxTokens` is the adapter-configured output cap for conversation requests and defaults to 8,192. A catalog entry may carry its own `maxTokens`, which wins for that model; an entry without one, and any unlisted pass-through id, resolve to the profile value. Exact-model resolution exposes the winner as `defaultMaxTokens`; `LlmRuntime` materializes that value into `GenerateOptions.maxTokens` before the agent loop writes `request/header`, so the wire request remains reconstructable. An explicit request or `AgentOptions.maxTokens` value wins and is serialized as `max_tokens`. The adapter does not clamp this request budget against `contextWindow`; deployments with a smaller context or provider output limit must configure a compatible `maxTokens`.

`streamIdleTimeoutMs` bounds each outstanding provider read, including the initial `fetch`, without counting time the consumer spends between chunks. SSE comments rearm an outstanding read as transport activity but never become `StreamChunk` values or session-log events. One stable abort signal reaches the request and body reader for the whole call; expiry stops the transport and throws `LlmError('TIMEOUT')`, while an earlier caller abort throws `LlmError('ABORTED')`. The adapter makes exactly one provider request per `stream()` call; it registers the configured policy as provider metadata, and `dsh-llm-retry` separately executes it at durable agent-step boundaries.

## Dynamic model discovery

`listModels` advertises the live chat listing, not a hand-maintained snapshot: it interrogates `GET {baseURL}/models?sub_type=chat` with the resolved key, keeps the reply in **endpoint order**, and caches it for five minutes. Discovery is advisory and best-effort — a missing key, an unreachable endpoint, a refused credential, or an unreadable reply all fall back to the configured `models` list rather than breaking the picker, because an empty catalog would hide the provider entirely. A successful listing also feeds exact-model resolution: a model the static catalog does not name still gets the context window and output cap its listing disclosed.

The config surface's "fetch available models" action uses the same interrogation through `ctx.llm.discoverModels('llm-siliconflow', …)`: a key typed into the form wins, otherwise the stored credential or the ambient environment is probed, and a route with no key is probed unauthenticated so the surface can still answer "what does this endpoint serve".

## Dynamic configuration (settings + credentials)

Connection facts are not frozen at load. `resolveAdapterOptions` is the one explicit resolve step from raw config to validated facts, and the adapter re-reads them through a thunk **once per operation**: base URL, catalog, request defaults, and idle budget all take effect on the next request, while an in-flight stream keeps the facts it started with. Two optional seams feed that thunk:

- **`ctx.settings`** — the plugin registers the `llm-siliconflow` namespace with this same `Config` schema and its `cordis.yml` entry as the composition `base`, so a `llm-siliconflow:` section in the user settings document overrides any field without a restart. Without a mounted settings service the entry config alone drives the adapter, unchanged. A live settings snapshot that passes the schema but fails a beyond-schema bound (a duplicate catalog id) keeps the last good facts and logs the failure; the entry config itself still fails plugin load.
- **`ctx.credentials`** — the API key resolves per stream call, from the *same* resolved snapshot that supplies the endpoint. Configuration carries only `apiKeyEnv`, never a literal key: the reference resolves through the credential seam, and without a mounted seam through the trusted environment layers. Because credential facts travel with the connection facts, a settings snapshot the resolver rejects contributes neither its endpoint nor its key. Every resolved key is format-checked before use, so a value no HTTP header can carry is refused with `LlmError('INVALID_CREDENTIAL')` naming the failing entry point — never any part of the key. A request with no key anywhere fails with `MISSING_CREDENTIAL` naming every configuration entry point, while the route stays registered and the catalog stays browsable.

The one registration-captured fact is the retry policy: when its resolved value changes, the plugin re-registers the route in place (same adapter instance, one synchronous section), so `ctx.llm.providerRetryPolicy('siliconflow')` always reports the current policy. The plugin also declares its route in the configurable-provider directory (`ctx.llm.listConfigurableProviders()`): provider `siliconflow`, settings namespace `llm-siliconflow`, empty settings path.

## App attribution

Every request carries the shared attribution header from dsh-llm's `attributionHeaders()` — the mandatory `User-Agent` baseline identifying the harness. After credential resolution, every provider request carries `x-siliconflow-harness-user-id` with the stable anonymous id from `@deepseek-ai/dsh-anonymous-user-id`; a request carrying `GenerateOptions.sessionId` also sends that exact value as `x-siliconflow-harness-session-id`, while a direct call without a session omits the session header. A request whose `GenerateOptions.purpose` is `compaction` additionally carries `x-siliconflow-harness-compact: 1`. All three headers go to the resolved `baseURL` and remain outside the request body and model-visible content.

## Wire-format notes

- Streaming only (`stream_options.include_usage` always on). `usage` may arrive attached to the finish chunk or as a trailing usage-only chunk — the translator defers both to `[DONE]`, so `usage` always precedes `finish` and nothing follows `finish`.
- The adapter never sends `thinking` or `reasoning_effort`; SiliconFlow's endpoint has no such knobs, and a reasoning model is chosen by id.
- The first reasoning-model chunk carries `reasoning_content: ""` — handled (no spurious reasoning block).
- **Reasoning passback rule**: on assistant turns that carried tool calls, `reasoning_content` is serialized back in history (required by hosted DeepSeek-R1-style models); on tool-call-free turns it is dropped (ignored anyway — saves tokens).
- Cache accounting: `cacheReadTokens` ← `prompt_cache_hit_tokens` / `prompt_tokens_details.cached_tokens`; SiliconFlow reports no cache-write metric.

## Errors

Non-2xx responses throw `LlmError` with stable codes: `AUTH` (401/403), `QUOTA` (a response whose provider details identify exhausted quota, balance, or credits), `RATE_LIMIT` (other 429s), `CONTEXT_WINDOW_EXCEEDED` (a 400 whose provider code, type, or message identifies context overflow), `INVALID_REQUEST` (other 400s), `SERVER` (5xx), `HTTP_<status>` otherwise. Its serializable `failure` retains the HTTP status plus a valid positive `Retry-After` seconds/date delay and `x-request-id` when present. A pre-response transport failure (DNS, refused connection, TLS, proxy) throws `TRANSPORT` naming the configured endpoint and chaining the original rejection as `cause`; caller aborts throw `ABORTED`, and the loop's cancellation signal remains authoritative. Protocol violations throw `STREAM_CLOSED` (no `[DONE]`) or `MALFORMED_RESPONSE` (bad JSON payload). Unknown wire `finish_reason`s (e.g. `content_filter`, `insufficient_system_resource`) become `finish {kind: 'error', failure}` chunks, and a completed stream whose `stop` (or absent) finish opened no content blocks becomes a `finish {kind: 'error'}` with code `EMPTY_RESPONSE` (retried by default policy).

## Model Experience

### SiliconFlow request

#### What the model sees

The selected SiliconFlow model receives the harness system prompt, message history, tool schemas, stop sequences, and call config without adapter-authored prompt prose. On a prior assistant turn with tool calls, its reasoning content is passed back as required; reasoning from tool-call-free turns is omitted.

#### Token effect

Provider tokenization governs exact input. Conditional reasoning passback increases tool-round-trip context, while dropping other reasoning avoids paying those tokens again; cache-read usage is reported when available.

#### KV Cache effect

An unchanged assembled prefix is eligible for provider cache reuse, which this adapter reports in usage. A model-route change or any upstream prompt, schema, prefix, or history change may prevent reuse from the first changed token; reasoning passback appends during tool round trips.

### SiliconFlow response

#### What the model sees

Reasoning, text, and raw-string tool arguments are translated into harness chunks for the loop to log and assemble.

#### Token effect

Generated tokens follow the request's logged `maxTokens`; only loop-retained blocks affect later input.

#### KV Cache effect

Loop-retained response blocks append to the next request and preserve its earlier reusable prefix; dropped blocks have no later cache effect. Changing the provider or model selects a different cache domain.

## Known Limitations and Deferred Work

- **The fallback `models` list is hand-maintained** — the six defaults are a small snapshot, shown only while discovery cannot run; the live listing is the authoritative catalog.
- **Discovery does not cache across a baseURL change** — the cache is keyed by endpoint, so repointing the route re-interrogates on the next `listModels`.
- **A settings `models` list replaces the composition list wholesale** — settings-layer merging is per-field, and arrays are one field; per-entry catalog merging would need a keyed shape.
- **`tool_choice` is not mapped** — not part of the core vocabulary (MVP cut, shared with the pi-ai and DeepSeek twins).
- **Requests use raw `fetch`, not `@cordisjs/plugin-http`** — no shared proxy/interception configuration; adoption is deferred until a second direct-fetch adapter wants it (`TODO(http)`).
- **Serialization flattens user and tool-result content to text blocks** — plugin-added block types are skipped, and empty tool output crosses the wire as the literal `(no output)`.
- **Image content is rejected** — the chat-completions wire route here is text-only; a multimodal SiliconFlow route would need its own content serializer.
