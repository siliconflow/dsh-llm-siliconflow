/**
 * Single-build cross-era compatibility for the DeepSeek Harness dsh family.
 *
 * One published plugin build must run inside TWO dsh runtime closures:
 *
 * - the NEW era (dsh 0.1.2+ LLM family, first shipped 2026-08-30):
 *   `ToolCallId` brand, `version: 1` credentials layout (`refs:` nesting,
 *   unknown top-level keys hard-fail), settings installed via the service
 *   method `installSection`, model discovery via `LlmModelDiscoveryOperation`
 *   (extends `LlmModelDiscoveryRequest` with optional `signal`).
 * - the OLD era (dsh-llm 0.1.1 and earlier; credentials-local 0.1.0):
 *   `CallId` brand, flat top-level credentials (no `version` key),
 *   settings installed via the module function `installSettingsSection`,
 *   model discovery via `LlmModelDiscoveryRequest` carrying `signal`.
 *
 * Compatibility rules this module relies on, established against real
 * published packages (npm tarballs), not assumptions:
 *
 * 1. ESM link-time export validation only applies to NAMED imports
 *    (`import { ToolCallId }`). A missing named export throws SyntaxError at
 *    link time and the module never evaluates — exactly the reported
 *    `SyntaxError: … does not provide an export named 'CallId'`. A NAMESPACE
 *    import (`import * as llm`) links unconditionally; missing members are
 *    simply `undefined` at runtime. Every era-sensitive value therefore comes
 *    from the namespace object, never from a named import.
 * 2. Exports BOTH eras ship (verified in 0.1.1-rc.2 and 0.1.5-rc.2):
 *    EMPTY_RESPONSE_CODE, LlmError, LlmAdapter, ProviderRequestId,
 *    QUOTA_EXCEEDED_CODE, RetryPolicySchema, assertUsableApiKey,
 *    resolveRetryPolicy, attributionHeaders, isQuotaExceededError,
 *    isContextWindowExceededError, CONTEXT_WINDOW_EXCEEDED_CODE. Named
 *    imports of these stay safe in both eras.
 * 3. `dsh-util-values` postdates the OLD era entirely (404 there); its one
 *    consumer in this plugin, `deepEqualJson`, is inlined below.
 *
 * The era probe: `typeof ns.ToolCallId === 'function'`. NEW era exports
 * ToolCallId; OLD era exports CallId instead. One probe classifies the
 * closure and stays stable under future renames.
 *
 * @module dsh-llm-siliconflow/dsh-era
 */

import * as llm from '@deepseek-ai/dsh-llm'
import type { ToolCallId as ToolCallIdType } from '@deepseek-ai/dsh-llm'

/**
 * The raw namespace, exported for era-sensitive call sites. Typed loosely:
 * the OLD-era variant of a member is unknown to the NEW-era .d.ts and vice
 * versa, so era-shape access happens through this record, never the typed
 * import.
 */
export const dshLLM = llm as unknown as Record<string, unknown>

/** True when the running dsh closure is the NEW era (ToolCallId present). */
export const isNewEra: boolean = typeof dshLLM['ToolCallId'] === 'function'

/**
 * Construct a provider tool-call id brand for the running era: `ToolCallId`
 * in the NEW era, `CallId` in the OLD. The constructor is resolved at call
 * time from the live namespace; a closure exposing neither fails loud.
 *
 * @param raw - the raw provider id, e.g. `block.callId ?? ''`.
 * @returns the era-correct branded id.
 */
export function toolCallIdOf(raw: string): ToolCallIdType {
  const ctor = (dshLLM['ToolCallId'] ?? dshLLM['CallId']) as ((raw: string) => ToolCallIdType) | undefined
  if (ctor === undefined) throw new Error('neither ToolCallId nor CallId is exported by @deepseek-ai/dsh-llm')
  return ctor(raw)
}

/**
 * Structural JSON equality over plain values (the shape settings documents
 * and retry policies are made of). Inlined from dsh-util-values so the plugin
 * carries no dependency on a NEW-era-only package: the OLD-era closure cannot
 * even install dsh-util-values, which used to break npm resolution of this
 * plugin under old dsh homes.
 *
 * Owned here over the value domain the plugin actually observes (plain JSON
 * documents): recursion by key set, so object member order never matters —
 * the upstream package compares structurally, and JSON.stringify would make
 * order significant, corrupting that contract.
 *
 * @param a - one plain JSON value.
 * @param b - another plain JSON value.
 * @returns true when both represent the same JSON value.
 */
export function deepEqualJson(a: unknown, b: unknown): boolean {
  if (a === b) return true
  if (a === null || b === null) return false
  if (typeof a !== typeof b) return false
  if (typeof a !== 'object') return a === b
  const ao = a as Record<string, unknown>
  const bo = b as Record<string, unknown>
  if (Array.isArray(ao) !== Array.isArray(bo)) return false
  if (Array.isArray(ao) && Array.isArray(bo)) {
    if (ao.length !== bo.length) return false
    return ao.every((item, i) => deepEqualJson(item, bo[i]))
  }
  const ak = Object.keys(ao)
  const bk = Object.keys(bo)
  if (ak.length !== bk.length) return false
  if (!ak.every((k) => k in bo)) return false
  return ak.every((k) => deepEqualJson(ao[k], bo[k]))
}

/**
 * The era-neutral discovery-request shape one callback can serve both eras
 * from: the fields the SiliconFlow discovery callback reads. In the OLD era
 * the caller passes `LlmModelDiscoveryRequest` (signal present as a field);
 * in the NEW era `LlmModelDiscoveryOperation` (same field, optional).
 */
export interface ModelDiscoveryShape {
  /** Route being edited, when it edits an existing one. */
  provider?: string
  /** Endpoint to interrogate. */
  baseURL?: string
  /** Wire protocol the endpoint speaks, when the draft names one. */
  api?: string
  /** Credential for this interrogation alone; never stored by the harness. */
  apiKey?: string
  /** Caller cancellation; present in both eras, optional in the new one. */
  signal?: AbortSignal
}

/**
 * The era-neutral settings-hooks shape. The OLD and NEW installers take
 * identical hook objects; only the mount point moved between eras.
 * Parameterized on the entry type so it flows through without variance
 * friction against exactOptionalPropertyTypes.
 */
export interface EraSettingsHooks<T> {
  setSource: (source: () => T) => void
  onChange: () => void
}

/**
 * Structural mirror of the NEW-era settings namespace input type (a lowercase
 * hyphenated identifier). Declared locally so this plugin typechecks under the
 * OLD-era closure, where `@deepseek-ai/dsh-settings` exports a differently
 * shaped `SettingsNamespace`; the namespace value itself is identical either way.
 */
export type SettingsNamespaceInput = string

/**
 * The era-union settings service seen by the plugin: the NEW era exposes
 * installSection as a service method; the OLD era exported
 * installSettingsSection as a module function (also re-exported by the
 * service object). Signatures are identical (owner, ns, schema, entry, hooks)
 * — only the mount point moved.
 */
export interface EraSettingsService {
  installSection?: <T>(owner: unknown, ns: SettingsNamespaceInput, schema: unknown, entry: T, hooks: EraSettingsHooks<T>) => void
  installSettingsSection?: <T>(owner: unknown, ns: SettingsNamespaceInput, schema: unknown, entry: T, hooks: EraSettingsHooks<T>) => void
}
