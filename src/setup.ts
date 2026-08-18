/**
 * Interactive setup wizard core for {@link @siliconflow-official/dsh-llm-siliconflow}:
 * the pure steps (harness-home and document-path resolution, credential and
 * settings read/write, model-choice parsing) plus the orchestration over an
 * injected I/O face, so the thin bin entry stays untested glue and the whole
 * flow is unit-testable without a terminal.
 *
 * The wizard guides a fresh install to a working route: confirm the default
 * channel, obtain an API key, interrogate the live chat-model listing, pick a
 * default model, then persist `agent-default-model` into `settings.yaml`.
 * @module @siliconflow-official/dsh-llm-siliconflow/setup
 */

import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { parseDocument } from 'yaml'
import type { SiliconFlowListingEntry } from './discovery.ts'
import { DEFAULT_API_KEY_ENV, DEFAULT_MODELS, PROVIDER } from './index.ts'

/** Directory name for the default DeepSeek Harness home under the OS home. */
export const DSH_HOME_DIR_NAME = '.dsh'
/** Environment variable that overrides the default DeepSeek Harness home. */
export const DSH_HOME_ENV = 'DSH_HOME'
/** Settings namespace the wizard writes the default model into. */
export const DEFAULT_MODEL_NAMESPACE = 'agent-default-model'

/** One model row the wizard offers, independent of the discovery source. */
export interface SetupModel {
  /** Provider-owned model id, exactly as written to `agent-default-model`. */
  id: string
  /** Optional display name; falls back to the id. */
  name?: string
}

/** Minimal terminal face the wizard talks through. */
export interface SetupIo {
  /** Ask one question and resolve with the trimmed-free answer. */
  question(prompt: string): Promise<string>
  /** Print one line of wizard progress. */
  log(message: string): void
}

/** Injected dependencies keeping the wizard testable and network-agnostic. */
export interface SetupDeps {
  /** Resolved harness home (`settings.yaml` and `.credentials.yaml` live here). */
  home: string
  /** The terminal face. */
  io: SetupIo
  /** Live chat-model listing; the wizard falls back to the static catalog on rejection. */
  discover: (apiKey: string | undefined) => Promise<readonly SiliconFlowListingEntry[]>
}

/** A persisted default-model selection. */
export interface DefaultModelSelection {
  provider: string
  model: string
}

/**
 * Resolve the DeepSeek Harness home: non-empty `$DSH_HOME`, else `~/.dsh`.
 * @param env - environment mapping; defaults to `process.env`.
 * @returns the normalized absolute home path.
 */
export function resolveHome(env: Record<string, string | undefined> = process.env): string {
  const fromEnv = env[DSH_HOME_ENV]
  return resolve(fromEnv !== undefined && fromEnv.trim().length > 0 ? fromEnv : join(homedir(), DSH_HOME_DIR_NAME))
}

/** The managed credential document path under a harness home. */
export function credentialsPath(home: string): string {
  return join(home, '.credentials.yaml')
}

/** The settings document path under a harness home. */
export function settingsPath(home: string): string {
  return join(home, 'settings.yaml')
}

function isEnoent(error: unknown): boolean {
  return (error as NodeJS.ErrnoException).code === 'ENOENT'
}

/**
 * Read one credential reference from a comment-preserving document.
 * @param path - the credentials document path.
 * @param keyEnv - the top-level reference name (e.g. `SILICONFLOW_API_KEY`).
 * @returns the stored value, or `undefined` when absent or the file is missing.
 */
export async function readCredential(path: string, keyEnv: string): Promise<string | undefined> {
  let text: string
  try {
    text = await readFile(path, 'utf8')
  } catch (error: unknown) {
    if (isEnoent(error)) return undefined
    throw error
  }
  const root: unknown = parseDocument(text).toJS()
  const value = (root as Record<string, unknown> | null)?.[keyEnv]
  return typeof value === 'string' && value.length > 0 ? value : undefined
}

/**
 * Set one credential reference, preserving every other entry and comment.
 * @param path - the credentials document path; created when absent.
 * @param keyEnv - the top-level reference name to write.
 * @param key - the value.
 */
export async function writeCredential(path: string, keyEnv: string, key: string): Promise<void> {
  const doc = await loadDocument(path)
  doc.set(keyEnv, key)
  await persistDocument(path, doc)
}

/**
 * Read the persisted default-model selection, if the section is a complete
 * `{provider, model}` map.
 * @param path - the settings document path.
 * @returns the selection, or `undefined` when absent or incomplete.
 */
export async function readDefaultModel(path: string): Promise<DefaultModelSelection | undefined> {
  let text: string
  try {
    text = await readFile(path, 'utf8')
  } catch (error: unknown) {
    if (isEnoent(error)) return undefined
    throw error
  }
  const root: unknown = parseDocument(text).toJS()
  const section = (root as Record<string, unknown> | null)?.[DEFAULT_MODEL_NAMESPACE]
  if (section === null || typeof section !== 'object') return undefined
  const provider = (section as Record<string, unknown>).provider
  const model = (section as Record<string, unknown>).model
  if (typeof provider !== 'string' || typeof model !== 'string') return undefined
  return { provider, model }
}

/**
 * Replace the default-model section, preserving every other section and comment.
 * The replacement carries only `provider` and `model`, so a previous
 * `reasoningEffort` (unsupported by SiliconFlow) is dropped with the section.
 * @param path - the settings document path; created when absent.
 * @param selection - the provider/model pair to persist.
 */
export async function writeDefaultModel(path: string, selection: DefaultModelSelection): Promise<void> {
  const doc = await loadDocument(path)
  doc.set(DEFAULT_MODEL_NAMESPACE, { provider: selection.provider, model: selection.model })
  await persistDocument(path, doc)
}

/**
 * Parse a 1-based model-choice answer.
 * @param raw - the user's answer.
 * @param count - the number of offered models.
 * @returns the 0-based index, or `undefined` when the answer is out of range.
 */
export function parseModelIndex(raw: string, count: number): number | undefined {
  const value = Number(raw.trim())
  if (!Number.isInteger(value) || value < 1 || value > count) return undefined
  return value - 1
}

/** A parsed YAML document, regardless of its root node kind. */
type ParsedDocument = ReturnType<typeof parseDocument>

async function loadDocument(path: string): Promise<ParsedDocument> {
  try {
    return parseDocument(await readFile(path, 'utf8'))
  } catch (error: unknown) {
    if (isEnoent(error)) return parseDocument('')
    throw error
  }
}

async function persistDocument(path: string, doc: ParsedDocument): Promise<void> {
  await mkdir(dirname(path), { recursive: true, mode: 0o700 })
  await writeFile(path, doc.toString(), { mode: 0o600 })
}

/** Render one model row as a display label. */
function labelOf(model: SetupModel): string {
  return model.name ?? model.id
}

/** Convert a discovery listing to wizard rows, keeping endpoint order. */
function rowsOf(entries: readonly SiliconFlowListingEntry[]): SetupModel[] {
  return entries.map(entry => ({ id: entry.id, ...entry.name === undefined ? {} : { name: entry.name } }))
}

/**
 * Run the wizard: confirm the default channel, obtain a key, list models,
 * pick the default, and persist the selection. Never throws for an unroutable
 * endpoint — discovery falls back to the static catalog — but a malformed
 * existing document or an unwritable one fails loud.
 * @param deps - home, terminal face, and discovery function.
 */
export async function runSetup(deps: SetupDeps): Promise<void> {
  const { home, io, discover } = deps
  const credentials = credentialsPath(home)
  const settings = settingsPath(home)

  const confirm = (await io.question('是否将 SiliconFlow 设为默认渠道？ [Y/n] ')).trim().toLowerCase()
  if (confirm === 'n') {
    io.log('已跳过。之后可在 web 的 Models 页配置，或手动编辑 settings.yaml。')
    return
  }

  let key = await readCredential(credentials, DEFAULT_API_KEY_ENV)
  if (key === undefined) {
    const answer = (await io.question(`未找到 ${DEFAULT_API_KEY_ENV}，请输入 API key（sk-...）：`)).trim()
    if (answer.length === 0) {
      io.log('未提供 key，将以未认证方式探测模型列表（列表可能受限）。')
    } else {
      key = answer
      await writeCredential(credentials, DEFAULT_API_KEY_ENV, key)
      io.log(`已写入 ${credentials}`)
    }
  } else {
    io.log(`已找到 ${DEFAULT_API_KEY_ENV}。`)
  }

  let models: SetupModel[]
  try {
    const listed = await discover(key)
    models = listed.length > 0 ? rowsOf(listed) : DEFAULT_MODELS
    if (listed.length > 0) io.log(`从端点获取到 ${String(listed.length)} 个模型。`)
  } catch (error: unknown) {
    models = DEFAULT_MODELS
    io.log(`实时获取模型列表失败，回退到默认目录（${error instanceof Error ? error.message : String(error)}）。`)
  }

  io.log('选择默认模型：')
  models.forEach((model, index) => {
    io.log(`  ${String(index + 1)}) ${labelOf(model)}`)
  })

  let index = 0
  for (;;) {
    const answer = (await io.question(`请选择 [1-${String(models.length)}]（默认 1）：`)).trim()
    if (answer.length === 0) break
    const parsed = parseModelIndex(answer, models.length)
    if (parsed === undefined) {
      io.log(`请输入 1-${String(models.length)} 之间的编号。`)
      continue
    }
    index = parsed
    break
  }

  const model = models[index]
  /* v8 ignore next 3 -- models is non-empty here: discovery rows or the non-empty static catalog both supply entries */
  if (model === undefined) {
    throw new Error('setup: no model selected')
  }
  await writeDefaultModel(settings, { provider: PROVIDER, model: model.id })
  io.log(`已写入 ${settings}：agent-default-model = ${PROVIDER} / ${model.id}`)
}
