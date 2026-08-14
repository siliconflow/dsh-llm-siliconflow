import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { createUserMessage } from '@deepseek-ai/dsh-llm'
import type { Message } from '@deepseek-ai/dsh-llm'
import { LocalCredentialProvider } from '@deepseek-ai/dsh-credentials-local'
import * as LlmSiliconFlow from '../src/index.ts'
import { assemble, type AssembledResult } from './assemble.ts'

/**
 * Real-API e2e for the direct-fetch adapter against a well-hosted text model.
 * Key-gated — skips entirely without $SILICONFLOW_API_KEY (see
 * vitest.e2e.config.ts).
 */

const MODEL = 'deepseek-ai/DeepSeek-V4-Flash'
const contexts: Context[] = []
let identityHome: string

beforeEach(async () => {
  identityHome = await mkdtemp(join(tmpdir(), 'dsh-e2e-siliconflow-'))
  vi.stubEnv('DSH_HOME', identityHome)
})

afterEach(async () => {
  await Promise.all(contexts.splice(0).map(ctx => ctx.fiber.dispose()))
  vi.unstubAllEnvs()
  await rm(identityHome, { recursive: true, force: true })
})

function ask(text: string): Message[] {
  return [createUserMessage({
    content: [{ type: 'text', text }],
    source: { kind: 'plugin', plugin: 'test' },
  })]
}

function textOf(result: AssembledResult): string {
  return result.message.content
    .filter(block => block.type === 'text')
    .map(block => block.text)
    .join('')
}

describe.skipIf(!process.env.SILICONFLOW_API_KEY)('llm-siliconflow e2e (real API)', () => {
  it('serves a real request with the key held only by a credentials-local document', async () => {
    const key = process.env.SILICONFLOW_API_KEY
    if (key === undefined) throw new Error('e2e ran without SILICONFLOW_API_KEY')
    const dir = await mkdtemp(join(tmpdir(), 'dsh-e2e-siliconflow-credentials-'))
    try {
      await writeFile(join(dir, '.credentials.yaml'), `SILICONFLOW_API_KEY: ${JSON.stringify(key)}\n`, { mode: 0o600 })
      vi.stubEnv('SILICONFLOW_API_KEY', '')
      const ctx = new Context()
      contexts.push(ctx)
      await ctx.plugin(LlmRuntime)
      await ctx.plugin(LocalCredentialProvider, { path: join(dir, '.credentials.yaml'), watch: false })
      await ctx.plugin(LlmSiliconFlow, {})

      const result = await assemble(ctx, {
        model: MODEL,
        messages: ask('Reply with exactly the word: pong'),
        maxTokens: 50,
      })
      expect(result.finish.kind).toBe('stop')
      expect(textOf(result).toLowerCase()).toContain('pong')
    } finally {
      vi.unstubAllEnvs()
      await rm(dir, { recursive: true, force: true })
    }
  })

  it('streams raw chunks in protocol order', async () => {
    const ctx = new Context()
    contexts.push(ctx)
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmSiliconFlow, {})

    const kinds: string[] = []
    for await (const chunk of ctx.llm.stream({
      provider: 'siliconflow',
      model: MODEL,
      messages: ask('Count from 1 to 5, digits only.'),
      maxTokens: 50,
    })) {
      kinds.push(chunk.type)
    }
    expect(kinds[0]).toBe('block-start')
    expect(kinds.at(-1)).toBe('finish')
    expect(kinds.filter(kind => kind === 'finish')).toHaveLength(1)
    // usage precedes finish (deferred-emit contract)
    expect(kinds.indexOf('usage')).toBeLessThan(kinds.indexOf('finish'))
  })
})
