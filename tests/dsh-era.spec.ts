import { describe, expect, it } from 'vitest'
import { deepEqualJson, dshLLM, isNewEra, toolCallIdOf, type ModelDiscoveryShape } from '../src/dsh-era.ts'
import * as llm from '@deepseek-ai/dsh-llm'
import { assembleSettingsSection } from '../src/index.ts'

describe('era probe', () => {
  it('classifies the running dsh closure consistently with its live exports', () => {
    const hasToolCallId = typeof (llm as unknown as Record<string, unknown>)['ToolCallId'] === 'function'
    const hasCallId = typeof (llm as unknown as Record<string, unknown>)['CallId'] === 'function'
    expect(isNewEra).toBe(hasToolCallId)
    expect(hasCallId || hasToolCallId).toBe(true)
  })

  it('dshLLM mirrors the live namespace', () => {
    expect(dshLLM['LlmError']).toEqual((llm as unknown as Record<string, unknown>)['LlmError'])
  })
})

describe('toolCallIdOf', () => {
  it('brands ids with the era-correct constructor', () => {
    const out = toolCallIdOf('x')
    expect(typeof out).toBe('string')
    // The brand must originate from the era-correct constructor, whichever ran.
    const ctor = (dshLLM['ToolCallId'] ?? dshLLM['CallId']) as (raw: string) => string
    expect(out).toBe(ctor('x'))
  })

  it('fails loud when neither constructor exists', () => {
    const saved = dshLLM['ToolCallId']
    const savedOld = dshLLM['CallId']
    delete dshLLM['ToolCallId']
    delete dshLLM['CallId']
    try {
      expect(() => toolCallIdOf('x')).toThrow('neither ToolCallId nor CallId')
    } finally {
      if (saved !== undefined) dshLLM['ToolCallId'] = saved
      if (savedOld !== undefined) dshLLM['CallId'] = savedOld
    }
  })

  it('returns undefined from a new-era constructor present-but-undefined probe path never produces branded ids', () => {
    // new era present case
    if (isNewEra) expect(toolCallIdOf('z')).toBe((dshLLM['ToolCallId'] as (r: string) => string)('z'))
  })
})

describe('deepEqualJson (inlined)', () => {
  it('equals for identical plain JSON values', () => {
    expect(deepEqualJson({ a: [1, { b: null }] }, { a: [1, { b: null }] })).toBe(true)
  })
  it('differs for different values, types, and key order within objects vs arrays', () => {
    expect(deepEqualJson({ a: 1 }, { a: 2 })).toBe(false)
    expect(deepEqualJson(1, '1')).toBe(false)
    expect(deepEqualJson(null, {})).toBe(false)
    expect(deepEqualJson([1, 2], [2, 1])).toBe(false)
  })
  it('treats object key order as insignificant', () => {
    expect(deepEqualJson({ x: 1, y: 2 }, { y: 2, x: 1 })).toBe(true)
  })
  it('compares arrays element-wise and length-wise', () => {
    expect(deepEqualJson([1, [2, { a: null }]], [1, [2, { a: null }]])).toBe(true)
    expect(deepEqualJson([1, 2], [1, 2, 3])).toBe(false)
    expect(deepEqualJson([1], {})).toBe(false)
  })
  it('rejects objects with different key sets and nested mismatches', () => {
    expect(deepEqualJson({ a: 1 }, { a: 1, b: 2 })).toBe(false)
    expect(deepEqualJson({ a: { b: [1] } }, { a: { b: [2] } })).toBe(false)
    expect(deepEqualJson({ a: undefined }, { a: undefined })).toBe(true)
    // same key count, but one key present only on the left
    expect(deepEqualJson({ a: 1, x: 2 }, { a: 1, y: 2 })).toBe(false)
  })
})

describe('ModelDiscoveryShape', () => {
  it('accepts both era payloads through one neutral shape', () => {
    const fromOld: ModelDiscoveryShape = { provider: 'siliconflow', baseURL: 'https://x', apiKey: 'sk', signal: new AbortController().signal }
    const fromNew: ModelDiscoveryShape = { baseURL: 'https://x' }
    expect(fromOld.signal?.aborted).toBe(false)
    expect(fromNew.signal).toBeUndefined()
  })
})

describe('assembleSettingsSection (era-adaptive settings install)', () => {
  const base = { owner: {} as unknown as import('@deepseek-ai/cordis').Context, ns: 'llm-siliconflow', schema: {}, entry: { a: 1 } }
  const hooks = { setSource: () => {}, onChange: () => {} }

  it('routes through installSection when the NEW-era method is present', () => {
    const calls: string[] = []
    const service = {
      installSection: (o: unknown, ns: string, sch: unknown, entry: unknown, h: object) => { void o; void ns; void sch; void entry; void h; calls.push('new') },
    }
    assembleSettingsSection(service, { ...base, setSource: hooks.setSource, onChange: hooks.onChange })
    expect(calls).toEqual(['new'])
  })

  it('routes through installSettingsSection for an OLD-era service', () => {
    const calls: string[] = []
    const service = {
      installSettingsSection: () => { calls.push('old') },
    }
    assembleSettingsSection(service, { ...base, setSource: hooks.setSource, onChange: hooks.onChange })
    expect(calls).toEqual(['old'])
  })

  it('fails loud when the service exposes neither installer', () => {
    expect(() => { assembleSettingsSection({}, { ...base, setSource: hooks.setSource, onChange: hooks.onChange }) }).toThrow('neither installSection')
  })

  it('prefers the NEW-era method when both exist (forward-correct under mixed closure)', () => {
    const calls: string[] = []
    const service = {
      installSection: () => { calls.push('new') },
      installSettingsSection: () => { calls.push('old') },
    }
    assembleSettingsSection(service, { ...base, setSource: hooks.setSource, onChange: hooks.onChange })
    expect(calls).toEqual(['new'])
  })
})
