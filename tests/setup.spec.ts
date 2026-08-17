import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_API_KEY_ENV, DEFAULT_MODELS, PROVIDER } from '../src/index.ts'
import {
  credentialsPath,
  parseModelIndex,
  readCredential,
  readDefaultModel,
  resolveHome,
  runSetup,
  settingsPath,
  writeCredential,
  writeDefaultModel,
  type SetupIo,
} from '../src/setup.ts'

let dir: string

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'dsh-setup-'))
})

afterEach(async () => {
  vi.restoreAllMocks()
  await rm(dir, { recursive: true, force: true })
})

function pathOf(name: string): string {
  return join(dir, name)
}

describe('resolveHome', () => {
  it('uses a non-empty DSH_HOME', () => {
    expect(resolveHome({ DSH_HOME: '/tmp/dsh-home' })).toBe('/tmp/dsh-home')
  })

  it('ignores a blank DSH_HOME', () => {
    expect(resolveHome({ DSH_HOME: '   ' }).endsWith(join('.dsh'))).toBe(true)
  })

  it('falls back to ~/.dsh without DSH_HOME', () => {
    expect(resolveHome({}).endsWith(join('.dsh'))).toBe(true)
  })
})

describe('document paths', () => {
  it('joins the credential and settings files under the home', () => {
    expect(credentialsPath('/home/dsh')).toBe(join('/home/dsh', '.credentials.yaml'))
    expect(settingsPath('/home/dsh')).toBe(join('/home/dsh', 'settings.yaml'))
  })
})

describe('credentials', () => {
  it('reads undefined when the file is absent', async () => {
    await expect(readCredential(pathOf('missing.yaml'), DEFAULT_API_KEY_ENV)).resolves.toBeUndefined()
  })

  it('reads undefined when the reference is absent or empty', async () => {
    const path = pathOf('credentials.yaml')
    await writeFile(path, 'OTHER_KEY: sk-other\n')
    await expect(readCredential(path, DEFAULT_API_KEY_ENV)).resolves.toBeUndefined()
  })

  it('writes a reference and preserves other entries and comments', async () => {
    const path = pathOf('credentials.yaml')
    await writeFile(path, '# keep me\nOTHER_KEY: sk-other\n')
    await writeCredential(path, DEFAULT_API_KEY_ENV, 'sk-new')
    const text = await readFile(path, 'utf8')
    expect(text).toContain('# keep me')
    expect(text).toContain('OTHER_KEY: sk-other')
    await expect(readCredential(path, DEFAULT_API_KEY_ENV)).resolves.toBe('sk-new')
  })

  it('creates the document when absent', async () => {
    const path = pathOf('credentials.yaml')
    await writeCredential(path, DEFAULT_API_KEY_ENV, 'sk-fresh')
    await expect(readCredential(path, DEFAULT_API_KEY_ENV)).resolves.toBe('sk-fresh')
  })

  it('rethrows a read error that is not ENOENT', async () => {
    await expect(readCredential(dir, DEFAULT_API_KEY_ENV)).rejects.toThrow()
    await expect(writeCredential(dir, DEFAULT_API_KEY_ENV, 'sk')).rejects.toThrow()
  })
})

describe('default model', () => {
  it('reads undefined when the file is absent', async () => {
    await expect(readDefaultModel(pathOf('missing.yaml'))).resolves.toBeUndefined()
  })

  it('reads undefined when the section is not an object', async () => {
    const path = pathOf('settings.yaml')
    await writeFile(path, 'agent-default-model: null\n')
    await expect(readDefaultModel(path)).resolves.toBeUndefined()
  })

  it('reads undefined when the section is incomplete', async () => {
    const path = pathOf('settings.yaml')
    await writeFile(path, 'agent-default-model:\n  provider: siliconflow\n')
    await expect(readDefaultModel(path)).resolves.toBeUndefined()
  })

  it('writes the selection and drops a previous reasoningEffort', async () => {
    const path = pathOf('settings.yaml')
    await writeFile(path, '# keep\nother: value\nagent-default-model:\n  provider: old\n  model: old-model\n  reasoningEffort: high\n')
    await writeDefaultModel(path, { provider: PROVIDER, model: 'deepseek-ai/DeepSeek-V4-Flash' })
    const text = await readFile(path, 'utf8')
    expect(text).toContain('# keep')
    expect(text).toContain('other: value')
    expect(text).not.toContain('reasoningEffort')
    await expect(readDefaultModel(path)).resolves.toEqual({ provider: PROVIDER, model: 'deepseek-ai/DeepSeek-V4-Flash' })
  })

  it('rethrows a read error that is not ENOENT', async () => {
    await expect(readDefaultModel(dir)).rejects.toThrow()
  })
})

describe('parseModelIndex', () => {
  it('parses a valid 1-based answer', () => {
    expect(parseModelIndex('2', 3)).toBe(1)
  })

  it('rejects zero, out-of-range, and non-integer answers', () => {
    expect(parseModelIndex('0', 3)).toBeUndefined()
    expect(parseModelIndex('4', 3)).toBeUndefined()
    expect(parseModelIndex('abc', 3)).toBeUndefined()
    expect(parseModelIndex('1.5', 3)).toBeUndefined()
  })
})

interface Harness {
  io: SetupIo
  logs: string[]
}

function harness(answers: string[]): Harness {
  const logs: string[] = []
  let index = 0
  return {
    logs,
    io: {
      question: vi.fn(async () => answers[index++] ?? ''),
      log: message => logs.push(message),
    },
  }
}

describe('runSetup', () => {
  it('skips when the user declines the default channel', async () => {
    const h = harness(['n'])
    await runSetup({
      home: dir,
      io: h.io,
      discover: vi.fn(async () => { throw new Error('unreached') }),
    })
    expect(h.logs.some(line => line.includes('已跳过'))).toBe(true)
    await expect(readDefaultModel(settingsPath(dir))).resolves.toBeUndefined()
  })

  it('persists the chosen model and key from a live listing', async () => {
    const h = harness(['', 'sk-live', '2'])
    const discover = vi.fn(async () => [
      { id: 'org/Alpha', name: 'Alpha' },
      { id: 'org/Beta' },
    ])
    await runSetup({ home: dir, io: h.io, discover })
    expect(discover).toHaveBeenCalledWith('sk-live')
    await expect(readDefaultModel(settingsPath(dir))).resolves.toEqual({ provider: PROVIDER, model: 'org/Beta' })
    await expect(readCredential(credentialsPath(dir), DEFAULT_API_KEY_ENV)).resolves.toBe('sk-live')
    expect(h.logs.some(line => line.includes('Alpha'))).toBe(true)
  })

  it('reuses an existing key and defaults to the first model on an empty answer', async () => {
    const cred = credentialsPath(dir)
    await writeCredential(cred, DEFAULT_API_KEY_ENV, 'sk-existing')
    const h = harness(['', ''])
    const discover = vi.fn(async () => [{ id: 'org/Alpha' }])
    await runSetup({ home: dir, io: h.io, discover })
    expect(discover).toHaveBeenCalledWith('sk-existing')
    expect(h.logs.some(line => line.includes('已找到'))).toBe(true)
    await expect(readDefaultModel(settingsPath(dir))).resolves.toEqual({ provider: PROVIDER, model: 'org/Alpha' })
  })

  it('falls back to the static catalog when discovery fails, listing without a key', async () => {
    const h = harness(['', '', '1'])
    const discover = vi.fn(async () => { throw new Error('down') })
    await runSetup({ home: dir, io: h.io, discover })
    expect(h.logs.some(line => line.includes('回退到默认目录'))).toBe(true)
    await expect(readDefaultModel(settingsPath(dir))).resolves.toEqual({
      provider: PROVIDER,
      model: DEFAULT_MODELS[0]?.id,
    })
  })

  it('falls back to the static catalog when discovery lists nothing', async () => {
    const h = harness(['', ''])
    const discover = vi.fn(async () => [])
    await runSetup({ home: dir, io: h.io, discover })
    await expect(readDefaultModel(settingsPath(dir))).resolves.toEqual({
      provider: PROVIDER,
      model: DEFAULT_MODELS[0]?.id,
    })
  })

  it('reports a non-Error discovery rejection', async () => {
    const h = harness(['', '', '1'])
    const discover = vi.fn(async () => { throw 'down' })
    await runSetup({ home: dir, io: h.io, discover })
    expect(h.logs.some(line => line.includes('down'))).toBe(true)
  })

  it('re-prompts on an out-of-range choice', async () => {
    const h = harness(['', '', '99', '1'])
    const discover = vi.fn(async () => [{ id: 'org/Alpha' }])
    await runSetup({ home: dir, io: h.io, discover })
    expect(h.logs.some(line => line.includes('之间的编号'))).toBe(true)
    await expect(readDefaultModel(settingsPath(dir))).resolves.toEqual({ provider: PROVIDER, model: 'org/Alpha' })
  })
})
