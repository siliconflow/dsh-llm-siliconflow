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

  it('upgrades a pre-release flat document in place, keeping every entry and comment', async () => {
    const path = pathOf('credentials.yaml')
    await writeFile(path, '# keep me\nOTHER_KEY: sk-other\nSILICONFLOW_API_KEY: sk-flat\n')
    await expect(readCredential(path, DEFAULT_API_KEY_ENV)).resolves.toBe('sk-flat')
    await writeCredential(path, DEFAULT_API_KEY_ENV, 'sk-new')
    const text = await readFile(path, 'utf8')
    expect(text).toContain('version: 1')
    expect(text).toContain('# keep me')
    expect(text).toContain('OTHER_KEY: sk-other')
    await expect(readCredential(path, DEFAULT_API_KEY_ENV)).resolves.toBe('sk-new')
    await expect(readCredential(path, 'OTHER_KEY')).resolves.toBe('sk-other')
    expect(text.indexOf('version: 1')).toBeLessThan(text.indexOf('refs:'))
  })

  it('reads a versioned document, its other refs, and a records-only document', async () => {
    const path = pathOf('credentials.yaml')
    await writeFile(path, 'version: 1\nrefs:\n  SILICONFLOW_API_KEY: sk-versioned\n  OTHER_KEY: sk-other\n')
    await expect(readCredential(path, DEFAULT_API_KEY_ENV)).resolves.toBe('sk-versioned')
    await expect(readCredential(path, 'OTHER_KEY')).resolves.toBe('sk-other')

    const recordsOnly = pathOf('records-only.yaml')
    await writeFile(recordsOnly, 'version: 1\nrecords:\n  a/b:\n    kind: api-key\n')
    await expect(readCredential(recordsOnly, DEFAULT_API_KEY_ENV)).resolves.toBeUndefined()

    const unversioned = pathOf('empty.yaml')
    await writeFile(unversioned, '')
    await expect(readCredential(unversioned, DEFAULT_API_KEY_ENV)).resolves.toBeUndefined()
  })

  it('writes into a versioned document without duplicating the version stamp', async () => {
    const path = pathOf('credentials.yaml')
    await writeFile(path, `version: 1\nrecords:\n  a/b:\n    kind: api-key\n    key: sk-env\n    env:\n      ${DEFAULT_API_KEY_ENV}: sk-record\n`)
    await writeCredential(path, DEFAULT_API_KEY_ENV, 'sk-added')
    const text = await readFile(path, 'utf8')
    expect(text).toContain('sk-added')
    expect(text.match(/version: 1/g)).toHaveLength(1)
    expect(text).toContain('kind: api-key')
    await expect(readCredential(path, DEFAULT_API_KEY_ENV)).resolves.toBe('sk-added')
  })

  it('rejects writing a document that is neither flat nor version-1', async () => {
    const path = pathOf('credentials.yaml')
    await writeFile(path, 'version: 2\nrefs:\n  KEY: v\n')
    await expect(writeCredential(path, DEFAULT_API_KEY_ENV, 'sk')).rejects.toThrow(/not a recognizable credentials document/)
    await expect(readCredential(path, DEFAULT_API_KEY_ENV)).resolves.toBeUndefined()
  })

  it('reads undefined for scalar roots, scalar refs sections, and non-identifier ref names', async () => {
    const scalarRoot = pathOf('scalar-root.yaml')
    await writeFile(scalarRoot, '42\n')
    await expect(readCredential(scalarRoot, DEFAULT_API_KEY_ENV)).resolves.toBeUndefined()
    await expect(writeCredential(scalarRoot, DEFAULT_API_KEY_ENV, 'sk')).rejects.toThrow()
    const scalarRefs = pathOf('scalar-refs.yaml')
    await writeFile(scalarRefs, 'version: 1\nrefs: 5\n')
    await expect(readCredential(scalarRefs, DEFAULT_API_KEY_ENV)).resolves.toBeUndefined()
    await expect(writeCredential(scalarRefs, DEFAULT_API_KEY_ENV, 'sk')).rejects.toThrow()
    const badName = pathOf('bad-name.yaml')
    await writeFile(badName, 'version: 1\nrefs:\n  BAD KEY: v\n')
    await expect(readCredential(badName, DEFAULT_API_KEY_ENV)).resolves.toBeUndefined()
    await expect(writeCredential(badName, DEFAULT_API_KEY_ENV, 'sk')).rejects.toThrow()
  })

  it('reads undefined for unparseable, non-mapping, and unknown-version documents', async () => {
    const parseError = pathOf('parse-error.yaml')
    await writeFile(parseError, 'key: [unclosed\n')
    await expect(readCredential(parseError, DEFAULT_API_KEY_ENV)).resolves.toBeUndefined()
    await expect(writeCredential(parseError, DEFAULT_API_KEY_ENV, 'sk')).rejects.toThrow()
    const listRoot = pathOf('list-root.yaml')
    await writeFile(listRoot, '- just\n- a\n- list\n')
    await expect(readCredential(listRoot, DEFAULT_API_KEY_ENV)).resolves.toBeUndefined()
    await expect(writeCredential(listRoot, DEFAULT_API_KEY_ENV, 'sk')).rejects.toThrow()
  })

  it('repairs the corrupted version-1 layout a pre-fix wizard wrote, folding its top-level key into refs', async () => {
    // exactly the file dsh >= 0.1.2 rejects with unknown top-level key
    const path = pathOf('credentials.yaml')
    await writeFile(path, 'version: 1\nrefs:\n  DEEPSEEK_API_KEY: sk-kept\nSILICONFLOW_API_KEY: sk-leftover\n')
    await expect(readCredential(path, DEFAULT_API_KEY_ENV)).resolves.toBeUndefined()
    await writeCredential(path, DEFAULT_API_KEY_ENV, 'sk-repair')
    const text = await readFile(path, 'utf8')
    expect(text).toContain('version: 1')
    expect(text.match(/version: 1/g)).toHaveLength(1)
    expect(text).toContain('DEEPSEEK_API_KEY: sk-kept')
    expect(text).toContain('SILICONFLOW_API_KEY: sk-repair')
    expect(text).not.toMatch(/^SILICONFLOW_API_KEY:/m)
    await expect(readCredential(path, DEFAULT_API_KEY_ENV)).resolves.toBe('sk-repair')
    await expect(readCredential(path, 'DEEPSEEK_API_KEY')).resolves.toBe('sk-kept')
  })

  it('refuses corrupted layouts whose extra top-level key is not a reference over a non-empty string', async () => {
    const badName = pathOf('extra-bad-name.yaml')
    await writeFile(badName, 'version: 1\nrefs:\n  A: a\nNOT A REF: v\n')
    await expect(readCredential(badName, DEFAULT_API_KEY_ENV)).resolves.toBeUndefined()
    await expect(writeCredential(badName, DEFAULT_API_KEY_ENV, 'sk')).rejects.toThrow(/not a recognizable/)
    const empty = pathOf('extra-empty.yaml')
    await writeFile(empty, 'version: 1\nrefs:\n  A: a\nEMPTY_KEY: ""\n')
    await expect(readCredential(empty, DEFAULT_API_KEY_ENV)).resolves.toBeUndefined()
    await expect(writeCredential(empty, DEFAULT_API_KEY_ENV, 'sk')).rejects.toThrow(/not a recognizable/)
    const numeric = pathOf('extra-numeric.yaml')
    await writeFile(numeric, 'version: 1\nrefs:\n  A: a\nNUMERIC_KEY: 42\n')
    await expect(readCredential(numeric, DEFAULT_API_KEY_ENV)).resolves.toBeUndefined()
    await expect(writeCredential(numeric, DEFAULT_API_KEY_ENV, 'sk')).rejects.toThrow(/not a recognizable/)
  })

  it('reads undefined for non-reference names and empty values in either layout', async () => {
    const badFlat = pathOf('bad-flat.yaml')
    await writeFile(badFlat, 'key with space: v\n')
    await expect(readCredential(badFlat, DEFAULT_API_KEY_ENV)).resolves.toBeUndefined()
    const badVersioned = pathOf('bad-versioned.yaml')
    await writeFile(badVersioned, 'version: 1\nrefs:\n  OTHER:\n    nested: map\n')
    await expect(readCredential(badVersioned, DEFAULT_API_KEY_ENV)).resolves.toBeUndefined()
    const emptyFlat = pathOf('empty-flat.yaml')
    await writeFile(emptyFlat, 'SILICONFLOW_API_KEY: ""\n')
    await expect(readCredential(emptyFlat, DEFAULT_API_KEY_ENV)).resolves.toBeUndefined()
    const typeError = pathOf('type-error.yaml')
    await writeFile(typeError, 'SILICONFLOW_API_KEY: 42\n')
    await expect(readCredential(typeError, DEFAULT_API_KEY_ENV)).resolves.toBeUndefined()
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
