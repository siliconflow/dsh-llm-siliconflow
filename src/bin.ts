#!/usr/bin/env node
/**
 * Executable entry for the SiliconFlow setup wizard: wire the real terminal
 * and the public endpoint's live listing into {@link runSetup}. Kept as thin
 * glue so the wizard core is testable without a terminal.
 * @module @siliconflow-official/dsh-llm-siliconflow/bin
 */

import { createInterface } from 'node:readline/promises'
import { stdin, stdout } from 'node:process'
import { discoverChatModels } from './discovery.ts'
import { PUBLIC_BASE_URL } from './index.ts'
import { resolveHome, runSetup } from './setup.ts'

async function main(): Promise<void> {
  const io = createInterface({ input: stdin, output: stdout, terminal: true })
  try {
    await runSetup({
      home: resolveHome(),
      io: {
        question: prompt => io.question(prompt),
        log: message => stdout.write(`${message}\n`),
      },
      discover: apiKey => discoverChatModels(PUBLIC_BASE_URL, apiKey),
    })
  } finally {
    io.close()
  }
}

main().catch((error: unknown) => {
  stdout.write(`setup failed: ${error instanceof Error ? error.message : String(error)}\n`)
  process.exitCode = 1
})
