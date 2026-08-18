import { rmSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

// Clear the build output before tsc + tsdown repopulate it, so a previous
// build's code-split chunk (whose filename carries a content hash) cannot
// linger in the published tarball unreferenced.
rmSync(fileURLToPath(new URL('../lib', import.meta.url)), { recursive: true, force: true })
