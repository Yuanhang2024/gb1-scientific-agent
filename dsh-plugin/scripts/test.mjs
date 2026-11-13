import { mkdir, readdir } from 'node:fs/promises'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'
import { build } from 'esbuild'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const pkg = resolve(root, 'packages/gb1')
const tests = (await readdir(resolve(pkg, 'test'))).filter(name => name.endsWith('.test.ts'))
const outputs = []
await mkdir(resolve(pkg, 'lib-test'), { recursive: true })
for (const name of tests) {
  const output = resolve(pkg, 'lib-test', name.replace(/\.ts$/, '.mjs'))
  await build({ entryPoints: [resolve(pkg, 'test', name)], outfile: output,
    bundle: true, packages: 'external', platform: 'node', format: 'esm', target: 'node24', sourcemap: false })
  outputs.push(output)
}
const result = spawnSync(process.execPath, ['--test', ...outputs], { cwd: root, stdio: 'inherit' })
process.exitCode = result.status ?? 1
