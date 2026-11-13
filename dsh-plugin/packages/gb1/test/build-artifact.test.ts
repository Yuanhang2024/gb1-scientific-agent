import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { publishHostArtifact, withCanonicalTypertExport } from '../../../scripts/host-artifact.mjs'
const canonical = { types: './lib/typert.host.d.ts', default: './lib/typert.host.js' }
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), 'gb1-build-artifact-'))
  await mkdir(join(root, 'lib'))
  const original = { name: 'dsh-gb1-scientific-agent', version: '0.4.0', exports: { './typert': { ...canonical, default: './lib/typert.host.0.4.0.123.js' } }, files: ['lib/typert.host.js'], description: 'Original' }
  await writeFile(join(root, 'package.json'), JSON.stringify(original))
  await writeFile(join(root, 'lib/typert.host.js'), 'export const TYPERT = { package: "test" }\n')
  return { root, path: join(root, 'package.json'), original }
}
test('repeated builds present canonical exports to the generator and publish fresh Host artifact URLs', async () => {
  const f = await fixture(), artifacts: string[] = []
  for (let index = 0; index < 2; index++) {
    await withCanonicalTypertExport(f.path, async () => {
      assert.deepEqual(JSON.parse(await readFile(f.path, 'utf8')).exports['./typert'], canonical)
      const output = await publishHostArtifact(f.root)
      const current = JSON.parse(await readFile(f.path, 'utf8'))
      assert.equal(current.exports['./typert'].default, './lib/' + output.hostArtifact)
      assert.equal(await readFile(join(f.root, 'lib', output.hostArtifact), 'utf8'), await readFile(join(f.root, 'lib/typert.host.js'), 'utf8'))
      assert.ok(current.files.includes('lib/typert.host.*.js')); artifacts.push(output.hostArtifact)
    })
  }
  assert.notEqual(artifacts[0], artifacts[1])
})
test('failed generation restores its prior type export while preserving other manifest edits', async () => {
  const f = await fixture()
  await assert.rejects(withCanonicalTypertExport(f.path, async () => {
    const current = JSON.parse(await readFile(f.path, 'utf8'))
    assert.deepEqual(current.exports['./typert'], canonical)
    current.description = 'Updated during build'
    await writeFile(f.path, JSON.stringify(current))
    throw new Error('Synthetic generator failure')
  }), /Synthetic generator failure/)
  const restored = JSON.parse(await readFile(f.path, 'utf8'))
  assert.deepEqual(restored.exports['./typert'], f.original.exports['./typert'])
  assert.equal(restored.description, 'Updated during build')
})
