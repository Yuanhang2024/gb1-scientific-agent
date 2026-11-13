import { randomUUID } from 'node:crypto'
import { copyFile, readFile, rename, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
const canonicalExport = { types: './lib/typert.host.d.ts', default: './lib/typert.host.js' }
let lastArtifactTimestamp = 0
async function saveManifest(path, manifest) {
  const temporary = `${path}.${randomUUID()}.tmp`
  await writeFile(temporary, JSON.stringify(manifest, null, 2) + '\n', { flag: 'wx' })
  await rename(temporary, path)
}
/** 官方生成器分析时使用标准路径；任何失败均恢复构建前的类型导出。 */
export async function withCanonicalTypertExport(manifestPath, action) {
  const original = JSON.parse(await readFile(manifestPath, 'utf8'))
  const previousExport = structuredClone(original.exports['./typert'])
  original.exports['./typert'] = { ...canonicalExport }
  await saveManifest(manifestPath, original)
  try { return await action() }
  catch (error) {
    const current = JSON.parse(await readFile(manifestPath, 'utf8'))
    current.exports['./typert'] = previousExport
    await saveManifest(manifestPath, current)
    throw error
  }
}
/** 独立文件URL允许已运行宿主显式导入本次生成的类型声明。 */
export async function publishHostArtifact(pkg) {
  const manifestPath = resolve(pkg, 'package.json')
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'))
  lastArtifactTimestamp = Math.max(Date.now(), lastArtifactTimestamp + 1)
  const hostArtifact = `typert.host.${manifest.version}.${lastArtifactTimestamp}.js`
  await copyFile(resolve(pkg, 'lib/typert.host.js'), resolve(pkg, 'lib', hostArtifact))
  manifest.exports['./typert'] = { ...canonicalExport, default: `./lib/${hostArtifact}` }
  if (!manifest.files.includes('lib/typert.host.*.js')) manifest.files.push('lib/typert.host.*.js')
  await saveManifest(manifestPath, manifest)
  return { manifest, hostArtifact }
}
