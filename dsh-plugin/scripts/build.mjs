import { mkdir, writeFile } from 'node:fs/promises'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'
import { createRequire } from 'node:module'
import { build } from 'esbuild'
import { WorkspaceTypertGenerator, writeWorkspaceArtifacts } from '@deepseek-ai/dsh-typert-generator'
import { prepareSdkTypes } from './sdk-types.mjs'
import { publishHostArtifact, withCanonicalTypertExport } from './host-artifact.mjs'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const pkg = resolve(root, 'packages/gb1')
const require = createRequire(import.meta.url)
function typecheck(project) {
  const result = spawnSync(process.execPath, [require.resolve('typescript/bin/tsc'), '--project', project], { cwd: root, stdio: 'inherit' })
  if (result.status !== 0) throw new Error(`TypeScript检查未通过: ${project}`)
}

await withCanonicalTypertExport(resolve(pkg, 'package.json'), async () => {
  // 官方生成器只在构建时运行，输出Host校验和Client Remote。
  await prepareSdkTypes(root)
  await mkdir(resolve(pkg, 'lib'), { recursive: true })
  typecheck('packages/gb1/tsconfig.host.json')
  const generator = new WorkspaceTypertGenerator(root, { rejectBuildOutputs: true })
  const artifacts = generator.generateContributors(['host'])
  if (artifacts.length !== 1) throw new Error(`Expected one Host contribution, received ${artifacts.length}`)
  writeWorkspaceArtifacts(root, artifacts)
  await build({ entryPoints: [resolve(pkg, 'src/index.ts')], outfile: resolve(pkg, 'lib/index.js'),
    bundle: true, packages: 'external', platform: 'node', format: 'esm', target: 'node24', sourcemap: false })
  await build({ entryPoints: [resolve(pkg, 'src/contracts.ts')], outfile: resolve(pkg, 'lib/types/contracts.js'),
    bundle: true, platform: 'neutral', format: 'esm', target: 'es2022', sourcemap: false })

  if (!process.argv.includes('--host-only')) {
    typecheck('packages/gb1/tsconfig.client.json')
    const shared = ['react', 'react-dom', 'react/jsx-runtime', '@deepseek-ai/cordis',
      '@deepseek-ai/dsh-client-store', '@deepseek-ai/dsh-client-ui-primitives', '@deepseek-ai/dsh-client-ui-slots']
    const output = await build({ entryPoints: [resolve(pkg, 'src/client/index.tsx')], bundle: true, write: false,
      platform: 'browser', format: 'cjs', target: 'es2022', jsx: 'automatic', external: shared,
      loader: { '.css': 'text' }, sourcemap: false })
    const code = output.outputFiles[0].text
    const wrapped = `window.__ModuleLoader__.load({id:"dsh-gb1-scientific-agent",factory:(require)=>{var module={exports:{}};var exports=module.exports;\n${code}\nreturn module.exports;}});\n`
    await writeFile(resolve(pkg, 'lib/client.js'), wrapped, 'utf8')
  }
  const { manifest, hostArtifact } = await publishHostArtifact(pkg)
  console.log(`Built ${manifest.name}@${manifest.version} for DSH ${manifest.engines.dsh}; Host types: ${hostArtifact}`)
})
