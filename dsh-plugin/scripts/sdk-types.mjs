import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { createRequire } from 'node:module'

// 将已安装SDK的原始声明放入构建工作区，供官方生成器识别框架符号的所属包。
// 这里只复制类型声明；发布插件的运行时依赖仍由DSH宿主提供。
export async function prepareSdkTypes(root) {
  const require = createRequire(import.meta.url)
  const library = dirname(require.resolve('@deepseek-ai/dsh-typert-protocol'))
  const manifest = JSON.parse(await readFile(join(library, '../package.json'), 'utf8'))
  if (manifest.version !== '0.2.2-rc.1') throw new Error('DSH SDK版本与插件声明不一致')
  const target = resolve(root, 'packages/_framework-protocol')
  async function copyDeclarations(source, output) {
    await mkdir(output, { recursive: true })
    for (const item of await readdir(source, { withFileTypes: true })) {
      if (item.isDirectory()) await copyDeclarations(join(source, item.name), join(output, item.name))
      else if (item.name.endsWith('.d.ts')) await writeFile(join(output, item.name), await readFile(join(source, item.name)))
    }
  }
  await copyDeclarations(join(library, 'types'), join(target, 'types'))
  await writeFile(join(target, 'package.json'), JSON.stringify({ name: manifest.name, version: manifest.version, private: true, type: 'module',
    exports: { '.': './types/index.d.ts', './types': './types/types.d.ts' } }, null, 2) + '\n')
  await writeFile(join(target, 'tsconfig.host.json'), JSON.stringify({ extends: '../../tsconfig.base.json',
    compilerOptions: { rootDir: './types', noEmit: true, emitDeclarationOnly: false }, include: ['types/**/*.d.ts'] }, null, 2) + '\n')
}
