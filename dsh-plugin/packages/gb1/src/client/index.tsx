import type { Context } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-api-gateway/client'
import type {} from '@deepseek-ai/dsh-client-ui-renderer/client'
import type {} from '@deepseek-ai/dsh-client-ui-layout/client'
import type {} from '@deepseek-ai/dsh-client-ui-sidebar/client'
import { TYPERT_REMOTE } from '../../lib/typert.remote-client.js'
import { createApi } from './api.ts'
import { FlaskIcon, Workbench } from './Workbench.tsx'
import styles from './style.css'

export const name = 'gb1-workbench-client'
export const inject = ['remote', 'slots', 'layout']
const PANEL = 'gb1-evolution'

// 先装配本插件的严格Remote，再启用依赖该接口的页面，避免注入循环。
export async function apply(ctx: Context) {
  const unmount = await ctx.remote.$mount(TYPERT_REMOTE)
  ctx.effect(() => unmount, 'gb1-workbench:remote')
  ctx.effect(() => {
    const style = document.createElement('style')
    style.dataset.pluginCss = 'dsh-gb1-scientific-agent'
    style.textContent = styles
    document.head.appendChild(style)
    return () => style.remove()
  }, 'gb1-workbench:style')
  await ctx.plugin({
    name: 'gb1-workbench-page',
    inject: ['remote', 'remote.gb1Workbench', 'slots'],
    apply(scope: Context) {
      const api = createApi(scope.remote)
      scope.slots.inject('main', () => scope.slots.register({ name: 'main', key: PANEL }, () => <Workbench api={api}/>))
      scope.slots.inject('sidebar.panellist', () => scope.slots.register({ name: 'sidebar.panellist', id: PANEL, order: 70, label: '蛋白质进化' }, FlaskIcon))
    },
  })
}
