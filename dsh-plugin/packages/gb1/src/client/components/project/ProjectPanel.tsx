import type { WorkbenchInfo } from '../../../contracts.ts'

export function ProjectPanel({ info }: { info: WorkbenchInfo | null }) {
  const positions = info?.positions.length ? info.positions.join('、') : '39、40、41、54'

  return <section className="gb1-project" aria-labelledby="gb1-project-heading">
    <div className="gb1-panel-heading">
      <div><h2 id="gb1-project-heading">项目说明</h2><p>基于公开实验数据的 GB1 虚拟定向进化。</p></div>
    </div>
    <section className="gb1-project-section">
      <h3>研究对象</h3>
      <p>当前任务使用 56 个氨基酸的 GB1 参考域，探索第 {positions} 位的氨基酸替换。fitness 表示公开数据中的结合选择相关分数，原数据中野生型分数为 1。</p>
      {info?.reference && <details className="gb1-project-reference">
        <summary>查看野生型参考序列</summary><code className="gb1-sequence-value">{info.reference}</code>
      </details>}
    </section>
    <section className="gb1-project-section">
      <h3>实验流程</h3>
      <ol className="gb1-project-steps">
        <li><strong>读取已有观测</strong><span>从选定随机种子的已知数据开始，整理历史变体与分数。</span></li>
        <li><strong>设计候选</strong><span>基础 Agent 或知识辅助 Agent 提出序列、设计假设和推荐理由。</span></li>
        <li><strong>预测与筛选</strong><span>本地适应度预测模型评分，默认每轮最多处理 {info?.scoreBudget ?? 40} 个候选，选择前 {info?.topK ?? 10} 条推荐；实际数量以本次实验设置为准。</span></li>
        <li><strong>查询实验反馈</strong><span>候选选定后，再从公开实验记录中查询对应分数，加入下一轮已知数据。</span></li>
      </ol>
      <p className="gb1-help">每次运行可输入自定义轮数，并由查询、请求与停止条件控制预算。工作对话展示实际调度、模型公开分析和工具操作。</p>
    </section>
    <section className="gb1-project-section">
      <h3>如何阅读结果</h3>
      <dl className="gb1-project-definitions">
        <div><dt>预测分数</dt><dd>本地预测模型在候选筛选时给出的估计值。</dd></div>
        <div><dt>实验数据分数</dt><dd>来自已有公开记录的测量分数，用于本次虚拟实验的反馈。</dd></div>
        <div><dt>设计假设与理由</dt><dd>模型的公开说明，可结合实际反馈查看推荐表现。</dd></div>
      </dl>
    </section>
    <section className="gb1-project-section">
      <h3>数据来源与信息范围</h3>
      <p>反馈数据来自 Wu 等人（2016）的 Supplementary file 1（S1）。当前运行结果与历史研究结果分别保存。</p>
      <div className="gb1-project-links">
        <a href="https://elifesciences.org/articles/16965" target="_blank" rel="noopener noreferrer">研究论文 ↗</a>
        <a href="https://cdn.elifesciences.org/articles/16965/elife-16965-supp1-v4.xlsx" target="_blank" rel="noopener noreferrer">原始补充数据 S1 ↗</a>
      </div>
      <p className="gb1-help">当前界面展示运行内的推荐、反馈和规则引用。独立测试集的预测指标及知识规则正文尚未接入此页面。</p>
    </section>
  </section>
}
