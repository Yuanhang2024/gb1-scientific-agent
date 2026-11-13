# 首轮报告生成器：读取已完成的实验记录，生成统计、图表和Word文档，不请求LLM。
"""Summarize completed experimental records and create the five-page report.

Run from the project directory after all seeds have completed:
    .venv/Scripts/python.exe build_report.py
No model requests are issued by this document script.
"""
from pathlib import Path
import json
import math
from statistics import mean, stdev

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reports'
METHODS = ['random', 'model', 'llm', 'knowledge']
NAMES = {'random':'随机推荐', 'model':'模型推荐', 'llm':'LLM Agent', 'knowledge':'知识增强 Agent'}
COLORS = {'random':'#68747c', 'model':'#257c9d', 'llm':'#b66027', 'knowledge':'#236f4e'}
FONT = 'Microsoft YaHei'
REFERENCE = 'MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def describe(values):
    values = [float(v) for v in values]
    return {'mean':mean(values), 'sd':stdev(values) if len(values)>1 else 0.0, 'n':len(values)}


def fmt(stats):
    return f"{stats['mean']:.3f} ± {stats['sd']:.3f}"


# 先检查各方法轮次完整，再读取推荐与指标；报告中的数值来自保存记录。
def collect():
    states, frames, rounds, selections, final = {}, [], [], [], []
    attempts = []
    for seed in [11,22,33]:
        directory=ROOT/'results'/f'seed-{seed}'/'run'
        state=read_json(directory/'state.json')
        assert all(state['methods'][m]['status']=='completed' for m in METHODS)
        assert state['dataset_manifest']['audit']['reference']==REFERENCE
        frame=pd.read_csv(directory/'recommendations.csv')
        frame['seed']=seed
        assert len(frame)==120
        assert not frame.duplicated(['method','variant']).any()
        frames.append(frame)
        metrics=pd.read_csv(directory/'round_metrics.csv')
        metrics['seed']=seed
        assert len(metrics)==12 and (metrics.assays==10).all()
        rounds.append(metrics)
        states[seed]=state
        selected=read_json(ROOT/'results'/f'seed-{seed}'/'selection'/'selection.json')
        spec=selected['spec']
        row=next(r for r in selected['comparison'] if r['family']==spec['family'] and r['parameters']==spec['parameters'])
        selections.append({'seed':seed, 'family':spec['family'], 'leaf':spec['parameters'].get('min_samples_leaf'), 'validation_spearman':row['spearman'], 'validation_pearson':row['pearson'], 'validation_mse':row['mse'], 'validation_top10':row['top_k_hit_rate'], 'fit_seconds':row['fit_seconds']})
        for method, values in read_json(directory/'final_test_metrics.json').items():
            final.append({'seed':seed,'method':method,**values})
        for method, arm in state['methods'].items():
            assert len(arm['rounds'])==3
            for record in arm['rounds']:
                assert record['stage']=='evaluated' and len(record['recommendations'])==10
                assert record['model_score_count']<=40
                for attempt in record['attempts']:
                    response=attempt.get('response',{})
                    usage=response.get('usage') or {}
                    attempts.append({'seed':seed,'method':method,'round':record['round'],'status':attempt['status'],'prompt_tokens':usage.get('prompt_tokens',usage.get('input_tokens',0)) or 0,'completion_tokens':usage.get('completion_tokens',usage.get('output_tokens',0)) or 0,'total_tokens':usage.get('total_tokens',0) or 0,'model':response.get('model_returned'), 'invalid':len(attempt.get('rejected',[]))})
    rec=pd.concat(frames,ignore_index=True)
    round_frame=pd.concat(rounds,ignore_index=True)
    final_frame=pd.DataFrame(final)
    usage_frame=pd.DataFrame(attempts)
    assert len(rec)==360 and len(round_frame)==36 and len(final_frame)==12
    summary={}
    for method in METHODS:
        r=round_frame[round_frame.method==method]
        subset=rec[rec.method==method]
        t=final_frame[final_frame.method==method]
        # 先求每个seed自身的新发现最高值，再汇总跨seed均值和标准差。
        best_by_seed=subset.groupby('seed').actual_fitness.max()
        initial=r[r['round']==1].set_index('seed').previous_best
        improvements=best_by_seed-initial
        summary[method]={
            'round_mean_fitness':{str(i):describe(r[r['round']==i].mean_fitness) for i in [1,2,3]},
            'best_new_by_seed':best_by_seed.to_dict(),
            'best_new':describe(best_by_seed),
            'cumulative_best_final':describe(r[r['round']==3].cumulative_best),
            'initial_best_by_seed':initial.to_dict(),
            'seeds_improved_initial':int((improvements>0).sum()),
            'mean_selected_fitness':describe(subset.groupby('seed').actual_fitness.mean()),
            'wt_above_count':int((subset.actual_fitness>1).sum()),
            'selected_count':len(subset),
            'min_candidates':int(r.model_score_count.min()),'max_candidates':int(r.model_score_count.max()),
            'score_count_total':int(r.model_score_count.sum()),
            'test_spearman':describe(t.spearman),'test_mse':describe(t.mse),
            'test_pearson':describe(t.pearson),'test_top10':describe(t.top_k_hit_rate)
        }
    total_usage={key:int(usage_frame[key].sum()) for key in ['prompt_tokens','completion_tokens','total_tokens']}
    total_usage.update(requests=len(usage_frame), model_ids=sorted(usage_frame.model.dropna().unique().tolist()))
    result={'seeds':[11,22,33], 'selections':selections,'methods':summary,'usage':total_usage, 'total_recommendations':len(rec), 'unique_recommended_variants':int(rec.variant.nunique()), 'rounds_completed':len(round_frame)}
    return result, rec, round_frame, final_frame, usage_frame, states


def make_figure(summary):
    fig, axes=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained')
    for method in METHODS:
        stats=[summary['methods'][method]['round_mean_fitness'][str(i)] for i in [1,2,3]]
        axes[0].errorbar([1,2,3],[s['mean'] for s in stats],yerr=[s['sd'] for s in stats],marker='o',capsize=3,label=method,color=COLORS[method])
    best=[summary['methods'][m]['best_new'] for m in METHODS]
    axes[1].bar(METHODS,[s['mean'] for s in best],yerr=[s['sd'] for s in best],capsize=4,color=[COLORS[m] for m in METHODS])
    axes[0].set(title='Round mean measured fitness',xlabel='Recommendation round',ylabel='S1 fitness (WT = 1)',xticks=[1,2,3])
    axes[1].set(title='Best newly assayed variant per seed',xlabel='Method',ylabel='S1 fitness (WT = 1)')
    axes[0].legend(fontsize=8)
    axes[1].tick_params(axis='x',labelsize=8)
    for ax in axes:
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('GB1 | 3 seeds, 3 rounds, 10 assays/round | mean ± sample SD',fontsize=11)
    fig.savefig(OUT/'experiment_overview.png',dpi=180)
    fig.savefig(OUT/'experiment_overview.svg')
    plt.close(fig)


def font(run,size=10.5,bold=False,color=None):
    run.font.name=FONT
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),FONT)
    run.font.size=Pt(size)
    run.bold=bold
    if color:
        run.font.color.rgb=RGBColor.from_string(color)
    return run


def paragraph(doc,text,bold=False,size=10.5):
    p=doc.add_paragraph()
    font(p.add_run(text),size,bold)
    return p


def heading(doc,text,page=False):
    p=doc.add_paragraph(style='Heading 1')
    p.paragraph_format.page_break_before=page
    font(p.add_run(text),17,True,'173F5F')
    return p


def table(doc,headers,rows,widths=None):
    t=doc.add_table(rows=1,cols=len(headers))
    t.style='Light Shading Accent 1'
    t.autofit=False
    for idx,h in enumerate(headers):
        font(t.rows[0].cells[idx].paragraphs[0].add_run(h),9,True)
    for values in rows:
        cells=t.add_row().cells
        for cell,value in zip(cells,values):
            font(cell.paragraphs[0].add_run(str(value)),9)
    for row in t.rows:
        no_split=OxmlElement('w:cantSplit')
        row._tr.get_or_add_trPr().append(no_split)
        for idx,cell in enumerate(row.cells):
            cell.width=Cm(widths[idx] if widths else 17/len(headers))
            for p in cell.paragraphs:
                p.paragraph_format.space_after=Pt(3)
                p.paragraph_format.space_before=Pt(2)
                p.paragraph_format.line_spacing=1.1
    doc.add_paragraph().paragraph_format.space_after=Pt(1)
    return t


def make_docx(summary,rec,rounds,final,usage,states):
    doc=Document()
    sec=doc.sections[0]
    sec.page_width,sec.page_height=Cm(21),Cm(29.7)
    sec.top_margin,sec.bottom_margin=Cm(1.6),Cm(1.6)
    sec.left_margin=sec.right_margin=Cm(2)
    sec.header_distance=sec.footer_distance=Cm(.7)
    normal=doc.styles['Normal']
    normal.font.name=FONT
    normal.font.size=Pt(10.5)
    normal._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),FONT)
    normal.paragraph_format.line_spacing=1.15
    normal.paragraph_format.space_after=Pt(6)
    header=sec.header.paragraphs[0]
    header.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    font(header.add_run('GB1 科学智能体 · 首轮实验报告 · 2026-11-09'),8,color='607080')
    footer=sec.footer.paragraphs[0]
    footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    font(footer.add_run('实验报告  ·  '),8)
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
    doc.core_properties.title='面向蛋白质定向进化的科学智能体：GB1首轮实验'

    heading(doc,'1  背景、数据与问题定义')
    paragraph(doc,'面向蛋白质定向进化的科学智能体',bold=True,size=16)
    paragraph(doc,'本实验用公开 GB1 四位点数据模拟实验反馈，比较随机推荐、适应度模型推荐、LLM Agent 和知识增强 Agent。研究问题是：在相同实验标签查询预算和模型评分上限下，LLM 的候选设计及知识提示能否改善推荐结果。全部结果来自实际运行，未进行湿实验。')
    paragraph(doc,'数据来源为 Wu 等人的 eLife 研究 Supplementary file 1 [1]。表中 149,361 条变体唯一且可解析，29,477 条 fitness 为零；零值保留，S2 插补表不纳入。76 条单点记录包含 Olson 等人实验数据的校准。分数是相对野生型的结合选择相关表现，WT=1，数值越高越好。')
    table(doc,['项目','本次设置'],[
        ['实验位点','39、40、41、54；野生型四字母 VDGV'],
        ['初始训练 / 验证 / 最终测试','2,000 / 1,000 / 5,000 条，各部分互不重叠'],
        ['隐藏候选池','141,361 条，每轮只揭示选定候选标签'],
        ['重复和迭代','随机种子 11、22、33；四方法各三轮'],
        ['每轮预算','最多 40 次模型评分，10 次标签查询']],[5,12])
    paragraph(doc,'实验模板根据原论文 Figure 1 supplement 1 翻译，为 56 aa；第 2 位为 Q。结构参照 1PGA 的第 2 位 T 不用于候选还原。')
    paragraph(doc,REFERENCE,size=9)
    paragraph(doc,'每组划分在完整测量表上随机产生，不按 fitness 筛选。四种方法使用相同初始数据；每种方法的新增反馈只进入自己的下一轮训练。独立最终测试在三轮结束后才打开。公开数据可能被 LLM 预训练接触，本实验只保证本次流程不提前提供隐藏标签。')

    heading(doc,'2  适应度模型、Agent 与知识增强',page=True)
    paragraph(doc,'四个变异位点编码为 80 维 one-hot。每种子比较七组配置：Ridge 的 alpha=0.1/1/10；Random Forest 与 Extra Trees 各使用 100 棵树，叶节点最少样本数为 1 或 3。按验证集 Spearman、Top-10 命中率、较低 MSE 依次选择，不使用最终测试集调参。[2]')
    table(doc,['种子','选定模型 / 叶样本数','验证 Spearman','验证 MSE','Top-10命中'],[[r['seed'],f"Random Forest / {r['leaf']}",f"{r['validation_spearman']:.3f}",f"{r['validation_mse']:.3f}",f"{r['validation_top10']:.0%}"] for r in summary['selections']],[1.2,5,3.2,3,3.6])
    paragraph(doc,'每轮固定模型类别和超参数，仅用本方法新增数据重新拟合。Spearman 衡量排序关系；MSE 为平均平方误差；Top-10 命中率定义为预测前十与真实前十的交集数量除以十，同分按固定输入顺序处理。[2,3]')
    table(doc,['方法','生成与选择'],[
        ['随机推荐','从可测候选池随机抽40个，再随机选择10个'],
        ['模型推荐','随机抽40个，按预测分数选择前10个'],
        ['LLM Agent','GPT-6 Luna 自主设计最多40个合法组合，本地模型选前10个'],
        ['知识增强 Agent','同一流程增加带来源的性质与上位性提示']],[4,13])
    paragraph(doc,'Agent 使用用户配置的 GPT-6 Luna chat API，JSON Schema 输出后经 Pydantic 检查。输入包含当前训练统计、top variants、最近反馈及无标签示例；输出包含位点分析、候选、假设与理由。Python 负责实际评分、排序、去重和隐藏评估，模型没有浏览和文件工具。[4]')
    paragraph(doc,'知识规则引用氨基酸侧链性质、电荷、甘氨酸/脯氨酸结构区别，以及 GB1 上位性。[1,5] 规则只影响建议与解释，不改变合法范围。每轮最多一次纠正请求；不以随机候选补足 LLM 输出。评分少于40时照实记录。该对照评价固定评分预算下的候选生成，未将模型方法设置为全池穷举。')

    heading(doc,'3  三轮对照结果',page=True)
    table(doc,['方法','第1轮均值','第2轮均值','第3轮均值'],[[NAMES[m]]+[fmt(summary['methods'][m]['round_mean_fitness'][str(r)]) for r in [1,2,3]] for m in METHODS],[4.1,4.3,4.3,4.3])
    paragraph(doc,'表中为每轮10个推荐的实测均值，再对三个种子计算平均值 ± 样本标准差。误差范围描述三次运行波动，不表示稳定的统计显著性。',size=9)
    doc.add_picture(str(OUT/'experiment_overview.png'),width=Cm(17))
    table(doc,['方法','三轮新发现最佳','超过初始最佳的种子数','实测>WT记录'],[[NAMES[m],fmt(summary['methods'][m]['best_new']),f"{summary['methods'][m]['seeds_improved_initial']}/3",f"{summary['methods'][m]['wt_above_count']}/90"] for m in METHODS],[4.2,4.8,4.8,3.2])
    paragraph(doc,'新发现最佳先在每个种子的30个新增推荐中取最高值，再计算三种子统计。累计最佳还包含原有2,000条训练数据，可能在三轮中保持不变；因此同时展示新增结果。各方法最多查询90条记录，跨方法重复序列分别计算为各自的查询。',size=9)

    heading(doc,'4  序列示例、失败分析与模型评价',page=True)
    best_rows=[]
    for m in METHODS:
        row=rec[rec.method==m].sort_values('actual_fitness',ascending=False).iloc[0]
        best_rows.append([NAMES[m],f"{int(row.seed)}/{int(row['round'])}",row.variant,f"{row.predicted_fitness:.3f}",f"{row.actual_fitness:.3f}"])
    table(doc,['方法','种子/轮','四位点','预测','实测'],best_rows,[4.5,2.5,3,3.5,3.5])
    success=rec.sort_values('actual_fitness',ascending=False).iloc[0]
    paragraph(doc,f"新增推荐最高示例：{success.variant}，突变为 {success.mutations}。其完整序列：",size=9.5)
    paragraph(doc,success.sequence,size=9)
    agent_rows=rec[rec.method.isin(['llm','knowledge'])]
    failure=agent_rows.assign(overprediction=agent_rows.predicted_fitness-agent_rows.actual_fitness).sort_values('overprediction',ascending=False).iloc[0]
    paragraph(doc,f"过高预测案例：{NAMES[failure.method]}，种子{int(failure.seed)}第{int(failure['round'])}轮的{failure.variant}，预测{failure.predicted_fitness:.3f}，实测{failure.actual_fitness:.3f}。差值说明模型在该组合上判断偏高；应结合训练覆盖及组合相互作用检查，不能仅凭此案例确定生化机制。")
    rationale=str(failure.rationale)
    paragraph(doc,'该候选的原始推荐理由（节选）：'+rationale[:280],size=9)
    table(doc,['方法','最终测试 Spearman','最终测试 MSE','最终 Top-10命中'],[[NAMES[m],fmt(summary['methods'][m]['test_spearman']),fmt(summary['methods'][m]['test_mse']),fmt(summary['methods'][m]['test_top10'])] for m in METHODS],[4,4.5,4.5,4])
    paragraph(doc,'每方法最终模型使用2,030条自身已知记录训练。测试评估的是同一四位点空间内的留出预测；不同随机种子的测试集可能重叠。全体每轮Top-10、理由及完整序列保存在配套结果表，共360条推荐记录。',size=9)
    llm_rows=rec[rec.method=='llm']; kg_rows=rec[rec.method=='knowledge']
    paragraph(doc,f"位点分布：基础 Agent 的第41位有{int((llm_rows.variant.str[2]=='C').sum())}/90条为C，第54位有{int((llm_rows.variant.str[3]=='A').sum())}/90条为A；知识增强分别偏向第41位L（{int((kg_rows.variant.str[2]=='L').sum())}/90）和第54位G（{int((kg_rows.variant.str[3]=='G').sum())}/90）。这描述候选偏好，未据此推断单点因果效应；实验只允许四个位点变化。",size=9)

    heading(doc,'5  解释、复现、成本与参考来源',page=True)
    averages='、'.join(f"{NAMES[m]} {summary['methods'][m]['mean_selected_fitness']['mean']:.3f}" for m in METHODS)
    paragraph(doc,'本次全部推荐的平均 fitness：'+averages+'。本次三轮曲线中，基础 Agent 的均值下降，知识增强第二轮较高而第三轮下降；这说明新增推荐表现需要逐轮评价。')
    paragraph(doc,f"知识增强有{summary['methods']['knowledge']['seeds_improved_initial']}/3种子超过初始最佳，基础 Agent为{summary['methods']['llm']['seeds_improved_initial']}/3；推荐均值与是否刷新最佳值回答不同问题。",size=9.5)
    paragraph(doc,'Agent 展示了根据当前统计和反馈提出候选的能力；其贡献需结合对照和具体假设检验判断。知识增强每轮有效候选为10—40个，基础 Agent为39—40个，实际选择机会不同。这一运行不足已保留，不能将其影响单独归因于知识本身。',size=9.5)
    budget_rows=[]
    for m in METHODS:
        s=summary['methods'][m]
        u=usage[usage.method==m]
        budget_rows.append([NAMES[m],s['score_count_total'],f"{s['min_candidates']}—{s['max_candidates']}",len(u),int(u.total_tokens.sum())])
    table(doc,['方法','总评分数','每轮有效数','API请求','tokens'],budget_rows,[4.3,3,3.5,2.7,3.5])
    paragraph(doc,f"正式实验共{summary['usage']['requests']}次API请求，服务返回总量{summary['usage']['total_tokens']:,} tokens；另有一次连接检查49 tokens。实际返回模型标识为 {', '.join(summary['usage']['model_ids'])}。没有可靠价格配置，因此不推算金额。环境使用Python3.12.14，关键依赖已固定，24项合成测试通过；数据划分与真实调用另有运行记录。",size=9.5)
    paragraph(doc,'改进方向：扩大重复次数；增加同预算的其他候选生成方法或单独报告全池模型评分；比较初始数据规模和训练覆盖；在独立蛋白任务上检验泛化；用新实验评估未测序列。任何新模型或依赖先查资料，再按统一划分比较。')
    paragraph(doc,'复现：按项目README安装requirements，依次运行prepare、select、run、finalize、plot、aggregate。原始数据下载链接、参数与种子已保存。独立API需用户自己的密钥；重新调用LLM可能产生不同输出，原响应已保留。GitHub发布尚需确定仓库与可见性，当前提供本地完整工程。',size=9.5)
    paragraph(doc,'参考来源与辅助工具',bold=True)
    refs=[
        '[1] Wu NC et al. eLife (2016), 5:e16965. doi:10.7554/eLife.16965. 数据S1及Figure 1 supplement 1。',
        '[2] scikit-learn: common_pitfalls.html；modules/grid_search.html；RandomForestRegressor官方文档。',
        '[3] SciPy: scipy.stats.spearmanr / pearsonr；scikit-learn: mean_squared_error。',
        '[4] OpenAI Python SDK: github.com/openai/openai-python；Pydantic官方JSON Schema文档。',
        '[5] Clark MA, Douglas M, Choi J. Biology 2e (2018), OpenStax, §3.4 Proteins。'
    ]
    for ref in refs: paragraph(doc,ref,size=8.5)
    paragraph(doc,'资料查证由GPT-6 Luna辅助，代码由GPT-6.1 Sol辅助，主代理负责设计、检查、运行与正式文稿。项目内部模型调用与开发辅助分别记录，文中的实验数字由已保存结果计算。',size=8.5)
    path=OUT/'GB1首轮实验报告.docx'
    doc.save(path)
    reopened=Document(path)
    body='\n'.join(p.text for p in reopened.paragraphs)
    for required in ['背景','知识增强','失败','149,361','GitHub']:
        assert required in body
    return path


def make_markdown(summary,rec,usage):
    lines=['# GB1 首轮实验结果','', '日期：2026-11-09。三随机种子 × 四方法 × 三轮全部完成，共360条推荐记录。','',
           '## 核心比较','', '| 方法 | 第1轮平均fitness | 第2轮 | 第3轮 | 三轮新增最高fitness | 超过初始最佳 |', '| --- | --- | --- | --- | --- | --- |']
    for m in METHODS:
        s=summary['methods'][m]
        lines.append('| '+ ' | '.join([NAMES[m]]+[fmt(s['round_mean_fitness'][str(i)]) for i in [1,2,3]]+[fmt(s['best_new']),str(s['seeds_improved_initial'])+'/3'])+' |')
    lines += ['', '数值为三种子均值 ± 样本标准差；新增最高分先按每个种子的30条推荐取最大值。', '',
              '## 预测模型选择','', '| 种子 | 模型 | 叶样本数 | 验证Spearman | 验证MSE |','| --- | --- | --- | --- | --- |']
    for s in summary['selections']:
        lines.append(f"| {s['seed']} | Random Forest | {s['leaf']} | {s['validation_spearman']:.4f} | {s['validation_mse']:.4f} |")
    lines += ['', '## 运行数量', '', f"- 实验轮次：{summary['rounds_completed']}，推荐记录：360，跨种子/方法唯一推荐序列：{summary['unique_recommended_variants']}。", f"- 正式API请求：{summary['usage']['requests']}；服务返回总用量：{summary['usage']['total_tokens']:,} tokens。", '- 连接检查额外49 tokens，费用未换算。', '- 实际LLM返回模型：'+', '.join(summary['usage']['model_ids'])+'。', '',
              '## 解释范围','', '随机和模型方法每轮先随机抽取40个候选，两种Agent自行生成最多40个候选。四者共享评分上限和标签查询数，比较的是有限评分预算下的生成策略；模型方法未穷举整个候选池。不同方法的实际有效候选数见完整结果。', '',
              '结果来自公开GB1四位点空间。仅三组种子，最终测试集可能重叠，且无法审查LLM预训练是否接触公开数据；这些结果不用于推断跨蛋白泛化或稳定的统计显著性。', '',
              '详见[实验方案](<../实验方案.md>)、[复现说明](<../README.md>)、[完整推荐表](<all_recommendations.csv>)和[统计摘要](<summary.json>)。']
    (OUT/'实验结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    summary,rec,rounds,final,usage,states=collect()
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    rec.to_csv(OUT/'all_recommendations.csv',index=False,encoding='utf-8-sig')
    rounds.to_csv(OUT/'all_round_metrics.csv',index=False,encoding='utf-8-sig')
    final.to_csv(OUT/'all_test_metrics.csv',index=False,encoding='utf-8-sig')
    usage.to_csv(OUT/'all_api_usage.csv',index=False,encoding='utf-8-sig')
    site_rows=[]
    for method,group in rec.groupby('method'):
        for index,(position,wt) in enumerate(zip([39,40,41,54],'VDGV')):
            counts=group.variant.str[index].value_counts()
            site_rows.append({'method':method,'position':position,'mutation_fraction':float((group.variant.str[index]!=wt).mean()),'most_frequent_residue':counts.index[0],'residue_count':int(counts.iloc[0]),'total':len(group)})
    pd.DataFrame(site_rows).to_csv(OUT/'site_statistics.csv',index=False)
    make_figure(summary)
    make_markdown(summary,rec,usage)
    output=make_docx(summary,rec,rounds,final,usage,states)
    print(json.dumps({'report':str(output),'recommendations':len(rec),'rounds':len(rounds),'usage':summary['usage']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
