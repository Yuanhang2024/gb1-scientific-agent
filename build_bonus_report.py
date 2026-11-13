# 加分报告生成器：汇总主动学习、区间评价、结构和图谱结果，不执行新的实验。
"""Build the bonus-task report from completed CPU and real API records only.

Run from the project directory after both study summaries are complete:
    .venv/Scripts/python.exe build_bonus_report.py --tests-passed 100
This script performs no API calls and never reads project credentials.
"""
from pathlib import Path
from collections import Counter
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Cm, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from build_report import paragraph, table, heading, font, FONT

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/bonus'
STRATEGIES=['random','greedy','std_only','mean_plus_std']
NAMES={'random':'随机','greedy':'预测均值','std_only':'集成分歧','mean_plus_std':'均值＋分歧'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


# 这里只对传入的统计量计算均值和样本标准差，单个样本没有可估计的样本标准差。
def stats(values):
    a=np.asarray(values,dtype=float)
    return {'mean':float(a.mean()),'sd':float(a.std(ddof=1)) if len(a)>1 else None,'n':len(a)}


def fstats(value):
    return f"{value['mean']:.3f} ± {value['sd']:.3f}" if value['sd'] is not None else f"{value['mean']:.3f}"


# 同时检查CPU实验和图谱对照已完成，再整理选样、区间和知识引用结果。
def collect():
    learning=ROOT/'results/bonus-learning/summary'
    manifest=read_json(learning/'summary.json')
    assert manifest['complete'] and sorted(manifest['seeds'])==[44,55,66]
    assert manifest['rounds']==108 and manifest['recommendations']==1080
    rounds=pd.read_csv(learning/'round_metrics.csv')
    rec=pd.read_csv(learning/'all_recommendations.csv')
    initial=pd.read_csv(learning/'initial_heldout_coverage.csv')
    final=pd.read_csv(learning/'final_heldout_coverage.csv')
    assert len(rounds)==108 and len(rec)==1080
    assert (rounds.assays==10).all()
    assert not rec.duplicated(['seed','arm','variant']).any()
    result={'seeds':[44,55,66],'cpu_rounds':108,'cpu_recommendations':1080,'global':{},'hd':{},'uncertainty':{},'graph':{}}
    # 全池策略与固定HD候选库的实验分开汇总，避免混淆不同搜索范围。
    global_rounds=rounds[rounds.hd.isna()]
    for strategy in STRATEGIES:
        r=global_rounds[global_rounds.strategy==strategy]
        last=r[r['round']==3]
        result['global'][strategy]={
            'round_means':{str(i):stats(r[r['round']==i].mean_fitness) for i in [1,2,3]},
            'best_new_after_30':stats(last.cumulative_new_best),
            'mean_selected':stats(r.groupby('seed').mean_fitness.mean()),
            'selected_abs_coverage':float(r.selected_abs_coverage.mean()),
            'selected_scaled_coverage':float(r.selected_scaled_coverage.mean()),
            'selected_abs_mean_width':float(r.selected_abs_mean_width.mean()),
            'selected_scaled_mean_width':float(r.selected_scaled_mean_width.mean())}
    for hd in [1,2,3,4]:
        result['hd'][str(hd)]={}
        for strategy in ['random','mean_plus_std']:
            r=rounds[(rounds.hd==hd)&(rounds.strategy==strategy)]
            result['hd'][str(hd)][strategy]={'best_new_after_30':stats(r[r['round']==3].cumulative_new_best),
                                           'mean_selected':stats(r.groupby('seed').mean_fitness.mean())}
    overall=initial[initial.test_hd=='overall']
    assert len(overall)==3 and (overall.n==5000).all()
    for column in ['abs_coverage','abs_mean_width','scaled_coverage','scaled_mean_width','abs_error_std_spearman','mse']:
        result['uncertainty']['initial_'+column]=stats(overall[column])
    result['uncertainty']['initial_single_test_n']=initial[initial.test_hd=='1'][['seed','n']].to_dict(orient='records')
    result['uncertainty']['final_global']={}
    for strategy in STRATEGIES:
        rows=final[(final.test_hd=='overall')&(final.arm==f'global/{strategy}')]
        assert len(rows)==3
        result['uncertainty']['final_global'][strategy]={col:stats(rows[col]) for col in ['abs_coverage','abs_mean_width','scaled_coverage','scaled_mean_width','mse']}
    graph=read_json(ROOT/'results/bonus-graph/summary.json')
    assert graph['complete'] and graph['total_requests']<=12 and len(graph['conditions'])==6
    graph_arms=pd.DataFrame(graph['conditions']); graph_pairs=pd.DataFrame(graph['paired_results'])
    assert (graph_arms.valid_count==40).all() and (graph_arms.assay_count==10).all()
    result['graph'].update(requests=graph['total_requests'],tokens=int(graph_arms.total_tokens.sum()),pairs=graph['paired_results'],
                           mean_paired_difference=float(graph_pairs.graph_minus_static_mean_fitness.mean()))
    for condition in ['static_json','structure_graph']:
        a=graph_arms[graph_arms.condition==condition]
        result['graph'][condition]={'mean_fitness':stats(a.mean_fitness),'max_fitness':stats(a.max_fitness),
                                    'requests':int(a.requests.sum()),'tokens':int(a.total_tokens.sum())}
    graph_counts=[]; citations=0; structure_citations=0
    for seed in [44,55,66]:
        g=read_json(ROOT/'results/bonus-graph'/f'seed-{seed}'/'knowledge_graph.json')
        counts=Counter(node['entity'] for node in g['nodes'])
        assert counts['Measurement']==2000 and counts['Variant']==2000
        queries=read_json(ROOT/'results/bonus-graph'/f'seed-{seed}'/'graph_queries.json')
        graph_counts.append({'seed':seed,'nodes':len(g['nodes']),'edges':len(g['edges']),'entities':dict(counts),'queries':len(queries)})
        state=read_json(ROOT/'results/bonus-graph'/f'seed-{seed}'/'structure_graph/state.json')
        for proposal in state['proposals']:
            ids=proposal['knowledge_rule_ids']
            citations+=any(i.startswith('graph-') for i in ids)
            structure_citations+=any(i.startswith(('graph-position-','graph-distance-')) for i in ids)
    result['graph'].update(graph_counts=graph_counts,graph_citing_proposals=citations,structure_citing_proposals=structure_citations,graph_proposal_count=120)
    structural=pd.read_csv(ROOT/'results/bonus-graph/structure/target_features.csv')
    result['structure']=structural[['position','sasa_100_angstrom2','sasa_960_angstrom2','neighbor_count_4.5']].to_dict(orient='records')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    rounds.to_csv(OUT/'learning_rounds.csv',index=False,encoding='utf-8-sig')
    rec.to_csv(OUT/'learning_recommendations.csv',index=False,encoding='utf-8-sig')
    initial.to_csv(OUT/'initial_coverage.csv',index=False,encoding='utf-8-sig')
    final.to_csv(OUT/'final_coverage.csv',index=False,encoding='utf-8-sig')
    graph_pairs.to_csv(OUT/'graph_agent_pairs.csv',index=False,encoding='utf-8-sig')
    return result,rounds,structural


def figures(result,rounds):
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.7),layout='constrained')
    g=rounds[rounds.hd.isna()]
    for strategy in STRATEGIES:
        r=g[g.strategy==strategy].groupby('round').cumulative_new_best.agg(['mean','std'])
        axes[0].errorbar(r.index*10,r['mean'],yerr=r['std'],marker='o',capsize=3,label=strategy)
    x=np.arange(4)
    for offset,strategy in [(-.18,'random'),(.18,'mean_plus_std')]:
        values=[result['hd'][str(hd)][strategy]['best_new_after_30'] for hd in [1,2,3,4]]
        axes[1].bar(x+offset,[s['mean'] for s in values],yerr=[s['sd'] for s in values],width=.35,capsize=3,label=strategy)
    axes[0].set(title='Full measured pool: best new fitness',xlabel='Queried labels per arm',ylabel='Measured S1 fitness (WT=1)',xticks=[10,20,30])
    axes[1].set(title='Equal 40-variant pools: best after 30 labels',xlabel='Mutation count relative to WT',ylabel='Measured S1 fitness (WT=1)',xticks=x,xticklabels=['1','2','3','4'])
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Source: saved GB1 S1 runs | 3 seeds | mean ± sample SD',fontsize=9)
    fig.savefig(OUT/'learning_overview.png',dpi=180);fig.savefig(OUT/'learning_overview.svg');plt.close(fig)


def make_doc(result,structural,tests):
    doc=Document();sec=doc.sections[0]
    sec.page_width,sec.page_height=Cm(21),Cm(29.7)
    sec.top_margin=sec.bottom_margin=Cm(1.5);sec.left_margin=sec.right_margin=Cm(2)
    sec.header_distance=sec.footer_distance=Cm(.65)
    normal=doc.styles['Normal'];normal.font.name=FONT;normal.font.size=Pt(10)
    normal._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),FONT)
    normal.paragraph_format.line_spacing=1.12;normal.paragraph_format.space_after=Pt(5)
    h=sec.header.paragraphs[0];h.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    font(h.add_run('GB1 科学智能体 · 加分项实验 · 2026-11-10'),8,color='607080')
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    font(footer.add_run('加分实验报告  ·  '),8)
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
    doc.core_properties.title='GB1加分项实验：主动学习、不确定性、结构与知识图谱'

    heading(doc,'1  目标、实现范围与实验设计')
    paragraph(doc,'GB1科学智能体加分项实验报告',bold=True,size=15)
    paragraph(doc,'本阶段完成题目第1、2、3、4、5、7项：主动学习、不确定性估计、突变数量比较、结构信息、知识图谱和自动化平台对接讨论。第1项选择主动学习路线，第4项选择结构信息路线；交互式Demo不在范围内。没有开展真实湿实验，也没有训练大型蛋白质模型。')
    table(doc,['加分项','实际实现与检查'],[
        ['主动学习','random、greedy、std_only、mean_plus_std，选后反馈、各臂独立重训'],
        ['不确定性','100棵树预测分歧；独立绝对/尺度残差校准；覆盖率与宽度'],
        ['突变数量','HD1、2、3、4各40候选，随机与主动策略同预算比较'],
        ['结构','1PGA参考距离、近邻阈值敏感性、100/960点SASA'],
        ['知识图谱','带来源有向图、查询与保存恢复；真实Agent增强对照'],
        ['平台讨论','SiLA2/Opentrons接口、状态、数据、质量和异常流程']],[3.5,13.5])
    paragraph(doc,'数据采用Wu等人的S1，149,361条测量/校准记录，S2插补表不纳入。[1] 使用新拆分种子44、55、66；每组训练2,000条含WT、16条单点及1,983条HD≥2，随机共同留出6,000条再分校准1,000/测试5,000，其余141,361条作候选。抽取不按fitness筛选，并预留至少40条单点候选。')
    paragraph(doc,'预测器固定为100棵树Random Forest、叶样本最少1、4线程，不在新校准/测试标签上选择参数。全部控制器只在选定候选后取得S1分数。全池4策略与HD分组8策略，每种子12臂、各3轮Top-10，共108轮、1,080条虚拟查询。图谱Agent另有6条件、60条推荐。')
    paragraph(doc,'新拆分没有用于此前提示调整，但仍来自同一公开蛋白数据；结果不代表跨蛋白泛化。独立校准与测试角色、候选资格、模型版本和运行参数均保存在文件中。')

    heading(doc,'2  主动选样与不确定性评价',page=True)
    paragraph(doc,'设森林预测均值为μ，100棵树预测的样本标准差为s。四种策略分别均匀随机、按μ、按s、按μ+s选样。系数1为预先固定的探索设置，不宣称最优或满足GP-UCB理论条件。[2] 每轮各方法仅加入自身10条新反馈，再重新训练。')
    table(doc,['全池策略','30条推荐均值 ± SD','30次查询后的最高新增 ± SD'],[[NAMES[s],fstats(result['global'][s]['mean_selected']),fstats(result['global'][s]['best_new_after_30'])] for s in STRATEGIES],[3.5,6.5,7])
    paragraph(doc,'s仅是集成分歧。用独立校准样本分别计算|y−μ|及|y−μ|/max(s,1e−6)，取排序后的第ceil((n+1)×0.9)项作为阈值，构造90%预测区间，下界按非负fitness截到0。每次重训后重新校准；采集分数不使用校准或测试标签。[3,4]')
    u=result['uncertainty']
    table(doc,['初始随机留出测试','实测覆盖率','平均区间宽度'],[
        ['绝对残差区间',f"{u['initial_abs_coverage']['mean']:.1%}",f"{u['initial_abs_mean_width']['mean']:.3f}"],
        ['尺度残差区间',f"{u['initial_scaled_coverage']['mean']:.1%}",f"{u['initial_scaled_mean_width']['mean']:.3f}"]],[7,5,5])
    paragraph(doc,f"初始误差绝对值与树间分歧的Spearman均值为{u['initial_abs_error_std_spearman']['mean']:.3f}。表中覆盖率是三个5,000条随机留出测试的均值；宽度单位为S1 fitness。原始分歧没有被直接称为置信区间。")
    table(doc,['策略','最终测试尺度覆盖','选中点尺度覆盖','选中点尺度区间宽度'],[[NAMES[s],f"{u['final_global'][s]['scaled_coverage']['mean']:.1%}",f"{result['global'][s]['selected_scaled_coverage']:.1%}",f"{result['global'][s]['selected_scaled_mean_width']:.3f}"] for s in STRATEGIES],[3.3,4.5,4.2,5])
    paragraph(doc,f"均值＋分歧选中点的绝对残差区间覆盖仅{result['global']['mean_plus_std']['selected_abs_coverage']:.1%}；尺度区间覆盖{result['global']['mean_plus_std']['selected_scaled_coverage']:.1%}，但宽度达{result['global']['mean_plus_std']['selected_scaled_mean_width']:.3f}。主动选择改变分布，普通split-conformal边际条件不能直接用于选中点。[4] 高覆盖不能单独表示高精度；HD分层测试有极少或空样本，原表保留其n。",size=9.5)

    heading(doc,'3  突变数量比较与参考结构',page=True)
    paragraph(doc,'HD为四个位点相对VDGV的不同字母数。各HD候选库均随机抽取40条，每策略三轮查询30条；训练数据相同，反馈独立。比较新增候选最高值，并在组内定义累计最佳，避免其他突变组的高分掩盖当前组。')
    table(doc,['HD','随机：最高新增均值 ± SD','均值＋分歧：最高新增均值 ± SD'],[[hd,fstats(result['hd'][str(hd)]['random']['best_new_after_30']),fstats(result['hd'][str(hd)]['mean_plus_std']['best_new_after_30'])] for hd in [1,2,3,4]],[2,7.5,7.5])
    doc.add_picture(str(OUT/'learning_overview.png'),width=Cm(17))
    paragraph(doc,'不同HD原始空间规模、训练覆盖和测量来源有差异，单点分数包含跨研究校准。因此这里评价同规模抽样库内的效果，不把差异全部解释为突变数量的因果作用。',size=9)
    table(doc,['位点','SASA 960点 / Å²','4.5 Å非相邻邻居数'],[[int(r['position']),f"{r['sasa_960_angstrom2']:.3f}",int(r['neighbor_count_4.5'])] for r in result['structure']],[3,7,7])
    paragraph(doc,'使用1PGA chain A标准残基，剔除水及hetero记录。计算Cα与最小重原子距离，并比较4.0/4.5/5.0 Å近邻阈值；SASA使用1.4 Å probe及100/960点。39–54、41–54最小重原子距离约3.810和3.670 Å。这些是参考结构几何，不是突变体结构或fitness因果。[5]')

    heading(doc,'4  可查询知识图谱及Agent对照',page=True)
    paragraph(doc,'用NetworkX MultiDiGraph表示Position、Mutation、AminoAcid、Property、Variant、Measurement和Source。每种子图恰有2,000个已知测量节点，校准、测试及未查询候选分数不进入图。未知候选可查询化学和结构事实，但fitness状态为unknown。[6]')
    doc.add_picture(str(ROOT/'results/bonus-graph/seed-44/knowledge_subgraph.png'),width=Cm(16.5))
    paragraph(doc,'图示为一个已知完整变体及一个突变路径；测量属于完整变体，不能归因于图中的单个替换。完整图JSON与查询来源单独保存。每种子执行4个位点、6个距离对和3个已知观察查询，再将结果加入Agent上下文。',size=9)
    table(doc,['种子','静态规则均值','结构＋图谱均值','差值'],[[int(r['seed']),f"{r['static_json_mean_fitness']:.3f}",f"{r['structure_graph_mean_fitness']:.3f}",f"{r['graph_minus_static_mean_fitness']:+.3f}"] for r in result['graph']['pairs']],[2.2,5,5.3,4.5])
    paragraph(doc,f"图增强条件的{result['graph']['graph_citing_proposals']}/120个有效候选引用了图查询事实ID，其中{result['graph']['structure_citing_proposals']}个引用结构事实。六条件都实际生成并评分40个候选、推荐10个；同一训练、初始RF、v3提示及候选范围配对。")
    paragraph(doc,'该实验测试结构与图查询知识的组合，不能单独归因于图存储格式。性能变化如表所示，保留下降结果；实现可查询知识和模型引用，并不保证推荐性能提高。')

    heading(doc,'5  平台对接、完成情况与限制',page=True)
    paragraph(doc,'真实平台对接按软件职责讨论：候选检查 → 人工审核批次 → 适配器提交 → 保存执行ID → 查询状态 → 导入原始测量 → 单位/映射/质量检查 → 接受结果 → 更新已知数据。SiLA 2提供Feature/Command/Property及Observable Command执行UUID；Opentrons提供运行、状态和数据文件API，具体测量能力依设备模块与协议。[7,8]')
    paragraph(doc,'本项目需要维护experiment/batch/operation ID、variant与sample映射、协议及接口版本、原始值和单位、QC状态、fitness归一化版本。提交超时先核对外部执行状态，不能盲重发；仪器失败不记为蛋白质fitness为0。只有被接受的结果用于训练。完整讨论已交付，未连接设备或执行湿实验。')
    paragraph(doc,'六项加分内容完成状态',bold=True)
    paragraph(doc,'主动选样已运行；不确定性已校准和评估；单/双/三/四点已对照；参考结构特征已计算并进入Agent知识；图谱已保存、查询和实际接入；自动化平台讨论已完成。交互式Demo按要求不制作。')
    paragraph(doc,f"本地CPU实验108轮、1,080条推荐不调用LLM。结构图谱对照实际{result['graph']['requests']}次API请求、{result['graph']['tokens']:,} tokens，未超过12次上限；模型请求/返回标识为gpt-6-luna。完整离线测试{tests}项通过，真实输入和结果另有独立核查。",size=9.5)
    paragraph(doc,'局限与下一步研究：仅一个公开蛋白及三种子；自适应选样覆盖没有普通随机留出保证；HD样本库大小及来源不等同；结构为孤立WT参照且pos2与实验模板不同；图增强增加上下文和知识，不能单独证明图格式贡献。所有结论以本次记录为准，不预设新方法胜出。',size=9.5)
    paragraph(doc,'复现：requirements-bonus保留原依赖并增加Biopython/NetworkX；run_bonus.py负责prepare/run/summarize，run_graph_agent.py负责prepare/run/compare。配置、模型、选前分数与区间、选后反馈、图JSON、查询和响应均保存，旧结果不覆盖。',size=9)
    paragraph(doc,'主要参考',bold=True)
    refs=[
        '[1] Wu NC et al. eLife 2016;5:e16965. doi:10.7554/eLife.16965.',
        '[2] Srinivas et al. GP optimization in the bandit setting, ICML 2010, arXiv:0912.3995（方法启发，未声称GP理论界）.',
        '[3] Angelopoulos & Bates. Conformal prediction introduction, arXiv:2107.07511, §2.2–2.3.',
        '[4] Wager et al. JMLR 2014;15:1625–1651；Prinster et al. JAWS-X, ICML 2023, PMLR 202.',
        '[5] RCSB 1PGA；Biopython Bio.PDB.SASA官方文档，Shrake–Rupley方法.',
        '[6] NetworkX MultiDiGraph与node-link文档；OpenStax Biology 2e §3.4.',
        '[7] sila-standard.com/standards/；SiLA Python Observable Command官方文档.',
        '[8] docs.opentrons.com/http/api_reference.html、python-api/versioning/及读板数据说明.'
    ]
    for text in refs:paragraph(doc,text,size=8)
    paragraph(doc,'资料核查由GPT-6 Luna辅助，后端由GPT-6.1 Sol辅助，主代理负责方案、审查和文稿。文献链接、数据用途与模型辅助方式在实施方案及平台讨论中详细记录。',size=8)
    path=OUT/'GB1加分项实验报告.docx';doc.save(path)
    check=Document(path)
    text='\n'.join(p.text for p in check.paragraphs)
    for required in ['不确定性','图谱','SiLA','1,080','Demo']:
        assert required in text
    return path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tests-passed',type=int,required=True);args=parser.parse_args()
    result,rounds,structural=collect()
    figures(result,rounds)
    path=make_doc(result,structural,args.tests_passed)
    print(json.dumps({'report':str(path),'global':result['global'],'uncertainty':result['uncertainty'],'graph':result['graph']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
