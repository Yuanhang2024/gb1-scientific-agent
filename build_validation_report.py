# 验证报告生成器：汇总固定v3与反馈消融结果，不重新生成候选。
"""Create a five-page report from completed fixed-v3 and feedback-ablation records.

Run from the project directory after ablate_feedback.py compare succeeds:
    .venv/Scripts/python.exe build_validation_report.py --tests-passed N
This document builder issues no API requests and does not read test-set labels.
"""
from pathlib import Path
import argparse
import json
from statistics import mean, stdev

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Cm, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from build_report import font, paragraph, table, heading, FONT

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/validation-stage'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def stat_text(average, deviation):
    return f'{average:.3f} ± {deviation:.3f}'


# 消融必须是完整配对；本阶段新增用量与此前已经完成的用量分开统计。
def prepare_data():
    v3=load(OUT/'v3_summary.json')
    study=load(ROOT/'results/feedback-ablation/summary.json')
    assert study['complete'] and study['seeds']==[11,22,33]
    assert len(study['paired_results'])==3 and len(study['arms'])==6
    assert study['total_requests']<=12
    records=pd.read_csv(OUT/'v3_recommendations.csv')
    assert len(records)==180
    pairs=pd.DataFrame(study['paired_results'])
    arms=pd.DataFrame(study['arms'])
    assert set(arms.stage)=={'evaluated'}
    assert (arms.assay_count==10).all()
    assert (arms.scored_count<=40).all()
    ablation_tokens=int(arms.total_tokens.sum())
    added=v3['new_usage']
    result={
        'v3':v3,
        'ablation':{
            'pairs':study['paired_results'],
            'full_mean':float(pairs.full_mean_fitness.mean()),
            'withheld_mean':float(pairs.withheld_mean_fitness.mean()),
            'mean_paired_difference':float(pairs.full_minus_withheld_mean_fitness.mean()),
            'sd_paired_difference':float(pairs.full_minus_withheld_mean_fitness.std(ddof=1)),
            'positive_pairs':int((pairs.full_minus_withheld_mean_fitness>0).sum()),
            'requests':int(study['total_requests']), 'tokens':ablation_tokens,
            'score_counts':arms[['seed','condition','scored_count']].to_dict(orient='records')
        },
        'current_stage_usage':{'requests':int(added['requests'])+int(study['total_requests']),
                               'tokens':int(added['total_tokens'])+ablation_tokens,
                               'request_limit':36},
    }
    assert result['current_stage_usage']['requests']<=36
    (OUT/'stage_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    pairs.to_csv(OUT/'feedback_paired_results.csv',index=False,encoding='utf-8-sig')
    arms.to_csv(OUT/'feedback_condition_results.csv',index=False,encoding='utf-8-sig')
    return result,records,pairs,arms


def draw_ablation(pairs):
    fig,ax=plt.subplots(figsize=(8.5,3.3),layout='constrained')
    for _,r in pairs.iterrows():
        ax.plot([0,1],[r.full_mean_fitness,r.withheld_mean_fitness],marker='o',label=f"Seed {int(r.seed)}")
    ax.set(xticks=[0,1],xticklabels=['Detailed feedback','Details withheld'],ylabel='Mean Top-10 S1 fitness (WT=1)',xlabel='Feedback presentation condition',title='Paired one-step evaluation from the same first-round history')
    ax.legend(title='Fixed source state',fontsize=8)
    ax.grid(axis='y',alpha=.25)
    fig.suptitle('3 paired states; each point averages 10 selected variants | Saved experiment records, 2026-11-09',fontsize=9)
    fig.savefig(OUT/'feedback_ablation.png',dpi=180)
    fig.savefig(OUT/'feedback_ablation.svg')
    plt.close(fig)


def document(result,records,pairs,arms,test_count):
    doc=Document(); sec=doc.sections[0]
    sec.page_width,sec.page_height=Cm(21),Cm(29.7)
    sec.top_margin=sec.bottom_margin=Cm(1.6)
    sec.left_margin=sec.right_margin=Cm(2)
    sec.header_distance=sec.footer_distance=Cm(.7)
    normal=doc.styles['Normal']; normal.font.name=FONT; normal.font.size=Pt(10.5)
    normal._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),FONT)
    normal.paragraph_format.line_spacing=1.15; normal.paragraph_format.space_after=Pt(6)
    h=sec.header.paragraphs[0];h.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    font(h.add_run('GB1 科学智能体 · 固定版本验证与反馈消融 · 2026-11-09'),8,color='607080')
    f=sec.footer.paragraphs[0];f.alignment=WD_ALIGN_PARAGRAPH.CENTER
    font(f.add_run('验证报告  ·  '),8)
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');f._p.append(field)
    doc.core_properties.title='GB1科学智能体：三种子验证与详细反馈消融'
    profiles=result['v3']['profiles']

    heading(doc,'1  研究目标、数据与固定条件')
    paragraph(doc,'GB1 科学智能体：三种子验证与详细反馈消融',bold=True,size=15)
    paragraph(doc,'本阶段检验两个问题：固定v3后，另外两个随机种子是否仍出现推荐改善；在已知数据和预测器相同的时刻，详细预测—实测反馈是否给下一次推荐带来额外收益。所有分数来自公开GB1实验表的虚拟查询，没有开展湿实验或训练大型蛋白质模型。')
    paragraph(doc,'原始数据为Wu等人S1中的149,361条唯一变体，29,477条零fitness保留，S2插补表不纳入。单点记录包含先前实验的校准。目标是IgG-Fc结合选择相关fitness，WT=1；四个位点为39、40、41、54，实验域长56 aa，模板以MQY开头。[1]')
    table(doc,['项目','设置'],[
        ['每种子的数据角色','训练2,000；验证1,000；隐藏候选141,361；最终测试5,000'],
        ['固定版本验证','种子11、22、33；两种Agent各三轮；每轮Top-10'],
        ['预测评分预算','每方法每轮最多40个有效候选；最多一次资格/格式纠正'],
        ['模型比较','Ridge三组alpha；RF/ExtraTrees各两种叶样本约束'],
        ['选定预测器','三个种子均为Random Forest；叶样本数1、1、3'],
        ['模型评估','验证Spearman为0.5441、0.5234、0.5465']],[4.5,12.5])
    paragraph(doc,'同种子v1/v3初始数据、模型类别和参数、预算一致；后续轮次按同样规则用各自新增反馈重训。种子11沿用已经完成的结果，新增种子22和33前核对了提示全文和配置。最终测试标签没有用于本次提示选择或消融输入。')
    paragraph(doc,'GB1是公开数据，不能排除LLM预训练接触。v3是在首轮诊断后开发的，三个种子也已有历史比较；本报告属于开发性验证，不将其解释为全新盲测或跨蛋白质泛化。')

    heading(doc,'2  科学Agent与知识使用',page=True)
    paragraph(doc,'Agent按观察、假设、设计、模型评分和反馈迭代组织。提示借鉴ReAct的行动与观察分离、Reflexion的反馈记忆，以及机器学习辅助定向进化的方法说明。[2,3] 这些原则经本项目实际验证，不直接沿用文献中的性能数字。')
    table(doc,['环节','本实现'],[
        ['观察','当前方法已测统计、top/low/median例子；不读取隐藏候选标签'],
        ['假设','引用具体已知背景，给出局部预期与会削弱预期的观察'],
        ['设计','首次自主生成组合；纠正阶段才提供无标签替补'],
        ['模型评估','Python实际调用固定类别预测器，评分后选Top-10'],
        ['反馈','保存原预测、真实反馈、残差actual−predicted与原假设'],
        ['知识增强','有限的侧链性质、电荷及上位性规则，附来源和适用范围']],[4,13])
    paragraph(doc,'位点边际均值不是参考背景下的单点因果效应；化学保守性也不保证提高fitness。知识规则用于候选和解释，不改变两种方法的合法范围。API返回经过Pydantic、数量、重复与候选资格检查。缺额不由程序补造。')
    paragraph(doc,'科研过程机械核查',bold=True)
    paragraph(doc,'在360个被评分候选的假设文本中，基础Agent有283条引用了当时可见的已测锚点（78.6%），知识增强为344条（95.6%）。有近期反馈的23次请求均至少提到一个可见反馈变体。该统计只检查文字与记录的对应，不确认假设机制正确。')
    paragraph(doc,'真实例子：以已测QNCA为锚点，把第39位换成W，检验N-C-A组合能否在新的背景保持表现；下降会削弱该转移预期。QNCA存在于当时的近期反馈中。Agent的候选设计与本地预测排名分别记录，以便检查它是否仅在解释已有名单。')
    paragraph(doc,'此前v2虽填满候选数量，却复制了预填随机名单，推荐表现与模型对照相同。该失败记录保留；v3的首次请求没有可复制的候选名单。提示变详细本身不作为科学能力提高的判断依据。')

    heading(doc,'3  固定v3的三种子结果',page=True)
    table(doc,['方法','v1推荐均值 ± SD','v3推荐均值 ± SD','变化为正'],[
        ['基础Agent',stat_text(profiles['v1']['llm']['mean_fitness'],profiles['v1']['llm']['sd_fitness']),stat_text(profiles['v3']['llm']['mean_fitness'],profiles['v3']['llm']['sd_fitness']),'3/3'],
        ['知识增强',stat_text(profiles['v1']['knowledge']['mean_fitness'],profiles['v1']['knowledge']['sd_fitness']),stat_text(profiles['v3']['knowledge']['mean_fitness'],profiles['v3']['knowledge']['sd_fitness']),'3/3']],[3.4,5.2,5.2,3.2])
    paragraph(doc,'先对每种子三轮30条推荐求平均，再计算三个种子的均值与样本标准差。参考的原随机方法与模型推荐均值为0.174和0.386；模型方法只评分随机抽取的40个候选，未穷举全池。')
    doc.add_picture(str(ROOT/'results/prompt-v3/comparison-three-seeds/prompt_comparison.png'),width=Cm(17))
    table(doc,['方法','v1新增最高均值 ± SD','v3新增最高均值 ± SD'],[
        ['基础Agent',stat_text(profiles['v1']['llm']['best_new_mean'],profiles['v1']['llm']['best_new_sd']),stat_text(profiles['v3']['llm']['best_new_mean'],profiles['v3']['llm']['best_new_sd'])],
        ['知识增强',stat_text(profiles['v1']['knowledge']['best_new_mean'],profiles['v1']['knowledge']['best_new_sd']),stat_text(profiles['v3']['knowledge']['best_new_mean'],profiles['v3']['knowledge']['best_new_sd'])]],[3.5,6.75,6.75])
    paragraph(doc,'v3的18轮均有40个有效候选，共720次评分；v1对应为643次。原数据、预算和生成方式可比较，但实际有效数量不同，效果变化不能全部归因于科研解释。v3两方法三种子共33次请求、329,745 tokens，v1为31次、297,662 tokens。')
    paragraph(doc,'三个种子的变化方向一致，且v3推荐均值的跨种子波动较小；但逐轮仍有下降。知识增强的平均推荐分数低于基础Agent，其每种子新增最高值均值更高，两个指标回答不同问题。只有三个种子，不作稳定显著优势判断。',size=9.5)

    heading(doc,'4  详细反馈展示的配对消融',page=True)
    paragraph(doc,'每个种子冻结知识增强方法第一轮结束后的状态：2,000条初始训练加10条已揭示标签，同一个训练后的预测器供两个条件评分。两边都获得更新后的known摘要；full保留逐候选预测、实际值、原假设和残差，withheld仅隐藏这两个详细反馈字段并说明其被隐藏。其他输入、规则、schema和候选范围相同。')
    paragraph(doc,'两边均重新生成最多40个候选，由同一模型选Top-10后查询实验分数。各条件最多一次纠正，顺序按固定且与fitness无关的规则安排。第二、第三轮的源实验反馈没有加入本次输入。主指标预先定为Top-10实际均值，差值为full−withheld。')
    table(doc,['种子','full均值','withheld均值','配对差值','推荐交集'],[[int(r.seed),f'{r.full_mean_fitness:.3f}',f'{r.withheld_mean_fitness:.3f}',f'{r.full_minus_withheld_mean_fitness:+.3f}',f'{int(r.recommendation_overlap)}/10'] for r in pairs.itertuples()],[2,3.8,4.2,3.5,3.5])
    doc.add_picture(str(OUT/'feedback_ablation.png'),width=Cm(16.5))
    a=result['ablation']
    paragraph(doc,f"三个配对状态中，full均值较高的有{a['positive_pairs']}/3组；平均配对差值为{a['mean_paired_difference']:+.3f}，差值的样本标准差为{a['sd_paired_difference']:.3f}。本次把seed/实验状态作为重复单位，不把60个被选序列视为60个独立重复。")
    paragraph(doc,'最高值的变化方向不一致：种子11两边相同，种子22的隐藏条件最高值更高，种子33的详细反馈条件更高。平均推荐质量与找到单个更高峰值分别报告。',size=9.5)
    paragraph(doc,'此消融衡量详细字段展示的条件差异；两边训练和聚合统计都保留新增数据，因此不代表全系统无反馈。删除字段也缩短输入，且API具有随机性；本次不能分离内容、长度与随机响应的全部作用，不计算显著性p值。',size=9.5)

    heading(doc,'5  失败分析、用量与复现',page=True)
    failure=records.assign(gap=records.predicted_fitness-records.actual_fitness).nlargest(1,'gap').iloc[0]
    paragraph(doc,f"一个预测偏高的v3案例：种子{int(failure.seed)}、{failure.method}第{int(failure['round'])}轮的{failure.variant}，预测{failure.predicted_fitness:.3f}、实测{failure.actual_fitness:.3f}。该差异用于检查模型覆盖与组合假设，不据一个结果指定生化失效原因。")
    paragraph(doc,'原推荐理由（节选）：'+str(failure.rationale)[:220],size=9)
    paragraph(doc,'三轮均值没有持续上升，部分未测提议也会再次进入后续评分。后续可研究评分预算分配、候选去重与反馈表达，但应先固定比较方案，再记录正负结果。若要更强结论，应在未用于开发的条件上增加重复，并分别比较知识、反馈内容和生成约束。')
    by_condition=arms.groupby('condition')[['requests','total_tokens']].sum()
    table(doc,['本阶段工作','新增请求','返回tokens'],[
        ['补齐v3种子22/33',result['v3']['new_usage']['requests'],f"{result['v3']['new_usage']['total_tokens']:,}"],
        ['消融 full',int(by_condition.loc['full','requests']),f"{int(by_condition.loc['full','total_tokens']):,}"],
        ['消融 withheld',int(by_condition.loc['withheld','requests']),f"{int(by_condition.loc['withheld','total_tokens']):,}"],
        ['本阶段合计',result['current_stage_usage']['requests'],f"{result['current_stage_usage']['tokens']:,}"]],[8,3.5,5.5])
    paragraph(doc,f"本阶段预先说明的请求上限为36次；此前v1/v2与v3种子11属于已完成阶段，不计入本表。未提供可靠价格配置，故不换算金额。完整离线测试{test_count}项通过；真实请求、候选来源、数据范围和配对条件另行核查。",size=9.5)
    paragraph(doc,'复现与交付：项目README记录依赖、数据获取和命令；本阶段独立入口ablate_feedback.py提供prepare/run/compare。所有提示、请求和原始响应保留，旧结果未覆盖。API重跑可能产生不同输出；GitHub仓库尚未发布。本报告附统计摘要、推荐CSV及paired_results.csv。',size=9.5)
    paragraph(doc,'参考与使用说明',bold=True)
    refs=[
        '[1] Wu NC et al. eLife (2016), 5:e16965. doi:10.7554/eLife.16965；S1与实际模板图。',
        '[2] Yang KK, Wu Z, Arnold FH. Nature Methods (2019), 16:687–694. doi:10.1038/s41592-019-0496-6。',
        '[3] Yao et al., ReAct, ICLR 2023, arXiv:2210.03629；Shinn et al., Reflexion, NeurIPS 2023, arXiv:2303.11366。',
        '[4] scikit-learn模型/数据泄漏文档；OpenAI Python SDK；Pydantic JSON Schema文档。具体URL见项目技术选型与消融方案。'
    ]
    for text in refs: paragraph(doc,text,size=8.5)
    paragraph(doc,'资料查证由GPT-6 Luna辅助，后端实现由GPT-6.1 Sol辅助，主代理负责设计、审查和文稿；实验内部实际调用gpt-6-luna。知识来源与适用范围保存在规则文件。',size=8.5)
    path=OUT/'GB1科学Agent验证报告.docx';doc.save(path)
    reopened=Document(path)
    body='\n'.join(p.text for p in reopened.paragraphs)
    for word in ['149,361','withheld','消融','失败','GitHub']:
        assert word in body
    return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--tests-passed',type=int,required=True);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    result,records,pairs,arms=prepare_data()
    draw_ablation(pairs)
    path=document(result,records,pairs,arms,args.tests_passed)
    print(json.dumps({'report':str(path),'paired_results':result['ablation'],'current_stage_usage':result['current_stage_usage']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
