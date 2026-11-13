# 扩展实验：运行与结果

本目录提供主动学习、不确定性估计、突变数量比较、结构信息、知识图谱和自动化平台对接讨论六项扩展内容。主动学习采用采集策略比较路线，结构信息采用参考结构特征路线。交互式Demo由[DSH插件](<../dsh-plugin/packages/gb1/README.md>)提供。

## 结果入口

- [5页扩展实验报告](<../reports/bonus/GB1扩展实验报告.pdf>)及[可编辑版本](<../reports/bonus/GB1扩展实验报告.docx>)
- [数值摘要](<../reports/bonus/summary.json>)、[1,080条本地实验推荐](<../reports/bonus/learning_recommendations.csv>)
- [结构与图谱Agent配对结果](<../reports/bonus/graph_agent_pairs.csv>)
- [实施方案与文献依据](<../扩展实验实施方案.md>)
- [自动化实验平台对接讨论](<../自动化实验平台对接讨论.md>)

## 依赖

在项目根目录 `gb1-evolution` 执行以下命令。原42个固定依赖不变，仅添加Biopython 1.88和NetworkX 3.7。

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-bonus.txt
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -m pytest -q
```

实际安装和检查信息见[environment-bonus.json](<../environment-bonus.json>)，完整固定清单见[requirements-bonus.txt](<../requirements-bonus.txt>)。

## 1. 主动学习与不确定性

配置见[bonus.json](<../configs/bonus.json>)，入口为[run_bonus.py](<../run_bonus.py>)。已运行种子44、55、66，每种子训练2,000、校准1,000、测试5,000、候选141,361。

RF保持100棵树，计算逐树预测均值μ和样本标准差s。比较random、greedy、std_only、mean_plus_std四种策略。每臂三轮Top-10，训练和反馈独立。绝对残差与按s缩放的split-conformal区间只用于校准和评价，不参与采集函数。

真实结果：

| 全池策略 | 推荐均值 ± 种子SD | 查询30条后最高新增值 ± 种子SD |
| --- | ---: | ---: |
| 随机 | 0.0587 ± 0.0205 | 1.0266 ± 0.3317 |
| 预测均值 | 3.4194 ± 0.6941 | 7.2558 ± 0.3151 |
| 集成分歧 | 2.2196 ± 0.6432 | 7.2939 ± 1.4366 |
| 均值＋分歧 | 3.4787 ± 0.3128 | 7.8671 ± 0.7871 |

初始随机留出测试的90%标称区间，绝对残差方法平均实测覆盖90.7%、平均宽度0.209；尺度残差方法覆盖89.9%、宽度0.245。主动选择点属于不同分布：mean_plus_std选中点的绝对区间经验覆盖仅6.7%，尺度区间为91.1%，平均宽度达8.037。

校准样本n=1000、alpha=0.1时，取排序后的第901项。HD1测试样本三个种子分别只有1、0、1条。

### 重新运行

以下以种子44为例，使用新目录，避免覆盖现有实验。55、66替换对应种子与目录即可。

```powershell
$reference = 'MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'
& .\.venv\Scripts\python.exe run_bonus.py prepare --source data/raw/elife-16965-supp1-v4.xlsx --reference $reference --data data/bonus-new/seed-44 --output results/bonus-learning-new/seed-44 --config configs/bonus.json --seed 44
& .\.venv\Scripts\python.exe run_bonus.py run --data data/bonus-new/seed-44 --output results/bonus-learning-new/seed-44 --config configs/bonus.json --seed 44
```

已完成轮次会保留；中断后使用同一 `run` 命令继续。准备目录不可覆盖。汇总已有三种子不会重新训练或调用LLM：

```powershell
& .\.venv\Scripts\python.exe run_bonus.py summarize --outputs results/bonus-learning/seed-44 results/bonus-learning/seed-55 results/bonus-learning/seed-66 --output results/bonus-learning/summary
```

## 2. 单点、双点、三点和四点对照

HD指相对VDGV的四个位点中发生替换的数量。每种子每HD预先抽取40条无标签候选，对random与mean_plus_std各查询三轮、每轮10条。单点原始记录只有76条，因此训练固定保留16条，划分时检查候选至少留40条。

| HD | 随机：30次查询后的最高新增均值 | 均值＋分歧：最高新增均值 |
| --- | ---: | ---: |
| 1 | 3.8981 | 3.8981 |
| 2 | 2.1764 | 2.9324 |
| 3 | 1.7461 | 1.7461 |
| 4 | 1.8235 | 1.8235 |

这是固定等规模抽样库内的效果；单点分数包含跨研究校准，各HD原始空间大小和训练覆盖不同。全部控制与分组共108轮、1,080条推荐，本地实验不调用API。

## 3. 结构信息

[structure.py](<structure.py>)读取1PGA首个模型的chain A标准氨基酸，去掉水与hetero记录，检查结构MTY与实验MQY只在第2位不同，研究位点仍是39/40/41/54。

已计算所有56个残基的绝对SASA、1540个位点对的Cα/最小重原子距离，以及4.0/4.5/5.0 Å非相邻近邻敏感性。SASA使用1.4 Å探针，比较100和960点；960点在39/40/41/54的值为11.824、130.237、28.373、0.978 Å²。

这些数值描述孤立参考构象的几何特征。结构信息已作为可查询事实进入图条件的Agent输入。实际表见[目标位点特征](<../results/bonus-graph/structure/target_features.csv>)和[目标位点距离](<../results/bonus-graph/structure/target_distances.csv>)。

## 4. 知识图谱与实际Agent接入

[knowledge_graph.py](<knowledge_graph.py>)用MultiDiGraph表示位点、氨基酸、性质、突变、变体、测量和来源；测量只来自显式传入的known训练表。各种子图有4,160个节点，其中2,000个Measurement和2,000个Variant。边来源采用source_id，保存/恢复不会与node-link端点字段混淆。

支持位点、距离、候选属性及已知测量查询。未测候选返回 `fitness_status=unknown`、空观测，不读取其隐藏分数。`add_observations`支持将新获得的测量加入新的已知图；其行为已测试。当前实际Agent对照为每种子一个初始状态的一轮。

示例只读查询：

```python
from bonus.knowledge_graph import load_graph, query_variant

graph = load_graph('results/bonus-graph/seed-44/knowledge_graph.json')
context = query_variant(graph, 'VDGV')
print(context['result']['fitness_status'])
```

图通过实际查询生成事实，再送入Agent上下文。三组静态JSON与结构＋图谱条件各一轮40候选、Top-10；真实调用11次、142,187 tokens，低于12次上限。图条件120个有效候选中90个引用图规则，其中只有4个引用结构事实，引用主要集中在已知训练观测。

三个种子图条件的推荐均值差（图减静态）为−0.3341、−0.5131、−0.1132，均差−0.3201。功能已实现并验证，本次没有观察到推荐收益；该条件同时增加了结构事实、查询结果和上下文长度，这些因素的作用可在后续实验中分别检验。

实际图示见[seed44子图](<../results/bonus-graph/seed-44/knowledge_subgraph.png>)，完整图为[knowledge_graph.json](<../results/bonus-graph/seed-44/knowledge_graph.json>)。

### 图Agent复现

下面创建新的对照输出；`run`会调用用户配置的真实模型，最多12次请求。不要把密钥写入代码或报告。

```powershell
& .\.venv\Scripts\python.exe run_graph_agent.py prepare --data-dirs data/bonus/seed-44 data/bonus/seed-55 data/bonus/seed-66 --model-files results/bonus-learning/seed-44/initial_model.joblib results/bonus-learning/seed-55/initial_model.joblib results/bonus-learning/seed-66/initial_model.joblib --pdb data/raw/1PGA.pdb --knowledge configs/knowledge.json --output results/bonus-graph-new
& .\.venv\Scripts\python.exe run_graph_agent.py run --output results/bonus-graph-new
& .\.venv\Scripts\python.exe run_graph_agent.py compare --output results/bonus-graph-new
```

## 5. 平台对接讨论与交付边界

[对接讨论](<../自动化实验平台对接讨论.md>)基于SiLA2和Opentrons官方资料，覆盖批次审核、外部执行ID、状态、未知结果、重复提交、单位、质量检查和回传更新。题目此项要求讨论，本阶段没有连接设备或实施湿实验。

完整离线测试100项通过；数据划分、校准、采集、结构、图序列化、未知查询、真实API候选来源及汇总结果均已独立核查。已有主实验结果不覆盖；GitHub发布另行处理。
