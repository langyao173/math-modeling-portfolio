# A题：程序、输入与运行方法

本说明对应报名编号 MCB2600347 的论文。支撑材料收录源程序、依赖规格、运行配置及必要许可证；C.6—C.8给出计算与预测方法。输入为竞赛提供的八份附件。论文中的数值结果可通过这些程序重新计算，新样本预测另有独立入口。

## 1. 程序目录

以下路径以research目录为起点。

| 路径 | 用途 |
|---|---|
| `run_submission.py` | 输入检查、运行目录准备及按依赖顺序执行计算阶段 |
| `prepare_fonts.py` | 下载并核验完整运行流程所需的固定版本中文字体 |
| `project/MathorCup_A/A_work/q1` | 六模型基准、折内预处理、折外预测与缺测对照 |
| `project/MathorCup_A/A_work/q2` | 50项感官Top-3、方向与重采样稳定性、化合物共变网络 |
| `project/MathorCup_A/A_work/q3` | 风格条件生成、感官谱与初始迁移方法 |
| `project/MathorCup_A/A_work/competition_revision` | 相对等级、模块稳定性、风格可行性与预算对照 |
| `project/MathorCup_A/frontier_revision/experiments` | 现代预测、内层配置选择、风格内解释、生成更新、稀疏优化与重拟合检验 |
| `project/MathorCup_A/frontier_revision/audit` | 矩阵、约束、来源身份及结果汇总的独立复算 |
| `project/MathorCup_A/frontier_revision/visuals` | 机制图与定量图的Python程序 |
| `SOURCE_MANIFEST.json` | 源程序与文本配置的相对路径、大小和SHA-256 |

原始附件由竞赛提供；公开预训练权重与字体由固定地址获取。训练上下文、模型、折外数组、结果表和图形由相应程序生成。程序保留数据来源及公开模型许可。

## 2. 附件、样本标识与尺度

将八份原始附件放在同一目录，例如`/path/to/A_attachments`，保持文件名和原文件字节不变。

| 文件 | 分析用途 |
|---|---|
| `1.csv` | 250款啤酒的231个化学特征、样本标识与风格标签 |
| `2.csv` | 化学特征关联矩阵与字段名称对齐 |
| `3.xlsx` | 品评卡结构与计分规则检查 |
| `4.csv` | 50维标准化感官响应 |
| `5.csv` | 感官关联矩阵与字段顺序核验 |
| `6.csv` | 化学与感官关联矩阵核验 |
| `7.csv` | 训练部分使用的消费者词频辅助信息 |
| `8.xlsx` | 化合物名称、单位、阈值、标定量及感官名称 |

训练附件以整数列`beer`连接，连接前检查唯一性和样本集合。`beer_id`为另一标识字段，附件4中存在舍入值，不能替代`beer`作为连接键。化学列沿用附件1的输入尺度，包括其中已经使用的对数变换；程序不把任意浓度单位自动转换为该尺度。

附件3的空白品评卡不提供250款啤酒逐人的原始0—6评分。程序输出的0—6等级由训练响应分位点定义，表示相对于训练样本的强度等级。折外评价使用各训练折的切点，独立部署使用全体250个训练样本的切点。

无训练检查命令：

```bash
python run_submission.py --help
python run_submission.py --list
python run_submission.py --check-input --data-dir /path/to/A_attachments
```

`--list`无需附件；`--check-input`逐一核验八份原始文件的SHA-256。重新另存CSV或Excel可能改变文件身份，应使用竞赛原附件。

## 3. 软件与字体

记录的科学计算环境为Linux、CPython 3.12.14。不同预测库使用各自的依赖规格，运行时通过四个Python解释器参数指定。

| 环境 | 依赖规格，相对`project/MathorCup_A` |
|---|---|
| 基础计算与树模型 | `requirements_nature_upgrade.txt`，其中包含`requirements.txt` |
| 稀疏优化 | `frontier_revision/experiments/sparse_transfer/requirements_solver.txt` |
| TabICL、嵌套选模与新样本预测 | `frontier_revision/experiments/modern_prediction/requirements_deployment.txt`；原研究规格同时见`requirements_tabicl.txt` |
| TabDPT对照 | `frontier_revision/experiments/modern_prediction/tabdpt/requirements_tabdpt.txt` |

例如，在相应环境中使用`python -m pip install -r 规格文件`。TabICL与TabDPT的CPU版PyTorch单独安装：

```bash
/path/to/tabicl/bin/python -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
/path/to/tabicl/bin/python -m pip install -r project/MathorCup_A/frontier_revision/experiments/modern_prediction/requirements_deployment.txt
```

TabICL固定为2.2.0；目标编码使用NumPy 2.3.5、pandas 2.2.3、SciPy 1.17.0及scikit-learn 1.8.0。TabDPT另用其规格文件，避免不同版本的`huggingface-hub`混装。各`environment_freeze.txt`记录原运行环境，完整流程所需的求解器与基础依赖以上表为准。

完整73阶段含中文诊断绘图。**先在源目录准备字体，再创建运行目录：**

```bash
python prepare_fonts.py
python prepare_fonts.py --verify-only
```

该标准库程序从Noto CJK官方仓库的`Sans2.004`与`Serif2.003`版本获取未修改字体，核验文件大小及SHA-256。默认位置为`project/MathorCup_A/assets/fonts`，其中包含`NotoSansCJKsc-Regular.otf`、`NotoSerifCJKsc-Regular.otf`与SIL Open Font License 1.1。可用`--output /path/to/fonts`指定另一路径；完整流程仍需将经过核验的文件放在上述默认位置，供准备程序复制。50项感官名称表`A_work/figures/sensory_names_zh.json`已作为文本配置收录。

独立新样本预测无需字体。单独运行图A1时，程序优先使用项目字体，也可解析系统安装的`Noto Sans CJK SC`或通过`--font-path`指定字体。论文排版程序涉及的Pandoc、LibreOffice和Inkscape用于文稿或图形导出；统一73阶段不执行Word文稿重建。字体的官方来源与许可见[Noto CJK仓库](https://github.com/notofonts/noto-cjk)。

## 4. 从附件执行完整计算

使用独立的新目录保存本次运行的模型、结果与日志。先执行：

```bash
python run_submission.py --prepare-only \
  --data-dir /path/to/A_attachments \
  --work-dir /path/to/fresh_run \
  --python /path/to/base/bin/python \
  --solver-python /path/to/solver/bin/python \
  --modern-python /path/to/tabicl/bin/python \
  --tabdpt-python /path/to/tabdpt/bin/python
```

这一步复制程序、配置、字体和已核验的附件，建立解释器别名及输入记录，不启动模型训练或权重下载。`fresh_run`必须为空，或属于此前由同一入口创建、输入与源清单身份一致的运行。新目录不能位于源目录内部，也不能是源目录的上级。

将上述命令中的`--prepare-only`替换为`--stages all`，即可依次请求73个阶段：基础三问与验证、模型对照、风格可行性、现代预测、内层配置选择、生成更新、稀疏优化、跨模型复评、重拟合及分组误差分析。`--threads`控制相应现代预测程序的线程参数；嵌套选模阶段按其登记的4个进程、每进程2线程执行。模型训练和优化使用固定随机种子及求解时限，运行时间随硬件变化。

也可从基础阶段开始：

```bash
python run_submission.py \
  --data-dir /path/to/A_attachments --work-dir /path/to/fresh_run \
  --stages baseline_audit,baseline_q1,baseline_q2,baseline_q3,baseline_validation \
  --python /path/to/base/bin/python
```

后续运行使用相同工作目录及相应解释器参数，按照`--list`所列的顺序选择阶段。单选后面的阶段不会自动补跑前置阶段。已完成阶段的记录在`execution_logs/completed_stages.jsonl`，详细日志在同目录的阶段日志中；选取尚未完成的阶段时须保留其前置结果。修改源程序或输入后应使用新目录。

`comparison_data_claims`由附件1、附件8及基础审计输出生成70字段编码表。融合模型所需的训练/留出数组由`comparison_holdout_baseline`调用原六模型的估计器与调参函数重新生成。CatBoost采用记录的深度网格`[3, 5]`；`reduced_timing_result.json`用于固定该配置，其历史耗时不作为本次机器的测量。

公开权重阶段下载并核验固定文件。也可在准备后的目录单独执行：

```bash
python /path/to/fresh_run/experiments/modern_prediction/fetch_public_weights.py
python /path/to/fresh_run/experiments/modern_prediction/fetch_public_weights.py --verify-only
```

本次结果以本次输入、程序、模型和候选的身份记录为准。保存结果的历史核验入口`reproduce_frontier_revision.py --verify-only`针对完整的既有研究结果，不能替代从附件开始的统一计算入口。

## 5. 独立准备固定预测器f⁺

新样本预测可直接使用已经选定的`raw231_y16_no_residual`配置，无需先重算全部三问。该配置在完成的五配置比较中依据合并外层留出结果选定；以下准备程序固定此选择，在250个训练样本上建立部署上下文，不重新选择模型，也不使用新样本的感官标签。

只需原始`1.csv`、`4.csv`以及固定的公开权重。按第3节安装TabICL环境后执行：

```bash
python project/MathorCup_A/frontier_revision/experiments/modern_prediction/fetch_public_weights.py --model tabicl

/path/to/tabicl/bin/python \
  project/MathorCup_A/frontier_revision/experiments/modern_prediction/prepare_tabicl_deployment.py \
  --data-dir /path/to/A_attachments \
  --output /path/to/deployment
```

`deployment`须为新目录或空目录。程序默认读取同目录下的`models/tabicl-regressor-v2-20260212.ckpt`；已有权重可通过`--checkpoint /path/to/weight.ckpt`指定。准备程序本身不下载权重。推理脚本默认使用同目录的`predict_tabicl_csv.py`，需要另存位置时可用`--inference-script`指定。

权重固定于公开仓库版本`4dcd344ece2c00be9e831fdd35bed57b5ad83e19`，SHA-256为`0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`。上下文包含231维化学输入、50维训练响应、16维目标编码和相对等级切点；准备程序检查附件身份、`beer`对齐、维度、数值及编码环境。生成的`manifest.json`记录参数、版本、上下文和权重身份。

只检查两份训练附件时可执行：

```bash
/path/to/tabicl/bin/python \
  project/MathorCup_A/frontier_revision/experiments/modern_prediction/prepare_tabicl_deployment.py \
  --data-dir /path/to/A_attachments --check-input
```

## 6. 新样本CSV的输入与输出

先检查输入，再运行预测：

```bash
/path/to/tabicl/bin/python \
  project/MathorCup_A/frontier_revision/experiments/modern_prediction/predict_tabicl_csv.py \
  /path/to/new_chemistry.csv /path/to/predictions.csv \
  --artifact /path/to/deployment --threads 2 --validate-only

/path/to/tabicl/bin/python \
  project/MathorCup_A/frontier_revision/experiments/modern_prediction/predict_tabicl_csv.py \
  /path/to/new_chemistry.csv /path/to/predictions.csv \
  --artifact /path/to/deployment --threads 2
```

| 项目 | 输入契约 |
|---|---|
| 化学字段 | 按附件1的231个字段名精确匹配，可改变列顺序；保持原输入尺度 |
| 标签 | 无需任何感官标签；额外列被忽略并记录 |
| 样本标识 | 默认使用`beer`；其他名称须显式给出，例如`--id-column BeerID`；未指定且无`beer`时按输入顺序生成从0开始的`input_row` |
| 标识内容 | 显式ID或默认`beer`必须非空且唯一；保留前导零、中文与字面文本`NA` |
| 单元格缺测 | 允许NaN；不接受无穷值、非数值化学内容或整行化学值全部缺失 |
| 整列缺失 | 默认报错；确有整项未测时可显式使用`--missing-columns nan`，整行仍须有观测值 |
| CSV表头 | 在pandas自动改名之前检查原始表头；重复或空列名报错 |

`--validate-only`不生成预测文件，也不执行神经模型。预测输出保持输入行顺序，包含标识列及附件4原顺序的50个感官字段。以`predictions.csv`为例，同时写出：

| 文件 | 内容 |
|---|---|
| `predictions.csv` | 50维连续值，尺度与附件4标准化响应一致 |
| `predictions.relative_grades_0_6.csv` | 用训练上下文六个七分位点划分的0—6相对等级 |
| `predictions.csv.audit.json` | 输入、上下文、权重及输出身份，缺测统计、忽略列和运行时间 |

相对等级的边界使用`searchsorted(..., side='right')`，每维共有七个等级。连续值不截断为0—6。需要移动部署目录时，可保留权重相对位置，或在预测命令中显式传入`--checkpoint`。

接口已通过23项预测CSV检验、11项训练准备检验，以及3行无标签输入的新旧推理比较。两次实际推理的150个连续输出在所核验的浮点精度范围内相容，150个相对等级一致。这些是接口与执行检查，不构成未公开测试集的预测成绩。

## 7. 结果位置与追加复核

以下路径相对新的`fresh_run`目录。结果表与数组由对应阶段生成，可直接读取，不需要另一个计算结果工作簿。

| 内容 | 主要结果位置 |
|---|---|
| 六模型折外预测 | `project/MathorCup_A/A_work/q1/oof_arrays.npz` |
| 现代预测对照 | `experiments/modern_prediction/model_comparison.csv` |
| 内层配置选择的外层预测 | `experiments/modern_prediction/nested_selection/nested_oof.npz` |
| 风格与缺测分组误差 | `experiments/modern_prediction/generalization_groups` |
| 全50维Top-3 | `project/MathorCup_A/A_work/q2/top3_compact_table.csv` |
| 风格内解释与方向稳定性 | `experiments/within_style_explanation` |
| 22风格生成化学谱 | `project/MathorCup_A/A_work/q3/revised_generated_chemical_profiles_complete.csv` |
| 生成感官谱 | `project/MathorCup_A/A_work/q3/generated_sensory_*.csv` |
| 11款主迁移与政策—距离选择 | `experiments/sparse_transfer/final_summary` |
| 200次重拟合稳定性 | `experiments/transfer_robustness/aggregate_stability.csv` |
| 跨预测器候选复评 | `experiments/modern_prediction/deployment_transfer_challenge` |

最后的`generalization_groups`阶段按`Y`、`beer`、`fold`及`targets`严格对齐已完成预测。它读取嵌套选模的`prediction`数组，并将固定Y16与内层选模流程分开报告。分组归一化误差使用同一250样本的逐目标方差作为评价尺度；原随机五折不解释为未见风格测试。

完成嵌套计算后，可额外执行独立复核与统计汇总；该命令不重新调用预测模型：

```bash
/path/to/tabicl/bin/python \
  /path/to/fresh_run/experiments/modern_prediction/audit_nested_selection.py \
  --project-dir /path/to/fresh_run/project/MathorCup_A \
  --modern-dir /path/to/fresh_run/experiments/modern_prediction \
  --nested-dir /path/to/fresh_run/experiments/modern_prediction/nested_selection \
  --output /path/to/fresh_run/experiments/modern_prediction/nested_selection/statistics
```

该输出目录须尚不存在。程序核验训练分区、45个内层视图、五个外层选择及完整250行预测，再计算固定折外预测的2000次配对啤酒重采样。未完成计算时不会输出完整结果指标。

图A1可从分组结果单独绘制，输出SVG保留可编辑文本和矢量图形：

```bash
/path/to/base/bin/python /path/to/fresh_run/visuals/plot_style_group_errors.py \
  --project-dir /path/to/fresh_run/project/MathorCup_A \
  --table /path/to/fresh_run/experiments/modern_prediction/generalization_groups/table_A2_core30_nmse.csv \
  --style-path /path/to/fresh_run/visuals/publication_style.py \
  --font-path /path/to/fresh_run/project/MathorCup_A/assets/fonts/NotoSansCJKsc-Regular.otf \
  --output-prefix /path/to/fresh_run/visuals/A1_style_core30_nmse \
  --require-nested
```

独立复核与图A1为73阶段之外的追加命令。生成与迁移的既有候选仍使用其登记的固定感官函数评价，固定f⁺的跨模型复评单列记录；数值证书、候选与模型之间的对应按各自身份检查。

## 8. 已执行检查的范围

已对统一入口进行脚本存在性、语法、源清单、输入身份及依赖顺序检查，并实际完成独立目录准备、基础数据审计和70字段编码重建。编码表与既有登记内容一致。字体程序完成了固定版本的实际下载、SHA-256核验及错误输入拒绝检查。

另以已有完整计算结果执行稀疏迁移、交叉政策、跨预测器及证据快照复核，检查来源身份、保存的矩阵、候选可行性和统计重算；损坏输入身份、计数或记录文件的反例会被拒绝。条件SDP只覆盖一个已登记试验的情形与主透视方法的22个案例分别检查。

这些检查不表示已在另一新环境重新运行全部73阶段训练和优化。重新计算时，数值库、并行安排和求解时限可能影响末位数值及搜索返回点；各阶段的身份、约束与完整性检查仍须逐项满足。实际新样本表现应由未公开测试数据评价。
