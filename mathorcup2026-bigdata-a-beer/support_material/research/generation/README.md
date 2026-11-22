# 条件生成复现说明

本目录复现固定区间高斯Copula下的清洁风味条件候选、内部留出评价、代表谱选择，以及两种等价条件抽样算法的计算开销。原项目和原始附件只读，所有新文件写入本目录下的结果文件夹。

## 项目布局与输入

解压后保留以下相对位置；在同时含有 `generation`、`project` 等目录的项目根目录运行后文命令。

| 相对路径 | 用途 |
|---|---|
| `generation/` | 本说明、脚本与 `results/` |
| `project/MathorCup_A/data/A/` | 用户放回原官方赛题附件的位置 |
| `project/MathorCup_A/A_work/q1/` | 原六模型感官程序和固定模型 |
| `project/MathorCup_A/A_work/q3/` | 原生成程序、原候选和辅助重建模型 |
| `project/MathorCup_A/A_work/audit/` | 已保存的校准表和化学字段元数据 |

包中不再复制官方 `data/A` 原始文件。完整项目需要用户从官方附件放回 `1.csv` 至 `8.xlsx`，文件名保持原样。本子模块直接读取 `data/A/1.csv`；校准核对使用项目中已有的两份导出CSV，感官描述使用既有固定模型。本模块不下载赛题数据，也不重新训练六模型感官集成。

默认项目根由脚本位置推导为 `../project/MathorCup_A`，不依赖当前工作目录。如果项目放在其他位置，在运行三个脚本前设置 `MATHORCUP_PROJECT_DIR` 为该目录，即同时包含 `data` 与 `A_work` 的 `MathorCup_A` 目录。例如POSIX shell：

```bash
export MATHORCUP_PROJECT_DIR="/path/to/MathorCup_A"
```

Windows PowerShell等价写法：

```powershell
$env:MATHORCUP_PROJECT_DIR="C:\path\to\MathorCup_A"
```

三个研究脚本共享此设置。结果目录默认固定为脚本旁的 `generation/results`；`--out` 可以选择 `generation` 内的其他目录。

## 运行环境

已执行环境为Python 3.12.14，依赖版本见 `requirements.txt`。冻结的scikit-learn模型按该版本加载；这部分生成研究不需要TabICL、PyTorch、GPU或付费API。

```bash
python -m venv .venv_generation
. .venv_generation/bin/activate
python -m pip install -r generation/requirements.txt
```

Windows激活环境可使用 `.venv_generation\Scripts\Activate.ps1`。若已有同版本Python环境，可直接安装依赖。脚本内部将BLAS和相关数值线程限制为1。

## 需要复用的原项目文件

必须保持以下原项目程序和缓存可用：

- `A_work/q3/conditional_model.py`、`transfer_profiles.py`、`run_q3.py`：区间生成器、辅助字段重建与其导入依赖。
- `A_work/q1/sensory_model.py`、`sensory_ensemble.joblib`：固定六模型集成。用于模型感官描述，不参与生成器留出训练。
- `A_work/q3/transfer_and_augmentation_models.joblib`：原225字段顺序、秩映射和四项辅助指数预测器。
- `A_work/q3/revised_generated_chemical_profiles_complete.csv`：原2200份无条件候选。
- `A_work/q3/requested_style_representatives.csv`：原Tripel、Kriek、Lambic代表，用于前后对照。
- `A_work/audit/calibrated_chemistry_source.csv`、`chemical_metadata.csv`：校准配对与单位审计。

不需要原 `revised_conditional_generator.joblib`：本轮按已固定的 `interval_gaussian, kappa=2, nu=20` 重新拟合生成器，完整复现三折评价。以上固定模型与原候选对应原官方附件；不要把其他数据替换进来而继续使用这些冻结模型。

## 完整复现顺序

从项目根依次运行：

```bash
python generation/run_generation.py
python generation/postprocess_generation.py
python generation/benchmark_sampling.py
python generation/audit_cached_generation.py
```

1. `run_generation.py` 生成二阈值主场景与四阈值敏感性候选，复算真实/原候选通过率、模型感官描述和近邻诊断，并执行仅在训练折拟合变换和参数的三折评价。固定随机种子和阈值都写入 `results/protocol.json`。
2. `postprocess_generation.py` 读取前一步结果，生成22类有效留出数与误差表，核查校准尺度，选择三个清洁风味代表，并输出化学和模型感官作图数据。它也回读所有主要CSV核查完整行数。
3. `benchmark_sampling.py` 与前两步统计独立，可单独运行；使用同一固定生成器比较225维全向量拒绝与2维边际拒绝加Schur补全，每类100合法谱、三个固定种子。时间只覆盖采样和经验逆变换，另列额外条件协方差准备时间。
4. `audit_cached_generation.py` 只读取已保存结果，不拟合或采样；核查2200谱、各类100谱、阈值、29个条件留出行及18组效率数组。

如需保护现有结果，给前三个脚本传入相同的新目录：

```bash
python generation/run_generation.py --out generation/results_replay
python generation/postprocess_generation.py --out generation/results_replay
python generation/benchmark_sampling.py --out generation/results_replay
python generation/audit_cached_generation.py --results generation/results_replay
```

## 只生成候选而跳过留出评价

`--skip-cv` 仍会生成候选、计算模型感官和新颖性描述，但不执行或刷新 `heldout_*` 文件，也不计算配对留出区间。应使用新目录，避免误读已有的旧评价文件：

```bash
python generation/run_generation.py --skip-cv --out generation/results_candidates
python generation/postprocess_generation.py --skip-cv --out generation/results_candidates
python generation/benchmark_sampling.py --out generation/results_candidates
python generation/audit_cached_generation.py --results generation/results_candidates
```

此时 `run_summary.json` 的 `cv_complete` 为 `false`；不能将该目录解释为已完成三折验证。普通后处理若找不到留出缓存会报错并提示传入 `--skip-cv`；带此参数则只做校准、代表与图数据，不汇总CV。

各研究脚本及只读审计均支持 `--help`，不会因为查看帮助而启动研究。

## 结果与解释边界

- 主场景只筛查附件锚定的DMS≤19 μg/L、双乙酰≤0.09原生字段标注mg/L。目标是固定拟合模型的 `P_style(X | 二阈值事件)`，不是改进无条件生成分布，也不是证明物理可酿造。
- 原2200候选通过255份；新条件候选22类各100份，全部通过。四阈值场景受固定proposal预算限制，只产生实际1876份，不补造不足样本。
- 条件留出有效样本是29款、14风格；Tripel/Kriek/Lambic分别为0/0/1。三个代表的感官与分类指标是模型描述，不能充作三种风格的独立真实品评验证。
- 校准表与附件1的少量实质差异另有逐行审计；E-2-nonenal外引阈值和风格依赖的4VG只作敏感性诊断。
- 效率实验减少的是丢弃候选上的225维生成与逆变换，不提高原阈值事件概率。共享环境下的毫秒墙钟只作同口径计算观察。

主要结果索引和已执行数值见 `results/generation_results_note.md`。所有主要CSV、效率NPZ采用完整缓冲、同步、原子替换和立即回读；目录可整体移动，脚本没有固定工作区绝对路径。
