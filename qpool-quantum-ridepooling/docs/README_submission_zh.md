# QPool：面向动态拼车的量子—经典协同优化

QPool 将动态拼车中的车辆、请求与路线约束转化为组合分配问题，通过可行候选生成、冲突图建模、模型约简、Ising 求解和业务回译，连接实时调度与量子优化。

## 环境与安装

核心工程使用 Python 3.10 及以上版本。官方 Kaiwu SA 入口使用 Python 3.10，并需在本机安装和授权官方 SDK。

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

## 目录与功能

| 路径 | 功能 |
|---|---|
| `src/qpool/replay.py` | 请求到达、车辆状态、可行候选与闭环调度回放 |
| `src/qpool/qubo.py` | 组合分配、QUBO/Ising 映射和矩阵导出 |
| `src/qpool/competition.py` | 图约简、条件邻域、量化与成本修正 |
| `src/qpool/solvers.py` | 经典求解与参考解计算 |
| `src/qpool/provider_reports.py` | 真机报告解析、能量复算、候选择优与分配接收 |
| `scripts/` | 数据获取、实验回放、模型生成和结果复核入口 |
| `tests/` | 建模、求解、回放和平台报告的自动测试 |
| `upload/03_application_batch/` | 六项应用实例的 Ising 矩阵、模型元数据和清单 |
| `results/stage2/` | 经典回放、冻结快照和应用实例的对照结果 |
| `results/stage3/provider_reports/` | 六份平台原始 PDF |
| `results/stage3/verified/` | 真机返回向量的复核结果与业务回译 |

## 复核六项真机实验

```bash
python scripts/import_provider_reports.py --output results/stage3/recheck
```

该入口逐项核对平台返回向量、Ising 能量、模型约束和原始业务效用，结果写入独立的 `recheck` 目录。报告中的六项任务各对应一个平台任务 ID；每项任务提供十组备选解。A01—A04 为完整周期经约简后的模型，A05—A06 为固定其余分配的条件邻域。

## 复现系统回放

数据采用 BUILTNYU/ridepooling 的 Sioux Falls 公开示例前 240 个请求，保留原到达时刻。获取脚本固定了源版本和文件校验值。依据上游许可完成确认后，执行：

```bash
python scripts/fetch_example_data.py --accept-source-license --prefix 240 --out data/private_example
python scripts/contest_round.py
python scripts/run_final_round.py
python scripts/build_next_batch.py
```

正文系统对照使用 `results/stage2/contest_replay/all_runs.csv` 中的贪心、局部搜索和 HiGHS 结果，结合 `results/stage2/final_replay/all_runs.csv` 中的 QPool 经典后端结果；共 4 种方法、24 组条件、96 条运行记录。车队规模为 8、16、32、48，决策周期为 60、300 秒，初始化种子为 11、22、33。

## 官方 Kaiwu SA 入口

矩阵预检查：

```bash
python scripts/run_kaiwu_batch.py
```

在完成官方 SDK 安装和授权的 Python 3.10 环境执行本地 SA：

```bash
python scripts/run_kaiwu_batch.py --execute
```

Windows 可使用 `RUN_KAIWU_SA.cmd`。官方 SA 为待执行的对照入口，输出目录为 `results/kaiwu_application_sa/`。

## 报告和支撑材料

项目报告以提交包中“项目报告”文件夹内的 PDF 为准。配套真机报告、实验数据、正文排版和图表源码收录于“支撑材料”。
