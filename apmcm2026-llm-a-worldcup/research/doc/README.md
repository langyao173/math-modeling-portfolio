# 运行说明 / Execution Guide

## 1. 项目定位

本项目完整实现 2026 APMCM A 题的三个强制任务，并在不改变强制基线的前提下增加时序验证、联合进球分布、概率校准、不确定性区间、泄漏检查和预测证书。

- 生产链路仅依赖 Python。
- MATLAB 仅用于候选方案筛选和数值交叉核验，不是最终运行依赖。
- 默认使用赛题提供的 1930–2014 三个 CSV；原始文件只读并通过 SHA-256 校验。
- 2026 输出是可复现的情景演示，因为赛题没有提供指定对阵或 2026 输入表。

## 2. 环境

- Python 3.11–3.13（项目锁定并验证 Python 3.12）
- `uv` 0.7 或更高
- Windows、Linux 或 macOS；最终 Python 链路不要求 MATLAB

在项目目录执行：

```powershell
uv sync --frozen --extra dev
uv run python main.py
```

官方最小链路：

```powershell
uv run python src/main.py --config official
```

完整冻结链路（默认）：

```powershell
uv run python src/main.py --config final
```

研究验证链路（必须写入与 final 分离的目录）：

```powershell
uv run python src/main.py --config research --output research_outputs
```

## 3. 赛题任务到代码的映射

| 赛题要求 | 主要模块 | 自动输出 |
|---|---|---|
| 问题 1 描述统计与趋势 | `reporting/tables.py`, `reporting/figures.py`, `reporting/narrative.py` | 统计 CSV、两类趋势图、数据驱动文字总结 |
| 问题 1 线性回归与 2018 审计 | `models/aggregate.py` | 系数/HC3 显著性/VIF/拟合与 LOOCV、2018 误差 |
| 问题 2 阶段统计与相关图 | `data/canonicalize.py`, `reporting/*` | 六类历史兼容阶段统计、相关系数、散点图 |
| 问题 2 单场线性回归 | `models/match.py`, `models/half_time.py` | 1930–2010/2014 时序评估、MAE/MSE/R²、半场情景接口 |
| 问题 3 历史特征 | `features/team_history.py` | 四类赛前历史特征、相关热力图、数据截止信息 |
| 问题 3 多项逻辑回归 | `models/outcome.py` | 2014 accuracy、混淆矩阵、三类概率及情景接口 |
| 研究增强 | `models/count.py`, `models/calibration.py`, `validation/*` | 外层时序折、proper scores、校准、联合比分分布、消融/压力测试 |
| 审计与复现 | `data/audit.py`, `uncertainty/certificates.py`, `reporting/manifest.py` | schema、缺失/重复、lineage、证书、manifest、SHA-256 |

## 4. 关键数据规则

1. `WorldCupMatches.csv` 从 852 行按 `(RoundID, MatchID)` 去重为 836 场；2014 从错误的 80 场恢复为 64 场。
2. `WorldCupPlayers.csv` 精确去重为 37,048 行，并通过 `(RoundID, MatchID)` 与比赛表连接。
3. `WorldCups_preview.csv` 的 `Attendance` 中句点是千位分隔符，例如 `3.386.810 = 3386810`。
4. 半场进球只允许进入 `half_time` 模型，绝不进入赛前胜平负模型。
5. 问题 3 的四个强制历史特征仅使用当前世界杯之前已完成的赛事。
6. 点球晋级不覆盖统计比分结果；平局仍标记为 `draw`，晋级目标另行标记。

## 5. 2018 输入歧义

赛题提供了 2018 真实总进球 169，但没有提供回归所需的 2018 总观众数。冻结的 closed-track 运行只使用 1930–2014 每场观众数的预先冻结对数线性外推，再乘以 64 场；不会把 2018 实际观众数偷偷写入模型，也不提供可改变这一审计口径的 CLI 覆盖入口。

## 6. 输出目录

`outputs/` 下包含：

- `audit/`: schema、missingness、duplicates、reconciliation、lineage；
- `tables/`: 问题 1–3 强制表格及研究比较表；
- `figures/`: PNG 300 DPI 与 PDF 矢量图；
- `logs/`: 一键运行日志及运行时间；
- `models/`: 冻结 Python 模型；
- `predictions/`: 2018 审计和 2026 情景 JSON；
- `certificates/`: 数据截止、schema/model/config hash、概率、区间和 OOD；
- `analysis/`: 严格分析报告、统计附录和图目录；
- `manifest.json`: 最后生成的完整文件清单与 SHA-256。

## 7. 验证

```powershell
uv run pytest
uv run ruff check .
uv run python verify.py
```

最终交付必须完成完整测试、final/research 双 manifest 验证、两次独立清洁室运行和
提交 ZIP 解压复核；只有实际测得的结果才写入 `doc/AUDIT.md`。
