# 支撑材料说明（APMCM 2026 B题：芯片歧管式微通道热管理系统优化）

本包为论文全部计算与图表的可复现代码。所有路径均为相对路径，解压到任意目录即可运行。

## 0. 审计修正版说明

本版本为 `v2_audit_fix`。相对原稿，已修复会影响结论可信度的三个实现层问题：

1. **严格 nested LOOCV**：`p2_surrogate.py` 的外层 84 次 LOOCV 中，标准化、特征变换、超参数搜索和模型拟合均只在 83 个训练样本内完成。旧协议结果仅保存在 `代码/p2_cv_results_legacy_reference.json`，不再作为正文主结果。
2. **γ 边界公差不外推**：`design_star=(0.22,4.5,4)` 位于 γ 上边界。`p5_sensitivity.py` 改用设计域内条件截断正态分布，审计文件 `代码/p5_mc_domain_audit.json` 中 `outside_domain_count=0`。
3. **Sobol 多 seed 置信区间**：Sobol 基础样本量提升至 `32768`，使用 `seed=0..9` 重复估计。正文表格应使用 `代码/p5_sobol_table_for_paper.csv`，不再使用单 seed 裸点估计。

统一配置位于 `代码/config.py`，集中管理设计域、离散 `N_levels=[2,4,6,8,10]`、随机种子、Sobol/MC 样本量和输出路径。默认 `domain_clip=False`，禁止隐式 clip；边界分布必须显式使用条件截断。

## 1. 环境要求

- Python ≥ 3.8
- 依赖库：`numpy scipy pandas scikit-learn joblib matplotlib pymoo`

安装示例：

```bash
pip install numpy scipy pandas scikit-learn joblib matplotlib pymoo
```

## 2. 一键运行

```bash
python run_all.py
```

脚本按问题 1→5 顺序执行，并在最后运行 `audit_tests.py`。正文审计默认检查支撑材料同级目录下的最终稿 `Bapmcm26202913.docx`，并筛查禁用措辞；如需强制检查正文是否逐字包含内部证据文件名，可设置 `STRICT_MANUSCRIPT_AUDIT=1` 后运行。

## 3. 目录结构

```text
答案支撑材料/
├── run_all.py
├── README.md
├── CHANGELOG.md
├── 附件2_data.tsv
├── 代码/
└── 图表/
```

## 4. 脚本与主要输出

| 脚本 | 功能 | 主要输出 |
|---|---|---|
| `config.py` | 统一设计域、随机种子、样本量和路径 | 被所有审计脚本引用 |
| `p1_mechanism_model.py` | 灰箱机理模型标定；误差、相关性、物性假设审计 | `p1_fit_results.json`, `p1_error_summary.json`, `p1_correlation_audit.json`, `p1_property_assumption.json` |
| `p1_ablation_identifiability.py` | 灰箱模型消融和参数多起点可辨识性审计 | `p1_ablation_identifiability.csv`, `p1_parameter_stability.csv`, `p1_eps0_identifiability_note.json` |
| `p1_make_figures.py` | 主效应、误差审计 parity、消融图、参数稳定性图 | `p1_fig1~7_*.png` |
| `p2_surrogate.py` | strict nested LOOCV；RSM-3 解析精度代理；GPR 不确定性代理 | `p2_nested_cv_results.json`, `p2_cv_protocol_audit.json`, `rsm3_precision_models.pkl`, `gpr_uncertainty_models.pkl`, `dual_proxy_check.csv` |
| `p2_make_figures.py` | 模型横评、旧协议对照、GPR/RSM-3 误差对比、校准图 | `p2_fig1~5_*.png`, `p2_fig1b_*.png`, `p2_fig1c_*.png` |
| `p3_optimize.py` | 高分辨率网格、分层 NSGA-II、网格收敛、局部精修、双代理 Pareto 对照 | `p3_results.json`, `p3_pareto_front.csv`, `p3_grid_convergence.csv`, `p3_local_refinement.csv` |
| `p3_make_figures.py` | Pareto、决策空间、双代理、网格收敛、局部精修图 | `p3_fig1~6_*.png` |
| `p4_weight_robust.py` | 权重扫描；min-max regret 与 ε-coverage 分开输出 | `p4_results.json`, `p4_robust_tradeoff.csv`, `p4_*.npy` |
| `p4_make_figures.py` | 权重三角图、鲁棒准则取舍、场景分析 | `p4_fig1~4_*.png` |
| `p5_sensitivity.py` | 弹性系数、Sobol CI、域内 MC、γ 边界风险对照、工况扫掠 | `p5_results.json`, `p5_sobol_ci.json`, `p5_sobol_table_for_paper.csv`, `p5_mc_domain_audit.json`, `p5_boundary_risk_compare.csv` |
| `p5_make_figures.py` | 敏感性、域内 MC、γ 分布修正、Sobol CI、工况波动 | `p5_fig1~6_*.png` |
| `audit_tests.py` | 检查 CV、MC、Sobol 表和正文禁用措辞 | 通过时输出 `audit tests passed` |

## 5. 关键算法实现要点

- **问题一**：灰箱机理模型只作为趋势、排序、极值与工况解释模型，不承担逐点精算。误差审计显示压降 MaxRE 为 8.535%，24/84 个样本超过 ±3%。
- **问题二**：RSM-3 作为解析精度代理，GPR 作为不确定性代理。所有正文 LOOCV 指标来自 strict nested 协议。
- **问题三**：优化限定在附件 2 覆盖的设计域内，`N` 始终取离散集合 `{2,4,6,8,10}`。使用高分辨率网格、分层 NSGA-II 和局部 SLSQP 精修互证，不声称连续域严格全局最优。
- **问题四**：现实偏好域中 min-max regret 解为 `(0.2175,3.55,4)`，ε-coverage 解为 `(0.22,4.5,4)`；最终方案是覆盖率主导鲁棒折中。
- **问题五**：γ=4.5 是边界设计。MC 公差传播使用域内条件截断正态，Sobol 使用 10 seed 置信区间，主文不使用违反 `S_T>=S_1` 的单 seed 裸点。

## 6. 结果核对

运行结束后应核对：

- `p1_error_summary.json`：`dP MaxRE_pct=8.535`，`outside_3pct_count=24`。
- `p2_cv_protocol_audit.json`：所有 outer fold 的 `train_count=83`，`test_count=1`。
- `p3_results.json`：综合折中推荐仍为 `(0.22,4.5,4)`，但限定为设计域内结论。
- `p4_results.json`：现实偏好域两鲁棒准则不同解，最终采用 coverage 解。
- `p5_mc_domain_audit.json`：`outside_domain_count=0`，`gamma_max_sample<=4.5`。
- `p5_sobol_table_for_paper.csv`：含 10 seed 均值、95% CI 和理论一致化 display 值。

## 7. 旧结果说明

原 `p2_cv_results.json` 的非嵌套调参结果已废弃为正文主结果。首次运行审计修正版时，脚本会将旧文件备份到 `p2_cv_results_legacy_reference.json`，仅用于绘制“旧协议 vs strict nested”对照图。
