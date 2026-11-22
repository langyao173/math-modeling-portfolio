# CHANGELOG

## v2_audit_fix - 2026-06-14

修复内容：

- 修复 `p2_surrogate.py` 的验证协议泄漏：所有正文 LOOCV 指标改为 strict nested LOOCV。
- 新增 `config.py`，统一设计域、随机种子、Sobol/MC 样本量和路径。
- 新增双代理输出：
  - `rsm3_precision_models.pkl`
  - `gpr_uncertainty_models.pkl`
  - `dual_proxy_check.csv`
- 新增问题一审计：
  - `p1_error_audit.csv`
  - `p1_error_summary.json`
  - `p1_correlation_audit.json`
  - `p1_property_assumption.json`
  - `p1_ablation_identifiability.csv`
  - `p1_parameter_stability.csv`
- 修复问题三连续域表述证据：
  - `p3_grid_convergence.csv`
  - `p3_local_refinement.csv`
- 修复问题四鲁棒准则混写：
  - `p4_robust_tradeoff.csv`
  - `p4_results.json` 中明确 `minimax` 与 `coverage` 不同解。
- 修复问题五 γ 边界外推：
  - `p5_mc_domain_audit.json`
  - `p5_boundary_risk_compare.csv`
  - MC 采样改为设计域内条件截断正态。
- 修复 Sobol 单 seed 理论矛盾：
  - `p5_sobol_replicates.csv`
  - `p5_sobol_ci.json`
  - `p5_sobol_table_for_paper.csv`
- 新增 `audit_tests.py`，检查 CV、MC、Sobol 和正文禁用措辞。

关键结果文件与生成脚本：

| 文件 | 生成脚本 | 用途 |
|---|---|---|
| `p1_error_summary.json` | `p1_mechanism_model.py` | 问题一误差审计表 |
| `p1_ablation_identifiability.csv` | `p1_ablation_identifiability.py` | 问题一消融表 |
| `p2_nested_cv_results.json` | `p2_surrogate.py` | 问题二代理横评表 |
| `p2_cv_protocol_audit.json` | `p2_surrogate.py` | 验证协议审计 |
| `p3_grid_convergence.csv` | `p3_optimize.py` | 问题三网格收敛表 |
| `p3_local_refinement.csv` | `p3_optimize.py` | 问题三局部精修表 |
| `p4_robust_tradeoff.csv` | `p4_weight_robust.py` | 问题四鲁棒准则取舍表 |
| `p5_mc_domain_audit.json` | `p5_sensitivity.py` | 问题五域内 MC 审计 |
| `p5_sobol_table_for_paper.csv` | `p5_sensitivity.py` | 问题五 Sobol 主文表 |

## v1_original

原稿结果链包含：

- GPR + RSM/SVR/KRR/RF/MLP 横评；
- p3 网格枚举 + 分层 NSGA-II；
- p4 权重单纯形扫描；
- p5 单 seed Sobol + 公差 MC。

已废弃为正文主依据的旧结果：

- 非嵌套调参 LOOCV；
- γ=4.5 周围对称 ±3σ 公差传播；
- 单 seed Sobol 裸点表；
- 旧版误差带、有限网格、连续域最优性与三视角一致性等过强表述。
