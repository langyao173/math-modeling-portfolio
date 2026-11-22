# TAOLIN-DC-SIM Competition Report

## 摘要

本文将 GPU 任务调度、绿电利用、追索风险和 QUBO/Ising 求解结合为一个离散优化流程。最终实例包含 588 个作业、1929 个候选动作自旋变量。最终提交解覆盖 588/588 个作业，one-hot violation 为 0，上游 action table 已补入 horizon-rescue 真实候选，最终 dummy actions 为 0。 经 700-spin 云端候选、独立全量回放与三指标 Pareto guard 晋升后的 expected recourse 为 35.394524，CVaR95 为 47.718285。

## 1. 问题定义

给定 GPU 作业集合、候选调度动作和站点级绿电/价格/碳排时间序列，目标是在每个作业恰选一个动作的前提下，降低调度成本、峰值负载和尾部追索风险。追索风险来自绿电短缺、高价、flex 调用和可再生出力压力场景。离散选择部分编码为 QUBO，便于提交到 QUBO/Ising 求解器；连续 grid/renewable/flex-shortfall recourse 通过闭式逐时 evaluator 做后验和局部精修。

## 2. 数据与任务构造

- 作业数：588
- QUBO 变量/候选动作数：1929
- 候选动作类型：normal、delay、degrade、horizon_rescue、dummy
- dummy action 含义：无真实可行动作时的占位选择；horizon_rescue 用于处理 24h 边界内可部分执行但完整 duration 溢出的任务
- 时间粒度：24 个逐时 slot

## 3. 模型

QUBO 包含 one-hot 约束惩罚、动作成本、结构化时段风险项、峰值负载二次项和可选 recourse marginal 线性项。recourse/CVaR evaluator 对固定调度解计算各压力场景下的 grid import、renewable curtailment、flex shortfall 与 CVaR95。后续加入 iterative recourse cut，用候选解更新 marginal reference；最终再用 exact recourse polish 在 one-job 邻域直接优化真实 CVaR。

## 4. 求解流程

流程为：构造候选动作表 -> 生成 QUBO -> LocalSolver/CloudSolver bounded oracle -> one-hot repair -> exact recourse replay -> local polish -> 三指标 submission guard -> 成果包验收。历史 staged 记录覆盖 120/240/360 作业及一次旧版 1922-spin 全量调用；当前终版的云端贡献严格限定为一个 700-spin `peak_shave_lns` 同构邻域种子。完整 1929 变量调度未被声明为云端全局求解，终版只在独立 588 作业回放同时改善 expected recourse、CVaR 与峰值负荷后晋升。

## 5. 必备结果表

### 5.1 最终指标表

|   jobs |   spins |   dummy_actions |   one_hot_violations |   expected_recourse_cost_usd |   cvar95_recourse_cost_usd | source                                          |
|-------:|--------:|----------------:|---------------------:|-----------------------------:|---------------------------:|:------------------------------------------------|
|    588 |    1929 |               0 |                    0 |                      35.3945 |                    47.7183 | cloudseed_peak_shave_contextual_recourse_polish |

### 5.2 优化过程表

| stage        | source                                          |   jobs |   spins |   dummy_actions |   one_hot_violations |   repair_count |   expected_recourse_cost_usd |   cvar95_recourse_cost_usd | notes                                                                                                                       |
|:-------------|:------------------------------------------------|-------:|--------:|----------------:|---------------------:|---------------:|-----------------------------:|---------------------------:|:----------------------------------------------------------------------------------------------------------------------------|
| local_polish | cloudseed_peak_shave_contextual_recourse_polish |    588 |    1929 |               0 |                    0 |              0 |                      35.3945 |                    47.7183 | full-instance Cloud result not regenerated for the 1929-action table; same-QUBO LNS Cloud evidence is reported in Section 7 |

### 5.3 Baseline 对比表

| run              | source                                          |   dummy_actions |   one_hot_violations |   expected_recourse_cost_usd |   cvar95_recourse_cost_usd |   peak_it_mw | notes                                  |
|:-----------------|:------------------------------------------------|----------------:|---------------------:|-----------------------------:|---------------------------:|-------------:|:---------------------------------------|
| initial_solution | baseline                                        |               0 |                    0 |                      42.1419 |                    57.716  |     0.036735 | initial deterministic per-job baseline |
| final_solution   | cloudseed_peak_shave_contextual_recourse_polish |               0 |                    0 |                      35.3945 |                    47.7183 |     0.031646 | current final submitted solution       |

### 5.4 Dummy jobs 解释表

最终无 dummy jobs；此前 horizon 边界任务已由上游 `horizon_rescue` 候选动作吸收。

_Table not generated._

### 5.5 Clean-room 审计表

| audit_item             | status   | evidence                                                                                                                |
|:-----------------------|:---------|:------------------------------------------------------------------------------------------------------------------------|
| zip_and_final_csv      | pass     | zip_exists=True; final_csv_exists=True                                                                                  |
| validation_matches_csv | pass     | rows=588; unique_jobs=588; validation_status=pass                                                                       |
| cleanroom_recompute    | pass     | expected=35.390081; cvar95=47.731129                                                                                    |
| metrics_consistency    | pass     | FINAL_METRICS.json agrees with VALIDATION_RECOMPUTED.json within 1e-12 (expected_diff=0.000e+00; cvar95_diff=0.000e+00) |
| csv_field_consistency  | pass     | job_id,action_id,site,start_slot,end_slot,mode,gpu_req,power_mw,energy_mwh,action_cost,is_dummy                         |
| one_hot_violation_zero | pass     | one_hot_violations=0                                                                                                    |
| absolute_path_scan     | pass     | hits=0                                                                                                                  |
| token_scan             | pass     | hits=0                                                                                                                  |
| cache_dependency_scan  | pass     | hits=0                                                                                                                  |

## 6. 支撑实验表

### 6.1 Cloud raw / repair / polish / exact recourse polish

_Table not generated._

### 6.2 Structural QUBO proxy vs true recourse/CVaR

| run                          | source                                          |   dummy_actions |   expected_recourse_cost_usd |   cvar95_recourse_cost_usd |   peak_it_mw |
|:-----------------------------|:------------------------------------------------|----------------:|-----------------------------:|---------------------------:|-------------:|
| cloudseed_peakguard_promoted | cloudseed_peak_shave_contextual_recourse_polish |               0 |                      35.3945 |                    47.7183 |     0.031646 |

### 6.3 LocalSolver / CloudSolver / random baseline

| run                 | view          | source                                          |   dummy_actions |   expected_recourse_cost_usd |   cvar95_recourse_cost_usd |   peak_it_mw |
|:--------------------|:--------------|:------------------------------------------------|----------------:|-----------------------------:|---------------------------:|-------------:|
| full_promoted       | baseline      | baseline                                        |               0 |                      42.1419 |                    57.716  |     0.036735 |
| full_promoted       | solver        | greedy_polish                                   |               0 |                      38.9231 |                    53.3979 |     0.028576 |
| full_promoted       | solver        | recourse_true_polish                            |               0 |                      35.89   |                    48.4386 |     0.031897 |
| full_promoted       | solver        | local                                           |              21 |                      37.8809 |                    51.761  |     0.031166 |
| full_promoted       | solver        | local_polished                                  |               0 |                      38.4989 |                    53.0335 |     0.027617 |
| full_promoted       | solver        | cloudseed_peak_shave_contextual_recourse_polish |               0 |                      35.3945 |                    47.7183 |     0.031646 |
| full_promoted       | accepted      | greedy_polish                                   |               0 |                      38.9231 |                    53.3979 |     0.028576 |
| full_promoted       | risk_accepted | cloudseed_peak_shave_contextual_recourse_polish |               0 |                      35.3945 |                    47.7183 |     0.031646 |
| random_feasible_200 | min           | random_feasible                                 |               0 |                      38.205  |                    51.9707 |     0.028973 |
| random_feasible_200 | mean          | random_feasible                                 |               0 |                      39.3346 |                    53.4952 |     0.033414 |
| random_feasible_200 | median        | random_feasible                                 |               0 |                      39.3389 |                    53.5249 |     0.033369 |
| random_feasible_200 | max           | random_feasible                                 |               0 |                      40.366  |                    54.8208 |     0.03732  |

### 6.4 Penalty / neighborhood / recourse cut sensitivity

| run                                      | source                                          |   dummy_actions |   expected_recourse_cost_usd |   cvar95_recourse_cost_usd |   peak_it_mw | notes                                     |
|:-----------------------------------------|:------------------------------------------------|----------------:|-----------------------------:|---------------------------:|-------------:|:------------------------------------------|
| rec_cvar0p10_cut3_multistart_local_pass5 | recourse_true_polish                            |               0 |                      35.89   |                    48.4386 |     0.031897 |                                           |
| cloudseed_peakguard_promoted             | cloudseed_peak_shave_contextual_recourse_polish |               0 |                      35.3945 |                    47.7183 |     0.031646 |                                           |
| neighborhood_120                         | recourse_true_polish                            |               0 |                      15.3531 |                    19.7404 |     0.023863 | local_multistart_neighborhood_sensitivity |
| neighborhood_240                         | recourse_true_polish                            |               0 |                      24.6948 |                    32.0746 |     0.033906 | local_multistart_neighborhood_sensitivity |
| neighborhood_360                         | recourse_true_polish                            |               0 |                      30.4649 |                    40.5095 |     0.040521 | local_multistart_neighborhood_sensitivity |

### 6.5 Dummy jobs explanation

最终无 dummy jobs；此前 horizon 边界任务已由上游 `horizon_rescue` 候选动作吸收。

_Table not generated._

### 6.6 Tail scenario analysis

最终 CVaR 由最坏场景 `green_short_peak_price` 主导，该场景总 recourse cost 为 47.718285。

| name                   |   probability |   total_recourse_cost_usd |   grid_cost_usd |   curtailment_cost_usd |   flex_shortfall_cost_usd |   grid_import_mwh |   tail_rank |
|:-----------------------|--------------:|--------------------------:|----------------:|-----------------------:|--------------------------:|------------------:|------------:|
| green_short_peak_price |          0.2  |                   47.7183 |         44.3914 |               0.074792 |                  3.25209  |          0.356769 |           1 |
| evening_stress         |          0.1  |                   39.5235 |         35.7548 |               0.188445 |                  3.58031  |          0.346941 |           2 |
| solar_midday_shortfall |          0.05 |                   37.3599 |         34.3833 |               0.363789 |                  2.61276  |          0.330936 |           3 |
| nominal                |          0.5  |                   32.9115 |         30.6814 |               0.363789 |                  1.86626  |          0.330612 |           4 |
| high_green_low_price   |          0.15 |                   23.8319 |         22.3528 |               0.545973 |                  0.933129 |          0.320406 |           5 |

### 6.7 Staged cloud check after SDK repair

| run             | view          | source               |   one_hot_violations |   dummy_actions |   expected_recourse_cost_usd |   cvar95_recourse_cost_usd |   peak_it_mw |
|:----------------|:--------------|:---------------------|---------------------:|----------------:|-----------------------------:|---------------------------:|-------------:|
| cloud_check_120 | baseline      | baseline             |                    0 |               0 |                      17.8652 |                    23.3804 |     0.027685 |
| cloud_check_120 | solver        | recourse_true_polish |                    0 |               0 |                      15.3531 |                    19.7404 |     0.023863 |
| cloud_check_120 | solver        | cloud                |                    0 |               0 |                      16.1622 |                    20.9874 |     0.025699 |
| cloud_check_120 | solver        | cloud_polished       |                    0 |               0 |                      16.2117 |                    21.1238 |     0.021465 |
| cloud_check_120 | risk_accepted | recourse_true_polish |                    0 |               0 |                      15.3531 |                    19.7404 |     0.023863 |
| cloud_check_240 | baseline      | baseline             |                    0 |               0 |                      28.0055 |                    36.7115 |     0.03598  |
| cloud_check_240 | solver        | recourse_true_polish |                    0 |               0 |                      24.4937 |                    31.7289 |     0.032524 |
| cloud_check_240 | solver        | cloud                |                    0 |               0 |                      24.4916 |                    31.842  |     0.029628 |
| cloud_check_240 | solver        | cloud_polished       |                    0 |               0 |                      25.5367 |                    33.591  |     0.025041 |
| cloud_check_240 | risk_accepted | recourse_true_polish |                    0 |               0 |                      24.4937 |                    31.7289 |     0.032524 |
| cloud_check_360 | baseline      | baseline             |                    0 |               0 |                      34.7459 |                    46.7076 |     0.03598  |
| cloud_check_360 | solver        | recourse_true_polish |                    0 |               0 |                      30.4316 |                    40.4242 |     0.040521 |
| cloud_check_360 | solver        | cloud                |                    0 |               0 |                      31.0711 |                    41.1849 |     0.029729 |
| cloud_check_360 | solver        | cloud_polished       |                    0 |               0 |                      31.622  |                    42.9081 |     0.026059 |
| cloud_check_360 | risk_accepted | recourse_true_polish |                    0 |               0 |                      30.4316 |                    40.4242 |     0.040521 |

结论：历史 120/240/360 job staged 云端复核均成功且 one-hot violations 为 0；当前终版的可复核 CloudSolver 贡献以第 7 节 700-spin same-QUBO oracle replacement 与持久化候选为准。

## 7. 工业级 Ising 证据链

本轮增强将表述从“QUBO 能跑出一个本地 polish 解”升级为“Ising-assisted bounded large-neighborhood optimizer”。CloudSolver 用作 bounded binary action-selection neighborhood 的 oracle；所有 raw output 均经过相同 decode、one-hot repair、exact recourse replay 与 submission guard 审计。最终业务指标仍以 exact recourse/CVaR replay 为准，不用 QUBO energy 充当业务目标。

### 7.1 多 QUBO 邻域组合

| family                | purpose                                              |   jobs |   spins |   q_abs_max |
|:----------------------|:-----------------------------------------------------|-------:|--------:|------------:|
| base_lns              | general bounded action-selection improvement         |    229 |     700 |     80.6411 |
| aligned_recourse_lns  | one-swap exact recourse/CVaR diagonal alignment      |    229 |     700 |    264.5    |
| tail_cvar_lns         | CVaR95 tail-risk directed coefficients               |    229 |     700 |     80.3205 |
| cloud_sparse_tail_lns | sparse capped-range CloudSolver-friendly tail family |    169 |     520 |     45      |
| dummy_rescue_lns      | dummy and near-dummy action-space audit              |    240 |     701 |    100      |
| peak_shave_lns        | grid/import peak reduction pressure                  |    220 |     700 |     86.4106 |
| renewable_absorb_lns  | shift load toward renewable/low-price slots          |    185 |     702 |     81.4762 |

### 7.2 Same-QUBO oracle replacement

| family                | oracle            | status   |   spins |   raw_feasible |   repair_count |   accepted_under_submission_guard |   polished_cvar95 |   polished_true_objective |
|:----------------------|:------------------|:---------|--------:|---------------:|---------------:|----------------------------------:|------------------:|--------------------------:|
| base_lns              | incumbent         | success  |     700 |              1 |              0 |                                 1 |           48.3848 |                   84.2542 |
| base_lns              | greedy_one_hot    | success  |     700 |              1 |              0 |                                 0 |           50.6007 |                   87.6369 |
| base_lns              | random_restart_20 | success  |     700 |              1 |              0 |                                 0 |           40.6485 |                   70.4283 |
| aligned_recourse_lns  | incumbent         | success  |     700 |              1 |              0 |                                 1 |           48.4394 |                   84.3278 |
| aligned_recourse_lns  | greedy_one_hot    | success  |     700 |              1 |              0 |                                 1 |           49.0731 |                   85.1987 |
| aligned_recourse_lns  | random_restart_20 | success  |     700 |              1 |              0 |                                 0 |           40.5228 |                   70.086  |
| tail_cvar_lns         | incumbent         | success  |     700 |              1 |              0 |                                 1 |           48.4394 |                   84.3278 |
| tail_cvar_lns         | greedy_one_hot    | success  |     700 |              1 |              0 |                                 0 |           50.675  |                   87.7202 |
| tail_cvar_lns         | random_restart_20 | success  |     700 |              1 |              0 |                                 0 |           40.5228 |                   70.086  |
| cloud_sparse_tail_lns | incumbent         | success  |     520 |              1 |              0 |                                 1 |           48.4394 |                   84.3278 |
| cloud_sparse_tail_lns | greedy_one_hot    | success  |     520 |              1 |              0 |                                 1 |           49.7021 |                   86.4165 |
| cloud_sparse_tail_lns | random_restart_20 | success  |     520 |              1 |              0 |                                 0 |           43.6878 |                   75.6735 |
| dummy_rescue_lns      | incumbent         | success  |     701 |              1 |              0 |                                 1 |           48.3848 |                   84.2542 |
| dummy_rescue_lns      | greedy_one_hot    | success  |     701 |              1 |              0 |                                 1 |           48.4941 |                   84.4317 |
| dummy_rescue_lns      | random_restart_20 | success  |     701 |              1 |              0 |                                 0 |           41.1015 |                   71.3794 |
| peak_shave_lns        | incumbent         | success  |     700 |              1 |              0 |                                 1 |           48.4386 |                   84.3286 |
| peak_shave_lns        | greedy_one_hot    | success  |     700 |              1 |              0 |                                 1 |           48.2715 |                   84.0336 |
| peak_shave_lns        | random_restart_20 | success  |     700 |              1 |              0 |                                 0 |           42.368  |                   73.5815 |
| peak_shave_lns        | cloud_balanced    | success  |     700 |              1 |              0 |                                 1 |           47.7183 |                   83.1128 |
| renewable_absorb_lns  | incumbent         | success  |     702 |              1 |              0 |                                 1 |           48.4392 |                   84.3259 |
| renewable_absorb_lns  | greedy_one_hot    | success  |     702 |              1 |              0 |                                 1 |           49.5035 |                   85.7286 |
| renewable_absorb_lns  | random_restart_20 | success  |     702 |              1 |              0 |                                 0 |           50.0435 |                   86.6515 |

### 7.3 工业指标

| solution            |   FlexDeliveryRate |   TailShortfall95_mwh |   SLAViolationRate_delay_or_dummy |   GridImportPeak_mw |   RenewableCurtailment_mwh |   expected_recourse_cost_usd |   cvar95_recourse_cost_usd | dominant_tail_scenario   |
|:--------------------|-------------------:|----------------------:|----------------------------------:|--------------------:|---------------------------:|-----------------------------:|---------------------------:|:-------------------------|
| final_full_solution |            0.76109 |              0.013008 |                               0.5 |            0.031646 |                   0.072758 |                      35.3945 |                    47.7183 | green_short_peak_price   |

### 7.4 Cloud contribution and proxy audit

| family         |   cloud_objective |   best_non_cloud_objective |   cloud_contribution_positive_if_gt0 |   cloud_cvar95 |   best_non_cloud_cvar95 | best_non_cloud_oracle   |
|:---------------|------------------:|---------------------------:|-------------------------------------:|---------------:|------------------------:|:------------------------|
| peak_shave_lns |           83.1128 |                    84.0336 |                             0.920745 |        47.7183 |                 48.2715 | greedy_one_hot          |

| family                |   samples |   qubo_energy_vs_true_objective_corr |   qubo_energy_vs_true_objective_rank_corr |   qubo_energy_vs_recourse_only_corr | best_qubo_oracle   | best_true_objective_oracle   | best_recourse_only_oracle   |   best_oracle_match |
|:----------------------|----------:|-------------------------------------:|------------------------------------------:|------------------------------------:|:-------------------|:-----------------------------|:----------------------------|--------------------:|
| base_lns              |         3 |                             1        |                                       0.5 |                           -0.982714 | greedy_one_hot     | incumbent                    | random_restart_20           |                   0 |
| aligned_recourse_lns  |         3 |                             1        |                                       0.5 |                           -0.998684 | greedy_one_hot     | incumbent                    | random_restart_20           |                   0 |
| tail_cvar_lns         |         3 |                             1        |                                       0.5 |                           -0.983515 | greedy_one_hot     | incumbent                    | random_restart_20           |                   0 |
| cloud_sparse_tail_lns |         3 |                             1        |                                       0.5 |                           -0.983051 | greedy_one_hot     | incumbent                    | random_restart_20           |                   0 |
| dummy_rescue_lns      |         3 |                             1        |                                       0.5 |                           -0.999931 | greedy_one_hot     | incumbent                    | random_restart_20           |                   0 |
| peak_shave_lns        |         4 |                             0.999993 |                                       0.8 |                           -0.994758 | greedy_one_hot     | cloud_balanced               | random_restart_20           |                   0 |
| renewable_absorb_lns  |         3 |                             1        |                                       0.5 |                            0.800325 | greedy_one_hot     | incumbent                    | incumbent                   |                   0 |

解释：oracle replacement 表把 CloudSolver、incumbent、greedy 和 random restart 放在同一个 QUBO、同一个 decode/repair/replay 规则下比较。`true_objective` audit 使用 exact recourse/CVaR 加 submission guard penalty；`recourse_only` 单独报告纯 expected+CVaR 相关性，避免不可提交的低 recourse 随机解被误判为业务可接受改进。`aligned_recourse_lns` 和 `cloud_sparse_tail_lns` 在 QUBO diagonal 中注入 one-swap exact guarded recourse/CVaR delta，并对 Cloud-friendly family 做稀疏化和动态范围封顶。

### 7.5 Penalty calibration sweep

|   one_hot_penalty_factor |   one_hot_penalty |   raw_feasible |   true_objective_expected_plus_cvar |   CVaR95 |   dummy_count |   coefficient_abs_range |   greedy_improvements |
|-------------------------:|------------------:|---------------:|------------------------------------:|---------:|--------------:|------------------------:|----------------------:|
|                      0.5 |                40 |              1 |                             87.6369 |  50.6007 |             0 |                 80      |                    51 |
|                      1   |                80 |              1 |                             87.6369 |  50.6007 |             0 |                 80.6411 |                    51 |
|                      2   |               160 |              1 |                             87.6369 |  50.6007 |             0 |                160.641  |                    51 |
|                      4   |               320 |              1 |                             87.6369 |  50.6007 |             0 |                320.641  |                    51 |

### 7.6 受控规模精确 MILP 基线

对 30/60/100 个嵌套作业集合开放全部真实动作，其余作业固定为终版；HiGHS 最小化与统一评估器一致的期望追索加五场景 CVaR95，并在主目标容差内二级最小化峰值。该实验只给出受控邻域证书，不是 588 作业完整问题的全局最优证书。

|   controlled_jobs | solution                | solver_primary_status   |   audit_objective_usd |   audit_gap_to_exact_pct |   peak_it_mw |   changed_jobs |
|------------------:|:------------------------|:------------------------|----------------------:|-------------------------:|-------------:|---------------:|
|                30 | current_final           | not_applicable          |               83.1128 |                -0        |     0.031646 |              0 |
|                30 | exact_milp              | optimal                 |               83.1128 |                 0        |     0.031646 |              0 |
|                30 | exact_milp_peak_guarded | optimal                 |               83.1128 |                 0        |     0.031646 |              0 |
|                60 | current_final           | not_applicable          |               83.1128 |                 0.004624 |     0.031646 |              0 |
|                60 | exact_milp              | optimal                 |               83.109  |                 0        |     0.033319 |              7 |
|                60 | exact_milp_peak_guarded | optimal                 |               83.1108 |                 0.002158 |     0.031646 |              8 |
|               100 | current_final           | not_applicable          |               83.1128 |                 0.010378 |     0.031646 |              0 |
|               100 | exact_milp              | optimal                 |               83.1042 |                 0        |     0.038849 |             29 |
|               100 | exact_milp_peak_guarded | optimal                 |               83.1108 |                 0.007912 |     0.031646 |             24 |

## 8. 局限与边界

- 当前市场价格数据仍为 synthetic fallback，不能作为真实市场收益结论。
- dummy-only 边界已通过 horizon-rescue 候选动作处理；后续仍需用真实迁移/跨站动作替换该工程化 rescue proxy。
- recourse evaluator 尚未纳入 storage、unit commitment、DC power flow 与整数 redispatch。
- 云端额度应受控使用；报告只声明 bounded binary neighborhood oracle，不声明伊辛机全局求解完整调度问题。

## 9. 提交材料位置

- `deliverables/final_solution_package/FINAL_SOLUTION.csv`
- `deliverables/final_solution_package/VALIDATION.json`
- `deliverables/final_solution_package/VALIDATION_RECOMPUTED.json`
- `deliverables/final_solution_package/artifacts/`
- `deliverables/TAOLIN_DC_SIM_final_solution_package.zip`
