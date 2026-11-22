# 互补感官预测

本目录包含有限预定实验、全部比较结果、独立算术审计及双支预测CSV接口。主实验是训练侧两目标组权重融合；固定等权是同时预定的简洁参照，可单独作为工程部署选择。

完整可移植命令、两个环境的依赖及52个外部输入清单见`REPRODUCE.md`与`REPRO_INPUTS.json`。薄拷贝目录的实际运行核验见`PORTABILITY_CHECK.json`。

主要结果与解释见`PREDICTION_FINDINGS.md`。实验方案在新融合外层得分计算前保存为`frozen_protocol.json`，没有根据外层得分追加候选或改权重规则。

| 文件或目录 | 内容 |
|---|---|
| `run_prespecified_fusion.py` | 使用每个外折专属inner OOF拟合标量权重，应用对应outer预测 |
| `fusion_replay_v1/pooled_metrics.csv` | 全部7种方案的统一pooled R² |
| `fusion_replay_v1/all_prespecified_comparisons.csv` | 全部36项固定预测配对比较 |
| `fusion_replay_v1/fold_metrics.csv` | 全部35行逐折得分 |
| `fusion_replay_v1/fitted_weights.csv` | 10个训练侧组权重及条件稳定性诊断 |
| `fusion_replay_v1/per_target_error_and_loss_covariance.csv` | 逐目标、逐外折的残差与平方损失协方差 |
| `fusion_replay_v1/relative_grade_metrics.csv` | 相对等级MAE、同档率和±1档率 |
| `fusion_replay_v1/style_metrics.csv` | 全部22种风格分组误差 |
| `fusion_replay_v1/outer_predictions.npz` | 原参照与全部新方案的250行真实外层预测 |
| `independent_audit/` | 不导入runner的独立重算代码、结果和网络共享源码说明 |
| `deployment_fixed_half/` | 250行训练上下文，固定1/2融合，真实OOF验证缓存 |
| `deployment_groupwise/` | 相同上下文，全训练OOF校准的最终两个部署权重 |
| `predict_fusion_csv.py` | 新输入神经预测或显式历史CV缓存重放 |
| `deployment_validation/validation_result.json` | 已完成的500行双模式CSV重放验证记录 |

## 新输入的预测

在本目录运行。先安装版本固定的依赖和单一公开checkpoint；下载不需要账户或付费API。

```bash
python -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements_deployment.txt
python fetch_public_weights.py --model tabicl
python predict_fusion_csv.py new_chemistry.csv predictions.csv --artifact deployment_fixed_half --threads 2
```

输入按231个化学字段名对齐，使用赛题附件1的原有尺度；额外元数据忽略。默认保留`beer`为输出标识，也可显式指定`--id-column BeerID`。无标识列时输出零起始`input_row`。标识必须非空且唯一。缺失单元格允许NaN；缺失整列默认报错，确实无法测量时可显式用`--missing-columns nan`。

输出为50维标准化感官值及独立的`.relative_grades_0_6.csv`。相对等级由250行训练上下文的七分位切点定义，不能解释为恢复出的原始品评分。`--write-branches`另保存Y16和Native50单支结果，便于核对角色。

新神经路径需要66个标量目标预测：16个潜在目标、50个原始感官目标。二者使用同一公开权重文件，目标上下文各自构建。当前没有启用跨目标KV或表征缓存。官方v2.2.0公开接口仍会在每次fit时加载网络；不得将“单一checkpoint”写成“已测得只加载一次”。

本轮环境没有torch/tabicl和checkpoint，因此尚未执行这条新250上下文神经路径。已导出的上下文、CSV契约和下述真实缓存重放均已检查；两者的验证范围分别记录。

仅核对新CSV契约而不加载神经模型：

```bash
python predict_fusion_csv.py new_chemistry.csv unused.csv --artifact deployment_fixed_half --validate-only
```

## 不需要torch的真实缓存重放

以下命令重放原250款啤酒的真实外层预测，不回答新的或改动后的化学输入，也不声称执行了新神经推理。输出审计文件明确标记`historical_outer_cv_cache_replay`。

```bash
python predict_fusion_csv.py validation_inputs/chemistry_250.csv replay.csv --artifact deployment_fixed_half --replay-cache deployment_fixed_half/outer_cv_validation_cache.npz --write-branches
python validate_deployment_replay.py --output validation_rerun
```

验证脚本检查固定等权、分组两种模式的全部250行，逐值比较已审计数组和相对等级；检查输入行列重排后的标识保留，以及改动化学值被缓存模式拒绝。请选择尚不存在的输出文件或空验证目录，以保留已有结果。

分组模式的缓存重放使用各outer专属inner权重；新神经部署使用全体训练OOF校准的最终权重。后者不用于回填CV得分。

## 重算有限实验

若完整历史工程放在相邻的`../project/MathorCup_A`，最新内层缓存放在`../checkpoint/unpacked/nested_selection`：

```bash
python run_prespecified_fusion.py --modern ../project/MathorCup_A/frontier_revision/experiments/modern_prediction --nested ../checkpoint/unpacked/nested_selection --output fusion_replay_reproduced
python independent_audit/audit.py
```

统计重算只需要NumPy、pandas及既存预测文件，不需要torch。权重分母与分子使用相应outer的200行训练标签方差；完整250行方差仅参与最终统一评分。配对bootstrap条件于已有预测，不重训模型。

`prepare_fusion_deployment.py`可从原全250行Y16上下文导出两种部署契约，不调用神经网络：

```bash
python prepare_fusion_deployment.py --source-artifact ../project/MathorCup_A/frontier_revision/experiments/modern_prediction/final_artifact --modern ../project/MathorCup_A/frontier_revision/experiments/modern_prediction --output deployment_reexported --mode fixed_half
```

## 与生成迁移实验的关系

新增融合属于问题一。原生成与迁移的感官映射仍是固定六模型f★；原Y16部署挑战属于单支f⁺。这两种下游结果不能自动归给新融合预测器。22种风格分组仍来自随机留出，不是未见风格或独立外部测试。
