# 依赖与精确输入清单

保留以下目录关系，解压后可以移动整个`deep`目录。所有命令均从`deep/`目录运行；代码没有写死原工作区地址。缓存复算已在Python 3.12.14、NumPy 2.3.5、pandas 2.2.3下完成。

| 目录 | 作用 |
|---|---|
| `project/MathorCup_A/` | 原现代模型OOF和完整外层视图缓存 |
| `checkpoint/unpacked/nested_selection/` | 最新完整内层预测及原硬选择结果 |
| `prediction/` | 新算法、固定协议、结果、部署上下文与独立审计 |

## 仅复算与审计

这一阶段只安装两个数值依赖，不需要PyTorch、TabICL、公开模型权重、经典joblib模型或任何付费API。

```bash
python -m pip install -r prediction/requirements_replay.txt
python prediction/list_reproduction_inputs.py --deep-root . --output prediction/repro_inputs_checked.json
```

下面三个命令分别完成全部预定融合实验、独立算术审计、真实缓存CSV部署契约验证。输出目录请选新路径，已有实验不覆盖。

```bash
python prediction/run_prespecified_fusion.py --modern project/MathorCup_A/frontier_revision/experiments/modern_prediction --nested checkpoint/unpacked/nested_selection --output prediction/repro_runs/fusion
python prediction/independent_audit/audit.py --modern project/MathorCup_A/frontier_revision/experiments/modern_prediction --nested checkpoint/unpacked/nested_selection --output prediction/repro_runs/fusion --result-json prediction/repro_runs/fusion/independent_audit.json
python prediction/validate_deployment_replay.py --output prediction/repro_runs/deployment_validation
```

最后一个命令使用包内已经审计的`fusion_replay_v1/outer_predictions.npz`作为CSV逐值参照，不依赖刚生成的结果目录。前两个命令已经独立保证重新生成结果与原方案一致。部署缓存验证不会加载神经模型或生成新神经预测。

新增重算结果的逐目标协方差，以及感官等级、风格分组诊断：

```bash
python prediction/error_covariance_diagnostics.py --results prediction/repro_runs/fusion
python prediction/summarize_fusion_diagnostics.py --results prediction/repro_runs/fusion --project-dir project/MathorCup_A
```

风格分组这一步额外读取`project/MathorCup_A/data/A/1.csv`。其余基础复算从既有OOF归档读取真值和行标识，不需要再次读取全部附件数据。

## 必须保留的52个外部文件

精确展开的路径、字节数、摘要及消费程序见`REPRO_INPUTS.json`。以下`f=0,…,4`，`i=0,…,2`。

| 根目录 | 文件模式 | 数量 | 用途 |
|---|---|---:|---|
| `project/MathorCup_A/frontier_revision/experiments/modern_prediction/` | `oof_arrays.npz` | 1 | 复算与独立审计的原完整OOF |
| 同上 | `checkpoints/raw231_y16_fold{f}_complete.npz` | 5 | 原外层Y16预测、索引及目标变换 |
| 同上 | `checkpoints/raw231_y50_fold{f}_complete.npz` | 5 | 原外层Native50预测及索引 |
| `checkpoint/unpacked/nested_selection/` | `nested_oof.npz` | 1 | 原硬选择外层预测 |
| 同上 | `outer_{f}/inner_oof.npz` | 5 | 每外折200行完整内层预测 |
| 同上 | `outer_{f}/selection.json` | 5 | 原硬选择候选及选中结果 |
| 同上 | `outer_{f}/inner_{i}/raw231_y16_complete.npz` | 15 | 独立重建Y16内层预测、PCA与分区 |
| 同上 | `outer_{f}/inner_{i}/raw231_y50_complete.npz` | 15 | 独立重建Native50内层预测与分区 |
| 合计 |  | **52** | **5.673 MiB** |

`run_prespecified_fusion.py`实际读取其中22个；独立审计读取其中47个；并集是52个。请保持原NPZ文件完整，不要删除归档内部的候选数组或改写压缩文件；OOF摘要和跨文件身份检查依赖原文件。

推荐额外保留四份很小的来源说明，不是本次代码执行依赖：

- `checkpoint/unpacked/nested_selection/experiment_spec.json`
- `checkpoint/unpacked/nested_selection/results.json`
- `checkpoint/unpacked/nested_selection/inner_candidate_scores.csv`
- `project/MathorCup_A/frontier_revision/experiments/modern_prediction/experiment_spec.json`

若要重新导出部署上下文，还需原`modern_prediction/final_artifact/manifest.json`和`model_context.npz`；已打包的两个`prediction/deployment_*/`上下文无需重新导出。若需风格诊断，再加原`data/A/1.csv`。

这些是**预测子任务**的精确依赖；其他生成、迁移或原完整研究审计所需文件由相应流程另行决定。

## 包内的部署验证输入

部署缓存验证只读取`prediction/`内的以下文件，不另读外部项目模型：

- `fusion_replay_v1/outer_predictions.npz`
- `deployment_fixed_half/`、`deployment_groupwise/`各自的`manifest.json`、`model_context.npz`、`outer_cv_validation_cache.npz`
- `validation_inputs/chemistry_250.csv`
- `predict_fusion_csv.py`、`fusion_model.py`、`single_branch_interface.py`

`validation_inputs/chemistry_250.csv`是原始化学CSV的副本。它保留原行顺序；预测输出按输入标识与顺序核对，不假定原附件按beer排序。

## 可选的新神经推理

可把唯一公开TabICL权重放在`prediction/models/tabicl-regressor-v2-20260212.ckpt`，大小为114,324,594字节（约114.3 MB）；无需在包内附带PyTorch或其他环境库。也可在运行时通过`--checkpoint`指定权重位置。该权重既供Y16也供Native50两支使用，不需要TabDPT权重或其他大模型。

安装环境和核验或下载该固定权重：

```bash
python -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r prediction/requirements_deployment.txt
python prediction/fetch_public_weights.py --model tabicl --verify-only
```

若包内未带checkpoint，最后一个命令去掉`--verify-only`后下载免费公开权重。新输入的实际神经路径示例：

```bash
python prediction/predict_fusion_csv.py new_chemistry.csv prediction/new_predictions.csv --artifact prediction/deployment_fixed_half --checkpoint prediction/models/tabicl-regressor-v2-20260212.ckpt --threads 2
```

此路径每批查询执行16个潜在目标和50个原始目标，共66个标量fit/predict。**本轮尚未执行这66个标量的新250行上下文神经forward；已执行的是历史真实外层预测的缓存重放与接口核验。** 单一checkpoint也不等于官方接口只加载一次网络；当前不跨目标复用KV或表征缓存。

本次缓存复算不需要打包：`A_work/q1/*.joblib`、TabDPT safetensors、秩输入分支的外层或内层checkpoint、任何partial checkpoint、训练日志、PyTorch安装目录。保留它们若服务于其他研究任务，应独立列明理由。
