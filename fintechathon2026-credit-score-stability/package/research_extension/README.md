# 业务客群、更新方法与输入范围

此目录提供技术附录 A.9—A.11 的逐行计算依据。风险评分沿用冻结模型；新业务分组、更新挑战者和训练范围检查均在相同客户上对齐。

## 复算入口

运行环境见根目录 `requirements.txt`。随包的元数据、账务、结局与冻结预测足以复算更新、风险指标、尾部指标、固定容量及配对区间：

```bash
python research_extension/code/compare_updates.py development
python research_extension/code/compare_updates.py historical
python research_extension/code/deep_diagnostics.py --bootstrap 2000
```

风险配置首先使用开发客户的交叉拟合输出筛选；历史队列用于固定配置比较。输出写回 `results/`。重新生成论文图表执行根目录的 `code/revision_figures.py` 和 `code/revision_appendix.py`。

## 主要文件

| 路径 | 内容 |
| --- | --- |
| `results/historical_meta.parquet`、`historical_edges.parquet` | 行标识、账户月份、前后期连接及业务分组 |
| `results/historical_observations.parquet` | 分组所用的当期可见账务 |
| `results/historical_outcomes.parquet` | 按既定窗口观察的未来结果，未知保留为空 |
| `results/historical/frozen_inputs.parquet` | 已冻结底模及既有方法的逐行输出 |
| `results/historical/predictions.parquet` | 各更新方法的统一评分 |
| `results/historical/risk.csv`、`tail.csv`、`capacity.csv` | 完整风险、尾部和六档固定容量 |
| `results/historical/tail_bootstrap.csv` | 12项配对比较及同时区间 |
| `results/development/selection_table.csv` | 开发阶段各项筛选条件 |
| `results/support_reference_quantiles.csv` | 396个特征的训练参考区间 |
| `results/historical/support_per_row.parquet` | 逐行输入越界与余额/参考额度 |

## 原始构造与数据角色

原始账务为3,840,312行、103,558名客户。历史提额队列为2,907人，相邻队列为2,677人；结局为未来三个月已观察DPD>0。重复使用的历史队列按复测解释，分组定义只读取当时可见信息。

训练范围使用107,880个实际拟合行。若需重新构建原始分组或训练分位区间，设置 `FINTECH_SOURCE_ROOT` 指向原研究数据根目录，其下包含 `FinTechathon_Task1_V17`、`Capacity_Research/capacity_research` 及 `joint_advantage_research`。然后运行 `prepare_inputs.py`；完整支持范围重建使用 `deep_diagnostics.py --rebuild-support`。未设置外部目录时，常规复算直接读取随包固定输入与已保存区间。

模型重新训练仍使用原研究包的训练管线；本目录的更新计算不重新拟合风险模型。公开账务在产品中只提供观察时点可见信息，未来结果仅供研究评价。
