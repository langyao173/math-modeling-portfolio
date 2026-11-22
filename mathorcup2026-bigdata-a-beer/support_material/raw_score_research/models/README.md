# 原始品评补充数据的预测与评分

本模块使用经身份审计的公开逐次品评记录补充竞赛附件。官方4.csv的标准化连续目标、原始0–6分面板均值与七档分布分别定义和评分；训练分位等级不充当真实原分。原分和标准化目标对应同250款啤酒，不称独立外部验证。

## 正式运行与结果

`results_v2/`是按冻结协议执行的正式运行。外层为既有五折，每折200款训练、50款留出；内层复用各外折原200款训练集的三折。所有同beer品评记录进入同一侧。原分主目标先在beer内对观察记录等权，所有评价再对beer及目标等权。`taster_equal_sensitivity`只改变目标聚合，不重新调参或训练。

- 原分点预测：训练均值、直接Ridge、直接Extra Trees、固定等权标准化预测的训练内OOF仿射及单调校准。
- 分布预测：训练频率、六阈值Ridge、六阈值Extra Trees、固定等权标准化预测的六阈值单调校准。CDF经同权PAVA与[0,1]限制后差分得到合法七档概率。
- 内选参目标固定为核心Aroma+Flavour30维，Ridge网格为1、10、100、1000，ET为240棵树、max_features=0.7、叶大小2或5。点模型按均值MSE、分布模型按CDF-MSE选择。后者与观察级RPS仅差不依赖模型的面板离散项。
- 官方目标的次级受控对照仅比较相同Ridge有无记录数权重，权重为n/(n+训练中位n)，并归一到训练均值1。它未达到既有强模型表现，完整保留在补充结果中。
- 固定神经校准只使用完整输入缓存。直接模型另按旧规则测试派生量隐藏及10%、30%、50%随机缺测；不以完整输入神经缓存冒充缺测预测。

校准基底固定为Y16和Native50预测的等权混合。外折训练校准只用该外折200款训练beer内部的OOF预测；不能取全250外层OOF的200行，因为那些模型可能训练过当前外层测试beer。校准方法固定分别报告，不按其拟合残差互相选择。

## 指标

原分均值是0–6范围的连续期望，允许小数。其RMSE、MAE和宏R²与官方标准化目标分别报告。

分布主指标`rps`是对真实逐次评级的期望排名概率评分：

\[
\operatorname{RPS}=\frac16\sum_{h=0}^5 (Q_h-P_h)^2+
\frac16\sum_{h=0}^5P_h(1-P_h).
\]

`cdf_mse`只保存第一项，`irreducible_rps`保存第二项，二者不能混称同一个绝对分数。NLL在独立评分副本上先将概率下限设为1e−6再归一；主概率和RPS不作此平滑。离散等级固定为模型后验中位数，即累计概率首次不小于1/2的最小等级，其逐品评MAE、同档率和±1档率与均值误差分别报告。

## 文件

| 路径 | 内容 |
|---|---|
| `inputs/inputs.npz` | 官方X/Yz、原分均值/分布、记录和品评员等权目标、固定内外折、完整内外OOF缓存 |
| `inputs/target_provenance.json` | 官方/raw/映射/身份审计及旧缓存来源哈希 |
| `results_v2/predictions.npz` | 全部预定点预测、概率预测、官方目标对照与缺测场景 |
| `results_v2/fold_record_masks.json` | 每外折训练/测试beer及原始记录索引 |
| `results_v2/fit_log.json` | 每次拟合的训练beer与记录索引 |
| `results_v2/inner_scores.csv` | 逐候选、逐内折核心30维选择损失 |
| `results_v2/selected_parameters.json` | 每外折全部参数分数和确定的选择 |
| `exports_v2/` | 所有完整输入候选的250×50原分期望、概率方法中位等级及长表七档分布 |
| `deployment/` | 全250训练数据重新内选参并拟合的独立CPU CDF-ET及schema |
| `deployment_validation/VALIDATION_COMPLETE.json` | 真实CSV接口及一外折OOF重放检查 |

`results_v1/`和`execution_snapshots/run_raw_models_v1.py.txt`保留一次已被独立审查发现的实现偏差：内选参误用50维而非冻结核心30维。修复后完整重跑v2，全部最终参数恰好相同，预测数组逐字节相同；正式结果仍以v2及其正确内选参日志为准，不把v1当作另一个预定模型。详见`IMPLEMENTATION_CORRECTION.json`。

正式运行330个基模型拟合，修正前运行另330个，部署内选参加最终拟合7个，CSV外折重放1个，实际合计668个；仿射和单调校准的目标级拟合另列，未新增神经推理或付费调用。完整计数见`ACTUAL_FIT_ACCOUNTING.json`。

`deployment/affine_context/`另保存全250训练beer的50个非负仿射校准参数，供既有真实化学输入→固定等权标准化预测接口后接。该部署上下文不产生新的神经推理，不将全250校准训练残差用作留出成绩；正式原分评价使用各外折200款的独立校准。它新增50个解析目标拟合，累计仿射目标拟合550个、单调目标拟合3500个，与基模型668个分别计数。

## 复现

使用Python 3.12、numpy 2.3.5、pandas 2.2.3、scikit-learn 1.8.0、scipy 1.17.0和joblib。代码均使用显式输入路径，工作目录可更换。先将身份审计后的映射、原始公开CSV及竞赛官方1/4.csv放到自选位置，再执行：

```bash
python prepare_inputs.py --chemistry /path/to/1.csv --official-targets /path/to/4.csv \
  --raw-scores '/path/to/Raw sensory data anonymized.csv' \
  --mapping /path/to/raw_to_competition_mapping.csv \
  --mapping-record /path/to/identity_result.json \
  --research-cache /path/to/previous_package/research --output inputs_reproduced
python run_raw_models.py --inputs inputs_reproduced/inputs.npz \
  --protocol-lock /path/to/PROTOCOL_LOCK.json --output results_reproduced
python export_predictions.py --inputs inputs_reproduced/inputs.npz \
  --predictions results_reproduced/predictions.npz --output exports_reproduced
```

原始CSV使用cp1252编码；两处字段单复数别名在`prepare_inputs.py`中明确列出。该脚本消费已冻结映射，不能自行搜索最大相关匹配身份。缓存只用于其已核对的原样本、折和表示，不作新化学输入的预测。

## 预测新化学输入

部署模型无需神经权重。CSV可重排列，缺少的化学列填NaN后由已存训练中位数填补；至少需要一个已识别的化学字段。化学值须沿用附件1给定尺度，不进行隐含单位替换。`beer`列可选，仅作为输出标识，不作预测特征。

```bash
python predict_raw_csv.py --artifact deployment --input new_chemistry.csv --output predictions
```

得到期望分`expected_scores.csv`、中位等级`median_grades.csv`及全部七档概率`score_probabilities.csv`。CSV中的0–6期望分允许小数，中位等级为整数。部署训练是全训练数据拟合，不能作为额外留出成绩。重新构建部署与检查实际接口：

```bash
python fit_raw_deployment.py --inputs inputs_reproduced/inputs.npz \
  --protocol-lock /path/to/PROTOCOL_LOCK.json --output deployment_reproduced
python validate_raw_deployment.py --inputs inputs_reproduced/inputs.npz \
  --results results_reproduced --artifact deployment_reproduced --output validation_reproduced
python fit_affine_context.py --inputs inputs_reproduced/inputs.npz \
  --output deployment_reproduced/affine_context
```

实际检查包含未带标签的新构造化学查询、列重排、缺列与显式NaN一致、概率和/范围、异常输入拒绝，以及第0外折按已冻结参数重拟合后通过CSV接口重现其OOF结果。构造查询仅用于软件接口检查，不冒充新物理啤酒实验。
