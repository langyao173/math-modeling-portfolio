# 审计与复现报告

## 科学结论

本项目严格分开三类估计量：1986--2010 七届扩展窗验证、一次性 2014 外部审计，以及
使用历史球队画像占位的 2026 官方赛制情景。

开发期高级模型相对 legacy Dixon--Coles 的等届平均 log loss/Brier 改善为
0.020423/0.013022，并在 5/7 届获胜。2014 外部审计没有复现该优势：高级校准模型为
1.022728/0.608988，Dixon--Coles 为 0.985056/0.587594。因此结论是“开发期有界改善，
但没有外部优越性”，不能写成普遍胜出。

## 一次性 2014 审计

- 2014 共 64 场；所有预测先持久化，随后只读取一次最小真值表。
- `target_reads = 1`、`selection_reopened = false`、`cached_repeats_only = true`。
- 审计后不得重新选择模型；任何后续访问只能校验缓存证书。
- 审计是软件隔离，不是研究者完全盲法；旧 2014 结果曾经可见。
- 候选 ID：`f12f0fc20700df70743ef3b4f27089c45f45989a93fa2bd19c1886c98d609a81`。
- 最终拟合 SHA-256：`10238a4c6d368e50acb9ac28e4523dbd1b4f1b64710667ba0e8eb511b8774f53`。

证据位于 `temp/advanced_evidence/audit/`，由 state、result certificate、artifact manifest
和盲预测 manifest 交叉绑定。

## 2026 情景模拟

错误的 5,000 次邻接淘汰图产物永久作废。替代引擎实现 FIFA Annex C 的 495 种第三名
组合映射和 M73--M104 淘汰图，并通过 14 项赛制门槛。

正式运行完成 220,000 届、22,880,000 场模拟；最大晋级/冠军概率 MCSE 为
0.001066/0.000999，冠军概率和为 1，平均赛事总进球为 181.1653（MCSE 0.02875）。
这些是官方赛制下的历史画像情景，不是真实 2026 参赛队预测。

## 数据、时间和概率契约

- 原始数据 3/3 SHA-256 与冻结值一致。
- 比赛表 852 行去除 16 条完全重复后为 836 场；球员表 37,784 行去重为 37,048 行。
- 高级选择与校准最大年份为 2010；2014 不参与选择或 promotion gate。
- 36 个候选统一输出 WDL、联合比分、总进球和 metadata；概率非负且归一。
- 24 个压力测试行全部通过；最大交换/simplex 误差为 1.67e-16/2.22e-16。
- 超出有限比分网格可靠支撑时显式拒绝。

## 清单与发布门槛

发布构建器在写 ZIP 前强制验证 final manifest、research manifest、论文 33 个生成产物、
全部论文源证据 hash 和一次性审计缓存证书。ZIP 自带 `MANIFEST.json` 与
`checksums.sha256`；测试还必须解压 ZIP，再验证两套内嵌 manifest、全部 checksum、
CRC 和两次独立构建的字节确定性。

## 最终实测验证

- 三个独立 pytest 进程：`342 passed, 1 skipped`，分别用时 210.11、202.45、204.00 秒。
- 唯一跳过项是未随包保留的旧可选 MATLAB parity 证据；独立 MATLAB prototype 测试已通过。
- Ruff 对 `src tests scripts` 全部通过。
- final manifest：67 个产物、43 个必需项、6 个证书、710,243 bytes，独立验证通过。
- research manifest：206 个产物、63 个必需项、6 个证书、85,058,116 bytes，独立验证通过。
- 论文 manifest：33 个生成产物、45 个源证据 hash，全部匹配。
- 引用审计：58/58 条记录解析并核验，0 个 metadata mismatch。
- XeLaTeX 论文：19 页；0 个 warning、undefined citation/reference、overfull/underfull box。
- MATLAB R2026a 独立原型：七折平均 log loss 1.034731，全部结构不变量通过；不进入生产依赖。
- 最终代码只读审查结论：APPROVE，无 BLOCK、CRITICAL 或 HIGH 问题。

最终 ZIP 的 CRC、字节确定性、解压后双 manifest、全部 checksum 和两次清洁室运行仍由
发布命令及包外 `release_report.json`/`.sha256` 给出；不能把 ZIP 自身 hash 循环写入包内。

## 复核命令

```powershell
uv sync --frozen --extra dev
uv run python main.py
uv run python verify.py
uv run python main.py --config research --output research_outputs
uv run python verify.py research_outputs/manifest.json
uv run pytest -q
uv run ruff check src tests scripts
uv run python scripts/build_release.py
```

`python main.py` 是评审默认入口，只运行冻结的必要模型。advanced evidence 在审计完成后
不得重算；研究链路只执行 checkpoint 与缓存证书验证。

## 尚存限制

- 只有七个外层届次和一个 2014 外部届次，统计功效有限。
- 2014 高级概率 argmax 的平局召回为 0；成本敏感决策提高平局召回但牺牲准确率。
- 2026 情景没有真实参赛队、未来状态、阵容或伤停信息。
- 方差分解是模型条件下的敏感性分解，不是频率学覆盖保证。
- 不支持博彩、实时决策、因果效应或普遍模型优越性声明。
