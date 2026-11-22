# 噪声自适应量子编译挑战赛

这是一个面向参赛者的 NISQ 编译映射挑战赛仓库。参赛程序需要把输入量子线路编译到给定的噪声后端上，在保持逻辑语义等价的前提下，提高带噪采样下的正确输出概率。

比赛分两条公开赛道：

- Track A：中小规模通用 NISQ 线路，使用 Cirq 带噪采样评分。
- Track B：较大的 Clifford-only 压力测试线路，使用 Stim 带噪采样评分。
- 总分：`0.75 * TrackA + 0.25 * TrackB`。

## 参赛者需要提交什么

你的编译器必须能作为黑盒命令运行。评测器只会传入三个参数：

```bash
your-compiler \
  --circuit path/to/circuit.json \
  --backend path/to/backend.json \
  --out path/to/compiled_circuit.json
```

你的程序必须写出两个文件：

1. `compiled_circuit.json`：编译后的 Cirq JSON 线路，量子比特必须是物理 `LineQubit`。
2. `compiled_circuit.layout.json`：和输出文件同目录的 layout sidecar，说明输入程序 qubit 一开始放在哪些物理 qubit 上。

sidecar 示例：

```json
{
  "schema_version": "1.0",
  "initial_layout": {
    "0": 5,
    "1": 8
  }
}
```

这里的 `"0"` 和 `"1"` 是输入线路里的程序 qubit 编号，不是纠错意义下的 logical qubit。评测器会用这个 layout 做隐藏的无噪等价性检查，再进入带噪评分。只准备公开样本的答案 bitstring，而不真正编译输入线路，会被判为无效。

## 安装

建议使用 Python 3.11 或更新版本：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

如果你的系统有多个 Python 版本，请确认创建虚拟环境的解释器满足 `pyproject.toml` 里的 `>=3.11` 要求。

## 快速开始

生成公开数据集和后端快照：

```bash
nah-gen-dataset --out .
```

运行一个内置 Cirq RouteCQC baseline，编译单个样本：

```bash
nah-run-baseline \
  --compiler cirq_cqc \
  --circuit benchmarks/track_a/public/track_a_001/circuit.json \
  --backend backends/medium/medium_snapshot_a.json \
  --out results/cirq_cqc_track_a_001.json
```

这个命令会同时写出：

- `results/cirq_cqc_track_a_001.json`
- `results/cirq_cqc_track_a_001.layout.json`

运行仓库里的参赛者示例编译器：

```bash
python examples/cirq_route_cqc_baseline/compiler.py \
  --circuit benchmarks/track_a/public/track_a_001/circuit.json \
  --backend backends/medium/medium_snapshot_a.json \
  --out results/example_cirq_track_a_001.json
```

对一个提交命令做完整公开集评分：

```bash
EXAMPLE_COMPILER="$(pwd)/examples/cirq_route_cqc_baseline/compiler.py"
nah-evaluate-submission \
  --root . \
  --shots-a 256 \
  --shots-b 1024 \
  --out results/public_summary.json \
  --command python "$EXAMPLE_COMPILER"
```

完整评测会在临时目录中运行提交命令，所以脚本路径建议写成绝对路径，或使用已经安装好的 `python -m your_package.module`。

也可以评测内置 baseline：

```bash
nah-evaluate-submission \
  --root . \
  --shots-a 256 \
  --shots-b 1024 \
  --out results/cirq_cqc_summary.json \
  --command python -m noise_adaptive_hackathon.cli.run_baseline --compiler cirq_cqc
```

## 输出线路要求

- 只能使用 `backend.json["basis_gates"]` 中列出的门。
- 所有 qubit 必须是合法物理 `LineQubit`，编号范围是 `[0, backend["num_qubits"])`。
- 两比特门必须落在 `backend.json["coupling_map"]` 允许的物理边上。
- 最终测量 key 必须是 `z`。
- 不能使用中途测量、reset、feedforward 或 classically controlled operation。
- v1 后端 basis 不包含 `swap`，如果需要移动 qubit，请分解成合法的 `cx`。
- Track B 输出必须保持在 Stim 支持的 Clifford 子集：`x`, `y`, `z`, `h`, `s`, `sdg`, `cx`, `measure`。
- 必须通过隐藏 compiler-equivalence 检查和无噪语义检查。

## 评分

单个样本的主指标是：

```text
success_rate = correct_shots / total_shots
```

Track 分数是该 Track 公开样本的平均 `success_rate`。总分是：

```text
Final Score = 0.75 * TrackA + 0.25 * TrackB
```

命令失败、超时、输出文件缺失、layout sidecar 缺失或非法、非法门、非法 coupling、Track B 非 Clifford、等价性检查失败、无噪语义检查失败，都会让对应样本得分为 `0.0`。

评测报告还会给出 depth、两比特门数量、估计 log reliability、shot 数、正确 shot 数和编译时间。这些是调试指标，不直接改变分数。

## 仓库结构

```text
backends/                         公开后端快照
benchmarks/                       公开 benchmark 输入
docs/                             中文参赛说明
examples/cirq_route_cqc_baseline/ Cirq baseline 参赛示例
formats/                          输入输出 JSON 格式说明
src/noise_adaptive_hackathon/     评测器、模拟器、baseline 和 CLI
tests/                            发布前回归测试
LICENSE                           发布包代码许可
THIRD_PARTY_NOTICES.md            第三方依赖说明
```

`results/`、虚拟环境、缓存和临时研究材料不应进入发布包。

## 更多说明

- [参赛指南](docs/participant_guide.md)
- [评分示例](docs/scoring_examples.md)
- [分数占比说明 / Score Weighting](docs/scoring_weights_bilingual.md)
- [噪声模型](docs/noise_model.md)
- [比赛规则与评测环境](docs/rules_and_environment.md)
- [作品提交说明 / Submission Instructions](docs/submission_instructions_bilingual.md)
- [后端格式](formats/backend_schema.md)
- [提交格式](formats/submission_schema.md)
- [Cirq baseline 示例](examples/cirq_route_cqc_baseline/README.md)
