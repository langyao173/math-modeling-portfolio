# A题：程序运行说明

## 环境依赖

- Python 3.8（判题环境）
- NumPy、pandas、SciPy、scikit-learn、Matplotlib
- 安装：`python -m pip install -r requirements.txt`
- 说明文档 PDF 重建另需 XeLaTeX（`ctex` / `xelatex`）

程序检测 Matplotlib 中文字体；没有中文字体时，图表自动改用英文标题。
运行不需要网络。

## 模块功能

- `main.py`：正式一键入口（默认：六问分析 + 写 `cla_pre.csv`）
- `analyze.py`：仅生成六问分析结果（开发用）
- `problem1_analysis.py`：四项描述统计、两张趋势图、2018 线性回归
- `problem2_analysis.py`：阶段统计、半场—全场相关、单场总进球回归
- `problem3_analysis.py`：四项球队历史特征、2014 逻辑回归与混淆矩阵
- `prediction_api.py`：胜负和单场进球复用接口
- `release_check.py`：测试、六问输出和端到端协议核对

问题三唯一正式模型是题目规定的多分类逻辑回归。只使用球队历史参赛次数、
最佳成绩、场均进球、场均失球四项特征的主客队差值；1930–2010 用于训练，
2014 只用于最终测试。

## 一键运行（判题）

```text
python3 main.py
```

读取 `/testdata`（或 `--testdata` / 环境变量 `TESTDATA_DIR`），在源码目录写出：

```text
./cla_pre.csv
```

`prediction` 编码：`0=主胜`，`1=平局`，`2=客胜`；输出行序与查询输入一致。

演示（无外部 query 时）：

```text
python3 main.py --demo --details
```

仅预测、不重跑分析（开发/检查用）：

```text
python3 main.py --prediction-only --testdata <目录>
```

## 六问对应输出

1. `1_1_四项描述统计.csv`、`图1_总进球趋势.png`、`图2_场均进球趋势.png`
2. `1_2_模型口径与2018预测.csv`
3. `2_1_比赛阶段进球统计.csv`、半场—全场相关与 `图4`
4. `2_2_2014历史开发基准指标.csv` 及可复用单场预测接口
5. `3_1_球队历史四特征.csv` 与特征诊断图 `图5`
6. `3_2_2014逻辑回归测试指标.csv`、混淆矩阵 `图6` 及 2026 接口

表格：`outputs/tables/`；图：`outputs/figures/`；日志：`outputs/logs/`。
`outputs/report_metrics.csv` 是 PDF 唯一事实源。

## 重建说明文档（LaTeX）

```text
cd doc
python build_latex.py
```

产物：`doc/说明文档.pdf`。
