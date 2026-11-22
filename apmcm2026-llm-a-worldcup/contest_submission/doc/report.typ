#import "report_metrics.typ": metrics
#let asset-root = sys.inputs.at("asset-root", default: "../outputs/figures")
#let doc-lang = sys.inputs.at("doc-lang", default: "zh")
#let doc-font = sys.inputs.at("doc-font", default: "Microsoft YaHei")

#let navy = rgb("#17324d")
#let blue = rgb("#2f6f9f")
#let pale = rgb("#edf4f8")
#let light = rgb("#f7f9fb")
#let ink = rgb("#18222c")
#let muted = rgb("#5d6975")

#let t(zh, en) = if doc-lang == "zh" { zh } else { en }
#let metric(key) = metrics.at(key)
#let f3(x) = str(calc.round(x, digits: 3))
#let pct1(x) = str(calc.round(x * 100, digits: 1)) + "%"
#let status(key) = if metric(key) {
  t([通过], [PASS])
} else {
  t([未通过], [NOT PASSED])
}

#set page(
  paper: "a4",
  margin: (x: 18mm, y: 16mm),
  header: context [
    #set text(size: 8pt, fill: muted)
    #grid(
      columns: (1fr, auto),
      t([APMCM 2026 · A题], [APMCM 2026 · Problem A]),
      t([8.0 程序说明], [8.0 Program Guide]),
    )
    #line(length: 100%, stroke: 0.4pt + rgb("#cad4dd"))
  ],
  footer: context [
    #line(length: 100%, stroke: 0.4pt + rgb("#d8e0e7"))
    #align(center)[
      #set text(size: 8pt, fill: muted)
      #t([第], [Page ]) #counter(page).display()
    ]
  ],
)
#set text(font: doc-font, size: 9.2pt, fill: ink)
#set par(justify: true, leading: 0.72em)
#set heading(numbering: none)
#show heading.where(level: 1): it => [
  #text(size: 17pt, weight: 700, fill: navy)[#it.body]
  #v(2pt)
  #line(length: 100%, stroke: 1pt + blue)
  #v(5pt)
]
#show heading.where(level: 2): it => [
  #v(4pt)
  #text(size: 11pt, weight: 700, fill: blue)[#it.body]
  #v(2pt)
]
#show raw: set text(font: ("Cascadia Mono", "Consolas", "DejaVu Sans Mono"), size: 8pt)

#let stat(label, value, note) = block(
  inset: 8pt,
  radius: 4pt,
  fill: pale,
  stroke: 0.5pt + rgb("#c8d8e4"),
)[
  #text(size: 8pt, fill: muted)[#label]
  #v(2pt)
  #text(size: 17pt, weight: 700, fill: navy)[#value]
  #v(1pt)
  #text(size: 7.5pt, fill: muted)[#note]
]

#align(center)[
  #v(8mm)
  #text(size: 24pt, weight: 700, fill: navy)[
    #t([世界杯比赛数据挖掘与预测], [World Cup Data Mining and Prediction])
  ]
  #v(4pt)
  #text(size: 13pt, fill: blue)[
    #t([程序运行说明 · 版本 8.0], [Program Guide · Version 8.0])
  ]
  #v(4pt)
  #text(size: 8.5pt, fill: muted)[
    #t(
      [题面逻辑回归 · 六问自动输出 · CSV单一事实源],
      [Official logistic regression · Six automated outputs · One CSV fact source],
    )
  ]
]
#v(8mm)
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 8pt,
  stat(t([问题一 R²], [Q1 R²]), f3(metric("p1_r2")), t([三特征线性回归], [Linear regression])),
  stat(t([问题二 MAE], [Q2 MAE]), f3(metric("p2_official_2014_mae")), t([2014验证], [2014 validation])),
  stat(t([问题三准确率], [Q3 accuracy]), pct1(metric("p3_2014_accuracy")), t([2014测试], [2014 test])),
)
#v(5mm)
#table(
  columns: (1.2fr, 2.6fr),
  inset: 6pt,
  stroke: 0.4pt + rgb("#d5dde4"),
  fill: (x, y) => if y == 0 { navy } else if calc.even(y) { light },
  table.header(
    [#text(fill: white, weight: 700)[#t([项目], [Item])]],
    [#text(fill: white, weight: 700)[#t([说明], [Description])]],
  ),
  [#t([数据], [Data])],
  [#str(int(metric("data_min_year")))–#str(int(metric("data_max_year")))
   · #str(int(metric("tournament_count"))) #t([届], [tournaments])
   · #str(int(metric("match_count"))) #t([场], [matches])],
  [#t([类别], [Classes])],
  [prediction: 0=#t([主胜], [home win]), 1=#t([平局], [draw]), 2=#t([客胜], [away win])],
  [#t([六问输出], [Six-question outputs])], [#status("six_question_outputs_complete")],
  [#t([问题三协议], [Q3 protocol])], [#status("p3_protocol_checks_met")],
)

== #t([一键运行], [One-command workflow])
```text
python -m pip install -r requirements.txt
python analyze.py
python main.py --demo --details
python release_check.py
```
#t(
  [正式预测使用 `python main.py --testdata <目录>`；无查询文件时程序直接报错。],
  [For formal prediction use `python main.py --testdata <directory>`; missing queries fail explicitly.],
)

#pagebreak()
= #t([环境、模块与六问对应], [Environment, modules, and six tasks])

== #t([环境依赖], [Environment])
#t(
  [支持 Python 3.8；核心依赖为 NumPy、pandas、SciPy、scikit-learn、Matplotlib。正式说明文档以 XeLaTeX 构建；Typst 仅作可选备用。程序先检测中文字体；无中文字体时图表自动英文回退。],
  [Supports Python 3.8. Core dependencies: NumPy, pandas, SciPy, scikit-learn, and Matplotlib. The formal brief is built with XeLaTeX; Typst is optional. Chinese fonts are detected first; English labels are used when none are available.],
)

== #t([模块功能], [Module map])
#table(
  columns: (1.35fr, 2.65fr),
  inset: 5pt,
  stroke: 0.4pt + rgb("#d5dde4"),
  fill: (x, y) => if y == 0 { navy } else if calc.even(y) { light },
  table.header(
    [#text(fill: white, weight: 700)[#t([文件], [File])]],
    [#text(fill: white, weight: 700)[#t([功能], [Purpose])]],
  ),
  [analysis_pipeline.py], [#t([六问编排与统一摘要], [Six-task orchestration and summary])],
  [problem1_analysis.py], [#t([描述统计、趋势图、2018线性回归], [Statistics, trends, 2018 regression])],
  [problem2_analysis.py], [#t([阶段统计、相关分析、单场进球回归], [Stage statistics, correlation, match-goal regression])],
  [problem3_analysis.py], [#t([四特征诊断、2014逻辑回归、混淆矩阵], [Feature diagnostics, 2014 logistic test, confusion matrix])],
  [prediction_api.py], [#t([2026胜负与单场进球接口], [2026 outcome and match-goal APIs])],
  [report_data.py], [#t([生成 report_metrics.csv 与交付说明], [Generates report_metrics.csv and delivery note])],
)

== #t([六问对应脚本与输出], [Task-to-output map])
#table(
  columns: (0.45fr, 1.45fr, 2.5fr),
  inset: 4.5pt,
  stroke: 0.35pt + rgb("#d5dde4"),
  fill: (x, y) => if y == 0 { navy } else if calc.even(y) { light },
  table.header(
    [#text(fill: white, weight: 700)[#t([问], [No.])]],
    [#text(fill: white, weight: 700)[#t([脚本], [Script])]],
    [#text(fill: white, weight: 700)[#t([自动输出], [Automated output])]],
  ),
  [1], [problem1_analysis.py], [#t([四项统计、两张趋势图、文字结论], [Four statistics, two trends, generated conclusion])],
  [2], [problem1_analysis.py], [#t([2018线性回归预测与误差], [2018 regression prediction and error])],
  [3], [problem2_analysis.py], [#t([阶段进球表、半场—全场散点图与相关系数], [Stage table, scatter plot, correlation])],
  [4], [problem2_analysis.py], [#t([单场总进球模型、MAE/MSE/R²与复用接口], [Match-goal model, metrics, reusable API])],
  [5], [problem3_analysis.py], [#t([四特征 corrgram+VIF 与三类结局指示相关], [Feature corrgram+VIF and outcome-indicator correlations])],
  [6], [problem3_analysis.py], [#t([2014逻辑回归准确率、混淆矩阵、2026接口], [2014 accuracy, confusion matrix, 2026 API])],
)

#pagebreak()
= #t([简短结果展示：问题一与问题二], [Short results: Problems 1 and 2])

== #t([问题一], [Problem 1])
#grid(
  columns: (1fr, 1fr),
  gutter: 7pt,
  image(asset-root + "/图1_总进球趋势.png", width: 100%),
  image(asset-root + "/图2_场均进球趋势.png", width: 100%),
)
#v(3pt)
#t(
  [三特征模型 R² 为],
  [Three-feature R² is],
) #f3(metric("p1_r2"));
#t([2018预测为], [the 2018 prediction is])
#f3(metric("p1_prediction_2018")),
#t([真实值为], [versus actual])
#str(int(metric("p1_actual_2018")))。

== #t([问题二], [Problem 2])
#grid(
  columns: (1fr, 1fr),
  gutter: 7pt,
  image(asset-root + "/图3_比赛阶段进球统计.png", width: 100%),
  image(asset-root + "/图4_半场与全场进球相关.png", width: 100%),
)
#v(3pt)
#t([半场—全场 Pearson 相关系数为], [Half/full-time Pearson correlation is])
#f3(metric("p2_pearson_half_full"))；
#t([题面线性回归在2014的 MAE/MSE/R² 为], [official 2014 MAE/MSE/R² are])
#f3(metric("p2_official_2014_mae")) /
#f3(metric("p2_official_2014_mse")) /
#f3(metric("p2_official_2014_r2"))。

#pagebreak()
= #t([简短结果展示：问题三与输出], [Short results: Problem 3 and outputs])

== #t([唯一正式模型], [Only official model])
#t(
  [问题三使用题目规定的多分类逻辑回归。输入只含四项历史差值：参赛次数、最佳成绩、场均进球、场均失球。正则化只在2014之前的历史滚动折中选择。],
  [Problem 3 uses the required multiclass logistic regression. Inputs are only four historical differences: participation count, best result, goals for, and goals against. Regularization is selected only on rolling folds before 2014.],
)
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 7pt,
  stat(t([正则化 C], [Regularization C]), str(metric("p3_c")), t([训练期选择], [development selection])),
  stat(t([2014正确], [2014 correct]), str(int(metric("p3_2014_correct_matches"))) + "/" + str(int(metric("p3_2014_match_count"))), t([固定测试集], [fixed test])),
  stat(t([2014准确率], [2014 accuracy]), pct1(metric("p3_2014_accuracy")), t([自动计算], [computed automatically])),
)
#v(5pt)
#t(
  [图5给出四特征多重共线性诊断（下三角散点、对角密度、上三角 Spearman 相关与显著性、侧栏 VIF），并单独展示特征与主胜/平局/客胜指示变量的相关，避免将胜负编码为 0/1/2 后做序数相关。],
  [Figure 5 reports four-feature multicollinearity diagnostics (scatter, density, Spearman with significance, and VIF) and correlations against Home/Draw/Away indicators, avoiding ordinal correlation on a 0/1/2 outcome code.],
)
#v(4pt)
#figure(
  image(asset-root + "/图5_问题三四特征相关热力图.png", width: 100%),
)
#v(4pt)
#align(center)[
  #image(asset-root + "/图6_2014逻辑回归混淆矩阵.png", width: 58%)
]

== #t([自动输出文件], [Generated files])
- `outputs/tables/`: #t([统计表、模型指标、2014预测明细、混淆矩阵 CSV], [statistics, metrics, 2014 predictions, confusion-matrix CSV])
- `outputs/figures/`: #t([6张规定图], [six required figures])
- `outputs/logs/`: #t([三问文字结论与六问摘要], [generated conclusions and six-task summary])
- `outputs/report_metrics.csv`: #t([PDF和交付说明的唯一指标源], [single metric source for PDF and delivery note])
- `cla_pre.csv`: #t([正式最小提交文件], [minimal formal submission])

== #t([复现与检查], [Reproduction and checks])
```text
python analyze.py
python main.py --demo --details
python release_check.py
cd doc
python build_report.py
```
#t(
  [构建脚本每次先重新运行分析，再临时生成 `report_metrics.typ`；PDF完成后立即删除该临时模块，发布包不携带可能过期的预生成指标。],
  [The build reruns analysis before creating a temporary `report_metrics.typ`; the temporary module is removed immediately after PDF compilation, so no stale generated metrics ship in the package.],
)
