# 说明文档构建

正式交付 PDF 使用 **XeLaTeX**：

```text
cd doc
python build_latex.py
```

产物：`说明文档.pdf`。数字来自 `../outputs/report_metrics.csv`（构建时临时生成 `report_metrics.tex` 宏文件）。

可选备用：Typst 流程 `python build_report.py`（非正式主路径）。
