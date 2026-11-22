# PDF 验收清单（不超过一页）

| 项 | 结果 |
|---|---|
| 文档页数 | 5 页 A4（≤8） |
| 编译方式 | XeLaTeX（`doc/build_latex.py` → `xelatex` ×2） |
| 事实源 | `outputs/report_metrics.csv` → 临时 `report_metrics.tex` 宏注入 |
| 中文字体 | ctex Windows 字体集（宋体/微软雅黑链路），已嵌入 |
| 主图数量 | 4（图1、图4、图5、图6） |
| 逐页渲染 | `doc/pdf_preview/page_01.png` … `page_05.png`（220 dpi） |
| 裁切/重叠 | 映射表重叠已修复；协议框与图题可读 |
| 文风 | 已按 writing-anti-ai / 技术说明体改写：去口语化与空泛评价，保留可核对事实 |
| 数字抽查 | 2018 预测 160.10、2014 Acc 0.578125、37/64、r=0.696、836 场 — 与 CSV 一致 |
| `python3 main.py --demo` | 生成 `cla_pre.csv` + 六问 `outputs/` |
| `release_check.py` | 全部 PASS |
| P0 已修复 | `main.py` 默认分析+预测；Python 宣称 3.8；四项统计含场均进球 |

**最终文件：** `doc/说明文档.pdf`，源文件 `doc/report.tex`，构建脚本 `doc/build_latex.py`。
