"""重绘当前六图并编译当前论文；不重新执行长期仿真。

在本文件所在目录运行：python -X utf8 复现图表与论文.py
"""
from pathlib import Path
import shutil
import subprocess
from src.v18_main_visuals import render_all as main_figures
from src.v18_evidence_visuals import render_all as evidence_figures

root = Path(__file__).resolve().parent
paper = root / 'paper/V18成果稿'
compiler = shutil.which('xelatex')
if compiler is None:
    raise FileNotFoundError('论文编译需要已安装的 XeLaTeX。')
records = {**main_figures(), **evidence_figures()}
assert len(records) == 6
for stem, item in records.items():
    shutil.copyfile(root / item['pdf'], paper / 'figures' / (stem + '.pdf'))
for _ in range(3):
    subprocess.run([compiler, '-interaction=nonstopmode', '-halt-on-error', 'main.tex'], cwd=paper, check=True)
shutil.copyfile(paper / 'main.pdf', root.parent / 'ACSF2605371.pdf')
print('已重绘六张图并生成当前论文。')
