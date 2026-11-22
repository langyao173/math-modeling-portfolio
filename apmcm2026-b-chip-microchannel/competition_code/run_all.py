# -*- coding: utf-8 -*-
"""一键复现全部计算与图表：python run_all.py
按 问题1→5 的顺序依次执行求解脚本与绘图脚本（约5~10分钟）。"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, '图表'), exist_ok=True)

SCRIPTS = [
    ('p1_mechanism_model.py', '问题1：机理模型标定'),
    ('p1_ablation_identifiability.py', '问题1：消融与可辨识性审计'),
    ('p1_make_figures.py',    '问题1：绘图'),
    ('p2_surrogate.py',       '问题2：严格nested LOOCV与双代理模型（最耗时）'),
    ('p2_make_figures.py',    '问题2：绘图'),
    ('p3_optimize.py',        '问题3：Pareto优化与三准则决策'),
    ('p3_make_figures.py',    '问题3：绘图'),
    ('p4_weight_robust.py',   '问题4：权重扫描与鲁棒判定'),
    ('p4_make_figures.py',    '问题4：绘图'),
    ('p5_sensitivity.py',     '问题5：敏感性与蒙特卡洛'),
    ('p5_make_figures.py',    '问题5：绘图'),
    ('audit_tests.py',        '审计测试：CV/边界/Sobol/正文禁用词'),
]

env = dict(os.environ, PYTHONIOENCODING='utf-8')
t_all = time.time()
for name, desc in SCRIPTS:
    print('=' * 64, flush=True)
    print('>>', desc, '(%s)' % name, flush=True)
    t0 = time.time()
    ret = subprocess.run([sys.executable, os.path.join(HERE, '代码', name)], env=env)
    if ret.returncode != 0:
        print('!! 脚本失败：', name, flush=True)
        sys.exit(ret.returncode)
    print('<< 完成，用时 %.1f 秒' % (time.time() - t0), flush=True)
print('=' * 64, flush=True)
print('全部完成，总用时 %.1f 分钟。结果在 代码/ 目录（json/csv/pkl），图在 图表/ 目录。'
      % ((time.time() - t_all) / 60), flush=True)
