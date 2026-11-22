# -*- coding: utf-8 -*-
"""问题4配图：权重三角图分区、覆盖率/regret对比、场景分析"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from figure_style import DESIGN_COLORS, NEUTRAL, apply_pub_style, panel_label, save_pub

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
apply_pub_style()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 包根目录（相对路径，解压即可运行）
FIG = BASE + '/图表'
os.makedirs(FIG, exist_ok=True)

pf = pd.read_csv(BASE + '/代码/p3_pareto_front.csv', encoding='utf-8-sig')
X = pf[['beta', 'gamma', 'N']].values
W = np.load(BASE + '/代码/p4_W.npy')
best_idx = np.load(BASE + '/代码/p4_best_idx.npy')
with open(BASE + '/代码/p4_results.json', encoding='utf-8') as f:
    res = json.load(f)

# ---------- 图1：权重三角图（最优设计分区） ----------
# 重心坐标: w=(wR,wdP,wphi) → 2D
def tern_xy(W):
    x = W[:, 1] + 0.5 * W[:, 2]
    y = np.sqrt(3) / 2 * W[:, 2]
    return x, y

selN = X[best_idx, 2]
selG = X[best_idx, 1]
# 分区类别: N=4&γ=4.5 / N=4&γ<4 / N=2 / N=6 / N=10
cat = np.zeros(len(W), int)
cat[(selN == 4) & (selG > 4.2)] = 0
cat[(selN == 4) & (selG <= 4.2)] = 1
cat[selN == 2] = 2
cat[selN == 6] = 3
cat[(selN == 8) | (selN == 10)] = 4
cat_names = ['N=4, γ=4.5（鲁棒方案区）', 'N=4, γ≈3.3~3.6', 'N=2, γ=4.5',
             'N=6, γ=4.5', 'N=8~10, γ≈3~3.4（重热阻区）']
cat_colors = ['#C1121F', '#E76F51', '#6A4C93', '#00897B', '#D9A441']

fig, ax = plt.subplots(figsize=(8.6, 7.6))
tx, ty = tern_xy(W)
cmap = ListedColormap(cat_colors)
norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)
ax.tricontourf(tx, ty, cat, levels=np.arange(-0.5, 5.5, 1),
               cmap=cmap, norm=norm, alpha=0.88)
ax.tricontour(tx, ty, cat, levels=np.arange(0.5, 4.6, 1),
              colors='white', linewidths=0.45, alpha=0.9)
ax.scatter(tx, ty, s=2.2, color='white', alpha=0.18, linewidths=0)
for c in range(5):
    m = cat == c
    if m.sum() > 20:
        cx, cy = np.median(tx[m]), np.median(ty[m])
        label = ['N=4\nγ=4.5', 'N=4\nγ≈3.3-3.6', 'N=2', 'N=6', 'N=8/10'][c]
        ax.text(cx, cy, label, ha='center', va='center', fontsize=8.2,
                color='white' if c in [0, 2, 3] else NEUTRAL['black'],
                fontweight='bold')
# 边框与顶点标注
tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
ax.plot(tri[:, 0], tri[:, 1], 'k-', lw=1.2)
ax.text(-0.04, -0.03, '$w_R=1$\n(只看热阻)', ha='center', va='top', fontsize=10)
ax.text(1.04, -0.03, '$w_{\\Delta P}=1$\n(只看压降)', ha='center', va='top', fontsize=10)
ax.text(0.5, np.sqrt(3) / 2 + 0.02, '$w_\\phi=1$（只看均温性）', ha='center', va='bottom', fontsize=10)
# 现实偏好域边界 wi>=0.1
Wr = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
rx, ry = tern_xy(Wr)
ax.plot(rx, ry, color=NEUTRAL['black'], ls='--', lw=1.4, label='现实偏好域 $w_i\\geq0.1$')
# 场景点
scen_w = {'高性能计算/射频': [0.6, 0.2, 0.2], '数据中心': [0.2, 0.6, 0.2],
          '车规功率器件': [0.2, 0.2, 0.6], '均衡': [1/3, 1/3, 1/3]}
for name, w in scen_w.items():
    sx, sy = tern_xy(np.array([w]))
    ax.plot(sx, sy, '*', ms=17, color='white', mec='k', mew=1.4, zorder=6)
    ax.annotate(name, (sx[0], sy[0]), textcoords='offset points', xytext=(10, 8),
                fontsize=8.5, arrowprops=dict(arrowstyle='-', lw=0.5, color=NEUTRAL['dark']))
ax.set_xlim(-0.18, 1.18); ax.set_ylim(-0.14, 1.0)
ax.set_aspect('equal'); ax.axis('off')
ax.text(0.5, -0.105,
        '现实偏好域内：x_minmax=(0.2175,3.55,4)，x_cover=(0.22,4.5,4)，最终采用覆盖率主导方案',
        ha='center', fontsize=9.2, color='#2c3e50')
handles = [Line2D([0], [0], marker='s', ls='', ms=8, mfc=cat_colors[i],
                  mec='none', label=cat_names[i]) for i in range(5)]
handles.append(Line2D([0], [0], color=NEUTRAL['black'], ls='--', lw=1.2,
                      label='现实偏好域 $w_i\\geq0.1$'))
ax.legend(handles=handles, fontsize=8.0, loc='upper left')
fig.tight_layout()
save_pub(fig, FIG + '/p4_fig1_权重三角图.png')
plt.close(fig)

# ---------- 图2：覆盖率曲线 + 候选设计regret/覆盖对比 ----------
fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.35),
                         gridspec_kw={'width_ratios': [1.25, 0.9, 0.9]})
ax = axes[0]
cc = res['cover_curves']
styles = {'robust': ('-', '#c0392b', 2.4, '鲁棒方案(0.22,4.5,4)=P3综合最优'),
          'minimax_full': ('--', '#16a085', 1.8, '全域minimax解(0.22,4.5,6)'),
          'N=2': (':', '#8e44ad', 1.5, 'N=2代表解'),
          'N=10': (':', '#f1c40f', 1.8, 'N=10代表解')}
for k, (ls, c, lw, lab) in styles.items():
    if k in cc:
        arr = np.array(cc[k])
        ax.plot(arr[:, 0], 100 * arr[:, 1], ls, color=c, lw=lw, label=lab)
ax.axvline(0.05, color='grey', ls='-.', lw=1)
ax.text(0.052, 18, 'paper threshold\\nε=0.05', fontsize=8.5, color='grey')
ax.set_xlabel('近优容差 ε（标准化得分）'); ax.set_ylabel('权重空间覆盖率 (%)')
ax.set_title('ε-coverage curve')
panel_label(ax, '(a)')
ax.grid(alpha=0.25, ls=':'); ax.legend(fontsize=7.4)

ax = axes[1]
cands = [('鲁棒方案\n(0.22,4.5,4)', res['restricted_domain']['coverage'], '#c0392b'),
         ('受限minimax解\n(0.218,3.55,4)', res['restricted_domain']['minimax'], '#e67e22'),
         ('全域minimax解\n(0.22,4.5,6)', res['robust_minimax'], '#16a085')]
xpos = np.arange(len(cands))
mr = [c[1]['max_regret'] for c in cands]
cv = [c[1]['coverage5_pct'] for c in cands]
bars = ax.bar(xpos, mr, 0.58, color=[c[2] for c in cands], alpha=0.86,
              edgecolor=[NEUTRAL['black'] if i == 0 else 'white' for i in range(len(cands))],
              linewidth=[1.2 if i == 0 else 0.3 for i in range(len(cands))])
for i, v1 in enumerate(mr):
    ax.text(i, v1 + 0.006, '%.3f' % v1, ha='center', fontsize=8.5)
ax.set_xticks(xpos); ax.set_xticklabels([c[0] for c in cands], fontsize=9)
ax.set_ylabel('max regret (lower is better)')
ax.set_ylim(0, 0.42)
ax.set_title('Worst-case criterion')
panel_label(ax, '(b)')
ax.grid(axis='y')

ax = axes[2]
bars = ax.bar(xpos, cv, 0.58, color=[c[2] for c in cands], alpha=0.48,
              hatch=['//' if i == 0 else '' for i in range(len(cands))],
              edgecolor=[NEUTRAL['black'] if i == 0 else 'white' for i in range(len(cands))],
              linewidth=[1.2 if i == 0 else 0.3 for i in range(len(cands))])
for i, v2 in enumerate(cv):
    ax.text(i, v2 + 1.2, '%.1f%%' % v2, ha='center', fontsize=8.5)
ax.set_xticks(xpos); ax.set_xticklabels([c[0] for c in cands], fontsize=9)
ax.set_ylabel('ε=5% coverage (higher is better)')
ax.set_ylim(0, 105)
ax.set_title('Coverage criterion')
panel_label(ax, '(c)')
ax.grid(axis='y')
fig.tight_layout()
save_pub(fig, FIG + '/p4_fig2_鲁棒判定.png')
plt.close(fig)

# ---------- 图2b：max regret - coverage 取舍散点 ----------
trade_path = BASE + '/代码/p4_robust_tradeoff.csv'
if os.path.exists(trade_path):
    tr = pd.read_csv(trade_path, encoding='utf-8-sig')
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.scatter(tr['max_regret_restricted'], tr['coverage5_restricted_pct'],
               s=90, color='#7f8c8d', alpha=0.65)
    highlight = {
        'x_minmax_restricted': '#e67e22',
        'x_cover_restricted_final': '#c0392b',
        'x_p3_recommended': '#2980b9',
    }
    for _, r in tr.iterrows():
        if r['candidate'] in highlight:
            ax.scatter([r['max_regret_restricted']], [r['coverage5_restricted_pct']],
                       s=180, color=highlight[r['candidate']], edgecolor='k',
                       label=r['candidate'])
            ax.annotate('(%0.3f,%0.2f,%d)' % (r['beta'], r['gamma'], r['N']),
                        (r['max_regret_restricted'], r['coverage5_restricted_pct']),
                        textcoords='offset points', xytext=(8, 6), fontsize=8.5)
    ax.set_xlabel('现实偏好域 max regret')
    ax.set_ylabel('ε=5% 覆盖率 (%)')
    ax.set_title('鲁棒准则取舍：最终方案牺牲少量 max regret 换取高覆盖率')
    ax.grid(alpha=0.25, ls=':')
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_pub(fig, FIG + '/p4_fig4_后悔值_覆盖率散点.png')
    plt.close(fig)

# ---------- 图3：场景分析 ----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
scen = res['scenarios']
names = list(scen.keys())
ax = axes[0]
for i, nm in enumerate(names):
    x = scen[nm]['opt_x']
    ax.scatter(x[1], x[2], s=160, marker='o', color=plt.cm.tab10(i), edgecolor='k',
               zorder=4, label='%s 最优 (β=%.2f)' % (nm.split('(')[0], x[0]))
xr = res['final_robust']['x']
ax.scatter([xr[1]], [xr[2]], s=420, marker='*', color='#c0392b', edgecolor='k',
           zorder=5, label='鲁棒方案 (0.22, 4.5, 4)')
ax.set_xlabel('歧管深高比 γ'); ax.set_ylabel('针肋排数 N')
ax.set_xlim(3.0, 4.7); ax.set_ylim(1, 11)
ax.grid(alpha=0.25, ls=':')
ax.set_title('(a) 各场景最优设计 vs 鲁棒方案（γ–N 平面）', fontsize=11)
ax.legend(fontsize=8, loc='upper left')

ax = axes[1]
regs = [100 * scen[nm]['robust_regret'] for nm in names]
bars = ax.bar(range(len(names)), regs, 0.55,
              color=['#7f8c8d', '#2980b9', '#27ae60', '#8e44ad'], alpha=0.85)
for i, v in enumerate(regs):
    ax.text(i, v + 0.04, '%.2f%%' % v, ha='center', fontsize=10)
ax.axhline(5, color='r', ls='--', lw=1.2)
ax.text(2.4, 5.15, 'ε=5% 近优阈值', color='r', fontsize=9)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.split('(')[0] for n in names], fontsize=9.5)
ax.set_ylabel('鲁棒方案的后悔值（% 标准化得分）')
ax.set_ylim(0, 6)
ax.set_title('(b) 鲁棒方案在各场景下的性能损失', fontsize=11)
ax.grid(alpha=0.25, ls=':', axis='y')
fig.tight_layout()
save_pub(fig, FIG + '/p4_fig3_场景分析.png')
plt.close(fig)
print('p4 figures saved')
