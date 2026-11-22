# -*- coding: utf-8 -*-
"""问题3配图：Pareto前沿、决策空间、方案对比、双代理互验"""
import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa

from figure_style import DESIGN_COLORS, NEUTRAL, apply_pub_style, panel_label, save_pub

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
apply_pub_style()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 包根目录（相对路径，解压即可运行）
FIG = BASE + '/图表'
os.makedirs(FIG, exist_ok=True)

pf = pd.read_csv(BASE + '/代码/p3_pareto_front.csv', encoding='utf-8-sig')
with open(BASE + '/代码/p3_results.json', encoding='utf-8') as f:
    res = json.load(f)
M = joblib.load(BASE + '/代码/p2_gpr_models.pkl')
scaler = M['scalerX']
TARGETS = ['R', 'dP', 'phi']

SEL = res['selected']
REC = res['recommended']
SB = res['single_best']

mark = {'熵权TOPSIS': ('P', DESIGN_COLORS['topsis']), '等权理想点': ('*', DESIGN_COLORS['ideal']),
        'Chebyshev': ('X', DESIGN_COLORS['cheby'])}
N_LEVELS = np.array([2, 4, 6, 8, 10])
N_CMAP = ListedColormap(['#484878', '#7884B4', '#B4C0E4', '#E4CCD8', '#F0C0CC'])
N_NORM = BoundaryNorm([1, 3, 5, 7, 9, 11], N_CMAP.N)

# ---------- 图1：Pareto 前沿 ----------
fig = plt.figure(figsize=(13.5, 10))
ax = fig.add_subplot(2, 2, 1, projection='3d')
sc = ax.scatter(pf['R'], pf['dP'], pf['phi'], c=pf['N'], cmap=N_CMAP, norm=N_NORM, s=10, alpha=0.62)
cb = fig.colorbar(sc, ax=ax, shrink=0.6, label='N')
cb.set_ticks(N_LEVELS)
for k, (mk, c) in mark.items():
    F = SEL[k]['F']
    ax.scatter([F[0]], [F[1]], [F[2]], marker=mk, s=200, color=c, edgecolor='k',
               linewidths=0.8, label=k, depthshade=False)
ax.set_xlabel('$R^*$'); ax.set_ylabel('$\\Delta P^*$'); ax.set_zlabel('$\\phi$')
ax.set_title('3D Pareto front')
panel_label(ax, '(a)')
ax.view_init(22, 40); ax.legend(fontsize=8, loc='upper left')

pairs = [('R', 'dP', '$R^*$', '$\\Delta P^*$', 0, 1),
         ('R', 'phi', '$R^*$', '$\\phi$', 0, 2),
         ('dP', 'phi', '$\\Delta P^*$', '$\\phi$', 1, 2)]
for i, (xk, yk, xl, yl, fi, fj) in enumerate(pairs):
    ax = fig.add_subplot(2, 2, i + 2)
    sc = ax.scatter(pf[xk], pf[yk], c=pf['N'], cmap=N_CMAP, norm=N_NORM, s=10, alpha=0.58)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, label='N')
    cb.set_ticks(N_LEVELS)
    for k, (mk, c) in mark.items():
        F = SEL[k]['F']
        ax.scatter([F[fi]], [F[fj]], marker=mk, s=190, color=c, edgecolor='k',
                   linewidths=0.8, label=k, zorder=5)
    for t, lab in [('R', '$R^*$最优'), ('dP', '$\\Delta P^*$最优'), ('phi', '$\\phi$最优')]:
        F = SB[t]['F']
        ax.scatter([F[fi]], [F[fj]], marker='o', s=60, facecolor='none',
                   edgecolor='red', linewidths=1.4, zorder=5)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title('%s–%s projection' % (xl, yl))
    panel_label(ax, '(%s)' % chr(98 + i))
    ax.grid(alpha=0.25, ls=':')
    if i == 0:
        ax.legend(fontsize=8)
fig.tight_layout()
save_pub(fig, FIG + '/p3_fig1_Pareto前沿.png')
plt.close(fig)

# ---------- 图2：决策空间 ----------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2),
                         gridspec_kw={'width_ratios': [1.3, 1, 1]})
ax = axes[0]
sc = ax.scatter(pf['beta'], pf['gamma'], c=pf['N'], cmap=N_CMAP, norm=N_NORM, s=16, alpha=0.68)
cb = fig.colorbar(sc, ax=ax, shrink=0.9, label='N')
cb.set_ticks(N_LEVELS)
for k, (mk, c) in mark.items():
    x = SEL[k]['x']
    ax.scatter([x[0]], [x[1]], marker=mk, s=210, color=c, edgecolor='k', label=k, zorder=5)
ax.set_xlabel('β'); ax.set_ylabel('γ')
ax.set_xlim(0.08, 0.32); ax.set_ylim(2.95, 4.55)
ax.set_title('Decision-space Pareto branches')
panel_label(ax, '(a)')
ax.text(0.228, 3.17, 'low-$R^*$ branch', fontsize=8, color=NEUTRAL['dark'])
ax.text(0.205, 4.38, 'low-$\\Delta P^*$ branch', fontsize=8, color=NEUTRAL['dark'])
ax.legend(fontsize=8, loc='lower left'); ax.grid(alpha=0.25, ls=':')

ax = axes[1]
cnt = pf.groupby('N').size()
ax.bar(cnt.index, cnt.values, 1.1, color='#4878a8', alpha=0.85)
ax.set_xlabel('N'); ax.set_ylabel('Pareto解数量')
ax.set_title('Discrete N support')
panel_label(ax, '(b)')
ax.grid(alpha=0.25, ls=':', axis='y')

ax = axes[2]
h = ax.hist2d(pf['beta'], pf['N'], bins=[24, 5], cmap='Blues')
fig.colorbar(h[3], ax=ax, shrink=0.9, label='解数量')
ax.set_xlabel('β'); ax.set_ylabel('N')
ax.set_title('β–N density')
panel_label(ax, '(c)')
fig.tight_layout()
save_pub(fig, FIG + '/p3_fig2_决策空间.png')
plt.close(fig)

# ---------- 图3：候选方案对比（归一化分组柱状图） ----------
df = pd.read_csv(BASE + '/附件2_data.tsv', sep='\t', skiprows=2, header=None,
                 names=['id', 'beta', 'gamma', 'N', 'R', 'dP', 'phi'],
                 encoding='utf-8-sig').dropna()
lo = df[TARGETS].min().values; hi = df[TARGETS].max().values

designs = [('推荐方案\n(0.22, 4.5, 4)', REC['x'],
            [REC['pred'][t]['mu'] for t in TARGETS], '#c0392b'),
           ('TOPSIS解\n(0.218, 4.5, 2)', SEL['熵权TOPSIS']['x'], SEL['熵权TOPSIS']['F'], '#8e44ad'),
           ('Chebyshev解\n(0.22, 4.5, 6)', SEL['Chebyshev']['x'], SEL['Chebyshev']['F'], '#16a085'),
           ('R*单指标最优\n(0.227, 3.0, 10)', SB['R']['x'], SB['R']['F'], '#7f8c8d'),
           ('ΔP*单指标最优\n(0.226, 4.5, 2)', SB['dP']['x'], SB['dP']['F'], '#aab7b8'),
           ('无肋基准\n(0, 4.5, 0)', [0, 4.5, 0], df[(df.beta == 0) & (df.gamma == 4.5)][TARGETS].values[0].tolist(), '#2c3e50')]

fig, ax = plt.subplots(figsize=(12.5, 4.8))
x = np.arange(3); w = 0.13
labels = ['$R^*$', '$\\Delta P^*$', '$\\phi$']
for k, (name, xx, F, c) in enumerate(designs):
    Z = (np.array(F) - lo) / (hi - lo)
    off = (k - (len(designs) - 1) / 2) * w
    ax.bar(x + off, Z, w * 0.9, label=name, color=c, alpha=0.88)
    for xi, z in zip(x, Z):
        ax.text(xi + off, z + 0.015, '%.2f' % z, ha='center', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel('归一化指标（全数据min-max, 越低越好）')
ax.set_title('图3-3 候选设计方案三项指标对比', fontsize=13)
ax.legend(fontsize=8.5, ncol=3, loc='upper center')
ax.set_ylim(0, 1.25); ax.grid(alpha=0.25, ls=':', axis='y')
fig.tight_layout()
save_pub(fig, FIG + '/p3_fig3_方案对比.png')
plt.close(fig)

# ---------- 图4：双代理互验 ----------
def predict(X, kind):
    X = np.atleast_2d(np.asarray(X, float))
    if kind == 'gpr':
        Xs = scaler.transform(X)
        return np.column_stack([M[t].predict(Xs) for t in TARGETS])
    return np.column_stack([M['rsm3_' + t].predict(X) for t in TARGETS])

def pareto_mask(F):
    n = len(F); mask = np.ones(n, bool)
    for s in range(0, n, 512):
        e = min(s + 512, n)
        le = (F[None, :, :] <= F[s:e][:, None, :]).all(-1)
        lt = (F[None, :, :] < F[s:e][:, None, :]).any(-1)
        mask[s:e] &= ~((le & lt).any(1))
    return mask

betas = np.arange(0.10, 0.3001, 0.0025)
gammas = np.arange(3.0, 4.5001, 0.025)
G = np.array([(b, g, n) for n in [2, 4, 6, 8, 10] for g in gammas for b in betas])
Fr = predict(G, 'rsm3')
PFr = Fr[pareto_mask(Fr)]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
chk_path = BASE + '/代码/dual_proxy_check.csv'
chk = pd.read_csv(chk_path, encoding='utf-8-sig') if os.path.exists(chk_path) else None
for ax, (fi, fj, xl, yl) in zip(axes, [(0, 1, '$R^*$', '$\\Delta P^*$'),
                                        (1, 2, '$\\Delta P^*$', '$\\phi$')]):
    ax.scatter(pf[TARGETS[fi]], pf[TARGETS[fj]], s=12, alpha=0.5,
               color='#4878a8', label='GPR前沿')
    ax.scatter(PFr[:, fi], PFr[:, fj], s=12, alpha=0.45, marker='^',
               color='#e67e22', label='RSM-3前沿')
    F1 = REC['pred']
    ax.scatter([[F1['R']['mu'], F1['dP']['mu'], F1['phi']['mu']][fi]],
               [[F1['R']['mu'], F1['dP']['mu'], F1['phi']['mu']][fj]],
               marker='*', s=260, color='#c0392b', edgecolor='k', zorder=5,
               label='推荐方案(GPR)')
    F2 = res['rsm3_selected']['等权理想点']['F']
    ax.scatter([F2[fi]], [F2[fj]], marker='*', s=260, facecolor='none',
               edgecolor='#c0392b', linewidths=1.6, zorder=5, label='理想点解(RSM-3)')
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    if chk is not None:
        max_d = chk[['R_abs_diff', 'dP_abs_diff', 'phi_abs_diff']].max().max()
        mean_d = chk[['R_abs_diff', 'dP_abs_diff', 'phi_abs_diff']].mean().mean()
        ax.text(0.98, 0.04, 'max |Δ|=%.2e\\nmean |Δ|=%.2e' % (max_d, mean_d),
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=NEUTRAL['light'], alpha=0.9))
    ax.grid(alpha=0.25, ls=':'); ax.legend(fontsize=8.0)
fig.tight_layout()
save_pub(fig, FIG + '/p3_fig4_双代理互验.png')
plt.close(fig)

# ---------- 图5：网格收敛 ----------
conv_path = BASE + '/代码/p3_grid_convergence.csv'
if os.path.exists(conv_path):
    conv = pd.read_csv(conv_path, encoding='utf-8-sig')
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    axes[0].plot(conv['grid'], conv['n_pareto'], 'o-', color='#2980b9', lw=2)
    axes[0].set_ylabel('Pareto 解数')
    axes[0].set_title('(a) 网格分辨率与前沿规模')
    axes[0].grid(alpha=0.25, ls=':')
    axes[1].plot(conv['grid'], conv['front_max_nearest_distance_to_current'],
                 's-', color='#c0392b', lw=2, label='max nearest distance')
    axes[1].plot(conv['grid'], conv['front_mean_nearest_distance_to_current'],
                 'o--', color='#7f8c8d', lw=1.8, label='mean nearest distance')
    axes[1].set_ylabel('归一化目标空间距离')
    axes[1].set_title('(b) 前沿到现有网格前沿的最近距离')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25, ls=':')
    fig.tight_layout()
    save_pub(fig, FIG + '/p3_fig5_网格收敛.png')
    plt.close(fig)

# ---------- 图6：局部精修 ----------
ref_path = BASE + '/代码/p3_local_refinement.csv'
if os.path.exists(ref_path):
    ref = pd.read_csv(ref_path, encoding='utf-8-sig')
    best = ref.sort_values('objective').groupby('N', as_index=False).first()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(best['N'], best['objective'], 'o-', color='#2c3e50', lw=2)
    for _, r in best.iterrows():
        ax.text(r['N'], r['objective'], 'β=%.3f\nγ=%.3f' % (r['refined_beta'], r['refined_gamma']),
                fontsize=8, ha='center', va='bottom')
    ax.set_xlabel('固定针肋排数 N')
    ax.set_ylabel('局部精修 Chebyshev 目标')
    ax.set_title('各 N 分层 SLSQP 局部精修结果')
    ax.grid(alpha=0.25, ls=':')
    fig.tight_layout()
    save_pub(fig, FIG + '/p3_fig6_局部精修.png')
    plt.close(fig)
print('p3 figures saved')
