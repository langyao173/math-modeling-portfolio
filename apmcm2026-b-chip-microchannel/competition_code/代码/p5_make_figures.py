# -*- coding: utf-8 -*-
"""问题5配图：敏感性总览、公差MC分布、方案稳定性对比、工况波动"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import truncnorm

from figure_style import METRIC_COLORS, NEUTRAL, apply_pub_style, panel_label, save_pub

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
apply_pub_style()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 包根目录（相对路径，解压即可运行）
FIG = BASE + '/图表'
os.makedirs(FIG, exist_ok=True)
with open(BASE + '/代码/p5_results.json', encoding='utf-8') as f:
    res = json.load(f)
npz = np.load(BASE + '/代码/p5_mc_samples.npz')

TARGETS = ['R', 'dP', 'phi']
TN = {'R': '$R^*$', 'dP': '$\\Delta P^*$', 'phi': '$\\phi$'}
VARS = ['beta', 'gamma', 'N']
VN = ['β', 'γ', 'N']
C3 = [METRIC_COLORS['R'], METRIC_COLORS['dP'], METRIC_COLORS['phi']]

# ---------- 图1：敏感性总览 ----------
fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.35),
                         gridspec_kw={'width_ratios': [0.95, 1.15, 1.15]})
ax = axes[0]
E = np.array([[res['elasticity'][t][v] for v in VARS] for t in TARGETS])
norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
im = ax.imshow(E, cmap='RdBu_r', norm=norm, aspect='auto')
ax.set_xticks(np.arange(3)); ax.set_xticklabels(VN)
ax.set_yticks(np.arange(3)); ax.set_yticklabels([TN[t] for t in TARGETS])
for i in range(E.shape[0]):
    for j in range(E.shape[1]):
        ax.text(j, i, f'{E[i, j]:+.3f}', ha='center', va='center',
                fontsize=8.5, color='white' if abs(E[i, j]) > 0.55 else NEUTRAL['black'])
ax.set_title('Local elasticity matrix')
ax.set_xlabel('perturbed parameter')
ax.set_ylabel('response')
panel_label(ax, '(a)')
cbar = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
cbar.set_label('$\\partial\\ln Y / \\partial\\ln X$')

sobol_path = BASE + '/代码/p5_sobol_table_for_paper.csv'
sob = pd.read_csv(sobol_path, encoding='utf-8-sig')

def plot_sobol_ci(ax, domain, title, xticklabels):
    subd = sob[sob['domain'] == domain]
    x = np.arange(len(VARS)); w = 0.22
    for k, t in enumerate(TARGETS):
        sub = subd[subd['target'] == t].set_index('variable').loc[VARS]
        xpos = x + (k - 1) * w
        st = sub['ST_display'].values
        st_low = sub['ST_ci95_low'].values
        st_high = sub['ST_ci95_high'].values
        st_yerr = np.vstack([np.maximum(st - st_low, 0), np.maximum(st_high - st, 0)])
        ax.bar(xpos, st, w * 0.9, color=C3[k], alpha=0.32,
               edgecolor=C3[k], linewidth=0.8)
        ax.errorbar(xpos, st, yerr=st_yerr, fmt='none', ecolor=C3[k],
                    elinewidth=0.8, capsize=2.6, zorder=4)
        s1 = sub['S1_mean'].values
        s1_low = sub['S1_ci95_low'].values
        s1_high = sub['S1_ci95_high'].values
        s1_yerr = np.vstack([np.maximum(s1 - s1_low, 0), np.maximum(s1_high - s1, 0)])
        ax.errorbar(xpos, s1, yerr=s1_yerr, fmt='o', ms=4.5,
                    color=C3[k], mec='white', mew=0.45,
                    elinewidth=0.8, capsize=2.6, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Sobol index')
    ax.set_title(title)
    ax.grid(axis='y')

plot_sobol_ci(axes[1], 'full_design_domain', 'Full design domain Sobol', VN)
plot_sobol_ci(axes[2], 'domain_truncated_tolerance', 'Tolerance-domain Sobol',
              ['β tol.', 'γ tol.\\n[3.0,4.5]', 'N fixed'])
panel_label(axes[1], '(b)')
panel_label(axes[2], '(c)')
legend_items = [
    Line2D([0], [0], marker='o', ls='', color=C3[k], markeredgecolor='white',
           label=TN[t] + ' $S_1$') for k, t in enumerate(TARGETS)
]
legend_items.append(Rectangle((0, 0), 1, 1, fc=NEUTRAL['mid'], alpha=0.32,
                              label='$S_T$ bar + 95% CI'))
axes[2].legend(handles=legend_items, loc='upper right', fontsize=7.2)
fig.tight_layout()
save_pub(fig, FIG + '/p5_fig1_敏感性总览.png')
plt.close(fig)

# ---------- 图2：公差MC分布（鲁棒方案, GPR vs RSM3） ----------
Fmc = npz['robust']; Fr3 = npz['robust_rsm3']
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for k, t in enumerate(TARGETS):
    ax = axes[k]
    st = res['mc_stats']['鲁棒方案(0.22,4.5,4)'][t]
    ax.hist(Fmc[:, k], bins=60, density=True, color=C3[k], alpha=0.65, label='GPR代理')
    ax.hist(Fr3[:, k], bins=60, density=True, histtype='step', color='k',
            lw=1.4, label='RSM-3交叉验证')
    ax.axvline(st['nominal'], color='k', ls='-', lw=1.6)
    ax.axvline(st['p95'], color='r', ls='--', lw=1.4)
    ax.text(st['nominal'], ax.get_ylim()[1] * 0.93, ' 标称', fontsize=9)
    ax.text(st['p95'], ax.get_ylim()[1] * 0.80, ' P95\n %+.2f%%' % st['p95_shift_pct'],
            fontsize=9, color='r')
    ax.set_xlabel(TN[t])
    ax.text(0.98, 0.94,
            'CV %.2f%%\\nP95 %+0.2f%%\\nworst %+0.2f%%' %
            (st['cv_pct'], st['p95_shift_pct'], st['worst_shift_pct']),
            transform=ax.transAxes, ha='right', va='top', fontsize=8.2,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=NEUTRAL['light'], alpha=0.9))
    ax.set_title('%s distribution' % TN[t], fontsize=9.5)
    ax.grid(alpha=0.25, ls=':')
    if k == 0:
        ax.legend(fontsize=8.5)
fig.tight_layout()
fig.text(0.5, 0.01, 'Domain-safe MC: γ∈[3.0,4.5], outside_domain_count=0, n=20000',
         ha='center', va='bottom', fontsize=8.5, color=NEUTRAL['dark'])
save_pub(fig, FIG + '/p5_fig2_公差MC分布.png')
plt.close(fig)

# ---------- 图3：候选方案稳定性对比 ----------
names = list(res['mc_stats'].keys())
short = ['鲁棒方案\nN=4', 'TOPSIS解\nN=2', 'Chebyshev解\nN=6', 'R*最优\nN=10,γ=3']
colors = ['#c0392b', '#8e44ad', '#16a085', '#7f8c8d']
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
x = np.arange(3); w = 0.19
for ax, key, ttl in [(axes[0], 'cv_pct', '(a) 变异系数 CV (%)'),
                     (axes[1], 'worst_shift_pct', '(b) 最坏情形偏移 (%)')]:
    for k, nm in enumerate(names):
        vals = [res['mc_stats'][nm][t][key] for t in TARGETS]
        ax.bar(x + (k - 1.5) * w, vals, w * 0.9, label=short[k], color=colors[k], alpha=0.88)
        for xi, v in zip(x, vals):
            ax.text(xi + (k - 1.5) * w, v + 0.02, '%.1f' % v if v > 0.3 else '%.2f' % v,
                    ha='center', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([TN[t] for t in TARGETS], fontsize=11)
    ax.set_title(ttl, fontsize=11)
    ax.grid(alpha=0.25, ls=':', axis='y')
axes[0].set_ylabel('CV (%)'); axes[0].legend(fontsize=8.5)
fig.tight_layout()
save_pub(fig, FIG + '/p5_fig3_方案稳定性对比.png')
plt.close(fig)

# ---------- 图4：工况波动（流量扫掠, 机理模型） ----------
mm, op_R, op_dP = npz['mm'], npz['op_R'], npz['op_dP']
R0 = res['operating']['R0_KW']; dp0 = res['operating']['dP0_Pa']
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
ax = axes[0]
ax.plot(mm, 100 * (op_R / R0 - 1), '-', color=METRIC_COLORS['R'], lw=2)
ax.axvspan(0.9, 1.1, color='grey', alpha=0.15)
ax.axhline(0, color='k', lw=0.8)
for xx0 in [0.9, 1.0, 1.1]:
    yy = np.interp(xx0, mm, 100 * (op_R / R0 - 1))
    ax.plot(xx0, yy, 'o', color=METRIC_COLORS['R'], ms=5, mec='white', mew=0.6)
ax.annotate('+3.8%', (0.9, 100 * (op_R[8] / R0 - 1)), textcoords='offset points',
            xytext=(8, 4), fontsize=10, color='#c0392b')
ax.annotate('−3.2%', (1.1, 100 * (op_R[32] / R0 - 1)), textcoords='offset points',
            xytext=(-30, -12), fontsize=10, color='#c0392b')
ax.set_xlabel('流量比 $\\dot m/\\dot m_0$'); ax.set_ylabel('热阻变化 (%)')
ax.set_title('(a) 热阻对流量波动的响应（机理模型）', fontsize=11)
ax.grid(alpha=0.25, ls=':')

ax = axes[1]
ax.plot(mm, 100 * (op_dP / dp0 - 1), '-', color=METRIC_COLORS['dP'], lw=2)
ax.axvspan(0.9, 1.1, color='grey', alpha=0.15)
ax.axhline(0, color='k', lw=0.8)
for xx0 in [0.9, 1.0, 1.1]:
    yy = np.interp(xx0, mm, 100 * (op_dP / dp0 - 1))
    ax.plot(xx0, yy, 'o', color=METRIC_COLORS['dP'], ms=5, mec='white', mew=0.6)
ax.annotate('−16.2%', (0.9, 100 * (op_dP[8] / dp0 - 1)), textcoords='offset points',
            xytext=(8, -4), fontsize=10, color='#2980b9')
ax.annotate('+17.3%', (1.1, 100 * (op_dP[32] / dp0 - 1)), textcoords='offset points',
            xytext=(-40, 6), fontsize=10, color='#2980b9')
ax.set_xlabel('流量比 $\\dot m/\\dot m_0$'); ax.set_ylabel('压降变化 (%)')
ax.set_title('(b) 压降对流量波动的响应（机理模型）', fontsize=11)
ax.grid(alpha=0.25, ls=':')
fig.tight_layout()
save_pub(fig, FIG + '/p5_fig4_工况波动.png')
plt.close(fig)

# ---------- 图5：γ 公差分布修正 ----------
rng = np.random.default_rng(0)
mu_g, sig_g = 4.5, 0.015 * 4.5
wrong = np.clip(rng.normal(mu_g, sig_g, 20000), mu_g - 3 * sig_g, mu_g + 3 * sig_g)
a, b = (3.0 - mu_g) / sig_g, (4.5 - mu_g) / sig_g
fixed = truncnorm.rvs(a, b, loc=mu_g, scale=sig_g, size=20000, random_state=np.random.default_rng(0))
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), sharey=True)
for ax, arr, title in [
    (axes[0], wrong, '原错误方案：均值±3σ，约一半越界'),
    (axes[1], fixed, '修正方案：域内条件截断正态'),
]:
    ax.hist(arr, bins=70, color='#2980b9', alpha=0.72, density=True)
    ax.axvline(4.5, color='r', ls='--', lw=1.4, label='γ=4.5 设计域上界')
    ax.axvspan(4.5, arr.max(), color='r', alpha=0.10)
    ax.set_xlabel('γ sampled')
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, ls=':')
axes[0].set_ylabel('density')
axes[0].legend(fontsize=8.5)
fig.suptitle('γ=4.5 边界公差传播的外推风险与修正', fontsize=13)
fig.tight_layout()
save_pub(fig, FIG + '/p5_fig5_gamma公差分布修正.png')
plt.close(fig)

# ---------- 图6：Sobol 置信区间 ----------
sobol_table = BASE + '/代码/p5_sobol_table_for_paper.csv'
if os.path.exists(sobol_table):
    st = pd.read_csv(sobol_table, encoding='utf-8-sig')
    full = st[st['domain'] == 'full_design_domain']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    for ax, target, c in zip(axes, TARGETS, C3):
        sub = full[full['target'] == target].set_index('variable').loc[VARS]
        xpos = np.arange(len(VARS))
        y = sub['S1_mean'].values
        yerr = np.vstack([y - sub['S1_ci95_low'].values,
                          sub['S1_ci95_high'].values - y])
        ax.errorbar(xpos - 0.08, y, yerr=yerr, fmt='o', color=c,
                    capsize=3, label='$S_1$ mean±95% CI')
        y2 = sub['ST_display'].values
        yerr2 = np.vstack([y2 - sub['ST_ci95_low'].values,
                           sub['ST_ci95_high'].values - y2])
        ax.errorbar(xpos + 0.08, y2, yerr=yerr2, fmt='s', color='#2c3e50',
                    capsize=3, label='$S_T$ display±95% CI')
        ax.set_xticks(xpos); ax.set_xticklabels(VN)
        ax.set_title(TN[target])
        ax.grid(alpha=0.25, ls=':')
        if target == 'R':
            ax.set_ylabel('Sobol index')
            ax.legend(fontsize=8)
    fig.suptitle('全设计域 Sobol 指数：10 seeds, n=32768, 95% CI', fontsize=13)
    fig.tight_layout()
    save_pub(fig, FIG + '/p5_fig6_Sobol置信区间.png')
    plt.close(fig)
print('p5 figures saved')
