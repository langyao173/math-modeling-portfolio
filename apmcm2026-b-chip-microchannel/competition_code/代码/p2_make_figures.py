# -*- coding: utf-8 -*-
"""问题2配图：模型横评、GPR parity、不确定性校准、响应面、1D切片CI带"""
import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figure_style import METRIC_COLORS, NEUTRAL, apply_pub_style, panel_label, save_pub

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
apply_pub_style()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 包根目录（相对路径，解压即可运行）
FIG = BASE + '/图表'
os.makedirs(FIG, exist_ok=True)

df = pd.read_csv(BASE + '/附件2_data.tsv', sep='\t', skiprows=2, header=None,
                 names=['id', 'beta', 'gamma', 'N', 'R', 'dP', 'phi'],
                 encoding='utf-8-sig').dropna()
loo = pd.read_csv(BASE + '/代码/p2_loocv_gpr.csv', encoding='utf-8-sig')
with open(BASE + '/代码/p2_nested_cv_results.json', encoding='utf-8') as f:
    cv = json.load(f)['loocv']
M = joblib.load(BASE + '/代码/p2_gpr_models.pkl')
scaler = M['scalerX']

TARGETS = [('R', '无量纲热阻 $R^*$'), ('dP', '无量纲压降 $\\Delta P^*$'),
           ('phi', '温度非均匀性 $\\phi$')]
TARGET_SHORT = {'R': '$R^*$', 'dP': '$\\Delta P^*$', 'phi': '$\\phi$'}
ORDER = ['GPR-Matern', 'RSM-3次', 'SVR-RBF', 'KRR-RBF', 'RSM-2次', '随机森林', 'MLP(16-8)']
MODEL_COLORS = {
    'RSM-3次': '#0F4D92',
    'GPR-Matern': '#B64342',
    'SVR-RBF': '#42949E',
    'KRR-RBF': '#9A4D8E',
    'RSM-2次': '#B4C0E4',
    '随机森林': '#A7A7A7',
    'MLP(16-8)': '#D6D6D6',
}
COLORS = [MODEL_COLORS[n] for n in ORDER]

def gpr_pred(t, Xq, std=False):
    return M[t].predict(scaler.transform(Xq), return_std=std)

# ---------- 图1：模型横评 ----------
# R² 直接作柱状图会掩盖高精模型之间的差距；这里改用 RMSE 倍率与 1-R² 对数尺度。
model_rank = {}
for name in ORDER:
    ranks = []
    for t, _ in TARGETS:
        rmses = np.array([cv[t][m]['RMSE'] for m in ORDER], dtype=float)
        ranks.append(1 + np.argsort(np.argsort(rmses))[ORDER.index(name)])
    model_rank[name] = float(np.mean(ranks))
ORDER_SORTED = sorted(ORDER, key=lambda m: (model_rank[m], ORDER.index(m)))
colors_sorted = [MODEL_COLORS[m] for m in ORDER_SORTED]

fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8), gridspec_kw={'wspace': 0.22})
x = np.arange(len(TARGETS))
w = 0.105
metric_labels = ['$R^*$', '$\\Delta P^*$', '$\\phi$']
for k, name in enumerate(ORDER_SORTED):
    off = (k - (len(ORDER_SORTED) - 1) / 2) * w
    rmse = np.array([cv[t][name]['RMSE'] for t, _ in TARGETS], dtype=float)
    best = np.array([min(cv[t][m]['RMSE'] for m in ORDER) for t, _ in TARGETS], dtype=float)
    ratio = np.maximum(rmse / np.maximum(best, 1e-8), 1e-4)
    r2gap = np.array([max(1 - cv[t][name]['R2'], 1e-6) for t, _ in TARGETS], dtype=float)
    edge = NEUTRAL['black'] if name in ['RSM-3次', 'GPR-Matern'] else 'white'
    lw = 1.2 if name in ['RSM-3次', 'GPR-Matern'] else 0.35
    hatch = '//' if name == 'GPR-Matern' else ('..' if name == 'RSM-3次' else None)
    axes[0].bar(x + off, ratio, w * 0.94, color=colors_sorted[k], edgecolor=edge,
                linewidth=lw, hatch=hatch, label=name)
    axes[1].bar(x + off, r2gap, w * 0.94, color=colors_sorted[k], edgecolor=edge,
                linewidth=lw, hatch=hatch)

axes[0].axhline(1, color=NEUTRAL['dark'], lw=0.8, ls='--')
axes[0].set_yscale('log')
axes[0].set_ylabel('RMSE / best RMSE')
axes[0].set_title('RMSE error multiplier')
axes[1].set_yscale('log')
axes[1].set_ylabel('$1-R^2$')
axes[1].set_title('Residual unexplained variance')
for ax, lab in zip(axes, ['(a)', '(b)']):
    panel_label(ax, lab)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.grid(axis='y')
axes[0].text(
    0.02, 0.05,
    'strict nested LOOCV\\nouter fold: 83 train / 1 blind test',
    transform=axes[0].transAxes,
    fontsize=8,
    color=NEUTRAL['dark'],
    ha='left',
    va='bottom',
)
handles = [Line2D([0], [0], marker='s', ls='', ms=7, mfc=MODEL_COLORS[m],
                  mec=NEUTRAL['black'] if m in ['RSM-3次', 'GPR-Matern'] else 'white',
                  label=m) for m in ORDER_SORTED]
axes[1].legend(handles=handles, ncol=2, loc='upper right', fontsize=7.2)
fig.tight_layout()
save_pub(fig, FIG + '/p2_fig1_模型横评.png')
plt.close(fig)

# ---------- 图1b：旧协议 vs 严格 nested LOOCV ----------
legacy_path = BASE + '/代码/p2_cv_results_legacy_reference.json'
if os.path.exists(legacy_path):
    with open(legacy_path, encoding='utf-8') as f:
        legacy = json.load(f).get('loocv', {})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    x = np.arange(len(TARGETS)); w2 = 0.11
    compare_models = ['GPR-Matern', 'RSM-3次', 'KRR-RBF', 'SVR-RBF']
    for k, name in enumerate(compare_models):
        off = (k - 1.5) * w2
        old = [legacy[t][name]['R2'] if t in legacy and name in legacy[t] else np.nan
               for t, _ in TARGETS]
        new = [cv[t][name]['R2'] for t, _ in TARGETS]
        axes[0].plot(x + off, old, 'o', color=COLORS[k], alpha=0.35)
        axes[0].plot(x + off, new, 's', color=COLORS[k], label=name)
        old_rmse = [legacy[t][name]['RMSE'] if t in legacy and name in legacy[t] else np.nan
                    for t, _ in TARGETS]
        new_rmse = [cv[t][name]['RMSE'] for t, _ in TARGETS]
        axes[1].plot(x + off, old_rmse, 'o', color=COLORS[k], alpha=0.35)
        axes[1].plot(x + off, new_rmse, 's', color=COLORS[k], label=name)
    axes[0].set_ylabel('$R^2$'); axes[0].set_title('旧协议(圆) vs strict nested(方)：$R^2$')
    axes[1].set_yscale('log'); axes[1].set_ylabel('RMSE'); axes[1].set_title('旧协议(圆) vs strict nested(方)：RMSE')
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([n for _, n in TARGETS])
        ax.grid(alpha=0.25, ls=':')
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_pub(fig, FIG + '/p2_fig1b_旧协议_vs_strict_nested.png')
    plt.close(fig)

# ---------- 图1c：GPR vs RSM-3 LOOCV误差 ----------
all_path = BASE + '/代码/p2_nested_loocv_all_models.csv'
if os.path.exists(all_path):
    allcv = pd.read_csv(all_path, encoding='utf-8-sig')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, (t, title) in zip(axes, TARGETS):
        eg = np.abs(allcv[t] - allcv[f'{t}_GPR-Matern_pred'])
        er = np.abs(allcv[t] - allcv[f'{t}_RSM-3次_pred'])
        ax.scatter(eg, er, s=26, alpha=0.7, color='#2c3e50')
        hi = max(eg.max(), er.max()) * 1.05
        ax.plot([0, hi], [0, hi], 'k--', lw=1)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_xlabel('GPR |LOOCV误差|'); ax.set_ylabel('RSM-3 |LOOCV误差|')
        ax.set_title(title)
        ax.grid(alpha=0.25, ls=':')
    fig.tight_layout()
    save_pub(fig, FIG + '/p2_fig1c_GPR_vs_RSM3误差.png')
    plt.close(fig)

# ---------- 图2：GPR LOOCV parity ----------
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1))
for ax, (t, name) in zip(axes, TARGETS):
    y, p = loo[t].values, loo[t + '_loocv'].values
    r2 = cv[t]['GPR-Matern']['R2']
    lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
    pad = 0.04 * (hi - lo); lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], 'k-', lw=1)
    ax.fill_between([lo, hi], [lo * 0.99, hi * 0.99], [lo * 1.01, hi * 1.01],
                    color='grey', alpha=0.2, label='±1%')
    sc = ax.scatter(y, p, s=24, alpha=0.78, c=loo['N'], cmap='viridis',
                    edgecolors='white', linewidths=0.25)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    ax.set_title('%s parity\\nstrict nested LOOCV $R^2$=%.4f' % (TARGET_SHORT[t], r2), fontsize=9.5)
    ax.set_xlabel('CFD sample value'); ax.set_ylabel('GPR uncertainty-proxy prediction')
    ax.grid(alpha=0.25, ls=':'); ax.legend(fontsize=7.5, loc='upper left')
    ax.text(0.98, 0.04, 'blind-tested', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, color=NEUTRAL['dark'])
fig.colorbar(sc, ax=axes, shrink=0.86, pad=0.015, label='N')
fig.tight_layout()
save_pub(fig, FIG + '/p2_fig2_GPR_parity.png')
plt.close(fig)

# ---------- 图3：不确定性校准 ----------
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
zs = {t: (loo[t] - loo[t + '_loocv']) / loo[t + '_loocv_sd'] for t, _ in TARGETS}
ax = axes[0]
bins = np.linspace(-4, 4, 33)
for (t, name), c in zip(TARGETS, ['#c0392b', '#2980b9', '#27ae60']):
    ax.hist(zs[t], bins=bins, density=True, alpha=0.4, color=c, label=name.split(' ')[0])
xx = np.linspace(-4, 4, 200)
ax.plot(xx, np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi), 'k-', lw=1.5, label='N(0,1)')
ax.set_title('(a) 标准化残差分布', fontsize=10.5)
ax.set_xlabel('z = (y−μ)/σ'); ax.legend(fontsize=8)

ax = axes[1]
names = [n for _, n in TARGETS]
covs = [cv[t]['GPR-Matern']['CI95_coverage_pct'] for t, _ in TARGETS]
ax.bar(names, covs, 0.5, color=['#c0392b', '#2980b9', '#27ae60'], alpha=0.8)
ax.axhline(95, color='k', ls='--', lw=1.2, label='名义95%')
for i, v in enumerate(covs):
    ax.text(i, v + 0.6, '%.1f%%' % v, ha='center', fontsize=10)
ax.set_ylim(80, 104); ax.set_title('(b) 95%置信区间经验覆盖率', fontsize=10.5)
ax.legend(fontsize=8)

ax = axes[2]
for (t, name), c in zip(TARGETS, ['#c0392b', '#2980b9', '#27ae60']):
    ax.scatter(loo[t + '_loocv'], loo[t] - loo[t + '_loocv'], s=14, alpha=0.6,
               color=c, label=name.split(' ')[0])
ax.axhline(0, color='k', lw=1)
ax.set_title('(c) 残差 vs 预测值（无系统模式）', fontsize=10.5)
ax.set_xlabel('LOOCV预测值'); ax.set_ylabel('残差')
ax.legend(fontsize=8); ax.grid(alpha=0.25, ls=':')
for a in axes:
    a.grid(alpha=0.25, ls=':')
fig.tight_layout()
save_pub(fig, FIG + '/p2_fig3_不确定性校准.png')
plt.close(fig)

# ---------- 图4：响应面（GPR） ----------
fig, axes = plt.subplots(3, 2, figsize=(11, 12.5))
for i, (t, name) in enumerate(TARGETS):
    # (β,N) @ γ=4
    bb, nn = np.meshgrid(np.linspace(0.1, 0.3, 60), np.linspace(2, 10, 60))
    Xq = np.c_[bb.ravel(), np.full(bb.size, 4.0), nn.ravel()]
    Z = gpr_pred(t, Xq).reshape(bb.shape)
    ax = axes[i, 0]
    cf = ax.contourf(bb, nn, Z, 22, cmap='RdYlBu_r')
    fig.colorbar(cf, ax=ax, shrink=0.9)
    sub = df[(df.gamma == 4.0) & (df.N > 0)]
    ax.scatter(sub['beta'], sub['N'], c='k', s=12)
    imin = np.unravel_index(np.argmin(Z), Z.shape)
    ax.plot(bb[imin], nn[imin], '*', color='lime', ms=16, mec='k')
    ax.set_xlabel('β'); ax.set_ylabel('N')
    ax.set_title(name + '  (β–N 切片, γ=4)', fontsize=10.5)
    # (γ,N) @ β=0.2
    gg, nn2 = np.meshgrid(np.linspace(3, 4.5, 60), np.linspace(2, 10, 60))
    Xq = np.c_[np.full(gg.size, 0.2), gg.ravel(), nn2.ravel()]
    Z2 = gpr_pred(t, Xq).reshape(gg.shape)
    ax = axes[i, 1]
    cf = ax.contourf(gg, nn2, Z2, 22, cmap='RdYlBu_r')
    fig.colorbar(cf, ax=ax, shrink=0.9)
    sub = df[(df.beta == 0.2)]
    ax.scatter(sub['gamma'], sub['N'], c='k', s=12)
    imin = np.unravel_index(np.argmin(Z2), Z2.shape)
    ax.plot(gg[imin], nn2[imin], '*', color='lime', ms=16, mec='k')
    ax.set_xlabel('γ'); ax.set_ylabel('N')
    ax.set_title(name + '  (γ–N 切片, β=0.2)', fontsize=10.5)
fig.tight_layout()
save_pub(fig, FIG + '/p2_fig4_响应面.png')
plt.close(fig)

# ---------- 图5：1D切片 + 95% CI带 ----------
anchor = {'beta': 0.2, 'gamma': 4.5, 'N': 4.0}
sweeps = [('beta', np.linspace(0.1, 0.3, 80), '针肋宽度比 β'),
          ('gamma', np.linspace(3, 4.5, 80), '歧管深高比 γ'),
          ('N', np.linspace(2, 10, 80), '针肋排数 N')]
fig, axes = plt.subplots(3, 3, figsize=(12.5, 10))
for i, (t, name) in enumerate(TARGETS):
    for j, (sv, xs, xlabel) in enumerate(sweeps):
        ax = axes[i, j]
        Xq = np.tile([anchor['beta'], anchor['gamma'], anchor['N']], (len(xs), 1))
        Xq[:, ['beta', 'gamma', 'N'].index(sv)] = xs
        mu, sd = gpr_pred(t, Xq, std=True)
        ax.plot(xs, mu, '-', color=METRIC_COLORS.get(t, '#B64342'), lw=1.8, label='GPR均值')
        ax.fill_between(xs, mu - 1.96 * sd, mu + 1.96 * sd, color='#c0392b',
                        alpha=0.18, label='95% CI')
        ax.axvline(anchor[sv], color=NEUTRAL['dark'], lw=0.9, ls='--')
        if sv == 'N':
            ax.plot(xs, mu, '--', color=METRIC_COLORS.get(t, '#B64342'), lw=1.2, alpha=0.6)
        others = [k for k in anchor if k != sv]
        sub = df[(df[others[0]] == anchor[others[0]]) & (df[others[1]] == anchor[others[1]])]
        ax.plot(sub[sv], sub[t], 'o', color='#1f3f66', ms=5, label='样本')
        if i == 2: ax.set_xlabel(xlabel)
        if j == 0: ax.set_ylabel(name, fontsize=10)
        if i == 0 and j == 0: ax.legend(fontsize=8)
        ax.grid(alpha=0.25, ls=':')
fig.tight_layout()
save_pub(fig, FIG + '/p2_fig5_切片与CI.png')
plt.close(fig)
print('p2 figures saved')
