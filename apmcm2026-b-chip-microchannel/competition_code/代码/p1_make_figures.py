# -*- coding: utf-8 -*-
"""问题1配图：主效应(数据vs模型)、parity校验、指标相关性/冲突、机理网络示意图、φ分解"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import itertools
import pandas as pd

import p1_mechanism_model as M
from figure_style import METRIC_COLORS, NEUTRAL, apply_pub_style, panel_label, save_pub

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 110
apply_pub_style()

FIG = M.BASE + '/图表'
import os
os.makedirs(FIG, exist_ok=True)

df = M.df
thetaR, thetaP = M.resR.x, M.resP.x
TAU, CPHI = M.TAU_FIT, M.CPHI

BETAS = [0.1, 0.15, 0.2, 0.3]
GAMMAS = [3.0, 3.5, 4.0, 4.5]
NS = [2, 4, 6, 8, 10]

def model_all(X):
    X = np.asarray(X, dtype=float)
    return (M.R_model(X, thetaR), M.dP_model(X, thetaP),
            M.phi_features(X, TAU) @ CPHI)

# ---------- 图1：主效应（数据 vs 模型） ----------
fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.5))
metrics = [('R', '无量纲热阻 $R^*$'), ('dP', '无量纲压降 $\\Delta P^*$'),
           ('phi', '无量纲温度非均匀性 $\\phi$')]
sweeps = [('beta', '针肋宽度比 $\\beta$'), ('gamma', '歧管深高比 $\\gamma$'),
          ('N', '针肋排数 $N$')]

for j, (sv, sname) in enumerate(sweeps):
    # 模型扫描曲线（对其余两个因子的数据水平取平均）
    if sv == 'beta':
        xs = np.linspace(0.08, 0.32, 49)
        Xs = [np.array([(x, g, n) for g in GAMMAS for n in NS]) for x in xs]
    elif sv == 'gamma':
        xs = np.linspace(3.0, 4.5, 31)
        Xs = [np.array([(b, x, n) for b in BETAS for n in NS]) for x in xs]
    else:
        xs = np.linspace(0.25, 10, 40)
        Xs = [np.array([(b, g, x) for b in BETAS for g in GAMMAS]) for x in xs]
    curves = {m: [] for m, _ in metrics}
    for Xx in Xs:
        r, p, f = model_all(Xx)
        curves['R'].append(r.mean()); curves['dP'].append(p.mean()); curves['phi'].append(f.mean())

    sub = df if sv == 'N' else df[df.N > 0]
    for i, (mv, mname) in enumerate(metrics):
        ax = axes[i, j]
        ax.scatter(sub[sv], sub[mv], s=13, alpha=0.28, color=NEUTRAL['mid'], label='附件2样本', zorder=2)
        gm = sub.groupby(sv)[mv].mean()
        ge = sub.groupby(sv)[mv].sem().reindex(gm.index).fillna(0.0)
        ax.errorbar(gm.index, gm.values, yerr=1.96 * ge.values, fmt='o--',
                    color=METRIC_COLORS.get(mv, '#0F4D92'), ecolor=METRIC_COLORS.get(mv, '#0F4D92'),
                    lw=1.4, elinewidth=0.8, capsize=2.4, ms=4.8,
                    label='组均值±95% CI', zorder=4)
        ax.plot(xs, curves[mv], '-', color=NEUTRAL['black'], lw=1.8, label='机理模型', zorder=3)
        extrema = {
            ('R', 'beta'): (0.247, r'$\beta_R^*$'),
            ('phi', 'beta'): (0.196, r'$\beta_\phi^*$'),
            ('phi', 'gamma'): (3.83, r'$\gamma_\phi^*$'),
            ('phi', 'N'): (4.0, r'$N_\phi^*$'),
        }
        if (mv, sv) in extrema:
            xv, lab = extrema[(mv, sv)]
            ax.axvline(xv, color=NEUTRAL['dark'], ls=':', lw=1.0)
            ax.annotate(lab, xy=(xv, ax.get_ylim()[0] + 0.12 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
                        xytext=(4, 6), textcoords='offset points', fontsize=7.8,
                        color=NEUTRAL['dark'])
        if sv == 'N':   # 基准点 N=0(β=0)
            Xb = np.array([(0.0, g, 0.0) for g in GAMMAS])
            r, p, f = model_all(Xb)
            yb = {'R': r.mean(), 'dP': p.mean(), 'phi': f.mean()}[mv]
            ax.plot([0], [yb], 's', color='#c0392b', ms=7, zorder=5)
        if i == 0:
            ax.set_title('沿 ' + sname + ' 的主效应', fontsize=11)
        if i == 2:
            ax.set_xlabel(sname, fontsize=10)
        if j == 0:
            ax.set_ylabel(mname, fontsize=10)
        ax.grid(alpha=0.25, ls=':')
for i in range(3):
    lo = min(axes[i, j].get_ylim()[0] for j in range(3))
    hi = max(axes[i, j].get_ylim()[1] for j in range(3))
    pad = 0.02 * (hi - lo)
    for j in range(3):
        axes[i, j].set_ylim(lo - pad, hi + pad)
        panel_label(axes[i, j], f'({chr(97 + i * 3 + j)})')
axes[0, 0].legend(fontsize=8, loc='lower left')
fig.tight_layout()
save_pub(fig, FIG + '/p1_fig1_主效应_数据vs模型.png')
plt.close(fig)

# ---------- 图2：parity 校验 ----------
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1))
preds = {'R': M.predR, 'dP': M.predP, 'phi': M.predPhi}
err_audit = pd.read_csv(M.BASE + '/代码/p1_error_audit.csv', encoding='utf-8-sig')
for ax, (mv, mname) in zip(axes, metrics):
    y, p = df[mv].values, preds[mv]
    r2 = 1 - np.sum((y - p)**2) / np.sum((y - np.mean(y))**2)
    lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
    pad = 0.04 * (hi - lo); lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], 'k-', lw=1)
    ax.fill_between([lo, hi], [lo * 0.97, hi * 0.97], [lo * 1.03, hi * 1.03],
                    color='grey', alpha=0.18, label='±3%')
    if mv == 'dP':
        ax.fill_between([lo, hi], [lo * 0.90, hi * 0.90], [lo * 1.10, hi * 1.10],
                        color='#f39c12', alpha=0.10, label='±10%')
    outside = err_audit[mv + '_outside_3pct'].values.astype(bool)
    ax.scatter(y[~outside], p[~outside], s=18, alpha=0.65, color='#c0392b',
               edgecolors='none', label='≤±3%')
    if outside.any():
        ax.scatter(y[outside], p[outside], s=30, alpha=0.9, color='#2c3e50',
                   marker='x', label='>±3%')
    if mv == 'dP':
        ax.text(0.98, 0.04, '24/84 > ±3%\\nMaxRE=8.535%',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8.2, color=NEUTRAL['dark'],
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=NEUTRAL['light'], alpha=0.92))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    from scipy.stats import spearmanr
    ax.set_title('%s\n$R^2$=%.3f, Spearman ρ=%.3f' % (mname, r2, spearmanr(y, p)[0]), fontsize=10)
    ax.set_xlabel('CFD样本值（附件2）', fontsize=9.5)
    ax.set_ylabel('机理模型预测', fontsize=9.5)
    ax.grid(alpha=0.25, ls=':'); ax.legend(fontsize=8, loc='upper left')
fig.tight_layout()
save_pub(fig, FIG + '/p1_fig2_模型校验parity.png')
plt.close(fig)

# ---------- 图3：三指标相关性与冲突结构 ----------
fig = plt.figure(figsize=(13, 4.2))
ax1 = fig.add_subplot(1, 3, 1)
C = df[['R', 'dP', 'phi']].corr().values
im = ax1.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
labels = ['$R^*$', '$\\Delta P^*$', '$\\phi$']
ax1.set_xticks(range(3)); ax1.set_yticks(range(3))
ax1.set_xticklabels(labels); ax1.set_yticklabels(labels)
for i in range(3):
    for jj in range(3):
        ax1.text(jj, i, '%.3f' % C[i, jj], ha='center', va='center',
                 color='white' if abs(C[i, jj]) > 0.55 else 'black', fontsize=11)
fig.colorbar(im, ax=ax1, shrink=0.85)
ax1.set_title('(a) 指标Pearson相关矩阵', fontsize=11)

ax2 = fig.add_subplot(1, 3, 2)
sc = ax2.scatter(df['R'], df['dP'], c=df['gamma'], cmap='viridis', s=26, alpha=0.85)
fig.colorbar(sc, ax=ax2, shrink=0.85, label='γ')
ax2.set_xlabel('$R^*$'); ax2.set_ylabel('$\\Delta P^*$')
ax2.set_title('(b) 热阻—压降冲突（ρ=−0.68）', fontsize=11)
ax2.grid(alpha=0.25, ls=':')

ax3 = fig.add_subplot(1, 3, 3)
sc = ax3.scatter(df['dP'], df['phi'], c=df['N'], cmap='plasma', s=26, alpha=0.85)
fig.colorbar(sc, ax=ax3, shrink=0.85, label='N')
ax3.set_xlabel('$\\Delta P^*$'); ax3.set_ylabel('$\\phi$')
ax3.set_title('(c) 压降—非均匀性关联（ρ=0.47）', fontsize=11)
ax3.grid(alpha=0.25, ls=':')
fig.tight_layout()
save_pub(fig, FIG + '/p1_fig3_指标相关性.png')
plt.close(fig)

# ---------- 图4：热阻网络 + 压降链 示意图 ----------
fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 4.6))
for ax in (axa, axb):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

def box(ax, x, y, w, h, text, fc='#dbe8f4', fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12',
                                fc=fc, ec='#2c3e50', lw=1.2))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs)

def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                 mutation_scale=16, lw=1.4, color='#2c3e50'))

# (a) 热阻网络
box(axa, 0.2, 7.8, 2.6, 1.4, '芯片结温 $T_j$\n(体热源 36 W)', fc='#f6c8c2')
box(axa, 3.8, 7.8, 2.6, 1.4, '$R_{cond}$\n盖层导热+扩散')
box(axa, 7.2, 7.8, 2.6, 1.4, '$R_{conv}$\n对流换热')
box(axa, 7.2, 4.6, 2.6, 1.4, '壁面: $h_w A_w \\psi(\\beta,N) F_{imp}(\\gamma)$', fc='#e8f0d8', fs=8.5)
box(axa, 7.2, 2.6, 2.6, 1.4, '针肋: $\\varepsilon_{wake} h_{pin} A_{pin}(\\beta,N)$', fc='#e8f0d8', fs=8.5)
box(axa, 3.8, 1.0, 2.6, 1.4, '$R_{cap}=1/(2\\dot m c_p)$\n容积热阻')
box(axa, 0.2, 1.0, 2.6, 1.4, '冷却液入口 $T_{in}$\n(293 K, 1 g/s)', fc='#c8def2')
arrow(axa, 2.8, 8.5, 3.8, 8.5); arrow(axa, 6.4, 8.5, 7.2, 8.5)
arrow(axa, 8.5, 7.8, 8.5, 6.0)
arrow(axa, 7.2, 5.3, 6.4, 2.1); arrow(axa, 7.2, 3.3, 6.4, 1.9)
arrow(axa, 3.8, 1.7, 2.8, 1.7)
axa.set_title('(a) 串联热阻网络  $R=R_{cond}+R_{conv}+R_{cap}$', fontsize=11.5)

# (b) 压降链
chain = [('入口歧管沿程\n$\\propto \\bar u_{mf}/D_{h,mf}^2$', '#dbe8f4'),
         ('90°转弯/收缩\n$K_{turn}\\rho u_{mf}^2/2 \\propto 1/\\gamma^2$', '#f4e7c8'),
         ('微通道摩擦+发展段\n$fRe\\cdot\\mu u_{ch} L/2D_h^2$', '#dbe8f4'),
         ('针肋阵列绕流\n$N C_D(\\beta) \\rho u_{max}^2 \\beta/2$', '#f4d2c8'),
         ('出口歧管沿程\n+出口损失', '#dbe8f4')]
xs0 = 0.15
for k, (t, c) in enumerate(chain):
    box(axb, xs0 + k * 2.0, 4.2, 1.78, 2.4, t, fc=c, fs=8.0)
    if k:
        arrow(axb, xs0 + k * 2.0 - 0.22, 5.4, xs0 + k * 2.0, 5.4)
axb.text(5, 8.6, '$\\Delta P = \\Delta P_{mf,in} + \\Delta P_{turn} + \\Delta P_{ch} + \\Delta P_{pin} + \\Delta P_{mf,out}$',
         ha='center', fontsize=11.5)
axb.text(5, 1.8, '歧管加深(γ↑) → $u_{mf}\\downarrow$ → 转弯/歧管损失骤降；'
                 '针肋加粗加密(β,N↑) → 绕流阻力上升', ha='center', fontsize=9.5)
axb.set_title('(b) 沿程压降串联分解', fontsize=11.5)
fig.tight_layout()
save_pub(fig, FIG + '/p1_fig4_机理网络示意.png')
plt.close(fig)

# ---------- 图5：φ 的机理项分解 ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
c = CPHI
# 沿 N 分解（β=0.2, γ=4）
Ns = np.linspace(0, 10, 41)
mix = c[3] * (1 - np.exp(-Ns / TAU))
pen = c[4] * Ns
base = np.full_like(Ns, c[0] + c[1]*0.2 + c[2]*0.04 + c[5]/16 + c[6]*4)
ax1.plot(Ns, base + mix + pen, 'k-', lw=2, label='φ 合计')
ax1.plot(Ns, base, '--', color='grey', label='基底+β项+γ项')
ax1.plot(Ns, base + mix, '-', color='#2980b9', label='+ 混合受益 $(1-e^{-N/\\tau})$')
ax1.fill_between(Ns, base + mix, base + mix + pen, color='#c0392b', alpha=0.25,
                 label='热尾流叠加惩罚 ∝N')
sub = df[(df.beta == 0.2) & (df.gamma == 4.0)]
ax1.plot(sub['N'], sub['phi'], 'o', color='#1f3f66', label='数据(β=0.2,γ=4)')
ax1.set_xlabel('针肋排数 N'); ax1.set_ylabel('$\\phi$')
ax1.set_title('(a) φ 沿 N 的“受益—惩罚”竞争 → N≈4 最优', fontsize=10.5)
ax1.grid(alpha=0.25, ls=':'); ax1.legend(fontsize=8)

# 沿 β 分解（N=4, γ=4）
bs = np.linspace(0, 0.32, 65)
fbeta = c[1] * bs + c[2] * bs**2
base2 = c[0] + c[3]*(1-np.exp(-4/TAU)) + c[4]*4 + c[5]/16 + c[6]*4
ax2.plot(bs, base2 + fbeta, 'k-', lw=2, label='φ 合计')
ax2.plot(bs, base2 + c[1]*bs, '--', color='#2980b9', label='绕流混合受益 ∝β')
ax2.plot(bs, base2 + c[2]*bs**2, '--', color='#c0392b', label='尾流死区惩罚 ∝β²')
sub2 = df[(df.N == 4) & (df.gamma == 4.0)]
ax2.plot(sub2['beta'], sub2['phi'], 'o', color='#1f3f66', label='数据(N=4,γ=4)')
bstar = -c[1] / (2 * c[2])
ax2.axvline(bstar, color='green', ls=':', lw=1.5)
ax2.text(bstar + 0.005, base2 + 0.001, 'β*=%.2f' % bstar, color='green', fontsize=9)
ax2.set_xlabel('针肋宽度比 β'); ax2.set_ylabel('$\\phi$')
ax2.set_title('(b) φ 沿 β 的竞争 → β*≈0.20 最优', fontsize=10.5)
ax2.grid(alpha=0.25, ls=':'); ax2.legend(fontsize=8)
fig.tight_layout()
save_pub(fig, FIG + '/p1_fig5_phi机理分解.png')
plt.close(fig)

# ---------- 图6：灰箱模型消融审计 ----------
ablation_path = M.BASE + '/代码/p1_ablation_identifiability.csv'
if os.path.exists(ablation_path):
    ab = pd.read_csv(ablation_path, encoding='utf-8-sig')
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
    for ax, target in zip(axes, ['R', 'phi']):
        sub = ab[ab['target'] == target].copy()
        sub = sub.sort_values('BIC')
        colors = ['#c0392b' if m.startswith('full') else '#7f8c8d' for m in sub['model']]
        ax.barh(sub['model'], sub['BIC'], color=colors, alpha=0.85)
        ax.set_xlabel('BIC（越低越好）')
        ax.set_title('%s 子模型消融审计' % ({'R': '$R^*$', 'phi': '$\\phi$'}[target]))
        ax.grid(alpha=0.25, ls=':', axis='x')
    fig.tight_layout()
    save_pub(fig, FIG + '/p1_fig6_灰箱模型消融审计.png')
    plt.close(fig)

# ---------- 图7：参数稳定性森林图 ----------
stab_path = M.BASE + '/代码/p1_parameter_stability.csv'
if os.path.exists(stab_path):
    stab = pd.read_csv(stab_path, encoding='utf-8-sig')
    keep = stab[((stab['submodel'] == 'R') & (stab['parameter'].isin(['c_imp', 'eps0', 'a1', 'a2', 'n0', 'q']))) |
                ((stab['submodel'] == 'dP') & (stab['parameter'].isin(['K_turn', 'cD_scale', 'p_blk'])))]
    rows = []
    for (sub, par), g in keep.groupby(['submodel', 'parameter']):
        rows.append({'label': sub + ':' + par, 'mean': g['value'].mean(),
                     'lo': g['value'].quantile(0.025), 'hi': g['value'].quantile(0.975),
                     'bound': (g['at_lower_bound'].mean() > 0.5) or (g['at_upper_bound'].mean() > 0.5)})
    forest = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5.0))
    y = np.arange(len(forest))
    ax.errorbar(forest['mean'], y,
                xerr=[forest['mean'] - forest['lo'], forest['hi'] - forest['mean']],
                fmt='o', color='#2c3e50', ecolor='#7f8c8d', capsize=3)
    for yi, bound in zip(y, forest['bound']):
        if bound:
            ax.scatter(forest.loc[yi, 'mean'], yi, s=85, facecolors='none',
                       edgecolors='#c0392b', linewidths=1.8, label='边界参数' if yi == 0 else None)
    ax.set_yticks(y); ax.set_yticklabels(forest['label'])
    ax.set_xlabel('多起点估计值与 95% 分位区间')
    ax.set_title('关键参数可辨识性审计（红圈：多数初值收敛到边界）')
    ax.grid(alpha=0.25, ls=':', axis='x')
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:1], labels[:1], fontsize=8)
    fig.tight_layout()
    save_pub(fig, FIG + '/p1_fig7_参数稳定性森林图.png')
    plt.close(fig)

print('figures saved to', FIG)
