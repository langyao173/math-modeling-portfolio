# -*- coding: utf-8 -*-
"""
问题4：指标权重敏感性分析与鲁棒设计
- 权重空间: 单纯形 {w1+w2+w3=1, wi>=0}, 步长0.01 → 5151组权重
- 对P3的888个Pareto解: 加权得分 S(x,w)=w·Z(x) (Z为前沿min-max标准化)
- 敏感性: 权重→最优设计映射(三角图分区), 设计随权重的迁移统计
- 鲁棒设计: ① min-max regret ② ε-近优覆盖率, 两准则联合判定
- Chebyshev 加权变体作为标量化方式稳健性检验; 典型场景分析
"""
import os
import json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 包根目录（相对路径，解压即可运行）
pf = pd.read_csv(BASE + '/代码/p3_pareto_front.csv', encoding='utf-8-sig')
X = pf[['beta', 'gamma', 'N']].values
F = pf[['R', 'dP', 'phi']].values
Z = (F - F.min(0)) / (F.max(0) - F.min(0))            # 前沿min-max标准化
n = len(Z)

# ---------------- 权重单纯形网格 ----------------
step = 0.01
ks = np.arange(0, 101)
W = np.array([(i, j, 100 - i - j) for i in ks for j in ks if i + j <= 100], float) / 100
m = len(W)
print('weights:', m, 'front:', n)

# ---------------- 加权得分与最优设计映射 ----------------
S = Z @ W.T                       # (n, m)
best_idx = S.argmin(0)            # 每组权重的最优解索引
S_min = S.min(0)

# 唯一被选中的设计（聚类标识: beta按0.005、gamma按0.05舍入, N精确）
key = np.column_stack([np.round(X[:, 0] / 0.005) * 0.005,
                       np.round(X[:, 1] / 0.05) * 0.05, X[:, 2]])
sel_keys, sel_counts = np.unique(key[best_idx], axis=0, return_counts=True)
order = np.argsort(-sel_counts)
print('--- 被选为加权最优的设计簇(前12, 按占据权重面积排序) ---')
for k in order[:12]:
    b, g, N = sel_keys[k]
    print('  (%.3f, %.2f, %2d)  share=%.1f%%' % (b, g, N, 100 * sel_counts[k] / m))

# N 的占比
selN = X[best_idx, 2]
shareN = {int(v): round(100 * np.mean(selN == v), 1) for v in [2, 4, 6, 8, 10]}
print('share by N:', shareN)

# ---------------- regret 分析与鲁棒设计 ----------------
regret = S - S_min[None, :]       # (n, m)
max_regret = regret.max(1)
i_minimax = int(np.argmin(max_regret))

EPS = 0.05                        # ε=标准化得分5个百分点
coverage = (regret <= EPS).mean(1)
i_cover = int(np.argmax(coverage))

print('--- 鲁棒准则 ---')
print('min-max regret: x=(%.4f, %.3f, %d)  maxRegret=%.4f  coverage(5%%)=%.1f%%'
      % (X[i_minimax, 0], X[i_minimax, 1], X[i_minimax, 2],
         max_regret[i_minimax], 100 * coverage[i_minimax]))
print('max ε-coverage: x=(%.4f, %.3f, %d)  maxRegret=%.4f  coverage(5%%)=%.1f%%'
      % (X[i_cover, 0], X[i_cover, 1], X[i_cover, 2],
         max_regret[i_cover], 100 * coverage[i_cover]))

# 与问题3推荐方案对比
i_p3 = int(np.argmin(np.abs(X - np.array([0.22, 4.5, 4])).sum(1)))
print('P3推荐解   : x=(%.4f, %.3f, %d)  maxRegret=%.4f  coverage(5%%)=%.1f%%'
      % (X[i_p3, 0], X[i_p3, 1], X[i_p3, 2], max_regret[i_p3], 100 * coverage[i_p3]))

# ---------------- 现实偏好域（任何指标不被完全忽略: wi>=0.1） ----------------
mask_r = (W >= 0.1 - 1e-9).all(1)
regret_r = regret[:, mask_r]
i_minimax_r = int(np.argmin(regret_r.max(1)))
coverage_r = (regret_r <= EPS).mean(1)
i_cover_r = int(np.argmax(coverage_r))
print('--- 现实偏好域 wi>=0.1 (%d组权重) ---' % mask_r.sum())
print('min-max regret: x=(%.4f, %.3f, %d)  maxRegret=%.4f  coverage(5%%)=%.1f%%'
      % (X[i_minimax_r, 0], X[i_minimax_r, 1], X[i_minimax_r, 2],
         regret_r[i_minimax_r].max(), 100 * coverage_r[i_minimax_r]))
print('max ε-coverage: x=(%.4f, %.3f, %d)  maxRegret=%.4f  coverage(5%%)=%.1f%%'
      % (X[i_cover_r, 0], X[i_cover_r, 1], X[i_cover_r, 2],
         regret_r[i_cover_r].max(), 100 * coverage_r[i_cover_r]))
print('P3推荐解      : maxRegret=%.4f  coverage(5%%)=%.1f%%'
      % (regret_r[i_p3].max(), 100 * coverage_r[i_p3]))

# ---------------- Chebyshev 加权变体（标量化稳健性检验） ----------------
Wc = np.maximum(W, 1e-9)
# 逐块计算 max_j(w_j * Z_ij)
Sc = np.empty((n, m))
chunk = 600
for s in range(0, m, chunk):
    e = min(s + chunk, m)
    Sc[:, s:e] = (Z[:, None, :] * Wc[None, s:e, :]).max(-1)
regret_c = Sc - Sc.min(0)[None, :]
i_minimax_c = int(np.argmin(regret_c.max(1)))
print('Chebyshev变体 min-max regret: x=(%.4f, %.3f, %d)'
      % (X[i_minimax_c, 0], X[i_minimax_c, 1], X[i_minimax_c, 2]))

# ---------------- 典型场景分析 ----------------
scenarios = {
    '高性能计算/射频(重热阻)': [0.6, 0.2, 0.2],
    '数据中心(重能耗/压降)': [0.2, 0.6, 0.2],
    '车规功率器件(重均温/可靠性)': [0.2, 0.2, 0.6],
    '均衡场景': [1 / 3, 1 / 3, 1 / 3],
}
# 鲁棒方案 = 现实偏好域内两准则共同指向的设计（取覆盖率最优者）
i_robust = i_cover_r
x_robust = X[i_robust]
print('==> 最终鲁棒方案: x=(%.4f, %.3f, %d)' % (x_robust[0], x_robust[1], x_robust[2]))

# 不同ε下的覆盖率曲线（全单纯形; 鲁棒解、全域minimax解、各N代表解）
eps_grid = np.linspace(0.0, 0.2, 81)
cover_curves = {}
reps = {'robust': i_robust, 'minimax_full': i_minimax}
for Nv in [2, 10]:
    idx = np.where(X[:, 2] == Nv)[0]
    reps['N=%d' % Nv] = int(idx[np.argmin(max_regret[idx])])
for name, i in reps.items():
    cover_curves[name] = [(float(e), float((regret[i] <= e).mean())) for e in eps_grid]

scen_out = {}
for name, w in scenarios.items():
    w = np.array(w)
    s = Z @ w
    i = int(np.argmin(s))
    rob_regret = float(Z[i_robust] @ w - s[i])
    scen_out[name] = {'w': w.round(3).tolist(),
                      'opt_x': X[i].tolist(), 'opt_F': F[i].tolist(),
                      'robust_regret': round(rob_regret, 4),
                      'robust_regret_pct_of_range': round(100 * rob_regret, 2)}
    print('%-16s opt=(%.3f, %.2f, %d)  robust_regret=%.4f'
          % (name, X[i, 0], X[i, 1], X[i, 2], rob_regret))

# ---------------- 审计输出：min-max 与 coverage 准则分开呈现 ----------------
candidate_indices = {
    'x_minmax_restricted': i_minimax_r,
    'x_cover_restricted_final': i_cover_r,
    'x_p3_recommended': i_p3,
    'x_minmax_full': i_minimax,
    'x_cover_full': i_cover,
}
for t in range(3):
    idx = int(np.argmin(F[:, t]))
    candidate_indices['single_best_' + ['R', 'dP', 'phi'][t]] = idx

trade_rows = []
for label, idx in candidate_indices.items():
    row = {
        'candidate': label,
        'beta': float(X[idx, 0]),
        'gamma': float(X[idx, 1]),
        'N': int(X[idx, 2]),
        'R': float(F[idx, 0]),
        'dP': float(F[idx, 1]),
        'phi': float(F[idx, 2]),
        'max_regret_full': float(max_regret[idx]),
        'mean_regret_full': float(regret[idx].mean()),
        'coverage5_full_pct': float(100 * coverage[idx]),
        'max_regret_restricted': float(regret_r[idx].max()),
        'mean_regret_restricted': float(regret_r[idx].mean()),
        'coverage5_restricted_pct': float(100 * coverage_r[idx]),
    }
    for scen_name, scen in scenarios.items():
        w = np.array(scen)
        s = Z @ w
        row['scenario_regret_' + scen_name] = float(Z[idx] @ w - s.min())
    trade_rows.append(row)
pd.DataFrame(trade_rows).to_csv(
    BASE + '/代码/p4_robust_tradeoff.csv', index=False, encoding='utf-8-sig')

# ---------------- 保存 ----------------
np.save(BASE + '/代码/p4_W.npy', W)
np.save(BASE + '/代码/p4_best_idx.npy', best_idx)
np.save(BASE + '/代码/p4_max_regret.npy', max_regret)
np.save(BASE + '/代码/p4_coverage.npy', coverage)

out = {
    'n_weights': int(m), 'eps': EPS,
    'share_by_N': shareN,
    'top_clusters': [{'x': sel_keys[k].tolist(),
                      'share_pct': round(100 * sel_counts[k] / m, 2)} for k in order[:12]],
    'robust_minimax': {'x': X[i_minimax].tolist(), 'F': F[i_minimax].tolist(),
                       'max_regret': round(float(max_regret[i_minimax]), 4),
                       'coverage5_pct': round(100 * float(coverage[i_minimax]), 1)},
    'robust_coverage': {'x': X[i_cover].tolist(), 'F': F[i_cover].tolist(),
                        'max_regret': round(float(max_regret[i_cover]), 4),
                        'coverage5_pct': round(100 * float(coverage[i_cover]), 1)},
    'p3_design': {'x': X[i_p3].tolist(), 'max_regret': round(float(max_regret[i_p3]), 4),
                  'coverage5_pct': round(100 * float(coverage[i_p3]), 1)},
    'restricted_domain': {
        'n_weights': int(mask_r.sum()),
        'decision_note': 'min-max regret and epsilon-coverage select different gamma values; final robust design is coverage-dominant.',
        'minimax': {'x': X[i_minimax_r].tolist(),
                    'max_regret': round(float(regret_r[i_minimax_r].max()), 4),
                    'coverage5_pct': round(100 * float(coverage_r[i_minimax_r]), 1)},
        'coverage': {'x': X[i_cover_r].tolist(),
                     'max_regret': round(float(regret_r[i_cover_r].max()), 4),
                     'coverage5_pct': round(100 * float(coverage_r[i_cover_r]), 1)},
        'p3': {'max_regret': round(float(regret_r[i_p3].max()), 4),
               'coverage5_pct': round(100 * float(coverage_r[i_p3]), 1)}},
    'final_robust': {'x': x_robust.tolist(), 'F': F[i_robust].tolist()},
    'chebyshev_minimax_x': X[i_minimax_c].tolist(),
    'scenarios': scen_out,
    'cover_curves': cover_curves,
    'indices': {'robust': int(i_robust), 'minimax_full': int(i_minimax),
                'cover_full': int(i_cover), 'p3': int(i_p3)},
}
with open(BASE + '/代码/p4_results.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('saved p4 results')
