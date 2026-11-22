# -*- coding: utf-8 -*-
"""
问题3：多目标优化与综合最优设计
- 优化模型: min (R*, dP*, phi)  s.t. β∈[0.1,0.3], γ∈[3,4.5], N∈{2,4,6,8,10}
- 求解: ① 高分辨率网格近似覆盖 ② 按N分层的NSGA-II(方法学互证) → 合并后取非支配集
- 决策: 熵权-TOPSIS / 等权理想点 / Chebyshev(min-max) 三准则
- 验证: GPR 与 RSM-3 双代理交叉验证; 无肋基准支配性检验; 推荐点 GPR 置信区间
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from config import CODE_DIR, INPUTS, TARGETS as CFG_TARGETS, grid_specs

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 包根目录（相对路径，解压即可运行）
M = joblib.load(BASE + '/代码/p2_gpr_models.pkl')
scaler = M['scalerX']
TARGETS = ['R', 'dP', 'phi']

def predict(X, kind='gpr'):
    """X: (n,3) [beta,gamma,N] → (n,3) [R,dP,phi]"""
    X = np.atleast_2d(np.asarray(X, float))
    if kind == 'gpr':
        Xs = scaler.transform(X)
        return np.column_stack([M[t].predict(Xs) for t in TARGETS])
    return np.column_stack([M['rsm3_' + t].predict(X) for t in TARGETS])

def predict_std(X):
    X = np.atleast_2d(np.asarray(X, float))
    Xs = scaler.transform(X)
    out = {}
    for t in TARGETS:
        mu, sd = M[t].predict(Xs, return_std=True)
        out[t] = (mu, sd)
    return out

# ---------------- 非支配集提取 ----------------
def pareto_mask(F):
    """F: (n,3) 全为成本型 → bool掩码(非支配)"""
    n = len(F)
    mask = np.ones(n, bool)
    chunk = 512
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        block = F[s:e]                                # (c,3)
        # 任意 j 支配 i: 所有目标 <= 且至少一个 <
        le = (F[None, :, :] <= block[:, None, :]).all(-1)      # (c,n)
        lt = (F[None, :, :] < block[:, None, :]).any(-1)
        dominated = (le & lt).any(1)
        mask[s:e] &= ~dominated
    return mask

# ---------------- ① 网格枚举 ----------------
betas = np.round(np.arange(0.10, 0.3001, 0.0025), 4)     # 81
gammas = np.round(np.arange(3.0, 4.5001, 0.025), 3)      # 61
Ns = np.array([2, 4, 6, 8, 10], float)
G = np.array([(b, g, n) for n in Ns for g in gammas for b in betas])
F_grid = predict(G, 'gpr')
print('grid points:', len(G))

# ---------------- ② NSGA-II（按N分层, pymoo） ----------------
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pmoo_min

class LayerProb(Problem):
    def __init__(self, Nfix):
        super().__init__(n_var=2, n_obj=3, xl=[0.1, 3.0], xu=[0.3, 4.5])
        self.Nfix = Nfix
    def _evaluate(self, x, out, *a, **k):
        X = np.column_stack([x, np.full(len(x), self.Nfix)])
        out['F'] = predict(X, 'gpr')

nsga_X, nsga_F = [], []
for Nfix in Ns:
    res = pmoo_min(LayerProb(Nfix), NSGA2(pop_size=80), ('n_gen', 120),
                   seed=1, verbose=False)
    Xl = np.column_stack([res.X, np.full(len(res.X), Nfix)])
    nsga_X.append(Xl); nsga_F.append(res.F)
nsga_X = np.vstack(nsga_X); nsga_F = np.vstack(nsga_F)
print('nsga candidates:', len(nsga_X))

# ---------------- 合并 → 全局Pareto集 ----------------
ALL_X = np.vstack([G, nsga_X]); ALL_F = np.vstack([F_grid, nsga_F])
mk = pareto_mask(ALL_F)
PX, PF = ALL_X[mk], ALL_F[mk]
src = np.array(['grid'] * len(G) + ['nsga'] * len(nsga_X))[mk]
print('pareto size:', len(PX), ' from nsga:', (src == 'nsga').sum())

# ---------------- 决策准则 ----------------
def entropy_weights(F):
    """成本型→效益化后熵权"""
    B = (F.max(0) - F) / (F.max(0) - F.min(0) + 1e-12)
    P = B / (B.sum(0) + 1e-12)
    m = len(F)
    logP = np.log(np.where(P > 0, P, 1.0))
    e = -(P * logP).sum(0) / np.log(m)
    d = 1 - e
    return d / d.sum()

def topsis(F, w):
    Rn = F / np.sqrt((F**2).sum(0))
    V = Rn * w
    pis, nis = V.min(0), V.max(0)          # 成本型: 小=优
    Dp = np.sqrt(((V - pis)**2).sum(1))
    Dn = np.sqrt(((V - nis)**2).sum(1))
    return Dn / (Dp + Dn + 1e-12)

Z = (PF - PF.min(0)) / (PF.max(0) - PF.min(0) + 1e-12)   # front内min-max标准化

w_ent = entropy_weights(PF)
C = topsis(PF, w_ent)
i_topsis = int(np.argmax(C))
i_ideal = int(np.argmin(np.sqrt((Z**2).sum(1))))          # 等权理想点(欧氏)
i_cheby = int(np.argmin(Z.max(1)))                        # Chebyshev min-max

sel = {'熵权TOPSIS': i_topsis, '等权理想点': i_ideal, 'Chebyshev': i_cheby}
print('entropy weights:', np.round(w_ent, 4))
for k, i in sel.items():
    print('%-8s x=(%.4f, %.3f, %d)  F=(%.4f, %.4f, %.4f)' %
          (k, PX[i, 0], PX[i, 1], PX[i, 2], *PF[i]))

# 推荐设计 = 等权理想点解：
# ① 熵权反映"区分度"而非"重要性", R*在前沿上变化小被系统性降权, 不宜单独定推荐;
# ② 三准则在 β、γ 上一致, N∈{2,4,6} 中理想点解取中位 N=4, 且 φ 已达全局最优水平;
# ③ 该点被附件2样本64(0.2,4.5,4)直接锚定。TOPSIS/Chebyshev 解作为偏好敏感性区间(→问题4)。
i_rec = i_ideal
x_rec = PX[i_rec]

# ---------------- RSM-3 交叉验证 ----------------
F_grid_r = predict(G, 'rsm3')
mk_r = pareto_mask(F_grid_r)
PX_r, PF_r = G[mk_r], F_grid_r[mk_r]
Zr = (PF_r - PF_r.min(0)) / (PF_r.max(0) - PF_r.min(0) + 1e-12)
w_ent_r = entropy_weights(PF_r)
ir_topsis = int(np.argmax(topsis(PF_r, w_ent_r)))
ir_ideal = int(np.argmin(np.sqrt((Zr**2).sum(1))))
ir_cheby = int(np.argmin(Zr.max(1)))
print('--- RSM3 cross-check ---')
for k, i in [('熵权TOPSIS', ir_topsis), ('等权理想点', ir_ideal), ('Chebyshev', ir_cheby)]:
    print('%-8s x=(%.4f, %.3f, %d)  F=(%.4f, %.4f, %.4f)' %
          (k, PX_r[i, 0], PX_r[i, 1], PX_r[i, 2], *PF_r[i]))

# ---------------- 无肋基准支配性检验 ----------------
Gb = np.array([(0.0, g, 0.0) for g in gammas])
Fb = predict(Gb, 'gpr')
dom_count = 0
for fb in Fb:
    if ((PF <= fb).all(1) & (PF < fb).any(1)).any():
        dom_count += 1
print('baseline dominated: %d / %d' % (dom_count, len(Fb)))

# ---------------- 单指标最优（参照点） ----------------
singles = {}
for j, t in enumerate(TARGETS):
    i = int(np.argmin(ALL_F[:, j]))
    singles[t] = {'x': ALL_X[i].tolist(), 'F': ALL_F[i].tolist()}
    print('single-best %-3s x=(%.4f, %.3f, %d) F=(%.4f, %.4f, %.4f)' %
          (t, ALL_X[i, 0], ALL_X[i, 1], ALL_X[i, 2], *ALL_F[i]))

# ---------------- 推荐点不确定性 + 最近样本锚定 ----------------
ps = predict_std(x_rec)
rec_pred = {t: {'mu': round(float(ps[t][0][0]), 5),
                'sd': round(float(ps[t][1][0]), 5),
                'ci95': [round(float(ps[t][0][0] - 1.96 * ps[t][1][0]), 5),
                         round(float(ps[t][0][0] + 1.96 * ps[t][1][0]), 5)]}
            for t in TARGETS}
df = pd.read_csv(BASE + '/附件2_data.tsv', sep='\t', skiprows=2, header=None,
                 names=['id', 'beta', 'gamma', 'N', 'R', 'dP', 'phi'],
                 encoding='utf-8-sig').dropna()
dn = np.abs(df[['beta', 'gamma', 'N']].values - x_rec) / np.array([0.2, 1.5, 8])
i_near = int(np.argmin(dn.sum(1)))
near = df.iloc[i_near]

# ---------------- 保存 ----------------
pd.DataFrame(np.column_stack([PX, PF]),
             columns=['beta', 'gamma', 'N', 'R', 'dP', 'phi']).to_csv(
    BASE + '/代码/p3_pareto_front.csv', index=False, encoding='utf-8-sig')

out = {
    'entropy_weights': dict(zip(TARGETS, np.round(w_ent, 4).tolist())),
    'pareto_size': int(len(PX)),
    'selected': {k: {'x': PX[i].tolist(), 'F': PF[i].tolist(),
                     'topsis_C': round(float(C[i]), 4)} for k, i in sel.items()},
    'recommended': {'x': x_rec.tolist(), 'pred': rec_pred},
    'rsm3_selected': {k: {'x': PX_r[i].tolist(), 'F': PF_r[i].tolist()}
                      for k, i in [('熵权TOPSIS', ir_topsis), ('等权理想点', ir_ideal),
                                   ('Chebyshev', ir_cheby)]},
    'baseline_dominated': '%d/%d' % (dom_count, len(Fb)),
    'single_best': singles,
    'nearest_sample': {'id': int(near['id']),
                       'x': near[['beta', 'gamma', 'N']].tolist(),
                       'F': near[['R', 'dP', 'phi']].tolist()},
}
with open(BASE + '/代码/p3_results.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('saved p3 results')

# ---------------- 审计增强：双代理差异、网格收敛、局部精修 ----------------
F_rsm3_on_pareto = predict(PX, 'rsm3')
dual = pd.DataFrame(PX, columns=['beta', 'gamma', 'N'])
for j, t in enumerate(TARGETS):
    dual[t + '_gpr'] = PF[:, j]
    dual[t + '_rsm3'] = F_rsm3_on_pareto[:, j]
    dual[t + '_abs_diff'] = np.abs(PF[:, j] - F_rsm3_on_pareto[:, j])
dual['source'] = 'p3_pareto_candidates'
dual.to_csv(BASE + '/代码/dual_proxy_check.csv', index=False, encoding='utf-8-sig')

def run_grid(beta_step, gamma_step, kind='gpr'):
    bs = np.round(np.arange(0.10, 0.3000001, beta_step), 6)
    gs = np.round(np.arange(3.0, 4.5000001, gamma_step), 6)
    ns = np.array([2, 4, 6, 8, 10], float)
    Xg = np.array([(b, g, n) for n in ns for g in gs for b in bs])
    Fg = predict(Xg, kind)
    mk = pareto_mask(Fg)
    Xp, Fp = Xg[mk], Fg[mk]
    Zp = (Fp - Fp.min(0)) / (Fp.max(0) - Fp.min(0) + 1e-12)
    w = entropy_weights(Fp)
    it = int(np.argmax(topsis(Fp, w)))
    ii = int(np.argmin(np.sqrt((Zp**2).sum(1))))
    ic = int(np.argmin(Zp.max(1)))
    return Xg, Fg, Xp, Fp, {'topsis': it, 'ideal': ii, 'cheby': ic}, w

grid_records = []
fronts = {}
for spec in grid_specs:
    Xg, Fg, Xp, Fp, idxs, w = run_grid(spec['beta_step'], spec['gamma_step'])
    fronts[spec['name']] = (Xp, Fp)
    rec = {
        'grid': spec['name'],
        'beta_step': spec['beta_step'],
        'gamma_step': spec['gamma_step'],
        'n_grid_points': int(len(Xg)),
        'n_pareto': int(len(Xp)),
        'entropy_w_R': float(w[0]),
        'entropy_w_dP': float(w[1]),
        'entropy_w_phi': float(w[2]),
    }
    for label, idx in idxs.items():
        rec[label + '_beta'] = float(Xp[idx, 0])
        rec[label + '_gamma'] = float(Xp[idx, 1])
        rec[label + '_N'] = int(Xp[idx, 2])
        rec[label + '_R'] = float(Fp[idx, 0])
        rec[label + '_dP'] = float(Fp[idx, 1])
        rec[label + '_phi'] = float(Fp[idx, 2])
    grid_records.append(rec)

current_F = fronts['current'][1]
current_tree = cKDTree((current_F - current_F.min(0)) / (current_F.max(0) - current_F.min(0) + 1e-12))
for rec in grid_records:
    Fp = fronts[rec['grid']][1]
    Fpn = (Fp - current_F.min(0)) / (current_F.max(0) - current_F.min(0) + 1e-12)
    dist, _ = current_tree.query(Fpn, k=1)
    rec['front_mean_nearest_distance_to_current'] = float(np.mean(dist))
    rec['front_max_nearest_distance_to_current'] = float(np.max(dist))
pd.DataFrame(grid_records).to_csv(
    BASE + '/代码/p3_grid_convergence.csv', index=False, encoding='utf-8-sig')

# Local SLSQP refinement with Chebyshev scalarization for each fixed N.
Fmin, Fmax = PF.min(0), PF.max(0)
scale = Fmax - Fmin + 1e-12

def cheby_obj(bg, Nfix):
    x = np.array([[bg[0], bg[1], Nfix]])
    f = predict(x, 'gpr')[0]
    return float(np.max((f - Fmin) / scale))

ref_rows = []
for Nfix in [2, 4, 6, 8, 10]:
    idx = np.where(PX[:, 2] == Nfix)[0]
    if len(idx) == 0:
        continue
    # Use evenly spaced Pareto starts for this N plus the global recommended point if applicable.
    starts_idx = idx[np.linspace(0, len(idx) - 1, min(12, len(idx))).astype(int)]
    best = None
    for si in starts_idx:
        x0 = PX[si, :2]
        res = minimize(
            lambda z: cheby_obj(z, Nfix),
            x0=x0,
            method='SLSQP',
            bounds=[(0.1, 0.3), (3.0, 4.5)],
            options={'maxiter': 200, 'ftol': 1e-12},
        )
        f_ref = predict([[res.x[0], res.x[1], Nfix]], 'gpr')[0]
        row = {
            'N': int(Nfix),
            'start_beta': float(x0[0]),
            'start_gamma': float(x0[1]),
            'refined_beta': float(res.x[0]),
            'refined_gamma': float(res.x[1]),
            'success': bool(res.success),
            'objective': float(res.fun),
            'R': float(f_ref[0]),
            'dP': float(f_ref[1]),
            'phi': float(f_ref[2]),
        }
        ref_rows.append(row)
        if best is None or row['objective'] < best['objective']:
            best = row
    if best:
        print('local refine N=%d best beta=%.5f gamma=%.5f obj=%.5f' %
              (Nfix, best['refined_beta'], best['refined_gamma'], best['objective']))

pd.DataFrame(ref_rows).to_csv(
    BASE + '/代码/p3_local_refinement.csv', index=False, encoding='utf-8-sig')
