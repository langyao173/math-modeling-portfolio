# -*- coding: utf-8 -*-
"""
问题1：芯片歧管式微通道热管理系统 机理模型（灰箱）
- 热阻：R = R_cond(导热+扩散) + R_conv(壁面+针肋, 含入口段与冲击增强) + R_cap(容积热阻)
- 压降：ΔP = 歧管沿程 + 转弯局部 + 微通道摩擦(含发展段) + 针肋阵列绕流
- 非均匀性：φ = 基底扩散项 + 入口段换热不均(随N衰减) + 尾流热斑(βN) + 歧管分配不均(1/γ²) + 对流温差项
模型结构与指数来自机理；少量 O(1) 有效系数由附件2数据标定（灰箱建模）。
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 包根目录（相对路径，解压即可运行）

# ---------------- 1. 几何与物性（附件1 + 假设） ----------------
# 物性（附件1）
rho, cp, kf, mu = 998.2, 4182.0, 0.6, 1.003e-3     # 去离子水
ks = 200.0                                          # 氮化铝
Pr = mu * cp / kf

# 工况（附件1）
mdot_total = 1e-3        # kg/s
Q = 5e9 * (6e-3 * 6e-3 * 200e-6)   # = 36 W 体热源总功率

# 几何（附件1图A1/A2 + 合理假设，论文假设表中说明）
L_chip = 10e-3           # 芯片散热区边长
M_units = 5              # 歧管单元数（图A1五个条带）
t_cap = 200e-6           # 通道顶面到芯片的氮化铝盖层厚度（图A2: h_mc+200μm）
L_seg = 800e-6           # 单元内微通道流动段长度（图A1标注800μm）
w_c, w_w = 200e-6, 200e-6  # 微通道宽度/壁厚（假设: 间距400μm，与图A1约19条/带一致）
h_mc = 400e-6            # 微通道层高度（假设）
w_mf = 1.6e-3            # 歧管通道宽度（图A2出口宽1.6mm）
n_ch = 19                # 每个歧管单元的并行微通道数 ≈ 7.6mm/0.4mm
n_seg = M_units * n_ch * 2          # 并联流动段总数（每通道分2段, 进中间出两端）
mdot_seg = mdot_total / n_seg       # 单段流量

# 微通道内的流动特征（与设计变量无关的部分）
Dh = 2 * w_c * h_mc / (w_c + h_mc)
u_ch = mdot_seg / (rho * w_c * h_mc)
Re_ch = rho * u_ch * Dh / mu

def fRe_rect(alpha):
    """矩形通道层流 Darcy fRe（Shah-London 多项式, alpha=短边/长边）"""
    a = alpha
    return 4 * 24 * (1 - 1.3553*a + 1.9467*a**2 - 1.7012*a**3 + 0.9564*a**4 - 0.2537*a**5)

def Nu_developing(L, Re, Nu_inf):
    """Hausen 热入口段平均 Nu"""
    Gz = Dh * Re * Pr / L
    return Nu_inf + 0.0668 * Gz / (1 + 0.04 * Gz**(2/3))

Nu_inf_ch = 4.12          # 矩形通道 α=0.5, H1 充分发展 Nu
fRe_ch = fRe_rect(w_c / h_mc)

# 与设计无关的固定热阻
R_cap = 1 / (2 * mdot_total * cp)                 # 容积热阻（出入口平均温升）
A_src = 6e-3 * 6e-3
R_1d = t_cap / (ks * A_src)                       # 盖层一维导热
# 扩散热阻（6×6→10×10, Lee等近似式的简化形式，固定值）
r1, r2 = np.sqrt(A_src/np.pi), np.sqrt(L_chip**2/np.pi)
eps_sp = r1 / r2
R_spread = (1 - eps_sp)**1.5 / (np.pi**0.5 * ks * r1)   # ≈ 集中扩散阻（Song/Lee简式）
R_cond = R_1d + R_spread

def manifold_velocity(gamma):
    """歧管内特征流速（单元流量 / 歧管截面）"""
    h_mf = gamma * h_mc
    return (mdot_total / M_units) / (rho * w_mf * h_mf), h_mf

# ---------------- 2. 三个机理子模型 ----------------
def crowding(beta, N, a_w):
    """密排尾流拥挤因子 ∈[0,1)：排间距/直径 s_L/d 越小、拥挤越严重"""
    if N <= 0 or beta <= 0:
        return 0.0
    s_over_d = (L_seg / N) / (beta * w_c)
    return float(np.exp(-a_w * np.maximum(s_over_d - 1.0, 0.0)))

def R_conv_model(beta, gamma, N, c_imp, eps0, a_w, a1, a2, n0, q):
    """对流热阻：平直壁(入口段Nu+冲击增强+针肋诱导壁面修正ψ) + 针肋(Žukauskas+尾流遮蔽)
    ψ(β,N)=1+(a1β+a2β²)(1-e^{-(N/n0)^q}): 间隙加速强化(∝β) 与 阻塞回流/死区损失(∝β²) 竞争,
    随排数S形启动并饱和（少排扰动被层流边界层耗散, 多排尾流协同累积）"""
    u_mf, _ = manifold_velocity(gamma)
    F_imp = 1 + c_imp * (u_mf / u_ch)                 # 转弯冲击/射流增强因子
    Nu_w = Nu_developing(L_seg, Re_ch, Nu_inf_ch)
    h_w = F_imp * Nu_w * kf / Dh
    m_f = np.sqrt(2 * h_w / (ks * w_w))
    eta_f = np.tanh(m_f * h_mc) / (m_f * h_mc)        # 侧壁肋效率
    A_wall = (2 * h_mc * eta_f + 2 * w_c) * L_seg     # 两侧壁 + 顶/底面
    hA = h_w * A_wall
    if N > 0 and beta > 0:
        d = beta * w_c
        u_max = u_ch / (1 - beta)                     # 阻塞加速
        Re_d = rho * u_max * d / mu
        Nu_d = 1.04 * Re_d**0.4 * Pr**0.36            # Žukauskas 低Re管束式
        cw = crowding(beta, N, a_w)
        eps_wake = eps0 * (1 - cw)                    # 热尾流遮蔽有效系数
        psi = 1 + (a1 * beta + a2 * beta**2) * (1 - np.exp(-(N / n0)**q))
        hA = psi * h_w * A_wall + eps_wake * (Nu_d * kf / d) * (N * np.pi * d * h_mc)
    return 1 / (n_seg * hA)

def R_model(X, theta):
    c_imp, eps0, a_w, a1, a2, n0, q, R_ref = theta
    out = [ (R_cond + R_cap + R_conv_model(b, g, N, c_imp, eps0, a_w, a1, a2, n0, q)) / R_ref
            for b, g, N in X ]
    return np.array(out)

def dP_model(X, theta):
    """ΔP = 歧管沿程(×2) + 转弯局部 + 微通道摩擦+发展段 + 针肋绕流(含阻塞放大)"""
    K_turn, cD, p_blk, dP_ref = theta
    out = []
    for b, g, N in X:
        u_mf, h_mf = manifold_velocity(g)
        Dh_mf = 2 * w_mf * h_mf / (w_mf + h_mf)
        alpha_mf = min(w_mf, h_mf) / max(w_mf, h_mf)
        # 歧管沿程（沿程分流, 平均速度取 u_mf/2; 进出歧管各一段）
        dp_mf = 2 * fRe_rect(alpha_mf) * mu * (0.5*u_mf) * L_chip / (2 * Dh_mf**2)
        # 90°转弯+收缩/扩张（动压尺度: 歧管流速）
        dp_turn = K_turn * 0.5 * rho * u_mf**2
        # 微通道摩擦 + Hagenbach 发展段附加
        dp_ch = fRe_ch * mu * u_ch * L_seg / (2 * Dh**2) + 1.2 * 0.5 * rho * u_ch**2
        # 针肋阵列绕流（单排单肋, 阻塞比=β; 受限圆柱阻力随阻塞强烈放大）
        dp_pin = 0.0
        if N > 0 and b > 0:
            d = b * w_c
            u_max = u_ch / (1 - b)
            Re_d = rho * u_max * d / mu
            CD = cD * (10.0 / Re_d**0.8 + 1.0) / (1 - b)**p_blk
            dp_pin = N * CD * 0.5 * rho * u_max**2 * b
        out.append((dp_mf + dp_turn + dp_ch + dp_pin) / dP_ref)
    return np.array(out)

def phi_features(X, tau):
    """φ 的机理特征（近似可分离结构, 由数据交互核对支持）:
    [1, β(绕流混合受益), β²(尾流死区∝d²), 1-exp(-N/τ)(边界层重启受益,饱和),
     N(热尾流叠加惩罚), 1/γ²(歧管动压分配不均), γ(歧管加深致冲击减弱)]"""
    rows = []
    for b, g, N in X:
        mix_N = 1 - np.exp(-N / tau)
        rows.append([1.0, b, b * b, mix_N, float(N), 1.0 / g**2, g])
    return np.array(rows)

# ---------------- 3. 读数据并标定 ----------------
df = pd.read_csv(BASE + '/附件2_data.tsv', sep='\t', skiprows=2, header=None,
                 names=['id', 'beta', 'gamma', 'N', 'R', 'dP', 'phi'],
                 encoding='utf-8-sig').dropna()
X = df[['beta', 'gamma', 'N']].values
yR, ydP, yphi = df['R'].values, df['dP'].values, df['phi'].values

# --- 标定热阻模型(多起点): theta = [c_imp, eps0, a_w, a1, a2, n0, q, R_ref] ---
boundsR = ([0, 0.005, 0.05, 0.0, -50.0, 0.5, 1.0, 0.3], [3, 1, 10, 20, 0.0, 20, 4, 3])
bestR = None
for x0 in [[0.3, 0.2, 1.0, 2.0, -5.0, 3.0, 2.0, 1.0],
           [0.1, 0.05, 4.0, 1.0, -2.5, 5.0, 1.5, 1.0],
           [0.2, 0.02, 2.0, 4.0, -10.0, 2.0, 3.0, 0.9],
           [0.5, 0.1, 1.0, 0.5, -1.0, 8.0, 1.0, 1.2],
           [0.1, 0.03, 4.5, 8.0, -20.0, 4.0, 2.5, 0.95]]:
    r = least_squares(lambda t: R_model(X, t) - yR, x0=x0, bounds=boundsR)
    if bestR is None or r.cost < bestR.cost:
        bestR = r
resR = bestR
C_IMP_FIT, EPS0_FIT, AW_FIT, A1_FIT, A2_FIT, N0_FIT, Q_FIT, RREF_FIT = resR.x
predR = R_model(X, resR.x)

# --- 标定压降模型: theta = [K_turn, cD, p_blk, dP_ref] ---
resP = least_squares(lambda t: dP_model(X, t) - ydP,
                     x0=[1.5, 1.0, 2.0, 500.0],
                     bounds=([0.1, 0.05, 0.0, 50], [30, 20, 8, 5000]))
predP = dP_model(X, resP.x)

# --- 标定非均匀性模型: τ 网格 + 线性最小二乘 ---
best = None
for tau in np.arange(0.3, 6.05, 0.05):
    F = phi_features(X, tau)
    c, *_ = np.linalg.lstsq(F, yphi, rcond=None)
    sse = np.sum((F @ c - yphi)**2)
    if best is None or sse < best[0]:
        best = (sse, tau, c)
_, TAU_FIT, CPHI = best
predPhi = phi_features(X, TAU_FIT) @ CPHI

def r2(y, p):
    return 1 - np.sum((y - p)**2) / np.sum((y - np.mean(y))**2)

stats = {}
for name, y, p in [('R', yR, predR), ('dP', ydP, predP), ('phi', yphi, predPhi)]:
    stats[name] = {'R2': round(float(r2(y, p)), 4),
                   'spearman': round(float(spearmanr(y, p)[0]), 4),
                   'rmse': round(float(np.sqrt(np.mean((y-p)**2))), 5),
                   'max_rel_err_pct': round(float(np.max(np.abs(y-p)/y))*100, 2)}

params = {
    'geometry_assumed': {'M_units': M_units, 'n_ch': n_ch, 'w_c_um': 200, 'w_w_um': 200,
                         'h_mc_um': 400, 'L_seg_um': 800, 'w_mf_mm': 1.6, 't_cap_um': 200},
    'flow_state': {'u_ch_m_s': round(float(u_ch), 4), 'Re_ch': round(float(Re_ch), 2),
                   'Pr': round(float(Pr), 2), 'Q_W': Q,
                   'R_cap_K_W': round(float(R_cap), 4), 'R_cond_K_W': round(float(R_cond), 4)},
    'R_fit': {'c_imp': round(float(C_IMP_FIT), 4), 'eps0': round(float(EPS0_FIT), 4),
              'a_w': round(float(AW_FIT), 4), 'a1': round(float(A1_FIT), 4),
              'a2': round(float(A2_FIT), 4), 'n0': round(float(N0_FIT), 4),
              'q': round(float(Q_FIT), 4), 'R_ref_K_W': round(float(RREF_FIT), 4)},
    'dP_fit': {'K_turn': round(float(resP.x[0]), 4), 'cD_scale': round(float(resP.x[1]), 4),
               'p_blk': round(float(resP.x[2]), 4), 'dP_ref_Pa': round(float(resP.x[3]), 2)},
    'phi_fit': {'tau': float(round(TAU_FIT, 3)),
                'coef': [round(float(c), 5) for c in CPHI],
                'features': ['1', 'beta', 'beta^2', '1-exp(-N/tau)', 'N', '1/gamma^2', 'gamma']},
    'goodness': stats,
}
with open(BASE + '/代码/p1_fit_results.json', 'w', encoding='utf-8') as f:
    json.dump(params, f, ensure_ascii=False, indent=2)

df_out = df.copy()
df_out['R_pred'], df_out['dP_pred'], df_out['phi_pred'] = predR, predP, predPhi
df_out.to_csv(BASE + '/代码/p1_predictions.csv', index=False, encoding='utf-8-sig')

# ---------------- 4. 审计输出：误差、相关性与物性假设 ----------------
audit = df.copy()
summary = {}
for name, pred in [('R', predR), ('dP', predP), ('phi', predPhi)]:
    err = pred - audit[name].values
    abs_err = np.abs(err)
    rel_err = abs_err / audit[name].values * 100
    audit[name + '_pred'] = pred
    audit[name + '_abs_error'] = abs_err
    audit[name + '_rel_error_pct'] = rel_err
    audit[name + '_outside_3pct'] = rel_err > 3
    audit[name + '_outside_5pct'] = rel_err > 5
    summary[name] = {
        'MAE': round(float(abs_err.mean()), 8),
        'RMSE': round(float(np.sqrt(np.mean(err**2))), 8),
        'MaxRE_pct': round(float(rel_err.max()), 3),
        'outside_3pct_count': int((rel_err > 3).sum()),
        'outside_5pct_count': int((rel_err > 5).sum()),
        'outside_3pct_ids': audit.loc[rel_err > 3, 'id'].astype(int).tolist(),
        'outside_5pct_ids': audit.loc[rel_err > 5, 'id'].astype(int).tolist(),
    }
audit.to_csv(BASE + '/代码/p1_error_audit.csv', index=False, encoding='utf-8-sig')
with open(BASE + '/代码/p1_error_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])

corr = {}
for a, b in [('R', 'dP'), ('R', 'phi'), ('dP', 'phi')]:
    corr[a + '_' + b] = {
        'Pearson_r': round(pearson(df[a].values, df[b].values), 6),
        'Spearman_rho_s': round(float(spearmanr(df[a].values, df[b].values)[0]), 6),
    }
with open(BASE + '/代码/p1_correlation_audit.json', 'w', encoding='utf-8') as f:
    json.dump(corr, f, ensure_ascii=False, indent=2)

def rho_water(Tc):
    """Fresh-water density correlation near room temperature, kg/m^3."""
    return 1000 * (1 - ((Tc + 288.9414) / (508929.2 * (Tc + 68.12963))) * (Tc - 3.9863)**2)

def mu_water(Tk):
    """Common dynamic-viscosity correlation for liquid water, Pa*s."""
    return 2.414e-5 * 10**(247.8 / (Tk - 140))

def k_water_approx(Tc):
    """Simple room-temperature water thermal-conductivity approximation, W/(m*K)."""
    return 0.561 + 0.0019 * Tc - 6.5e-6 * Tc**2

Tin_K = 293.0
dT = Q / (mdot_total * cp)
Tout_K = Tin_K + dT
Tin_C, Tout_C = Tin_K - 273.15, Tout_K - 273.15
rho_in, rho_out = rho_water(Tin_C), rho_water(Tout_C)
mu_in, mu_out = mu_water(Tin_K), mu_water(Tout_K)
k_in, k_out = k_water_approx(Tin_C), k_water_approx(Tout_C)
property_assumption = {
    'Tin_K': round(float(Tin_K), 4),
    'Tout_K_estimated': round(float(Tout_K), 4),
    'delta_T_K': round(float(dT), 4),
    'basis': '293 K properties are used as calibration and Reynolds-number baseline.',
    'density_change_pct_estimated': round(100 * float(rho_out / rho_in - 1), 3),
    'cp_change_pct_assumed_small': '<0.5% near room temperature',
    'thermal_conductivity_change_pct_estimated': round(100 * float(k_out / k_in - 1), 3),
    'dynamic_viscosity_293K_Pa_s_from_correlation': round(float(mu_in), 8),
    'dynamic_viscosity_Tout_Pa_s_from_correlation': round(float(mu_out), 8),
    'dynamic_viscosity_change_pct_estimated': round(100 * float(mu_out / mu_in - 1), 3),
    'audit_conclusion': (
        'Density, cp and thermal conductivity are relatively weakly varying over the estimated '
        'temperature rise, but dynamic viscosity changes materially; the constant-property '
        'assumption should not claim all water properties vary by less than 2%.'
    ),
}
with open(BASE + '/代码/p1_property_assumption.json', 'w', encoding='utf-8') as f:
    json.dump(property_assumption, f, ensure_ascii=False, indent=2)

print('=== goodness of fit ===')
for k, v in stats.items():
    print(k, v)
print('=== fitted params ===')
print('R_fit  :', params['R_fit'])
print('dP_fit :', params['dP_fit'])
print('phi_fit: tau =', round(TAU_FIT, 3), 'coef =', params['phi_fit']['coef'])
print('flow   :', params['flow_state'])

# 主效应趋势核对：数据均值 vs 模型均值
print('=== main-effect check (data | model) ===')
for var in ['beta', 'gamma', 'N']:
    sub = df_out if var == 'N' else df_out[df_out.N > 0]
    gd = sub.groupby(var)[['R', 'R_pred', 'dP', 'dP_pred', 'phi', 'phi_pred']].mean()
    print('--- by', var, '---')
    print(gd.round(4).to_string())
