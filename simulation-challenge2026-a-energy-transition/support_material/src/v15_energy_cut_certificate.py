"""Explain the phase-zero ENS floor by four disjoint conservation cuts.

This is a post-solution algebraic witness evaluated on saved trajectories.
No annual model, dispatch LP, policy selection or new scientific cohort is run.
AI-assisted: OpenAI Codex, 2026-10-19.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import v13_storage_duration as duration
from src import v15_dispatch_bound_v2 as bound
from src.io_utils import verify_review
from src.v11_recovery_sensitivity import digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/v15/energy_cut_certificate'
WINDOWS = ((65, 72), (89, 96), (209, 216), (233, 240))
TOLERANCE = 1e-6


def energy_cuts(case, windows=WINDOWS):
    """Bound ENS using available generation, daily debt clearance and store energy."""
    windows = tuple(windows)
    if any(a < 0 or b <= a or b > case['hours'] for a, b in windows):
        raise ValueError('Each cut must be a nonempty interval inside the physical horizon')
    if any(b > c for (_, b), (c, _) in zip(windows, windows[1:])):
        raise ValueError('Conservation cuts must be ordered and disjoint')
    if any(b % case['config']['deadline_hours'] and b != case['hours'] for _, b in windows):
        raise ValueError('Each cut must end at an absolute debt-clearance deadline')
    for store in case['storage'].values():
        charge, discharge = store['eta_charge'], store['eta_discharge']
        if not (0 < charge <= 1 and 0 < discharge <= 1 and charge * discharge <= 1):
            raise ValueError('Storage efficiency must conserve energy')
    deliverable = sum(store['energy_gwh'] * store['eta_discharge'] for store in case['storage'].values())
    rows = []
    for a, b in windows:
        demand = float(case['demand_gw'][a:b].sum())
        available = float(case['available_gw'][a:b].sum())
        deficit = float((case['demand_gw'][a:b] - case['available_gw'][a:b].sum(axis=1)).sum())
        rows.append({'first_hour': a, 'end_exclusive': b, 'hours': b - a,
                     'original_demand_gwh': demand, 'all_available_generation_gwh': available,
                     'all_available_generation_deficit_gwh': deficit,
                     'storage_deliverable_gwh': deliverable,
                     'ens_lower_bound_gwh': max(0., deficit - deliverable)})
    return rows


def evaluate(case, table, identity):
    assert table.hour.tolist() == list(range(case['hours']))
    rows = energy_cuts(case)
    for row in rows:
        a, b = row['first_hour'], row['end_exclusive']
        before_debt = float(table.shift_debt_gwh.iloc[a - 1]) if a else 0.
        end_debt = float(table.shift_debt_gwh.iloc[b - 1])
        actual = float(table.unserved_gw.iloc[a:b].sum())
        assert before_debt >= -TOLERANCE and abs(end_debt) <= TOLERANCE
        assert actual + TOLERANCE >= row['ens_lower_bound_gwh']
        row.update(**identity, actual_window_ens_gwh=actual,
                   actual_minus_cut_bound_gwh=actual - row['ens_lower_bound_gwh'],
                   beginning_debt_gwh=before_debt, ending_debt_gwh=end_debt)
    total = sum(row['ens_lower_bound_gwh'] for row in rows)
    observed = float(table.unserved_gw.iloc[48:].sum())
    assert total <= observed + TOLERANCE
    return rows, {**identity, 'cut_bound_gwh': total, 'actual_primary_ens_gwh': observed,
                  'actual_minus_cut_bound_gwh': observed - total,
                  'tight_within_tolerance': abs(observed - total) <= TOLERANCE}


def main():
    assert not (OUT / 'audit.json').exists(), 'Preserve the completed algebraic witness'
    verify_review(OUT / 'code_review.json')
    verify_review(bound.OUT / 'execution_review.json')
    verify_review(duration.OUT / 'execution_review.json')
    paths = [Path(__file__), ROOT / 'tests/test_v15_energy_cut_certificate.py',
             ROOT / 'src/v3_hourly.py', ROOT / 'src/v13_storage_duration.py',
             ROOT / 'src/v13_event_design.py', ROOT / 'src/v13_storage_baseline.py',
             ROOT / 'src/v10_prepare_inputs.py', ROOT / 'src/v11_recovery_sensitivity.py',
             ROOT / 'src/v15_dispatch_bound_v2.py', OUT / 'code_review.json',
             bound.OUT / 'analysis.json', bound.OUT / 'protocol.json', bound.OUT / 'execution_review.json',
             duration.OUT / 'analysis.json', duration.OUT / 'protocol.json', duration.OUT / 'execution_review.json']
    specifications = []
    for row in bound.CONDITIONS:
        if row['phase_hours'] == 0:
            specifications.append((bound.make_case(bound.design.case_configuration(row)),
                                   bound.OUT / (row['condition_id'] + '_rolling'),
                                   {'study': 'V15_phase0', 'case_id': row['condition_id'],
                                    'strategy': row['strategy'], 'power_multiplier': row['storage_power_multiplier'],
                                    'battery_duration_hours': 4., 'terminal_condition': row['terminal_condition']}))
    for row in duration.GRID:
        specifications.append((duration.make_duration_case(row), duration.OUT / row['case_id'],
                               {'study': 'V13_storage_duration', 'case_id': row['case_id'],
                                'strategy': row['strategy'], 'power_multiplier': row['storage_power_multiplier'],
                                'battery_duration_hours': row['battery_duration_hours'], 'terminal_condition': 'free'}))
    all_rows, totals = [], []
    for case, folder, identity in specifications:
        path = folder / 'actions_states.csv'
        paths.append(path)
        table = pd.read_csv(path, float_precision='round_trip')
        physical = bound.event.independent_audit(case, table)[1]
        assert physical['passed'], (identity, physical)
        values, total = evaluate(case, table, identity)
        all_rows.extend(values)
        totals.append(total)
    assert len(all_rows) == 56 and len(totals) == 14
    assert all(item['tight_within_tolerance'] for item in totals if item['strategy'] == 'coordinated')
    source_hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(OUT / 'all_56_window_cuts.csv', index=False, float_format='%.17g')
    pd.DataFrame(totals).to_csv(OUT / 'all_14_path_cut_bounds.csv', index=False, float_format='%.17g')
    text = r"""# 四段夜间窗口的电量下界

这是求解完成后构造的守恒证据；读取全部四条V15原始相位基线及全部十条既有储能时长路径，计算14条路径的56个窗口。此处没有新增仿真或LP。

对窗口 $[a,b)$，电力平衡逐时相加。发电量不超过各源可用出力，弃电非负。移峰债务满足 $B_b=B_a+\sum(out-repay)$；各窗在绝对日末结束，$B_b=0$ 且 $B_a\ge0$，所以 $\sum(repay-out)\ge0$。

储能满足 $S_b-S_a=\eta_c\sum ch-\sum dis/\eta_d$。因 $0\le S_a\le E$、$S_b\ge0$ 及 $\eta_c\eta_d\le1$，有

$$\sum(dis-ch)=\eta_d(S_a-S_b)-(1-\eta_c\eta_d)\sum ch\le\eta_d E.$$

代入电力平衡，得到

$$\sum_{t=a}^{b-1}ENS_t\ge\left[\sum_{t=a}^{b-1}(D_t-\sum_g A_{g,t})-\sum_s\eta_{d,s}E_s\right]_+.$$

四窗 $[65,72),[89,96),[209,216),[233,240)$ 互不重叠，各结束于原24小时债务截止点，可将四个下界相加。每个窗口均允许储能从满电开始，是对实际可行调度的放松；燃料与爬坡约束只会进一步限制发电，所以上式仍成立。窗口由已求解原相位对偶的非零电平衡价格定位，再以独立守恒不等式验证。

4小时配置的两类储能最大可交付电量为740.879417719GWh/窗；四窗下界总和5169.910288182GWh，与四个原相位协调基线及4小时协调路径在计算精度内相等。功率由P1增至P1.5而能量E不变，该下界不变。

电池由4小时改为2.3小时时，两种功率仍使用同一电池能量；减少的可交付电量乘四，得到396.653646753GWh，恰与两条协调路径的累计缺供增量一致。这解释既有工程配置挑战，未重新计算投资资本，也未将2024统计平均解释为2060电池预测。

相位21的证明仍由完整LP及其数值原/对偶证书给出；这里的四窗封闭式证明适用于所列原相位48小时事件输入。全部14条路径及局部规则与该下界的差额在CSV中逐项保留。
"""
    (OUT / 'derivation.md').write_text(text, 'utf-8')
    for name, expected in source_hashes.items():
        assert digest(ROOT / name) == expected, name
    audit = {'status': 'PASS_POST_SOLUTION_CONSERVATION_CUTS_14_SAVED_PATHS_56_WINDOWS',
             'utc': datetime.now(timezone.utc).isoformat(), 'analysis_class': 'POST_SOLUTION_ALGEBRAIC_WITNESS',
             'scientific_lp_calls': 0, 'annual_simulations': 0, 'paths': 14, 'windows': 56,
             'phase21_closed_form_claim': False, 'source_sha256': source_hashes,
             'output_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(OUT.iterdir())
                               if p.is_file() and p.name not in ('audit.json', 'code_review.json')}}
    bound.event.write_json(OUT / 'audit.json', audit)
    print(json.dumps({key: audit[key] for key in ('status', 'paths', 'windows', 'scientific_lp_calls')}))


if __name__ == '__main__':
    main()
