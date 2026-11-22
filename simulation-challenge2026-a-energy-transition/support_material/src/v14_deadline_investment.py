"""Four fixed oil/gas order mechanisms on the original 48 development designs.

The frozen annual physics is retained verbatim. Only proposal and funding rules
change; the complete generated step is saved before evaluation. AI-assisted:
OpenAI Codex, 2026-10-19.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import pandas as pd
import scipy

from src import v6_annual as annual
from src.v5_planning import decode, digest, reliability_feasible, summarize
from src.v11_robust_plans import perturbation, yearly_reliability

ROOT = Path(__file__).resolve().parents[1]
REL_OUT = Path("outputs/v14/deadline_investment")
POLICIES = ("P0", "P1", "P2", "P3")
DEADLINE = 2030
FUEL_ORDER_INDICES = (10, 11)
OLD_SIGNATURE = "def step_numeric(state, p, scenario, year, gdp_ratio, mechanisms=None):"
NEW_SIGNATURE = "def step_deadline_numeric(state, p, scenario, year, gdp_ratio, mechanisms=None, policy='P0'):"
OLD_PROPOSAL = "    wanted=np.minimum(wanted,limit)"
NEW_PROPOSAL = """    raw_wanted=wanted.copy()
    legacy_wanted=np.minimum(wanted,limit)
    wanted,priority_mask=policy_proposals(legacy_wanted,limit,year,delays,policy)"""
OLD_FUNDING = "    spend=float(wanted@costs);orders=wanted*min(1,budget/max(spend,1e-12));spend=float(orders@costs)"
NEW_FUNDING = """    orders=fund_orders(wanted,costs,budget,priority_mask,policy)
    spend=float(orders@costs)"""
OLD_RETURN = "    return n,metrics"
NEW_RETURN = """    return n,metrics,{
        'raw_wanted':raw_wanted,'legacy_wanted':legacy_wanted,'wanted':wanted,
        'build_limit':limit,'costs':costs,'budget':budget,'orders':orders,
        'priority_mask':priority_mask,'delays':delays,'requested':requested,
        'ordinary':ordinary,'domestic_available':domestic_available,
        'potential_imports':imports,'release':release,'recharge':recharge,
        'direct_request':direct_request,'direct_delivered':direct_delivered,
        'power_use':powerfuel,'domestic':domestic,'actual_imports':actual_imports,
        'stocks':stocks,'retired':retired,'commissioned':commissioned,
        'service':service,'gdp_ratio':gdp_ratio} """


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default, allow_nan=False), "utf-8")


def sha_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def policy_proposals(legacy_wanted, limit, year, delays, policy):
    """Use only this year's known lead and the fixed 2030 commissioning date."""
    if policy not in POLICIES:
        raise ValueError("Unknown fixed policy: " + str(policy))
    wanted = np.asarray(legacy_wanted, float).copy()
    priority = np.zeros(12, bool)
    if policy != "P0":
        for j in FUEL_ORDER_INDICES:
            priority[j] = policy == "P3" or year + int(delays[j]) <= DEADLINE
            if priority[j] and policy in ("P2", "P3"):
                wanted[j] = limit[j]
    return wanted, priority


def fund_orders(wanted, costs, budget, priority, policy):
    """One current-year budget; priority groups share short money proportionally."""
    if policy == "P0" or not np.any(priority):
        spend = float(wanted @ costs)
        return wanted * min(1, budget / max(spend, 1e-12))
    orders = np.zeros(12)
    priority_spend = float(wanted[priority] @ costs[priority])
    priority_scale = min(1, budget / max(priority_spend, 1e-12))
    orders[priority] = wanted[priority] * priority_scale
    remaining = max(0., budget - float(orders[priority] @ costs[priority]))
    other = ~priority
    other_spend = float(wanted[other] @ costs[other])
    orders[other] = wanted[other] * min(1, remaining / max(other_spend, 1e-12))
    return orders


def generated_step_source():
    """Strict replacements keep every other executable physics statement intact."""
    original = inspect.getsource(annual.step_numeric)
    replacements = ((OLD_SIGNATURE, NEW_SIGNATURE), (OLD_PROPOSAL, NEW_PROPOSAL),
                    (OLD_FUNDING, NEW_FUNDING), (OLD_RETURN, NEW_RETURN))
    generated = original
    for old, new in replacements:
        if generated.count(old) != 1:
            raise AssertionError("Frozen step block is no longer unique: " + old)
        generated = generated.replace(old, new)
    restored = generated
    for old, new in reversed(replacements):
        if restored.count(new) != 1:
            raise AssertionError("Generated step block is no longer unique")
        restored = restored.replace(new, old)
    if ast.dump(ast.parse(restored), include_attributes=False) != ast.dump(ast.parse(original), include_attributes=False):
        raise AssertionError("A physics statement changed outside the two policy blocks")
    return original, generated


def compile_step(generated):
    # Source is generated solely from the frozen local function and fixed local
    # literals above. Runtime data never becomes executable source.
    namespace = dict(vars(annual))
    namespace.update(policy_proposals=policy_proposals, fund_orders=fund_orders)
    exec(compile(generated, "<v14_deadline_generated_step>", "exec"), namespace)
    return namespace["step_deadline_numeric"]


SOURCE_NAMES = [
    "src/v14_deadline_investment.py", "tests/test_v14_deadline_investment.py",
    "src/v6_annual.py", "src/v5_annual.py", "src/v5_planning.py", "src/v11_robust_plans.py",
    "src/v3_annual.py", "src/v3_annual_calibrated.py", "src/v3_calibration_candidate.py",
    "config/v3_transition.json", "config/v6_annual_planning.json", "config/v6_annual_experiments.json",
    "data/annual_china.csv", "data/ember_power_china.csv", "data/energy_production_china.csv",
    "data/fuel_supply_trade.csv", "data/supplement_carriers_2024.csv", "data/v6_fuel_import_channels.csv",
    "outputs/v6/annual/joint_design.csv", "outputs/v11/robust_repair/candidate_scores.csv",
    "outputs/v11/robust_repair/candidate_design_metrics.csv",
    "outputs/v12/native/python_states.csv", "outputs/v12/native/python_raw_metrics.csv",
    "outputs/v12/native/python_actions.csv", "outputs/v12/native/python_trace.csv",
    "outputs/v12/native/python_summary.json", "outputs/v12/native/native_states.csv",
    "outputs/v12/native/native_raw_metrics.csv", "outputs/v12/native/native_actions.csv",
]


def prepare(root=ROOT):
    root = Path(root).resolve()
    out = root / REL_OUT
    if (out / "analysis.json").exists():
        raise FileExistsError("Formal evaluation exists; preserve its sources and prepared inputs")
    out.mkdir(parents=True, exist_ok=True)
    before = {name: digest(root / name) for name in SOURCE_NAMES}
    planning = json.loads((root / "config/v6_annual_planning.json").read_text("utf-8"))
    scores = pd.read_csv(root / "outputs/v11/robust_repair/candidate_scores.csv")
    chosen = scores.loc[scores.candidate_id == "repair_005"]
    if len(chosen) != 1:
        raise AssertionError("Exactly one original repair_005 is required")
    chosen = chosen.iloc[0]
    vector = chosen[planning["decision_names"]].to_numpy(float)
    if not np.array_equal(vector[:3], [3., 3., 3.]) or vector[8] != 2.:
        raise AssertionError("Original 005 budget/fuel controls changed")
    designs = pd.read_csv(root / "outputs/v6/annual/joint_design.csv")
    if len(designs) != 48 or designs["case"].duplicated().any():
        raise AssertionError("The original complete 48 development designs are required")
    dev_reference = pd.read_csv(root / "outputs/v11/robust_repair/candidate_design_metrics.csv")
    dev_reference = dev_reference.loc[dev_reference.candidate_id == "repair_005"].copy()
    if len(dev_reference) != 48 or set(dev_reference.design_id) != set(designs["case"]):
        raise AssertionError("Complete original 005 development summaries are required")
    designs.to_csv(out / "development_design_frozen.csv", index=False)
    dev_reference.to_csv(out / "development_reference_frozen.csv", index=False)
    write_json(out / "nominal_reference_frozen.json", {key.removeprefix("nominal_"): value for key, value in chosen.items() if key.startswith("nominal_")})
    original, generated = generated_step_source()
    (out / "frozen_original_step.py").write_text(original, "utf-8")
    (out / "executed_step.py").write_text(generated, "utf-8")
    compile_step(generated)
    data = annual.load_data(root)
    params = annual.calibrate(data, 2020)
    write_json(out / "calibration.json", params)
    inputs = out / "inputs"
    inputs.mkdir(exist_ok=True)
    records = [{"case": "nominal"}, *designs.to_dict("records")]
    identities = []
    for design in records:
        design_id = design["case"]
        if design_id == "nominal":
            scenario, fitted, overrides = decode(vector, "repair_005"), params, None
        else:
            scenario, fitted, overrides = perturbation(data, params, vector, design, "repair_005__" + design_id)
        case_data = annual._engineering_data(data, overrides)
        scenario = annual._decision_scenario(case_data, scenario)
        rules = annual.resolve_mechanisms()
        initial, runtime = annual.initialize_state(case_data, fitted, 2024, mechanisms=rules)
        initial[212] = scenario.get("carbon_price_start_cny_t", 100.)
        if scenario.get("clear_initial_pipeline", False):
            initial[20:annual.QEND] = 0.
        write_json(inputs / (design_id + ".json"), {"design_id": design_id, "scenario": scenario,
                   "runtime": runtime, "initial_state": initial, "rules": rules, "factors": design})
        leads = np.ceil(np.array(runtime["engineering"]["construction_years"]) * scenario.get("construction_delay_multiplier", 1)).astype(int)
        identities.append({"design_id": design_id, "cohort": "nominal" if design_id == "nominal" else "development",
                           "oil_lead_years": int(leads[10]), "gas_lead_years": int(leads[11]),
                           "oil_last_order_year_for_2030": DEADLINE - int(leads[10]),
                           "gas_last_order_year_for_2030": DEADLINE - int(leads[11])})
    grid = pd.DataFrame([{**row, "policy": policy, "case_id": policy + "__" + row["design_id"]}
                         for policy in POLICIES for row in identities])
    grid.to_csv(out / "case_grid.csv", index=False)
    prepared = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob("*")
                if p.is_file() and p.name not in ("protocol.json", "preparation.json", "code_review.json", "tests.txt")}
    after = {name: digest(root / name) for name in SOURCE_NAMES}
    if before != after:
        raise AssertionError("Source/input changed during preparation")
    protocol = {
        "status": "FIXED_FOUR_MECHANISM_DEVELOPMENT_COHORT_PREPARED_NO_FORMAL_PATHS",
        "frozen_before_evaluation_utc": utc_now(), "formal_paths_run_at_preparation": 0,
        "command_prepare": "python -B -X utf8 -m src.v14_deadline_investment --prepare",
        "command_run": "python -B -X utf8 -m src.v14_deadline_investment --run",
        "expected_paths": 196, "policies": {
            "P0": "Original repair_005 unchanged proposal and proportional funding",
            "P1": "Only oil/gas orders able to commission by 2030 receive funding priority; proposals stay unchanged",
            "P2": "Same priority as P1; eligible oil/gas proposals filled to original annual physical build limits",
            "P3": "Every year oil/gas proposals filled to original build limits and funded first"},
        "fixed_controls": dict(zip(planning["decision_names"], vector.tolist())),
        "construction_deadline_year": DEADLINE,
        "eligibility": "Order year + the actual known construction lead <= 2030, independently for oil and gas; expired windows return to original proposals and proportional funding for P1/P2",
        "funding": "Oil/gas priority-group costs receive current-year budget first, pro rata inside that group if insufficient; remaining budget scales all other proposals proportionally; no carryover",
        "budget_contract": "All policies use the unchanged original base_budget*(service/initial_service)**0.7*annual_budget_multiplier*period_multiplier, with original 005 period multipliers 3. State feedback may change each policy's amount; same budget rule is not equal realised budget or equal paid capital.",
        "information_contract": "Only this-year state, fixed 2030 date, known lead, existing proposed orders, unit costs and current budget enter policy. No P0 future realised trajectory, future state, future funding or independent design is consumed.",
        "scope": "P2/P3 introduce proposal mechanisms beyond the original ten-dimensional control search. No new optimization or candidate reselection. All 48 old engineering designs are development evidence without probability weights.",
        "bridge_gate": "Execute all 49 P0 paths first. At each year compare all 217 states and 132 raw metrics with frozen original step from the identical prestate; derive and compare all original action columns. Compare all 14 summary fields against existing 48 development references and nominal anchor. Nominal full states/raw metrics/actions/trace also bridge to frozen V12 Python references and native state/raw-metric/action anchors. Proceed with other 147 paths only if all 49 pass.",
        "bridge_shadow_cost": "49*36=1764 frozen original one-step comparisons, explicitly additional computation; not independent observations or additional policy paths",
        "years": [2025, 2060], "state_dimension": 217, "raw_metric_dimension": 132,
        "expected_per_path": {"annual_rows": 36, "action_rows": 432, "all_year_carrier_rows": 180, "assessment_year_carrier_rows": 155},
        "failure_rule": "Preserve every attempted prefix and original exception, continue all fixed P0 cases; if any P0 fails, do not run the 147 non-P0 paths. After bridge pass continue all 147 predefined cases even if one fails. Failed or skipped cases retain their rows and null metrics. No policy range claims from incomplete groups.",
        "reporting": "Report all four policies, full 48 denominator, nominal separately, complete 2030-2060 five-carrier assessments, 2025-2029 early service, 2030 resource accounts, all proposals/funding/build caps, actual paid and discounted capital and 2060 net CO2; split complete development outputs by known oil lead 4/5/6 and gas lead, not favourable-subset selection.",
        "tolerances": {"physical_normalized": 1e-10, "physical_absolute_nonnegative": 1e-7, "bridge_rtol": 1e-11, "bridge_atol": 1e-7},
        "planning_protocol": planning, "source_input_hashes": before,
        "prepared_artifact_sha256": prepared, "frozen_original_step_sha256": sha_text(original),
        "executed_step_sha256": sha_text(generated),
        "ast_restoration": "PASS_ONLY_SIGNATURE_TWO_UNIQUE_POLICY_BLOCKS_AND_DIAGNOSTIC_RETURN_REPLACED",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
    }
    write_json(out / "protocol.json", protocol)
    receipt = {"status": protocol["status"], "formal_paths_run": 0, "case_count": len(grid),
               "initial_state_count": len(identities), "protocol_sha256": digest(out / "protocol.json"),
               "source_input_hashes_after": after, "prepared_artifact_sha256": prepared}
    write_json(out / "preparation.json", receipt)
    return receipt


def augment_row(state, metrics, scenario, year, gdp, runtime):
    row, actions = annual.frozen._row(state, metrics, scenario["name"], year)
    row["gdp_real_index"] = gdp
    row["energy_intensity_index_base_year_100"] = 100 * row["energy_total_mtce"] / runtime["base_primary_energy_mtce"] / (gdp / runtime["base_gdp"])
    row["carbon_intensity_index_base_year_100"] = 100 * row["gross_national_co2_mt"] / runtime["base_national_co2"] / (gdp / runtime["base_gdp"])
    row["fixed_sink_coverage_rate"] = runtime["engineering"]["sink_mt"] / max(row["gross_national_co2_mt"], 1e-12)
    row["carbon_neutralization_rate"] = min(runtime["engineering"]["sink_mt"] / max(row["gross_national_co2_mt"], 1e-12), 1.)
    row["carbon_neutral_target_met"] = bool(row["net_co2_mt"] <= 0.)
    for j, fuel in enumerate(annual.FUELS):
        row[fuel + "_import_capacity_mtce_per_year"] = float(state[213 + j])
        row[fuel + "_domestic_retention_fraction"] = runtime["domestic_retention_fraction"][j]
        row[fuel + "_production_capacity_mtce_per_year"] = float(state[14 + j])
    row["biomass_generation_budget_twh_per_year"] = runtime["initial_biomass_generation_twh"] * np.exp(scenario.get("biomass_feedstock_growth_per_year", .02) * (year - 2024))
    return row, actions


def assert_close(actual, expected, label, *, rtol=1e-11, atol=1e-7):
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(label + " shape/finite mismatch")
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=label)
    return float(np.max(np.abs(a - b) / np.maximum(1., np.maximum(np.abs(a), np.abs(b))), initial=0.))


def audit_step(prestate, state, metrics, diagnostic, runtime, scenario, policy, year):
    """Independent queue, material, budget and mechanism identities each year."""
    if state.shape != (217,) or metrics.shape != (132,) or not np.isfinite(state).all() or not np.isfinite(metrics).all():
        raise AssertionError("Incomplete/nonfinite physical state or raw metrics")
    e = runtime["engineering"]
    checks = {}
    def equality(name, actual, expected):
        checks[name] = assert_close(actual, expected, name, rtol=1e-10, atol=1e-8)
    old = np.r_[prestate[5:14], prestate[14:17]]
    queue = prestate[20:annual.QEND].reshape(12, annual.QDEPTH)
    orders, costs, budget = diagnostic["orders"], diagnostic["costs"], diagnostic["budget"]
    equality("commissioning", metrics[54:66], queue[:, 0])
    equality("retirement", metrics[66:78], old * np.array(e["retirement_fraction"]))
    equality("installed_assets", np.r_[state[5:14], state[14:17]], old - metrics[66:78] + metrics[54:66])
    expected_queue = np.column_stack([queue[:, 1:], np.zeros(12)])
    for j, lead in enumerate(diagnostic["delays"]):
        expected_queue[j, int(lead) - 1] += orders[j]
    equality("construction_queue", state[20:annual.QEND].reshape(12, annual.QDEPTH), expected_queue)
    equality("fuel_stock_flow", state[17:20], prestate[17:20] + metrics[30:33] + metrics[33:36] - metrics[24:27] - metrics[27:30])
    equality("fuel_delivery", metrics[24:27] + metrics[27:30], np.minimum(diagnostic["requested"], diagnostic["ordinary"] + diagnostic["release"]))
    equality("fuel_inflow", metrics[30:33] + metrics[33:36], metrics[24:27] + metrics[27:30] + diagnostic["recharge"] - diagnostic["release"])
    equality("electricity_balance", metrics[1] + metrics[2], metrics[0])
    equality("capital_spend", metrics[8], orders @ costs)
    period = 0 if year <= 2030 else 1 if year <= 2040 else 2
    expected_budget = runtime["base_budget"] * (state[0] / runtime["initial_service"]) ** .7 * scenario.get("annual_budget_multiplier", 1)
    expected_budget *= scenario.get("budget_multipliers_by_period", [1., 1., 1.])[period]
    equality("budget_rule", budget, expected_budget)
    equality("budget_metric", metrics[9], expected_budget)
    equality("share_sum", state[1:5].sum(), 1.)
    expected_wanted, expected_priority = policy_proposals(diagnostic["legacy_wanted"], diagnostic["build_limit"], year, diagnostic["delays"], policy)
    equality("policy_proposal", diagnostic["wanted"], expected_wanted)
    if not np.array_equal(diagnostic["priority_mask"], expected_priority):
        raise AssertionError("Deadline eligibility differs from fixed policy")
    equality("funding_allocation", orders, fund_orders(expected_wanted, costs, budget, expected_priority, policy))
    # Metrics[39:42] are signed stock changes, rather than physical stocks.
    if np.min(np.r_[state[1:20], state[20:annual.QEND], orders, metrics[14:39]]) < -1e-7:
        raise AssertionError("Negative physical assets/flow/stock")
    if np.any(orders > diagnostic["wanted"] + 1e-7) or np.any(orders > diagnostic["build_limit"] + 1e-7):
        raise AssertionError("Proposal or original physical build limit exceeded")
    if metrics[8] > budget + 1e-6:
        raise AssertionError("Current-year shared budget exceeded")
    if np.any(metrics[30:33] > diagnostic["domestic_available"] + 1e-7) or np.any(metrics[33:36] > diagnostic["potential_imports"] + 1e-7):
        raise AssertionError("Fuel source capacity exceeded")
    if np.any(diagnostic["release"] > prestate[17:20] * e["max_stock_release_fraction_per_year"] + 1e-7):
        raise AssertionError("Fuel stock release limit exceeded")
    return {"year": year, "max_normalized_residual": max(checks.values()), **checks}


def diagnostics_rows(year, d):
    rows = []
    for j, asset in enumerate(annual.ASSETS):
        proposal_spend = float(d["wanted"] @ d["costs"])
        rows.append({"year": year, "asset": asset, "unit": "GW" if j < 9 else "Mtce/year",
                     "lead_years": int(d["delays"][j]), "commission_year_if_ordered": year + int(d["delays"][j]),
                     "eligible_for_2030": bool(j in FUEL_ORDER_INDICES and year + int(d["delays"][j]) <= DEADLINE),
                     "priority_funding": bool(d["priority_mask"][j]),
                     "native_prebuild_proposal": d["raw_wanted"][j], "native_postbuild_proposal": d["legacy_wanted"][j],
                     "policy_proposal": d["wanted"][j], "build_limit": d["build_limit"][j],
                     "actual_order": d["orders"][j], "unit_cost_million_cny": d["costs"][j],
                     "capital_paid_million_cny": d["orders"][j] * d["costs"][j],
                     "current_budget_million_cny": d["budget"], "total_proposal_cost_million_cny": proposal_spend,
                     "native_build_cap_binding": bool(d["raw_wanted"][j] > d["build_limit"][j] + 1e-9),
                     "policy_fills_build_cap": bool(abs(d["wanted"][j] - d["build_limit"][j]) <= 1e-9),
                     "funding_binding": bool(d["orders"][j] < d["wanted"][j] - 1e-9)})
    fuel_rows = []
    for j, fuel in enumerate(annual.FUELS):
        used = d["direct_delivered"][j] + d["power_use"][j]
        fuel_rows.append({"year": year, "fuel": fuel, "requested_total_mtce": d["requested"][j],
                          "direct_requested_mtce": d["direct_request"][j], "direct_delivered_mtce": d["direct_delivered"][j],
                          "power_use_mtce": d["power_use"][j], "ordinary_available_mtce": d["ordinary"][j],
                          "domestic_available_mtce": d["domestic_available"][j], "potential_imports_mtce": d["potential_imports"][j],
                          "actual_domestic_mtce": d["domestic"][j], "actual_imports_mtce": d["actual_imports"][j],
                          "stock_release_mtce": d["release"][j], "recharge_mtce": d["recharge"][j],
                          "end_stock_mtce": d["stocks"][j], "unserved_total_mtce": max(0., d["requested"][j] - used),
                          "direct_shortfall_fraction": max(0., 1. - d["direct_delivered"][j] / d["direct_request"][j]) if d["direct_request"][j] > 0 else 0.})
    return rows, fuel_rows


def compare_frames(actual, expected, label):
    result = {}
    for column in expected.select_dtypes(include=np.number).columns:
        if column not in actual:
            raise AssertionError(label + " missing column " + column)
        a, b = actual[column].to_numpy(float), expected[column].to_numpy(float)
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            raise AssertionError(label + " missing-value pattern " + column)
        finite = ~np.isnan(a)
        result[column] = assert_close(a[finite], b[finite], label + " " + column)
    return {"numeric_columns": len(result), "max_normalized_error": max(result.values(), default=0.)}


def run_case(root, identity, step, planning, reference):
    out = root / REL_OUT
    case_out = out / "cases" / identity["case_id"]
    case_out.mkdir(parents=True, exist_ok=False)
    packet = json.loads((out / "inputs" / (identity["design_id"] + ".json")).read_text("utf-8"))
    runtime, scenario, rules = packet["runtime"], packet["scenario"], packet["rules"]
    state = np.asarray(packet["initial_state"], float)
    initial = state.copy()
    states, raw_metrics, rows, actions, audits, proposals, fuels, bridge_steps = [], [], [], [], [], [], [], []
    result = {**identity, "status": "RUNNING", "started_utc": utc_now()}
    original_error = None
    started = time.perf_counter()
    try:
        gdp = runtime["base_gdp"]
        for year in range(2025, 2061):
            next_gdp = runtime["gdp_history"].get(str(year), gdp * (1 + scenario["gdp_growth"][0 if year <= 2030 else 1 if year <= 2040 else 2]))
            before = state.copy()
            state, metrics, d = step(before, runtime, scenario, year, next_gdp / gdp, rules, identity["policy"])
            if identity["policy"] == "P0":
                old_state, old_metrics = annual.step_numeric(before, runtime, scenario, year, next_gdp / gdp, rules)
                state_error = assert_close(state, old_state, "P0 complete state")
                metric_error = assert_close(metrics, old_metrics, "P0 complete raw metrics")
                _, old_actions = annual.frozen._row(old_state, old_metrics, scenario["name"], year)
                _, new_actions = annual.frozen._row(state, metrics, scenario["name"], year)
                action_check = compare_frames(pd.DataFrame(new_actions), pd.DataFrame(old_actions), "P0 original actions")
                bridge_steps.append({"year": year, "states_compared": 217, "raw_metrics_compared": 132,
                                     "state_max_normalized_error": state_error, "metric_max_normalized_error": metric_error, **action_check})
            gdp = next_gdp
            row, acts = augment_row(state, metrics, scenario, year, gdp, runtime)
            discount = (1 + planning["discount_rate"]) ** (year - planning["discount_origin_year"])
            row["capital_discount_factor"] = discount
            row["discounted_capital_million_cny"] = metrics[8] / discount
            row["unused_current_budget_million_cny"] = max(0., metrics[9] - metrics[8])
            row["cumulative_discounted_capital_trillion_cny"] = (sum(r["discounted_capital_million_cny"] for r in rows) + row["discounted_capital_million_cny"]) / 1e6
            row["cumulative_paid_capital_trillion_cny"] = (sum(r["investment_spend_million_cny"] for r in rows) + metrics[8]) / 1e6
            states.append(state.copy()); raw_metrics.append(metrics.copy()); rows.append(row); actions.extend(acts)
            proposal_rows, fuel_rows = diagnostics_rows(year, d)
            proposals.extend(proposal_rows); fuels.extend(fuel_rows)
            audits.append(audit_step(before, state, metrics, d, runtime, scenario, identity["policy"], year))
        frame, action_frame = pd.DataFrame(rows), pd.DataFrame(actions)
        data = annual.load_data(root)
        summary = summarize(frame, action_frame, data, planning)
        yearly = yearly_reliability(frame, data, planning)
        all_year_protocol = {**planning, "reliability": {**planning["reliability"], "first_year": 2025}}
        all_yearly = yearly_reliability(frame, data, all_year_protocol)
        if len(frame) != 36 or len(action_frame) != 432 or len(yearly) != 155 or len(all_yearly) != 180:
            raise AssertionError("Incomplete annual/action/service cohort")
        pd.DataFrame(all_yearly).to_csv(case_out / "all_year_service.csv", index=False)
        yearly.to_csv(case_out / "assessment_service.csv", index=False)
        all_yearly.loc[all_yearly.year <= 2029].to_csv(case_out / "early_2025_2029_service.csv", index=False)
        if identity["policy"] == "P0":
            errors = {name: assert_close(value, reference[name], "P0 summary " + name) for name, value in summary.items()}
            result["summary_bridge"] = {"fields_compared": len(errors), "max_normalized_error": max(errors.values())}
            if identity["cohort"] == "nominal":
                nominal_checks = {}
                for prefix in ("python", "native"):
                    expected_states = pd.read_csv(root / f"outputs/v12/native/{prefix}_states.csv", header=None).to_numpy(float)
                    expected_metrics = pd.read_csv(root / f"outputs/v12/native/{prefix}_raw_metrics.csv", header=None).to_numpy(float)
                    nominal_checks[prefix + "_states"] = assert_close(states, expected_states, prefix + " nominal full states")
                    nominal_checks[prefix + "_raw_metrics"] = assert_close(raw_metrics, expected_metrics, prefix + " nominal raw metrics")
                    old_actions = pd.read_csv(root / f"outputs/v12/native/{prefix}_actions.csv")
                    base_columns = list(action_frame.select_dtypes(include=np.number).columns)
                    nominal_checks[prefix + "_actions"] = compare_frames(action_frame, old_actions[base_columns], prefix + " nominal actions")
                old_frame = pd.read_csv(root / "outputs/v12/native/python_trace.csv")
                nominal_checks["python_annual_frame"] = compare_frames(frame, old_frame, "nominal annual frame")
                result["nominal_full_bridge"] = nominal_checks
        worst = yearly.loc[yearly.shortfall_fraction.idxmax()]
        result.update(status="PASS_COMPLETE_PATH_PHYSICS_AND_PROTOCOL", **summary,
                      annual_reliability_feasible=reliability_feasible(summary, planning),
                      worst_shortfall_fraction=float(worst.shortfall_fraction), worst_year=int(worst.year), worst_carrier=worst.carrier,
                      violating_year_carrier_count=int(yearly.violates_reliability.sum()),
                      max_physical_normalized_residual=max(a["max_normalized_residual"] for a in audits),
                      early_2025_2029_max_shortfall_fraction=float(all_yearly.loc[all_yearly.year <= 2029, "shortfall_fraction"].max()))
        if result["annual_reliability_feasible"] != (not bool(yearly.violates_reliability.any())):
            raise AssertionError("Feasibility and complete carrier ledger disagree")
    except Exception as error:
        original_error = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        result.update(status="FAILED_PRESERVED_PREFIX", error=original_error)
    # Prefix evidence must never replace the scientific failure or stop the
    # remaining predeclared cases if its own serialization fails.
    evidence_errors = []
    evidence = {
        "initial_state.json": lambda: initial,
        "annual.csv": lambda: pd.DataFrame(rows), "actions.csv": lambda: pd.DataFrame(actions),
        "states.csv": lambda: pd.DataFrame(states, columns=[f"state_{i:03d}" for i in range(217)]),
        "raw_metrics.csv": lambda: pd.DataFrame(raw_metrics, columns=[f"metric_{i:03d}" for i in range(132)]),
        "physical_audit.csv": lambda: pd.DataFrame(audits), "proposal_funding.csv": lambda: pd.DataFrame(proposals),
        "fuel_account.csv": lambda: pd.DataFrame(fuels), "p0_step_bridge.csv": lambda: pd.DataFrame(bridge_steps),
    }
    for name, build_value in evidence.items():
        try:
            value = build_value()
            if isinstance(value, pd.DataFrame):
                value.to_csv(case_out / name, index=False)
            else:
                write_json(case_out / name, value)
        except Exception as error:
            evidence_errors.append({"artifact": name, "type": type(error).__name__, "message": str(error)})
    if evidence_errors:
        result["evidence_errors"] = evidence_errors
        if original_error is None:
            result["status"] = "FAILED_EVIDENCE_PRESERVED"
    result.update(completed_utc=utc_now(), runtime_seconds=time.perf_counter() - started,
                  completed_annual_rows=len(rows), completed_action_rows=len(actions), completed_audited_years=len(audits),
                  completed_p0_shadow_steps=len(bridge_steps),
                  preserved_original_error=original_error)
    result["artifact_sha256"] = {p.name: digest(p) for p in case_out.iterdir() if p.is_file()}
    write_json(case_out / "result.json", result)
    return result


def cohort_summary(results):
    rows = []
    for policy in POLICIES:
        development = [r for r in results if r["policy"] == policy and r["cohort"] == "development"]
        groups = [("all_48", development)]
        groups.extend((f"oil_lead_{lead}", [r for r in development if r["oil_lead_years"] == lead]) for lead in sorted({r["oil_lead_years"] for r in development}))
        groups.extend((f"gas_lead_{lead}", [r for r in development if r["gas_lead_years"] == lead]) for lead in sorted({r["gas_lead_years"] for r in development}))
        for label, group in groups:
            valid = [r for r in group if r["status"] == "PASS_COMPLETE_PATH_PHYSICS_AND_PROTOCOL"]
            complete = len(valid) == len(group)
            row = {"policy": policy, "group": label, "design_count": len(group), "passed_path_count": len(valid),
                   "complete_group": complete, "reliable_design_count": sum(r["annual_reliability_feasible"] for r in valid) if complete else None}
            for field in ("worst_shortfall_fraction", "net_emissions_2060_mt", "discounted_capital_trillion_cny", "paid_capital_trillion_cny"):
                row[field + "_min"] = min(r[field] for r in valid) if complete and valid else None
                row[field + "_max"] = max(r[field] for r in valid) if complete and valid else None
            rows.append(row)
    return rows


def run(root=ROOT):
    root = Path(root).resolve()
    out = root / REL_OUT
    if (out / "analysis.json").exists():
        raise FileExistsError("Formal evaluation already attempted; preserve its evidence")
    protocol = json.loads((out / "protocol.json").read_text("utf-8"))
    for name, expected in protocol["source_input_hashes"].items():
        if digest(root / name) != expected:
            raise AssertionError("Frozen source/input drift before evaluation: " + name)
    for name, expected in protocol["prepared_artifact_sha256"].items():
        if digest(out / name) != expected:
            raise AssertionError("Prepared artifact drift before evaluation: " + name)
    original, generated = generated_step_source()
    if sha_text(original) != protocol["frozen_original_step_sha256"] or sha_text(generated) != protocol["executed_step_sha256"]:
        raise AssertionError("Executed source no longer matches the prepared source")
    step = compile_step(generated)
    grid = pd.read_csv(out / "case_grid.csv").to_dict("records")
    if len(grid) != 196 or [r["policy"] for r in grid[:49]] != ["P0"] * 49:
        raise AssertionError("Complete ordered 196-path cohort required")
    development_reference = pd.read_csv(out / "development_reference_frozen.csv").set_index("design_id")
    nominal_reference = json.loads((out / "nominal_reference_frozen.json").read_text("utf-8"))
    results = [{**identity, "status": "PENDING"} for identity in grid]
    receipt = {"status": "RUNNING_P0_49_BRIDGE", "started_utc": utc_now(), "protocol_sha256": digest(out / "protocol.json"),
               "argv": sys.argv, "command": protocol["command_run"], "planned_paths": 196, "attempted_paths": 0,
               "completed_paths": 0, "frozen_original_shadow_steps": 0, "environment": protocol["environment"]}
    write_json(out / "analysis.json", receipt)
    started = time.perf_counter()
    for i, identity in enumerate(grid):
        if i == 49:
            if not all(r["status"] == "PASS_COMPLETE_PATH_PHYSICS_AND_PROTOCOL" for r in results[:49]):
                for remaining in results[49:]:
                    remaining["status"] = "NOT_ATTEMPTED_P0_BRIDGE_FAILED"
                receipt["status"] = "STOPPED_AFTER_FAILED_P0_BRIDGE"
                break
            receipt["p0_gate"] = "PASS_ALL_49_FULL_STATE_RAW_METRIC_ACTION_SUMMARY_BRIDGES"
            receipt["status"] = "RUNNING_REMAINING_147_PREDECLARED_PATHS"
            write_json(out / "analysis.json", receipt)
        reference = nominal_reference if identity["cohort"] == "nominal" else development_reference.loc[identity["design_id"]].to_dict()
        results[i] = run_case(root, identity, step, protocol["planning_protocol"], reference)
        receipt["attempted_paths"] += 1
        receipt["completed_paths"] = sum(r["status"] == "PASS_COMPLETE_PATH_PHYSICS_AND_PROTOCOL" for r in results)
        receipt["frozen_original_shadow_steps"] = sum(r.get("completed_p0_shadow_steps", 0) for r in results[:49])
        pd.DataFrame(results).to_csv(out / "case_metrics.csv", index=False)
        write_json(out / "analysis.json", receipt)
        if (i + 1) % 7 == 0 or i in (0, 48, 195):
            print(identity["case_id"], results[i]["status"], "completed", i + 1, "/196", flush=True)
    summaries = cohort_summary(results)
    pd.DataFrame(results).to_csv(out / "case_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "complete_cohort_and_lead_summaries.csv", index=False)
    after = {name: digest(root / name) for name in protocol["source_input_hashes"]}
    unchanged = after == protocol["source_input_hashes"]
    if receipt["attempted_paths"] == 196:
        receipt["status"] = "COMPLETED_196_VALIDATED_ANNUAL_PATHS" if receipt["completed_paths"] == 196 and unchanged else "COMPLETED_FIXED_COHORT_WITH_RETAINED_FAILURES_OR_DRIFT"
    receipt.update(completed_utc=utc_now(), runtime_seconds=time.perf_counter() - started,
                   source_input_hashes_after=after, source_input_unchanged=unchanged,
                   planned_case_rows=len(results), complete_cohort_and_lead_summaries=summaries,
                   full_annual_rows=sum(r.get("completed_annual_rows", 0) for r in results),
                   full_action_rows=sum(r.get("completed_action_rows", 0) for r in results),
                   artifact_sha256={p.relative_to(out).as_posix(): digest(p) for p in out.rglob("*") if p.is_file() and p.name != "analysis.json"})
    write_json(out / "analysis.json", receipt)
    print(receipt["status"], receipt["completed_paths"], "of", receipt["attempted_paths"], flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = prepare(args.root) if args.prepare else run(args.root)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
