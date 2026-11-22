"""Execute the predeclared V6 gross-import and mechanism cohort and joint engineering design.

No parameter selection or 2025 validation tuning is performed here. All cases,
including null and adverse results, are saved before comparisons are computed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import pandas as pd
import scipy
from scipy.stats import qmc

from src import v3_annual as frozen
from src import v6_annual as model


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(root):
    files = ["src/v3_annual.py", "src/v3_annual_calibrated.py", "src/v3_calibration_candidate.py",
             "src/v5_annual.py", "src/v6_annual.py", "src/v6_annual_experiments.py", "config/v3_transition.json",
             "config/v6_annual_experiments.json", "tests/test_v5_annual.py", "src/v4_factorial.py",
             "data/annual_china.csv", "data/ember_power_china.csv", "data/supplement_carriers_2024.csv",
             "data/energy_production_china.csv", "data/fuel_supply_trade.csv", "data/v6_fuel_import_channels.csv", "tests/test_v6_annual.py"]
    return {name: sha(root / name) for name in files}


def required_metrics(frame,data):
    """Required contest outputs, with boundary maxima distinguished from peaks."""
    initial=data["annual"].loc[data["annual"].year==2024].iloc[0]
    rows=[]
    for scenario,group in frame.groupby("scenario",sort=False):
        last=group.loc[group.year==2060].iloc[0]
        row={"scenario":scenario,"evaluation_start_year":2024,"evaluation_end_year":2060,
            "energy_intensity_base_2024_100":float(last.energy_intensity_index_base_year_100),
            "energy_intensity_decline_from_2024_pct":float(100-last.energy_intensity_index_base_year_100),
            "carbon_intensity_base_2024_100":float(last.carbon_intensity_index_base_year_100),
            "carbon_intensity_decline_from_2024_pct":float(100-last.carbon_intensity_index_base_year_100),
            "gross_national_co2_2060_mt":float(last.gross_national_co2_mt),
            "fixed_sink_2060_mt":float(data["config"]["engineering"]["sink_mt"]),
            "carbon_neutralization_rate_2060_pct":float(100*last.carbon_neutralization_rate),
            "remaining_net_co2_2060_mt":float(last.net_co2_mt),
            "additional_removals_required_2060_mt":float(last.required_removal_mt),
            "carbon_neutral_target_met":bool(last.carbon_neutral_target_met),
            "cumulative_useful_service_unserved_twh":float(last.cumulative_useful_service_unserved_twh)}
        base_values={"energy_co2_mt":float(initial.energy_co2_mt),
            "gross_national_co2_mt":float(initial.energy_co2_mt+initial.cement_co2_mt+initial.other_industry_co2_mt)}
        for metric,observed in base_values.items():
            points=pd.Series([observed,*group[metric].tolist()],index=[2024,*group.year.tolist()])
            peak_year=int(points.idxmax())
            row[metric+"_peak_year"]=peak_year;row[metric+"_peak_mt"]=float(points.max())
            row[metric+"_peak_status"]=("simulation_end_boundary_not_confirmed_peak" if peak_year==2060 else
                "origin_boundary_peak_in_evaluation_window" if peak_year==2024 else "interior_maximum_in_evaluation_window")
        rows.append(row)
    return pd.DataFrame(rows)


def enrich(frame, actions, data):
    """Derive service losses independently from realised deliveries, not labels."""
    result = frame.copy()
    efficiency = data["config"]["engineering"]["direct_use_efficiency"]
    service = sum(result[f + "_direct_delivered_mtce"] * model.TWH_PER_MTCE * efficiency[j]
                  for j, f in enumerate(model.FUELS))
    service += result.electricity_generation_twh * efficiency[3]
    result["useful_service_delivered_twh"] = service
    result["useful_service_unserved_twh"] = np.maximum(result.useful_service_demand_twh - service, 0.)
    result["useful_service_unserved_fraction"] = result.useful_service_unserved_twh / result.useful_service_demand_twh
    result["direct_fuel_unserved_mtce"] = sum(result[f + "_direct_demand_mtce"] - result[f + "_direct_delivered_mtce"] for f in model.FUELS)
    result["cumulative_investment_million_cny"] = result.investment_spend_million_cny.cumsum()
    result["cumulative_energy_co2_mt"] = result.energy_co2_mt.cumsum()
    result["cumulative_useful_service_unserved_twh"] = result.useful_service_unserved_twh.cumsum()
    result["cumulative_electricity_unserved_twh"] = result.electricity_unserved_twh.cumsum()
    result["cumulative_direct_fuel_unserved_mtce"] = result.direct_fuel_unserved_mtce.cumsum()
    nonfossil_caps = sum(result[t + "_capacity_gw"] for t in model.TECHS[2:])
    result["nonfossil_electric_capacity_gw"] = nonfossil_caps
    result["nonfossil_electric_capacity_share"] = nonfossil_caps / sum(result[t + "_capacity_gw"] for t in model.TECHS)
    # Investments in annual fuel-production capacity carry Mtce/year units and
    # must never be added to GW orders. Monetary spending is comparable.
    nonfossil_spend = actions.loc[actions.agent.isin(model.TECHS[2:])].groupby("year").capital_spend_million_cny.sum()
    result["nonfossil_investment_million_cny"] = result.year.map(nonfossil_spend).fillna(0.)
    result["cumulative_nonfossil_investment_million_cny"] = result.nonfossil_investment_million_cny.cumsum()
    return result


def audit_saved(frame, actions, gate):
    """Recompute physical and action invariants from persisted CSV fields."""
    fuel_residual = max(float((frame[f + "_domestic_supply_mtce"] + frame[f + "_imports_mtce"]
                               - frame[f + "_stock_change_mtce"] - frame[f + "_direct_delivered_mtce"]
                               - frame[f + "_power_use_mtce"]).abs().max()) for f in model.FUELS)
    power_residual = float((frame.electricity_generation_twh + frame.electricity_unserved_twh - frame.electricity_demand_twh).abs().max())
    minimum_stock = min(float(frame[f + "_stock_mtce"].min()) for f in model.FUELS)
    budget_excess = float(np.maximum(frame.investment_spend_million_cny - frame.investment_budget_million_cny, 0.).max())
    spend_by_year = actions.groupby("year").capital_spend_million_cny.sum().reindex(frame.year).to_numpy()
    action_error = float(np.abs(spend_by_year - frame.investment_spend_million_cny.to_numpy()).max())
    desired_rows = gate["expected_rows_per_run"]
    if len(frame) != desired_rows or len(actions) != desired_rows * len(model.ASSETS):
        raise AssertionError("Incomplete annual cohort member")
    for values in (frame.select_dtypes(include=np.number), actions.drop(columns=["simulated_profit_million_cny", "simulated_revenue_million_cny", "simulated_operating_cost_million_cny", "annual_capital_charge_million_cny"]).select_dtypes(include=np.number)):
        if not np.isfinite(values.to_numpy()).all(): raise AssertionError("Nonfinite persisted annual result")
    if fuel_residual > gate["fuel_balance_atol_mtce"] or power_residual > gate["power_balance_atol_twh"]:
        raise AssertionError("Saved annual physical ledger fails")
    if minimum_stock < -gate["negative_stock_atol_mtce"] or budget_excess > gate["budget_atol_million_cny"] or action_error > gate["budget_atol_million_cny"]:
        raise AssertionError("Saved annual stock/capital/action invariant fails")
    return {"status": "PASS", "years": len(frame), "actions": len(actions), "fuel_balance_max_mtce": fuel_residual,
            "power_balance_max_twh": power_residual, "minimum_fuel_stock_mtce": minimum_stock,
            "budget_excess_million_cny": budget_excess, "action_spend_error_million_cny": action_error}


def recent_expectations(data, params, expectation_end=None):
    result = deepcopy(params)
    end = params["train_end"] if expectation_end is None else expectation_end
    a, _, _, service, _, _ = frozen._observations(data, end)
    for key, values in (("service_fit", service), ("electricity_fit", a.electricity_demand_twh.to_numpy(float))):
        result["expectations"][key] = {"mode": "increment3", "increment": float(np.diff(values)[-3:].mean())}
    result["v4_expectation_fit_end"] = int(end)
    return result


def run(root):
    root = Path(root).resolve()
    out = root / "outputs/v6/annual"
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((root / "config/v6_annual_experiments.json").read_text("utf-8"))
    before = source_hashes(root)
    receipt = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
               "command": "python -X utf8 -m src.v6_annual_experiments --root <project-root>", "argv": sys.argv,
               "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
               "source_input_hashes_before": before, "protocol": protocol, "cases": []}
    receipt_path = out / "run_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")
    try:
        data = model.load_data(root)
        params = model.calibrate(data, protocol["calibration_end"])
        (out / "calibration.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), "utf-8")
        (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), "utf-8")
        frames, action_frames, endpoints = [], [], []

        _,initial_runtime=model.initialize_state(data,params,protocol["initial_year"])
        pd.DataFrame(initial_runtime["initial_fuel_account"]).to_csv(out/"initial_fuel_account.csv",index=False)

        def execute(case_id, kind, scenario, case_params, mechanisms, overrides=None, design_values=None, import_reference_multiplier=1.):
            case_data=data
            if import_reference_multiplier!=1.:
                case_data=deepcopy(data)
                case_data["import_channels"]["gross_imports_mtce"]*=import_reference_multiplier
            frame, actions = model.simulate(case_data, case_params, scenario, start_year=protocol["initial_year"],
                                             end_year=protocol["final_year"], mechanisms=mechanisms,
                                             engineering_overrides=overrides)
            frame = enrich(frame, actions, data)
            frame.insert(0, "case", case_id); frame.insert(1, "experiment_kind", kind)
            actions.insert(0, "case", case_id); actions.insert(1, "experiment_kind", kind)
            traces = out / "trajectories"; traces.mkdir(exist_ok=True)
            frame_path = traces / (case_id + ".csv"); action_path = traces / (case_id + "_actions.csv")
            frame.to_csv(frame_path, index=False); actions.to_csv(action_path, index=False)
            saved_frame = pd.read_csv(frame_path); saved_actions = pd.read_csv(action_path)
            audit = audit_saved(saved_frame, saved_actions, protocol["execution_gate"])
            receipt["cases"].append({"case": case_id, "kind": kind, "scenario": scenario if isinstance(scenario, str) else scenario,
                                      "mechanisms": model.resolve_mechanisms(mechanisms), "engineering_overrides": overrides or {},
                                      "expectation_fit_end": case_params.get("v4_expectation_fit_end", case_params["train_end"]),
                                      "expectations": case_params["expectations"],
                                      "design_values": design_values or {}, "import_reference_multiplier":import_reference_multiplier, "audit": audit,
                                      "trajectory": str(frame_path.relative_to(root)).replace("\\", "/"), "trajectory_sha256": sha(frame_path),
                                      "actions": str(action_path.relative_to(root)).replace("\\", "/"), "actions_sha256": sha(action_path)})
            frames.append(frame); action_frames.append(actions)
            endpoint = frame.loc[frame.year.isin(protocol["endpoint_years"])].copy()
            if design_values:
                for key, value in design_values.items(): endpoint["design_" + key] = value
            endpoints.append(endpoint)
            # Keep failure evidence recoverable even if a later case raises.
            receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")

        for scenario in protocol["scenarios"]:
            for case in protocol["mechanism_cases"]:
                fit_end = protocol["initial_year"] if case.get("expectations") == "initial_year_recent_increment3" else params["train_end"]
                case_params = recent_expectations(data, params, fit_end) if case.get("expectations") else params
                execute(scenario + "__" + case["case"], "mechanism", scenario, case_params, case["mechanisms"])

        for scenario in protocol["scenarios"]:
            for tier in protocol["scarcity_price_sensitivity"]["step_tiers_cny_mwh"]:
                execute(scenario+"__scarcity_step_"+str(tier),"price_sensitivity",scenario,params,{},
                        {"market_scarcity_price_cny_mwh":tier})
            execute(scenario+"__scarcity_smooth_1200","price_sensitivity",scenario,params,
                    {"market_price_rule":"smooth"},{"market_scarcity_price_cny_mwh":1200})
            for multiplier in protocol["import_conversion_sensitivity"]["multipliers"]:
                execute(scenario+"__import_reference_"+str(multiplier),"import_reference_sensitivity",scenario,params,{},
                        import_reference_multiplier=multiplier)

        design = protocol["joint_design"]
        keys = list(design["parameters"])
        bounds = np.array([design["parameters"][key] for key in keys])
        unit_design = qmc.LatinHypercube(d=len(keys), scramble=True, seed=design["seed"]).random(n=design["n_parameter_vectors"])
        values = qmc.scale(unit_design, bounds[:, 0], bounds[:, 1])
        design_frame = pd.DataFrame(values, columns=keys)
        design_frame.insert(0, "case", ["joint_%03d" % j for j in range(len(values))])
        design_frame.to_csv(out / "joint_design.csv", index=False)
        engineering = data["config"]["engineering"]
        for _, vector in design_frame.iterrows():
            factors = {key: float(vector[key]) for key in keys}
            scenario = deepcopy(data["config"]["scenarios"][design["scenario"]])
            scenario["gdp_growth"] = [v * factors["gdp_growth_multiplier"] for v in scenario["gdp_growth"]]
            scenario["carbon_price_drift_cny_t"] *= factors["carbon_price_drift_multiplier"]
            for key in ("technology_cost_multiplier", "cost_decline_per_year", "annual_budget_multiplier", "construction_delay_multiplier", "supply_multiplier", "import_multiplier"):
                scenario[key] = factors[key]
            overrides = {"cost_response": engineering["cost_response"] * factors["consumer_response_multiplier"],
                         "shortage_response": engineering["shortage_response"] * factors["consumer_response_multiplier"],
                         "investment_feedback": engineering["investment_feedback"] * factors["investor_response_multiplier"]}
            case_params = deepcopy(params)
            case_params["expectations"]["utilisation_response"] *= factors["investor_response_multiplier"]
            execute(str(vector["case"]), "joint_design", scenario, case_params, {}, overrides, factors)

        screening=[]
        _,origin_runtime=model.initialize_state(data,params)
        production=np.array([a["observed_production_mtce"] for a in origin_runtime["initial_fuel_account"]])
        consumption=np.array(origin_runtime["base_fuel"])
        reference=data["import_channels"].set_index("fuel").loc[list(model.FUELS),"gross_imports_mtce"].to_numpy(float)
        for scenario in protocol["scenarios"]:
            for multiplier in protocol["import_conversion_sensitivity"]["initial_attempt_multipliers"]:
                retained=(consumption-reference*multiplier)/production
                screening.append({"scenario":scenario,"initial_attempt_multiplier":multiplier,
                    "status":"REJECTED_DOMESTIC_RETENTION_ABOVE_ONE" if (retained>1+1e-12).any() else "PHYSICALLY_FEASIBLE_INPUT",
                    **{fuel+"_domestic_retention_fraction":float(retained[j]) for j,fuel in enumerate(model.FUELS)},
                    "preserved_invalid_attempt_directory":"rejected_import_reference_attempts" if (retained>1+1e-12).any() else ""})
        pd.DataFrame(screening).to_csv(out/"import_reference_initial_attempt_screening.csv",index=False)
        trajectories = pd.concat(frames, ignore_index=True)
        all_actions = pd.concat(action_frames, ignore_index=True)
        all_endpoints = pd.concat(endpoints, ignore_index=True)
        trajectories.to_csv(out / "all_trajectories.csv", index=False)
        all_actions.to_csv(out / "all_actions.csv", index=False)
        mechanism_endpoints = all_endpoints.loc[all_endpoints.experiment_kind == "mechanism"].copy()
        mechanism_endpoints.to_csv(out / "mechanism_endpoints.csv", index=False)
        all_endpoints.loc[all_endpoints.experiment_kind=="price_sensitivity"].to_csv(out/"scarcity_price_sensitivity.csv",index=False)
        all_endpoints.loc[all_endpoints.experiment_kind=="import_reference_sensitivity"].to_csv(out/"import_reference_sensitivity.csv",index=False)
        full=trajectories.loc[trajectories["case"].str.endswith("__full")].copy()
        full.to_csv(out / "annual_scenarios.csv", index=False)
        required_metrics(full,data).to_csv(out/"contest_required_metrics.csv",index=False)
        (out/"required_metrics_definitions.json").write_text(json.dumps({
            "peak_window":"Observed 2024 origin plus simulated 2025-2060; report an end-boundary maximum as not confirmed peaked.",
            "energy_intensity":"Energy total Mtce divided by the scenario real-GDP index, normalized to 2024=100.",
            "carbon_intensity":"Gross energy-plus-industry CO2 divided by the scenario real-GDP index, normalized to 2024=100.",
            "neutralization_rate":"min(fixed 800 Mt engineering sink / gross energy-plus-industry CO2,1); target met iff net emissions <=0.",
            "remaining_emissions":"Gross energy-plus-industry CO2 minus the fixed engineering sink; additional removals=max(net,0).",
        },ensure_ascii=False,indent=2),"utf-8")
        snapshots={f"{row.scenario}_{int(row.year)}":model.snapshot_from_row(row)
                   for _,row in full.loc[full.year.isin(protocol["endpoint_years"])].iterrows()}
        (out / "snapshots.json").write_text(json.dumps(snapshots,ensure_ascii=False,indent=2),"utf-8")
        joint_endpoints = all_endpoints.loc[all_endpoints.experiment_kind == "joint_design"].copy()
        joint_endpoints.to_csv(out / "joint_endpoints.csv", index=False)
        deltas = []
        for scenario in protocol["scenarios"]:
            baseline = mechanism_endpoints.loc[mechanism_endpoints["case"] == scenario + "__full"].set_index("year")
            for _, row in mechanism_endpoints.loc[mechanism_endpoints.scenario == scenario].iterrows():
                record = {"case": row["case"], "scenario": scenario, "year": int(row.year)}
                for metric in protocol["comparisons"]["metrics"]:
                    record[metric] = float(row[metric]); record["full_" + metric] = float(baseline.loc[row.year, metric])
                    record["delta_" + metric] = float(row[metric] - baseline.loc[row.year, metric])
                record["delta_nonfossil_percentage_points"] = record["delta_nonfossil_share"] * 100
                deltas.append(record)
        pd.DataFrame(deltas).to_csv(out / "mechanism_contributions.csv", index=False)
        range_rows, associations = [], []
        for year, group in joint_endpoints.groupby("year"):
            for metric in protocol["comparisons"]["metrics"]:
                range_rows.append({"year": int(year), "metric": metric, "n_design_cases": len(group),
                                   "minimum": float(group[metric].min()), "median": float(group[metric].median()),
                                   "maximum": float(group[metric].max()), "interpretation": "range over declared engineering design; not probability interval"})
                for key in keys:
                    # Pearson correlation of ranks is Spearman's descriptive
                    # association, used here without significance/causal claims.
                    association = group["design_" + key].rank().corr(group[metric].rank())
                    associations.append({"year": int(year), "metric": metric, "parameter": key,
                                         "spearman_rank_association": float(association) if np.isfinite(association) else None,
                                         "interpretation": "descriptive within this joint design; not causal variance allocation"})
        pd.DataFrame(range_rows).to_csv(out / "joint_design_ranges.csv", index=False)
        pd.DataFrame(associations).to_csv(out / "joint_design_rank_associations.csv", index=False)
        counts = pd.Series([case["kind"] for case in receipt["cases"]]).value_counts()
        gate = protocol["execution_gate"]
        if int(counts.get("mechanism", 0)) != gate["expected_mechanism_runs"] or int(counts.get("joint_design", 0)) != gate["expected_joint_runs"]:
            raise AssertionError("Incomplete fixed cohort; no partial comparison accepted")
        if int(counts.get("price_sensitivity",0))!=16 or int(counts.get("import_reference_sensitivity",0))!=8:
            raise AssertionError("Incomplete sensitivity cohort")
        receipt["source_input_hashes_after"] = source_hashes(root)
        if receipt["source_input_hashes_after"] != before: raise AssertionError("Source/input changed during fixed cohort")
        receipt.update(status="PASS_COMPLETE_FIXED_COHORT", completed_utc=datetime.now(timezone.utc).isoformat(),
                       mechanism_runs=int(counts["mechanism"]), joint_design_runs=int(counts["joint_design"]),
                       annual_rows=len(trajectories), investment_action_rows=len(all_actions),
                       worst_fuel_balance_mtce=max(c["audit"]["fuel_balance_max_mtce"] for c in receipt["cases"]),
                       worst_power_balance_twh=max(c["audit"]["power_balance_max_twh"] for c in receipt["cases"]))
        receipt["output_sha256"] = {str(path.relative_to(out)).replace("\\", "/"): sha(path)
                                    for path in sorted(out.rglob("*")) if path.is_file() and path != receipt_path}
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")
        print(json.dumps({key: receipt[key] for key in ("status", "mechanism_runs", "joint_design_runs", "annual_rows", "investment_action_rows", "worst_fuel_balance_mtce", "worst_power_balance_twh")}, ensure_ascii=False, indent=2))
    except Exception:
        receipt.update(status="FAILED_PARTIAL_NOT_FOR_SELECTION", failed_utc=datetime.now(timezone.utc).isoformat(), error=traceback.format_exc())
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")
        raise



def supplementary(root):
    """Two predeclared consumer-off definitions, shared-component history."""
    from src import v5_annual as legacy
    from src.v4_factorial import CELLS,PLAYERS,decompose
    root=Path(root);out=root/"outputs/v6/annual"
    protocol=json.loads((root/"config/v6_annual_experiments.json").read_text("utf-8"))
    data=model.load_data(root);params=model.calibrate(data,2020);before=source_hashes(root)
    gate=protocol["execution_gate"]
    for disabled,directory in (("exogenous_trend","factorial"),("frozen_initial","factorial_frozen_shares")):
        factorout=out/directory;factorout.mkdir(exist_ok=True)
        values,shapley,interactions,records=[],[],[],[]
        for scenario in protocol["scenarios"]:
            endpoints={}
            for bits in CELLS:
                rules={"carbon_feedback":bits[0]=="1","consumer_choice":"responsive" if bits[1]=="1" else disabled,
                       "investor_profit_response":bits[2]=="1","investor_utilisation_response":bits[2]=="1"}
                frame,actions=model.simulate(data,params,scenario,mechanisms=rules)
                frame=enrich(frame,actions,data);name=scenario+"__factorial_"+bits
                trace=factorout/(name+".csv");act=factorout/(name+"_actions.csv")
                frame.to_csv(trace,index=False);actions.to_csv(act,index=False)
                physical=audit_saved(pd.read_csv(trace),pd.read_csv(act),gate)
                records.append({"scenario":scenario,"bits":bits,"origin":"fresh_v6_run","audit":physical,
                    "consumer_disabled_definition":disabled,"trajectory_sha256":sha(trace),"actions_sha256":sha(act)})
                for year in protocol["endpoint_years"]:
                    row=frame.loc[frame.year==year].iloc[0];endpoints[bits,year]=row
                    values.append({"scenario":scenario,"bits":bits,"year":year,"origin":"fresh_v6_run",
                        "consumer_disabled_definition":disabled,**{m:float(row[m]) for m in protocol["comparisons"]["metrics"]}})
            for year in protocol["endpoint_years"]:
                for metric in protocol["comparisons"]["metrics"]:
                    v={bits:float(endpoints[bits,year][metric]) for bits in CELLS};result=decompose(v)
                    assert abs(result["shapley_sum_error"])<1e-7 and abs(result["reconstruction_error"])<1e-7
                    for player,contribution in result["shapley"].items():
                        shapley.append({"scenario":scenario,"year":year,"metric":metric,"player":player,
                            "consumer_disabled_definition":disabled,"contribution":contribution,"all_disabled_value":v["000"],
                            "full_value":v["111"],"full_minus_disabled":result["total_effect"],"sum_error":result["shapley_sum_error"]})
                    for bits,interaction in result["interactions"].items():
                        interactions.append({"scenario":scenario,"year":year,"metric":metric,
                            "consumer_disabled_definition":disabled,"mechanisms":"+".join(PLAYERS[j] for j,b in enumerate(bits) if b=="1"),
                            "order":bits.count("1"),"interaction":interaction,"full_minus_disabled":result["total_effect"],
                            "reconstruction_error":result["reconstruction_error"]})
        pd.DataFrame(values).to_csv(factorout/"factorial_values.csv",index=False)
        pd.DataFrame(shapley).to_csv(factorout/"shapley_contributions.csv",index=False)
        pd.DataFrame(interactions).to_csv(factorout/"nonadditive_interactions.csv",index=False)
        assert len(records)==32
        (factorout/"run_receipt.json").write_text(json.dumps({"status":"PASS_32_FRESH_V6_FACTORIAL_RUNS",
            "factorial_cells":32,"new_runs":32,"reused_runs":0,"bit_order":list(PLAYERS),
            "consumer_disabled_definition":disabled,"cases":records},indent=2),"utf-8")
    historical=[]
    for origin in range(2017,2024):
        fitted=legacy.calibrate(data,origin)
        a,g,k,svc,shares,direct=frozen._observations(data,origin)
        observed=data["annual"].set_index("year")
        for h in (1,3):
            if origin+h>2024:continue
            scenario={"name":"baseline","cost_decline_per_year":0,"carbon_feedback":False,"carbon_price_drift_cny_t":0}
            frame,_=legacy.simulate(data,fitted,scenario,origin+h,origin)
            future=data["power"].loc[data["power"].year==origin+h].set_index("technology")
            for target in [t+"_capacity_gw" for t in model.TECHS]+["electricity_demand_twh"]:
                if target=="electricity_demand_twh":
                    actual=float(observed.loc[origin+h,target]);base=float(observed.loc[origin,target]);slope=float(np.diff(a.electricity_demand_twh.tail(4)).mean())
                else:
                    tech=target.removesuffix("_capacity_gw");j=model.TECHS.index(tech)
                    actual=float(future.loc[tech,"capacity_gw"]);base=float(k[-1,j]);slope=float(np.diff(k[-4:,j]).mean())
                for method,pred in {"dynamic_abm":float(frame.iloc[-1][target]),"persistence":base,"recent_increment3":base+h*slope}.items():
                    historical.append({"origin_year":origin,"horizon":h,"target_year":origin+h,"target":target,"method":method,
                        "validation_scope":"Inherited V5 expectations/asset component; observed GDP conditional; excludes V6 gross-import remapping",
                        "observed":actual,"predicted":pred,"ape_percent":abs(pred/actual-1)*100})
    hist=pd.DataFrame(historical);hist.to_csv(out/"historical_rolling.csv",index=False)
    hist.groupby(["target","method","horizon"]).ape_percent.agg(["mean","count"]).reset_index().to_csv(out/"historical_metrics.csv",index=False)
    comparison=[]
    for scenario in protocol["scenarios"]:
        current,current_actions=model.simulate(data,params,scenario);oldframe,oldactions=legacy.simulate(data,params,scenario)
        oldframe=enrich(oldframe,oldactions,data)
        oldframe.to_csv(out/"trajectories"/(scenario+"__legacy_v5.csv"),index=False)
        oldactions.to_csv(out/"trajectories"/(scenario+"__legacy_v5_actions.csv"),index=False)
        current=enrich(current,current_actions,data)
        for year in protocol["endpoint_years"]:
            n=current.loc[current.year==year].iloc[0];o=oldframe.loc[oldframe.year==year].iloc[0]
            for metric in ["energy_co2_mt","net_co2_mt","nonfossil_share","electricity_unserved_twh","useful_service_unserved_fraction","cumulative_useful_service_unserved_twh","investment_spend_million_cny"]:
                comparison.append({"scenario":scenario,"year":year,"metric":metric,"v5_value":float(o[metric]),
                    "v6_value":float(n[metric]),"v6_minus_v5":float(n[metric]-o[metric])})
    pd.DataFrame(comparison).to_csv(out/"v5_v6_comparison.csv",index=False)
    assert source_hashes(root)==before
    report={"status":"PASS_FRESH_RESPONSE_ONLY_AND_FROZEN_SHARE_FACTORIALS_AND_SHARED_COMPONENT_HISTORY",
        "factorial_fresh_runs":64,"historical_origin_horizon_pairs":12,"historical_predictions":len(hist),
        "historical_condition":"Actual future GDP conditional; inherited V5 expectations/assets only. V6 2024 gross channels are never backfilled into earlier origins.",
        "source_hashes":before,"output_sha256":{str(p.relative_to(out)).replace(chr(92),"/"):sha(p) for p in out.rglob("*")
            if p.is_file() and p.name!="supplementary_audit.json"}}
    (out/"supplementary_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),"utf-8")
    print(report["status"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();run(args.root);supplementary(args.root)
