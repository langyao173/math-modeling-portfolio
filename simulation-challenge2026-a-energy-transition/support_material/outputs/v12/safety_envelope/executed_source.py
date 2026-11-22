"""A frozen 11x11 import/new-order-delay diagnostic for four existing controls.

No control is optimized; the V11 independent-design cohort is never read.
AI-assisted: OpenAI Codex, 2026-10-19.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src import v6_annual as annual
from src.v5_planning import decode, digest, reliability_feasible, summarize
from src.v11_robust_plans import assert_metrics, yearly_reliability

ROOT = Path(__file__).resolve().parents[1]
PLAN_IDS = ("plan_02349", "repair_005", "repair_006", "repair_007")
NAMES = {"plan_02349": "原折中", "repair_005": "既有终选005", "repair_006": "开发取舍006", "repair_007": "开发取舍007"}
BLUE, GREEN, ORANGE, INK, LIGHT = "#0072B2", "#009E73", "#D55E00", "#26343D", "#E4E9ED"


def fixed_scenario(vector, name, imports, delay):
    return {**decode(vector, name), "import_multiplier": float(imports), "construction_delay_multiplier": float(delay)}


def extract_boundaries(grid, anchors=None):
    """Report observed minima and every feasible tested point, preserving holes."""
    rows = []
    for (plan, delay), group in grid.groupby(["plan_id", "construction_delay_multiplier"], sort=False):
        ordered = group.sort_values("import_multiplier")
        extra = anchors.loc[anchors.plan_id.eq(plan) & anchors.construction_delay_multiplier.eq(delay)] if anchors is not None else group.iloc[0:0]
        all_tested = pd.concat([group, extra], ignore_index=True).sort_values("import_multiplier")
        feasible = all_tested.loc[all_tested.annual_reliability_feasible, "import_multiplier"].tolist()
        feasible_grid = ordered.loc[ordered.annual_reliability_feasible, "import_multiplier"].tolist()
        minimum = min(feasible) if feasible else None
        rows.append({"plan_id": plan, "construction_delay_multiplier": float(delay),
                     "minimum_grid_feasible_import_multiplier": min(feasible_grid) if feasible_grid else None,
                     "minimum_tested_feasible_import_multiplier": minimum,
                     "feasible_tested_import_multipliers": feasible,
                     "feasible_grid_import_multipliers": feasible_grid,
                     "upward_closed_on_tested_grid": bool(ordered.loc[ordered.import_multiplier.ge(minimum), "annual_reliability_feasible"].all()) if minimum is not None else None,
                     "upward_closed_on_all_tested_points": bool(all_tested.loc[all_tested.import_multiplier.ge(minimum), "annual_reliability_feasible"].all()) if minimum is not None else None,
                     "grid_feasible_count": len(feasible_grid), "grid_tested_count": len(group),
                     "additional_nominal_anchors": len(extra)})
    return pd.DataFrame(rows)


def independent_audit(frame, actions, initial, parameters, vector, scenario):
    """Reconstruct physical and capital identities from primitive result columns."""
    assert frame.year.tolist() == list(range(2025, 2061)) and len(actions) == 432
    assert not actions.duplicated(["year", "agent"]).any()
    assert actions.groupby("year").agent.apply(set).map(lambda assets: assets == set(annual.ASSETS)).all()
    physical = actions[["order", "commissioned", "retired", "capital_spend_million_cny"]].to_numpy()
    assert np.isfinite(physical).all() and physical.min() >= -1e-8
    residuals = {}
    generated = frame[[tech + "_generation_twh" for tech in annual.TECHS]].sum(axis=1)
    residuals["electricity_reconstructed_twh"] = float((generated + frame.electricity_unserved_twh - frame.electricity_demand_twh).abs().max())
    residuals["generation_sum_twh"] = float((generated - frame.electricity_generation_twh).abs().max())
    for i, fuel in enumerate(annual.FUELS):
        balance = (frame[fuel + "_domestic_supply_mtce"] + frame[fuel + "_imports_mtce"]
                   - frame[fuel + "_stock_change_mtce"] - frame[fuel + "_direct_delivered_mtce"] - frame[fuel + "_power_use_mtce"])
        stocks = frame[fuel + "_stock_mtce"].to_numpy()
        continuity = stocks - np.r_[initial[17 + i], stocks[:-1]] - frame[fuel + "_stock_change_mtce"].to_numpy()
        residuals[fuel + "_reconstructed_balance_mtce"] = float(balance.abs().max())
        residuals[fuel + "_stock_continuity_mtce"] = float(np.max(np.abs(continuity)))
        assert stocks.min() >= -1e-8
    spent = actions.groupby("year").capital_spend_million_cny.sum().reindex(frame.year).to_numpy()
    residuals["action_to_annual_spend_million_cny"] = float(np.max(np.abs(spent - frame.investment_spend_million_cny.to_numpy())))
    order_cost_error = 0.
    for j, asset in enumerate(annual.ASSETS):
        suffix = "_unit_cost_million_cny_per_gw" if j < 9 else "_unit_cost_million_cny_per_mtce_per_year"
        part = actions.loc[actions.agent.eq(asset)].sort_values("year")
        expected = part.order.to_numpy() * frame[asset + suffix].to_numpy()
        order_cost_error = max(order_cost_error, float(np.max(np.abs(expected - part.capital_spend_million_cny.to_numpy()))))
    residuals["order_times_unit_cost_million_cny"] = order_cost_error
    periods = np.where(frame.year <= 2030, 0, np.where(frame.year <= 2040, 1, 2))
    expected_budget = (parameters["base_budget"] * (frame.useful_service_demand_twh.to_numpy() / parameters["initial_service"]) ** .7
                       * scenario["annual_budget_multiplier"] * np.asarray(vector[:3])[periods])
    residuals["budget_formula_million_cny"] = float(np.max(np.abs(expected_budget - frame.investment_budget_million_cny.to_numpy())))
    residuals["budget_exceedance_million_cny"] = float(np.maximum(spent - expected_budget, 0).max())
    assert all(np.isfinite(v) and v < (1e-5 if name.endswith("million_cny") else 1e-7) for name, v in residuals.items()), residuals
    return residuals


def controls(root, decision_names):
    selected = pd.read_csv(root / "outputs/v6/planning/selected_cases.csv")
    candidates = pd.read_csv(root / "outputs/v11/robust_repair/candidate_scores.csv")
    frozen = json.loads((root / "outputs/v11/robust_repair/selected_before_independent_test.json").read_text("utf-8"))
    assert frozen["candidate"]["candidate_id"] == "repair_005"
    source_rows = {"plan_02349": selected.loc[selected.case_id.eq("plan_02349")].iloc[0]}
    source_rows.update({name: candidates.loc[candidates.candidate_id.eq(name)].iloc[0] for name in PLAN_IDS[1:]})
    vectors = {name: [float(source_rows[name][field]) for field in decision_names] for name in PLAN_IDS}
    assert vectors["repair_005"] == [float(frozen["candidate"][field]) for field in decision_names]
    return vectors, source_rows


def plot(root=ROOT):
    root = Path(root)
    out, figures = root / "outputs/v12/safety_envelope", root / "figures/V12/safety_envelope"
    results = pd.read_csv(out / "grid_metrics.csv")
    anchors = pd.read_csv(out / "nominal_anchors.csv")
    boundary = pd.read_csv(out / "safety_boundary.csv")
    protocol = json.loads((out / "protocol.json").read_text("utf-8"))
    imports, delays = protocol["import_multipliers"], protocol["construction_delay_multipliers"]
    font = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": [font_manager.FontProperties(fname=font).get_name(), "DejaVu Sans"],
                         "font.size": 8.8, "axes.labelsize": 8.8, "axes.titlesize": 9.2,
                         "xtick.labelsize": 8.8, "ytick.labelsize": 8.8, "legend.fontsize": 8.8,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.unicode_minus": False, "text.color": INK,
                         "axes.labelcolor": INK, "axes.linewidth": .6, "legend.frameon": False})
    fig = plt.figure(figsize=(160 / 25.4, 133 / 25.4))
    gs = fig.add_gridspec(2, 2, left=.11, right=.85, bottom=.12, top=.94, hspace=.64, wspace=.40, height_ratios=[1, .85])
    a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    pair = results.loc[results.plan_id.isin(PLAN_IDS[:2])]
    vmax = max(float(pair.worst_shortfall_fraction.max()) * 100, 1.01)
    norm = Normalize(0, vmax)
    for letter, name, ax in [("a", PLAN_IDS[0], a), ("b", PLAN_IDS[1], b)]:
        q = results.loc[results.plan_id.eq(name)]
        fail = q.loc[~q.annual_reliability_feasible]
        passed = q.loc[q.annual_reliability_feasible]
        image = ax.scatter(fail.import_multiplier, fail.construction_delay_multiplier,
                           c=fail.worst_shortfall_fraction * 100, cmap="Oranges", norm=norm,
                           marker="s", s=24, edgecolors="none")
        ax.scatter(passed.import_multiplier, passed.construction_delay_multiplier, marker="s", s=24, color=GREEN)
        ax.plot(1, 1, marker="*", ms=7.2, ls="none", markerfacecolor="white",
                markeredgecolor=INK, markeredgewidth=.8, clip_on=False, zorder=5)
        ax.set_title(f"{letter}  {NAMES[name]}：{len(passed)}/121格点达标", loc="left", fontweight="bold", pad=8)
        ax.set(xlabel="进口可用倍率", ylabel="新订单工期倍率", xlim=(.47, 1.13), ylim=(.95, 2.05),
               xticks=[.5, .8, 1.1], yticks=[1, 1.5, 2])
    colorbar = fig.colorbar(image, cax=fig.add_axes([.895, .62, .018, .29]))
    colorbar.set_label("未达标格点的最大缺失率 / %")
    for name, color, marker in [(PLAN_IDS[0], BLUE, "o"), (PLAN_IDS[1], ORANGE, "s")]:
        q = boundary.loc[boundary.plan_id.eq(name)]
        available = q.minimum_tested_feasible_import_multiplier.notna()
        c.scatter(q.loc[available, "construction_delay_multiplier"], q.loc[available, "minimum_tested_feasible_import_multiplier"],
                  s=40 if name == PLAN_IDS[0] else 24, marker=marker, facecolor="none", edgecolor=color,
                  linewidth=1.1, label=NAMES[name], zorder=4)
        missing = q.loc[~available]
        if len(missing):
            c.scatter(missing.construction_delay_multiplier, np.full(len(missing), 1.17),
                      marker="+" if name == PLAN_IDS[0] else "x", color=color, s=22)
    c.axhline(1, color=LIGHT, lw=.7, ls=":")
    c.set_title("c  每种工期下的最小已检验可行进口值", loc="left", fontweight="bold", pad=8)
    c.set(xlabel="新订单工期倍率", ylabel="最小已测进口倍率", xlim=(.95, 2.05), ylim=(.45, 1.22),
          xticks=[1, 1.2, 1.4, 1.6, 1.8, 2], yticks=[.5, .8, 1, 1.1, 1.17])
    c.set_yticklabels(["0.50", "0.80", "1.00", "1.10", "未找到"])
    c.grid(axis="y", color=LIGHT, lw=.4)
    c.legend(ncol=2, loc="lower right", bbox_to_anchor=(1, 1.0), handlelength=1)
    fig.legend(handles=[Line2D([], [], marker="s", color=GREEN, ls="none", label="全部五服务达标", ms=4),
                        Line2D([], [], marker="*", markerfacecolor="white", color=INK, ls="none", label="单列名义锚点", ms=7)],
               loc="lower center", ncol=2, bbox_to_anchor=(.52, .005), columnspacing=2, handlelength=1)
    figures.mkdir(parents=True, exist_ok=True)
    outputs = {}
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    visible_ticks, all_ticks = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                all_ticks.update((id(tick.label1), id(tick.label2)))
            for tick in axis._update_ticks():
                visible_ticks.update((id(tick.label1), id(tick.label2)))
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text() and (id(text) not in all_ticks or id(text) in visible_ticks):
            box = text.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
                outside.append(text.get_text())
    assert not outside, outside
    for ext in ("pdf", "svg", "png"):
        path = figures / f"figK_fixed_control_safety_envelope.{ext}"
        fig.savefig(path, dpi=300, facecolor="white")
        outputs[path.relative_to(root).as_posix()] = digest(path)
    plt.close(fig)
    audit = {"status": "DRAWN_AWAITING_ACTUAL_VISUAL_INSPECTION", "paper_width_mm": 160, "paper_height_mm": 133,
             "minimum_requested_font_pt": 8.8, "source_sha256": {p.name: digest(p) for p in [out / "grid_metrics.csv", out / "nominal_anchors.csv", out / "safety_boundary.csv", out / "protocol.json"]},
             "artifact_sha256": outputs, "renderer_sha256": digest(__file__), "out_of_canvas_text": outside,
             "semantics": "Squares are 484 tested grid points; only the two displayed controls are plotted. Stars are separate actually run nominal anchors. Boundary markers include these anchors at delay=1 and do not interpolate, connect or claim continuous feasibility. Missing solutions remain null in CSV/JSON; x in the labelled missing band is not an import value."}
    (out / "figures_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), "utf-8")
    return audit


def run(root=ROOT):
    root = Path(root).resolve()
    out = root / "outputs/v12/safety_envelope"
    if (out / "protocol.json").exists():
        raise FileExistsError("Fixed protocol already exists; preserve this cohort and use a fresh root for reproduction")
    out.mkdir(parents=True, exist_ok=True)
    planning = json.loads((root / "config/v6_annual_planning.json").read_text("utf-8"))
    joint = json.loads((root / "config/v6_annual_experiments.json").read_text("utf-8"))["joint_design"]
    vectors, expected_nominal = controls(root, planning["decision_names"])
    imports = np.linspace(*joint["parameters"]["import_multiplier"], 11).tolist()
    delays = np.linspace(*joint["parameters"]["construction_delay_multiplier"], 11).tolist()
    assert len(imports) == len(delays) == 11
    names = ["src/v12_safety_envelope.py", "tests/test_v12_safety_envelope.py", "src/v11_robust_plans.py", "src/v6_annual.py",
             "src/v5_planning.py", "src/v3_annual.py", "src/v3_annual_calibrated.py", "src/v3_calibration_candidate.py",
             "config/v3_transition.json", "config/v6_annual_experiments.json", "config/v6_annual_planning.json",
             "data/annual_china.csv", "data/ember_power_china.csv", "data/energy_production_china.csv", "data/fuel_supply_trade.csv",
             "data/supplement_carriers_2024.csv", "data/v6_fuel_import_channels.csv", "outputs/v6/planning/selected_cases.csv",
             "outputs/v11/robust_repair/candidate_scores.csv", "outputs/v11/robust_repair/selected_before_independent_test.json"]
    sources = {name: digest(root / name) for name in names}
    protocol = {"status": "FROZEN_TWO_AXIS_FIXED_CONTROL_DIAGNOSTIC", "frozen_before_first_solve_utc": datetime.now(timezone.utc).isoformat(),
                "expected_grid_paths": 484, "expected_additional_nominal_anchors": 4,
                "import_multipliers": imports, "construction_delay_multipliers": delays,
                "controls": vectors, "decision_names": planning["decision_names"], "source_sha256": sources,
                "joint_design_bounds_source": "config/v6_annual_experiments.json:joint_design.parameters",
                "nominal_anchor": {"import_multiplier": 1., "construction_delay_multiplier": 1., "separate_from_grid": True},
                "reliability": planning["reliability"], "comparison_tolerance": 1e-9,
                "semantics": "2025–2060 import-availability and new-order construction-delay multipliers; all other economic, engineering and behavioural factors nominal. Initial construction queue unchanged. ceil(native lead*delay) also affects order anticipation and biomass horizon via the unchanged model.",
                "boundary": "Minimum observed feasible import at each delay, with grid-only and all-tested (including nominal anchor) minima separately recorded. No interpolation, reoptimization, probability weighting or global-optimum claim.",
                "independent_evaluation": "The independent 48-design file and all its result tables are not opened or used. These four controls predate this diagnostic."}
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), "utf-8")
    grid = pd.DataFrame([{"grid_id": f"grid_{i:02d}_{j:02d}", "import_multiplier": x, "construction_delay_multiplier": y}
                         for j, y in enumerate(delays) for i, x in enumerate(imports)])
    grid.to_csv(out / "grid_frozen.csv", index=False)
    pd.DataFrame([{"plan_id": name, **dict(zip(planning["decision_names"], vector))} for name, vector in vectors.items()]).to_csv(out / "controls_frozen.csv", index=False)
    shutil.copyfile(__file__, out / "executed_source.py")
    started = time.perf_counter()
    receipt = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(), "command": "python -B -X utf8 -m src.v12_safety_envelope",
               "argv": sys.argv, "protocol_sha256": digest(out / "protocol.json"), "grid_sha256": digest(out / "grid_frozen.csv"),
               "controls_sha256": digest(out / "controls_frozen.csv"), "source_input_hashes_before": sources,
               "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}
    (out / "analysis.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")
    data = annual.load_data(root)
    parameters = annual.calibrate(data, 2020)
    initial, engineering = annual.initialize_state(data, parameters, 2024)
    oil_index = annual.ASSETS.index("fuel_oil")
    oil_lead = float(data["config"]["engineering"]["construction_years"][oil_index])
    initial_oil_pipeline_2030 = float(initial[20:annual.QEND].reshape(12, annual.QDEPTH)[oil_index, :6].sum())
    (out / "calibration.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), "utf-8")
    all_metrics, all_yearly, audits, diagnostic, anchors = [], [], [], [], []
    for plan_id, vector in vectors.items():
        nominal_scenario = {**data["config"]["scenarios"]["baseline"], **fixed_scenario(vector, plan_id, 1, 1)}
        frame, actions = annual.simulate(data, parameters, nominal_scenario, investor_ledger=False)
        metrics = summarize(frame, actions, data, planning)
        expected = expected_nominal[plan_id]
        if plan_id == "plan_02349":
            error = assert_metrics(metrics, expected, plan_id)
        else:
            error = assert_metrics(metrics, {key: expected["nominal_" + key] for key in metrics}, plan_id)
        yearly = yearly_reliability(frame, data, planning)
        worst = yearly.loc[yearly.shortfall_fraction.idxmax()]
        anchors.append({"plan_id": plan_id, "import_multiplier": 1., "construction_delay_multiplier": 1., **metrics,
                        "annual_reliability_feasible": reliability_feasible(metrics, planning), "worst_year": int(worst.year),
                        "worst_carrier": worst.carrier, "worst_shortfall_fraction": float(worst.shortfall_fraction), "reconstruction_max_metric_error": error})
        audits.append({"plan_id": plan_id, "grid_id": "nominal_anchor", **independent_audit(frame, actions, initial, engineering, vector, nominal_scenario)})
        for cell in grid.to_dict("records"):
            identity = {"plan_id": plan_id, **cell}
            scenario = {**data["config"]["scenarios"]["baseline"], **fixed_scenario(vector, plan_id + "__" + cell["grid_id"], cell["import_multiplier"], cell["construction_delay_multiplier"])}
            frame, actions = annual.simulate(data, parameters, scenario, investor_ledger=False)
            metrics = summarize(frame, actions, data, planning)
            yearly = yearly_reliability(frame, data, planning)
            feasible = reliability_feasible(metrics, planning)
            assert feasible == (not yearly.violates_reliability.any())
            worst = yearly.loc[yearly.shortfall_fraction.idxmax()]
            all_metrics.append({**identity, **metrics, "annual_reliability_feasible": feasible,
                                "worst_year": int(worst.year), "worst_carrier": worst.carrier,
                                "worst_shortfall_fraction": float(worst.shortfall_fraction), "violating_year_carrier_count": int(yearly.violates_reliability.sum())})
            all_yearly.append(yearly.assign(**identity))
            audits.append({**identity, **independent_audit(frame, actions, initial, engineering, vector, scenario)})
            oil = yearly.loc[yearly.year.eq(2030) & yearly.carrier.eq("oil")].iloc[0]
            state = frame.loc[frame.year.eq(2030)].iloc[0]
            commissioned = float(actions.loc[actions.agent.eq("fuel_oil") & actions.year.le(2030), "commissioned"].sum())
            diagnostic.append({**identity, "oil_new_order_lead_years": int(np.ceil(oil_lead * cell["construction_delay_multiplier"])),
                               "earliest_new_2025_oil_order_commission_year": 2025 + int(np.ceil(oil_lead * cell["construction_delay_multiplier"])),
                               "oil_2030_shortfall_fraction": float(oil.shortfall_fraction),
                               "oil_2030_requested_mtce": float(oil.requested), "oil_2030_delivered_mtce": float(oil.delivered),
                               "oil_2030_domestic_supply_mtce": float(state.oil_domestic_supply_mtce),
                               "oil_2030_imports_mtce": float(state.oil_imports_mtce), "oil_2030_stock_mtce": float(state.oil_stock_mtce),
                               "oil_2030_import_capacity_mtce_per_year": float(state.oil_import_capacity_mtce_per_year),
                               "oil_2030_production_capacity_mtce_per_year": float(state.oil_production_capacity_mtce_per_year),
                               "oil_initial_pipeline_commissioned_through2030": initial_oil_pipeline_2030,
                               "oil_new_order_commissioned_through2030": commissioned - initial_oil_pipeline_2030})
        print(plan_id, "completed 121 fixed grid paths and nominal anchor", flush=True)
    metrics = pd.DataFrame(all_metrics)
    anchor_frame = pd.DataFrame(anchors)
    yearly = pd.concat(all_yearly, ignore_index=True)
    assert len(metrics) == 484 and len(yearly) == 484 * 31 * 5 and len(anchors) == 4
    boundaries = extract_boundaries(metrics, anchor_frame)
    metrics.to_csv(out / "grid_metrics.csv", index=False)
    anchor_frame.to_csv(out / "nominal_anchors.csv", index=False)
    yearly.to_csv(out / "annual_reliability.csv", index=False)
    pd.DataFrame(audits).to_csv(out / "independent_physical_capital_audits.csv", index=False)
    pd.DataFrame(diagnostic).to_csv(out / "oil_2030_diagnostic.csv", index=False)
    boundaries.to_csv(out / "safety_boundary.csv", index=False)
    clean_boundaries = json.loads(boundaries.to_json(orient="records"))
    (out / "safety_boundary.json").write_text(json.dumps(clean_boundaries, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")
    summaries = []
    for name, part in metrics.groupby("plan_id", sort=False):
        worst = part.loc[part.worst_shortfall_fraction.idxmax()]
        summaries.append({"plan_id": name, "grid_paths": len(part), "feasible_grid_paths": int(part.annual_reliability_feasible.sum()),
                          "worst_shortfall_fraction": float(worst.worst_shortfall_fraction), "worst_year": int(worst.worst_year),
                          "worst_carrier": worst.worst_carrier, "net_emissions_2060_mt_min": float(part.net_emissions_2060_mt.min()),
                          "net_emissions_2060_mt_max": float(part.net_emissions_2060_mt.max()),
                          "discounted_capital_trillion_cny_min": float(part.discounted_capital_trillion_cny.min()),
                          "discounted_capital_trillion_cny_max": float(part.discounted_capital_trillion_cny.max())})
    pd.DataFrame(summaries).to_csv(out / "plan_summary.csv", index=False)
    after = {name: digest(root / name) for name in names}
    assert sources == after
    receipt.update(status="PASS_FROZEN_484_GRID_PATHS_FOUR_NOMINAL_ANCHORS_AND_INDEPENDENT_PRIMITIVE_AUDITS",
                   completed_utc=datetime.now(timezone.utc).isoformat(), runtime_seconds=time.perf_counter() - started,
                   actual_grid_paths=len(metrics), actual_nominal_anchors=len(anchors), assessed_year_carrier_rows=len(yearly),
                   source_input_hashes_after=after, plan_summaries=summaries,
                   independent_audit_maxima=pd.DataFrame(audits).select_dtypes(include=np.number).drop(columns=["import_multiplier", "construction_delay_multiplier"]).max().to_dict(),
                   safety_boundary=clean_boundaries, figure_status="PENDING_RENDER")
    (out / "analysis.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")
    plot(root)
    receipt["figure_status"] = "DRAWN_AWAITING_ACTUAL_VISUAL_INSPECTION"
    receipt["artifact_sha256"] = {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != "analysis.json"}
    (out / "analysis.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")
    print(receipt["status"], flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot(args.root)
    else:
        run(args.root)
