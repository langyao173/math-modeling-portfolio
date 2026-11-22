from __future__ import annotations

import argparse
import importlib.util
import itertools
import math
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "07_solve_isling" / "build_exact_controlled_baseline.py"


def load_module():
    spec = importlib.util.spec_from_file_location("exact_controlled_baseline_test_target", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_incumbent_replays_published_metrics() -> None:
    module = load_module()
    data = module.load_frozen_data()
    metrics = module.metrics_for_rows(data, data.incumbent)

    assert len(data.incumbent) == 588
    assert data.incumbent["job_id"].nunique() == 588
    assert metrics["one_hot_violations"] == 0
    assert metrics["dummy_actions"] == 0
    assert math.isclose(metrics["expected_recourse_cost_usd"], 35.39452387390732, abs_tol=5e-9)
    assert math.isclose(metrics["cvar95_recourse_cost_usd"], 47.718284903473034, abs_tol=5e-9)
    assert math.isclose(metrics["peak_it_mw"], 0.03164630399999999, abs_tol=5e-12)


def test_two_job_exact_milp_matches_brute_force() -> None:
    module = load_module()
    data = module.load_frozen_data()
    jobs = module.choose_nested_jobs(data, [2], seed=20260720)[2]
    model = module.build_exact_model(data, jobs)
    assert all(lower <= upper for lower, upper in zip(model.variables.lower, model.variables.upper))

    brute: list[dict[str, float]] = []
    groups = [group for _, group in model.actions.groupby("job_id", sort=True)]
    for selected_tuple in itertools.product(*[list(group.index) for group in groups]):
        selected = model.actions.loc[list(selected_tuple)]
        rows = pd.concat([model.fixed, selected], ignore_index=True)
        if float(rows["action_cost"].astype(float).sum()) > data.action_cost_guard_usd + 1e-9:
            continue
        brute.append(module.metrics_for_rows(data, rows))
    assert brute
    best_audit = min(item["audit_objective_usd"] for item in brute)
    best_peak = min(
        item["peak_it_mw"]
        for item in brute
        if item["audit_objective_usd"] <= best_audit + 5e-7
    )
    incumbent_peak = float(data.incumbent_load_mw.max())
    best_peak_guarded_audit = min(
        item["audit_objective_usd"]
        for item in brute
        if item["peak_it_mw"] <= incumbent_peak + 5e-10
    )

    solved = module.solve_controlled_case(
        data,
        jobs,
        time_limit=30.0,
        mip_rel_gap=0.0,
        lexicographic_tolerance=1e-7,
    )
    assert solved["primary"]["label"] == "optimal"
    assert solved["primary"]["mip_gap"] == 0.0
    assert solved["peak_tiebreak"]["label"] == "optimal"
    assert solved["peak_guarded_primary"]["label"] == "optimal"
    assert solved["metrics"] is not None
    assert solved["peak_guarded_metrics"] is not None
    assert math.isclose(solved["metrics"]["audit_objective_usd"], best_audit, abs_tol=5e-7)
    assert math.isclose(solved["metrics"]["peak_it_mw"], best_peak, abs_tol=5e-10)
    assert solved["milp_recourse_crosscheck"]["absolute_expected_error_usd"] <= 2e-5
    assert solved["milp_recourse_crosscheck"]["absolute_cvar95_error_usd"] <= 2e-5
    assert solved["peak_guarded_metrics"]["peak_it_mw"] <= data.incumbent_load_mw.max() + 5e-10
    assert math.isclose(
        solved["peak_guarded_metrics"]["audit_objective_usd"],
        best_peak_guarded_audit,
        abs_tol=5e-7,
    )
    assert solved["peak_guarded_milp_recourse_crosscheck"]["absolute_expected_error_usd"] <= 2e-5
    assert solved["peak_guarded_milp_recourse_crosscheck"]["absolute_cvar95_error_usd"] <= 2e-5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("time_limit", 0.0),
        ("time_limit", -1.0),
        ("time_limit", math.inf),
        ("time_limit", math.nan),
        ("mip_rel_gap", -1e-9),
        ("mip_rel_gap", 1.0),
        ("mip_rel_gap", math.inf),
        ("mip_rel_gap", math.nan),
        ("lexicographic_tolerance", -1e-9),
        ("lexicographic_tolerance", math.inf),
        ("lexicographic_tolerance", math.nan),
    ],
)
def test_main_rejects_invalid_solver_controls_before_loading_data(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: float,
) -> None:
    module = load_module()
    arguments = argparse.Namespace(
        sizes=[30],
        seed=20260720,
        time_limit=120.0,
        mip_rel_gap=0.0,
        lexicographic_tolerance=1e-7,
        output_csv=module.DEFAULT_CSV,
        output_json=module.DEFAULT_JSON,
    )
    setattr(arguments, field, value)
    monkeypatch.setattr(module, "parse_args", lambda: arguments)
    monkeypatch.setattr(
        module,
        "load_frozen_data",
        lambda: pytest.fail("invalid arguments reached data loading"),
    )

    with pytest.raises(ValueError, match=field.replace("_", "-")):
        module.main()


@pytest.mark.parametrize(
    ("csv_name", "json_name", "message"),
    [
        ("result.txt", "result.json", "--output-csv"),
        ("result.csv", "result.txt", "--output-json"),
        ("same.csv", "same.csv", "different resolved paths"),
    ],
)
def test_output_validation_rejects_wrong_extensions_and_same_path(
    csv_name: str,
    json_name: str,
    message: str,
) -> None:
    module = load_module()
    csv_path = module.TABLES_ROOT / csv_name
    json_path = module.TABLES_ROOT / json_name

    with pytest.raises(ValueError, match=message):
        module.validate_output_paths(csv_path, json_path)


def test_output_validation_rejects_paths_outside_tables(tmp_path: Path) -> None:
    module = load_module()

    with pytest.raises(ValueError, match="competition_report/tables"):
        module.validate_output_paths(tmp_path / "result.csv", module.DEFAULT_JSON)


def test_publish_output_pair_precreates_both_parents(tmp_path: Path) -> None:
    module = load_module()
    csv_path = tmp_path / "csv-parent" / "result.csv"
    json_path = tmp_path / "json-parent" / "result.json"

    module.publish_output_pair(
        pd.DataFrame([{"value": 1}]),
        {"status": "pass"},
        csv_path,
        json_path,
    )

    assert csv_path.read_text(encoding="utf-8-sig").strip().splitlines() == ["value", "1"]
    assert json_path.read_text(encoding="utf-8") == '{\n  "status": "pass"\n}'


def test_publish_output_pair_rolls_back_when_second_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    csv_path = tmp_path / "result.csv"
    json_path = tmp_path / "result.json"
    csv_path.write_text("old csv", encoding="utf-8")
    json_path.write_text("old json", encoding="utf-8")
    original_replace = Path.replace
    failed = False

    def fail_json_publish_once(source: Path, target: Path) -> Path:
        nonlocal failed
        if Path(target) == json_path and source.name.startswith(".result.json.") and not failed:
            failed = True
            raise OSError("simulated JSON publish failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_json_publish_once)

    with pytest.raises(OSError, match="simulated JSON publish failure"):
        module.publish_output_pair(
            pd.DataFrame([{"value": 2}]),
            {"status": "new"},
            csv_path,
            json_path,
        )

    assert csv_path.read_text(encoding="utf-8") == "old csv"
    assert json_path.read_text(encoding="utf-8") == "old json"
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))
