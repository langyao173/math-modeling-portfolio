"""Build exact controlled MILP baselines around the frozen final schedule.

The experiment fixes all jobs outside a deterministic high-leverage
neighborhood to the submitted incumbent.  Inside the neighborhood, one binary
variable selects one real action per job.  SciPy's HiGHS-backed ``milp`` solves
the exact five-scenario recourse model used by ``solve_qubo_platform.py``.

The primary objective is the auditable scalar ``Expected + CVaR95``.  A second
MILP minimizes peak IT load while constraining the primary objective to its
certified optimum (within a documented numerical tolerance).  The action-cost
guard is the minimum-real-action schedule cost plus USD 1.00, matching the
frozen workflow's guard.  No dummy action is admitted to the controlled model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "deliverables" / "final_solution_package"
BASELINE_SCRIPT = Path(__file__).resolve()
SOLVER_PATH = Path(__file__).with_name("solve_qubo_platform.py")
ACTIONS_PATH = PACKAGE / "validation_data" / "actions_for_validation.csv"
SITE_PATH = PACKAGE / "validation_data" / "site_timeseries_for_validation.csv"
SELECTION_SITE_PATH = ROOT / "data" / "interim" / "site_timeseries.parquet"
INCUMBENT_PATH = PACKAGE / "FULL_SELECTED_ACTIONS.csv"
METRICS_PATH = PACKAGE / "FINAL_METRICS.json"
DEFAULT_CSV = ROOT / "deliverables" / "competition_report" / "tables" / "15_exact_controlled_baseline.csv"
DEFAULT_JSON = DEFAULT_CSV.with_suffix(".json")
TABLES_ROOT = DEFAULT_CSV.parent

POWER_SCALE = 1000.0  # Work in kW for better HiGHS numerical conditioning.
ALPHA = 0.95
SELECTION_RULE = (
    "solve_qubo_platform.choose_jobs: descending max real-action "
    "power dot (1 + slot_risk), stable seed jitter; same-seed nested sets"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def root_relative_output(path: Path) -> tuple[Path, str]:
    """Resolve an output inside ROOT and return its portable recorded path."""
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"Output path must stay inside {ROOT}: {resolved}") from exc
    return resolved, relative.as_posix()


def validate_output_paths(csv_path: Path, json_path: Path) -> tuple[tuple[Path, str], tuple[Path, str]]:
    """Validate and resolve the two output paths before any solver work."""
    csv_resolved = csv_path.resolve()
    json_resolved = json_path.resolve()
    tables_resolved = TABLES_ROOT.resolve()
    if csv_resolved == json_resolved:
        raise ValueError("CSV and JSON outputs must use different resolved paths.")
    if csv_resolved.suffix.lower() != ".csv":
        raise ValueError("--output-csv must have the .csv extension.")
    if json_resolved.suffix.lower() != ".json":
        raise ValueError("--output-json must have the .json extension.")
    for resolved in (csv_resolved, json_resolved):
        try:
            resolved.relative_to(tables_resolved)
        except ValueError as exc:
            raise ValueError(
                "Output paths must stay inside deliverables/competition_report/tables."
            ) from exc
    return root_relative_output(csv_resolved), root_relative_output(json_resolved)


def validate_solver_controls(
    time_limit: float,
    mip_rel_gap: float,
    lexicographic_tolerance: float,
) -> None:
    """Reject invalid HiGHS controls before loading data or solving."""
    if not math.isfinite(time_limit) or time_limit <= 0.0:
        raise ValueError("--time-limit must be finite and greater than zero.")
    if not math.isfinite(mip_rel_gap) or not 0.0 <= mip_rel_gap < 1.0:
        raise ValueError("--mip-rel-gap must be finite and satisfy 0 <= value < 1.")
    if not math.isfinite(lexicographic_tolerance) or lexicographic_tolerance < 0.0:
        raise ValueError("--lexicographic-tolerance must be finite and nonnegative.")


def prepare_output_directories(csv_path: Path, json_path: Path) -> None:
    """Create and independently preflight both destination directories."""
    for parent in dict.fromkeys((csv_path.parent, json_path.parent)):
        parent.mkdir(parents=True, exist_ok=True)
        if not parent.is_dir():
            raise NotADirectoryError(f"Output parent is not a directory: {parent}")
        file_descriptor, probe_name = tempfile.mkstemp(
            prefix=".exact-baseline-write-probe.",
            suffix=".tmp",
            dir=parent,
        )
        os.close(file_descriptor)
        Path(probe_name).unlink()


def _temporary_path(destination: Path, suffix: str) -> Path:
    file_descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=suffix,
        dir=destination.parent,
    )
    os.close(file_descriptor)
    return Path(name)


def publish_output_pair(
    frame: pd.DataFrame,
    payload: dict[str, Any],
    csv_path: Path,
    json_path: Path,
) -> None:
    """Publish CSV and JSON from same-directory temporaries with rollback."""
    prepare_output_directories(csv_path, json_path)
    temporary_csv = _temporary_path(csv_path, ".tmp")
    temporary_json = _temporary_path(json_path, ".tmp")
    backups: dict[Path, Path] = {}
    try:
        frame.to_csv(temporary_csv, index=False, encoding="utf-8-sig")
        temporary_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for destination in (csv_path, json_path):
            if destination.exists():
                backup = _temporary_path(destination, ".bak")
                shutil.copy2(destination, backup)
                backups[destination] = backup
        try:
            temporary_csv.replace(csv_path)
            temporary_json.replace(json_path)
        except Exception:
            for destination in (csv_path, json_path):
                backup = backups.get(destination)
                if backup is not None:
                    backup.replace(destination)
                elif destination.exists():
                    destination.unlink()
            raise
    finally:
        for path in (temporary_csv, temporary_json, *backups.values()):
            if path.exists():
                path.unlink()


def load_solver_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("taolin_solve_qubo_platform", SOLVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator from {SOLVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class FrozenData:
    actions: pd.DataFrame
    site: pd.DataFrame
    selection_site: pd.DataFrame
    incumbent: pd.DataFrame
    reference_load_mw: np.ndarray
    incumbent_load_mw: np.ndarray
    action_cost_guard_usd: float
    evaluator: ModuleType


def _with_power(actions: pd.DataFrame, evaluator: ModuleType) -> pd.DataFrame:
    result = actions.copy()
    result["is_dummy"] = result["is_dummy"].astype(bool)
    result["_power"] = result["power_vector_mw"].map(
        lambda value: np.asarray(evaluator.parse_vector(value), dtype=float)
    )
    return result


def _sum_power(rows: pd.DataFrame, horizon: int) -> np.ndarray:
    if rows.empty:
        return np.zeros(horizon, dtype=float)
    return np.vstack(rows["_power"].to_list()).sum(axis=0)


def load_frozen_data() -> FrozenData:
    evaluator = load_solver_module()
    actions = _with_power(pd.read_csv(ACTIONS_PATH, encoding="utf-8-sig"), evaluator)
    site = pd.read_csv(SITE_PATH, encoding="utf-8-sig").sort_values("slot").reset_index(drop=True)
    selection_site = pd.read_parquet(SELECTION_SITE_PATH).sort_values("slot").reset_index(drop=True)
    incumbent = _with_power(pd.read_csv(INCUMBENT_PATH, encoding="utf-8-sig"), evaluator)
    horizon = len(site)

    if actions["job_id"].nunique() != 588 or len(actions) != 1929:
        raise ValueError("Frozen action table is not the expected 588-job / 1929-action instance.")
    if actions["action_id"].duplicated().any():
        raise ValueError("Frozen action ids are not unique.")
    if incumbent["job_id"].nunique() != 588 or len(incumbent) != 588:
        raise ValueError("Incumbent is not one action for each of the 588 jobs.")
    if any(len(vector) != horizon for vector in actions["_power"]):
        raise ValueError("Action power-vector horizon mismatch.")
    shared_site_fields = [
        "slot",
        "lmp_usd_mwh",
        "carbon_kgco2_mwh",
        "penalty_short_usd_mwh",
        "flex_call_prob",
        "renewable_mw",
    ]
    if not np.allclose(
        site[shared_site_fields].to_numpy(dtype=float),
        selection_site[shared_site_fields].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Selection site table is not aligned with the frozen validation site table.")

    canonical = set(zip(actions["job_id"].astype(str), actions["action_id"].astype(str)))
    incumbent_pairs = set(zip(incumbent["job_id"].astype(str), incumbent["action_id"].astype(str)))
    if not incumbent_pairs <= canonical:
        raise ValueError("Incumbent contains a job/action pair outside the frozen action table.")
    if incumbent["is_dummy"].any():
        raise ValueError("The frozen incumbent contains a dummy action.")

    real = actions.loc[~actions["is_dummy"]].copy()
    baseline_rows = (
        real.sort_values(["job_id", "action_cost", "action_id"])
        .groupby("job_id", sort=True, as_index=False)
        .first()
    )
    if len(baseline_rows) != 588:
        raise ValueError("At least one frozen job has no real action.")
    reference_load = _sum_power(baseline_rows, horizon)
    incumbent_load = _sum_power(incumbent, horizon)
    guard = float(baseline_rows["action_cost"].astype(float).sum() + 1.0)

    reported = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    replay = evaluator.evaluate_recourse(incumbent_load, reference_load, site, alpha=ALPHA)
    checks = {
        "expected": (replay["expected_recourse_cost_usd"], reported["expected_recourse_cost_usd"]),
        "cvar95": (replay["cvar_recourse_cost_usd"], reported["cvar95_recourse_cost_usd"]),
        "peak": (float(incumbent_load.max()), reported["peak_it_mw"]),
        "action_cost": (
            float(incumbent["action_cost"].astype(float).sum()),
            reported["true_action_cost_usd"],
        ),
    }
    mismatches = {
        key: values
        for key, values in checks.items()
        if not math.isclose(float(values[0]), float(values[1]), rel_tol=0.0, abs_tol=5e-9)
    }
    if mismatches:
        raise ValueError(f"Frozen incumbent does not replay FINAL_METRICS: {mismatches}")

    return FrozenData(
        actions=actions,
        site=site,
        selection_site=selection_site,
        incumbent=incumbent,
        reference_load_mw=reference_load,
        incumbent_load_mw=incumbent_load,
        action_cost_guard_usd=guard,
        evaluator=evaluator,
    )


class Variables:
    def __init__(self) -> None:
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.integrality: list[int] = []
        self.names: list[str] = []

    def add(
        self,
        name: str,
        size: int,
        lower: float | np.ndarray = 0.0,
        upper: float | np.ndarray = np.inf,
        integer: bool = False,
    ) -> np.ndarray:
        start = len(self.lower)
        lo = np.broadcast_to(np.asarray(lower, dtype=float), (size,)).tolist()
        hi = np.broadcast_to(np.asarray(upper, dtype=float), (size,)).tolist()
        self.lower.extend(lo)
        self.upper.extend(hi)
        self.integrality.extend([1 if integer else 0] * size)
        self.names.extend([f"{name}[{idx}]" for idx in range(size)])
        return np.arange(start, start + size, dtype=int)


class Rows:
    def __init__(self, variable_count: int) -> None:
        self.variable_count = variable_count
        self.row: list[int] = []
        self.col: list[int] = []
        self.data: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []

    def add(self, coefficients: dict[int, float], lower: float, upper: float) -> None:
        row_id = len(self.lower)
        for col, value in coefficients.items():
            if abs(value) > 1e-14:
                self.row.append(row_id)
                self.col.append(int(col))
                self.data.append(float(value))
        self.lower.append(float(lower))
        self.upper.append(float(upper))

    def constraint(self) -> LinearConstraint:
        matrix = coo_matrix(
            (self.data, (self.row, self.col)),
            shape=(len(self.lower), self.variable_count),
        ).tocsr()
        return LinearConstraint(matrix, np.asarray(self.lower), np.asarray(self.upper))


@dataclass
class ExactModel:
    variables: Variables
    rows: Rows
    actions: pd.DataFrame
    fixed: pd.DataFrame
    controlled_jobs: list[str]
    power_kw: np.ndarray
    fixed_load_kw: np.ndarray
    x: np.ndarray
    scenario_cost: np.ndarray
    expected: int
    cvar: int
    peak_kw: int
    action_cost_cap_remaining: float
    crossing_flex_binaries: int


def _add_vector(coefficients: dict[int, float], indices: np.ndarray, values: np.ndarray, factor: float = 1.0) -> None:
    for index, value in zip(indices.tolist(), values.tolist()):
        if abs(value) > 1e-14:
            coefficients[int(index)] = coefficients.get(int(index), 0.0) + factor * float(value)


def build_exact_model(data: FrozenData, controlled_jobs: list[str]) -> ExactModel:
    jobs = sorted(set(controlled_jobs))
    if len(jobs) != len(controlled_jobs):
        raise ValueError("Controlled job ids must be unique.")
    real = data.actions.loc[(data.actions["job_id"].isin(jobs)) & (~data.actions["is_dummy"])].copy()
    real = real.sort_values(["job_id", "action_id"]).reset_index(drop=True)
    if real["job_id"].nunique() != len(jobs):
        raise ValueError("A controlled job has no real action.")
    fixed = data.incumbent.loc[~data.incumbent["job_id"].isin(jobs)].copy()
    horizon = len(data.site)
    scenarios = data.evaluator.recourse_scenarios(horizon)
    scenario_count = len(scenarios)
    inputs = data.evaluator.recourse_inputs(data.site, data.reference_load_mw)

    power_kw = np.vstack(real["_power"].to_list()) * POWER_SCALE
    fixed_load_kw = _sum_power(fixed, horizon) * POWER_SCALE
    reference_kw = data.reference_load_mw * POWER_SCALE
    min_load_kw = fixed_load_kw.copy()
    max_load_kw = fixed_load_kw.copy()
    for _, group in real.groupby("job_id", sort=True):
        matrix = np.vstack(group["_power"].to_list()) * POWER_SCALE
        min_load_kw += matrix.min(axis=0)
        max_load_kw += matrix.max(axis=0)

    variables = Variables()
    x = variables.add("x", len(real), 0.0, 1.0, integer=True)
    grid: list[np.ndarray] = []
    curtail: list[np.ndarray] = []
    delivered: list[np.ndarray] = []
    renewable_by_scenario: list[np.ndarray] = []
    target_by_scenario: list[np.ndarray] = []
    price_by_scenario: list[np.ndarray] = []

    for scenario in scenarios:
        renewable = (
            np.asarray(inputs["base_green_mw"], dtype=float)
            * np.asarray(scenario["renewable_factor"], dtype=float)
            * POWER_SCALE
        )
        target = (
            np.asarray(inputs["flex_target_base"], dtype=float)
            * float(scenario["flex_factor"])
            * POWER_SCALE
        )
        price = (
            np.asarray(inputs["lmp"], dtype=float)
            * np.asarray(scenario["price_factor"], dtype=float)
            / POWER_SCALE
        )
        renewable_by_scenario.append(renewable)
        target_by_scenario.append(target)
        price_by_scenario.append(price)
        grid.append(variables.add("grid_kw", horizon, 0.0, np.maximum(max_load_kw - renewable, 0.0)))
        curtail.append(variables.add("curtail_kw", horizon, 0.0, np.maximum(renewable - min_load_kw, 0.0)))
        delivered.append(variables.add("flex_delivered_kw", horizon, 0.0, target))

    q_min = reference_kw - max_load_kw
    q_max = reference_kw - min_load_kw
    # At q == 0 either branch is valid; choose z = 0.  Strict separation is
    # important because using overlapping tolerances can create lower > upper.
    always_above_reference = q_max < -1e-10
    always_below_or_equal_reference = q_min >= -1e-10
    crossing = ~(always_above_reference | always_below_or_equal_reference)
    z_lower = np.where(always_above_reference, 1.0, 0.0)
    z_upper = np.where(always_above_reference, 1.0, np.where(always_below_or_equal_reference, 0.0, 1.0))
    z = variables.add("load_at_or_above_reference", horizon, z_lower, z_upper, integer=True)

    scenario_cost_upper = []
    penalty = np.asarray(inputs["penalty_short"], dtype=float) / POWER_SCALE
    for scenario_index in range(scenario_count):
        scenario_cost_upper.append(
            float(
                np.dot(price_by_scenario[scenario_index], np.maximum(max_load_kw - renewable_by_scenario[scenario_index], 0.0))
                + (5.0 / POWER_SCALE) * np.maximum(renewable_by_scenario[scenario_index] - min_load_kw, 0.0).sum()
                + np.dot(penalty, target_by_scenario[scenario_index])
            )
            + 1.0
        )
    maximum_cost = max(scenario_cost_upper)
    scenario_cost = variables.add("scenario_cost_usd", scenario_count, 0.0, scenario_cost_upper)
    expected = int(variables.add("expected_usd", 1, 0.0, maximum_cost)[0])
    eta = int(variables.add("cvar_eta_usd", 1, 0.0, maximum_cost)[0])
    excess = variables.add("cvar_excess_usd", scenario_count, 0.0, maximum_cost)
    cvar = int(variables.add("cvar95_usd", 1, 0.0, maximum_cost)[0])
    peak_kw = int(variables.add("peak_kw", 1, 0.0, float(max_load_kw.max()))[0])

    rows = Rows(len(variables.lower))
    job_positions = real.groupby("job_id", sort=True).indices
    for job_id in jobs:
        indices = x[np.asarray(job_positions[job_id], dtype=int)]
        rows.add({int(index): 1.0 for index in indices}, 1.0, 1.0)

    fixed_cost = float(fixed["action_cost"].astype(float).sum())
    remaining = data.action_cost_guard_usd - fixed_cost
    rows.add(
        {int(index): float(cost) for index, cost in zip(x, real["action_cost"].astype(float))},
        -np.inf,
        remaining,
    )

    for slot in range(horizon):
        load_coeff = power_kw[:, slot]
        big_m = max(float(-q_min[slot]), float(q_max[slot]), 1e-7)
        coeff = {int(z[slot]): big_m}
        _add_vector(coeff, x, load_coeff, factor=-1.0)
        rows.add(coeff, float(fixed_load_kw[slot] - reference_kw[slot]), np.inf)
        rows.add(coeff, -np.inf, float(big_m - reference_kw[slot] + fixed_load_kw[slot]))

        peak_coeff = {peak_kw: 1.0}
        _add_vector(peak_coeff, x, load_coeff, factor=-1.0)
        rows.add(peak_coeff, float(fixed_load_kw[slot]), np.inf)

        for scenario_index in range(scenario_count):
            renewable = renewable_by_scenario[scenario_index][slot]
            grid_coeff = {int(grid[scenario_index][slot]): 1.0}
            _add_vector(grid_coeff, x, load_coeff, factor=-1.0)
            rows.add(grid_coeff, float(fixed_load_kw[slot] - renewable), np.inf)

            curtail_coeff = {int(curtail[scenario_index][slot]): 1.0}
            _add_vector(curtail_coeff, x, load_coeff, factor=1.0)
            rows.add(curtail_coeff, float(renewable - fixed_load_kw[slot]), np.inf)

            target = float(target_by_scenario[scenario_index][slot])
            delivery_coeff = {
                int(delivered[scenario_index][slot]): 1.0,
                int(z[slot]): -big_m,
            }
            _add_vector(delivery_coeff, x, load_coeff, factor=1.0)
            rows.add(delivery_coeff, -np.inf, float(reference_kw[slot] - fixed_load_kw[slot]))
            rows.add(
                {
                    int(delivered[scenario_index][slot]): 1.0,
                    int(z[slot]): target,
                },
                -np.inf,
                target,
            )

    for scenario_index, scenario in enumerate(scenarios):
        coeff = {int(scenario_cost[scenario_index]): 1.0}
        _add_vector(coeff, grid[scenario_index], price_by_scenario[scenario_index], factor=-1.0)
        _add_vector(
            coeff,
            curtail[scenario_index],
            np.full(horizon, 5.0 / POWER_SCALE),
            factor=-1.0,
        )
        _add_vector(coeff, delivered[scenario_index], penalty, factor=1.0)
        constant = float(np.dot(penalty, target_by_scenario[scenario_index]))
        rows.add(coeff, constant, constant)

    probabilities = np.asarray([float(scenario["probability"]) for scenario in scenarios])
    expected_coeff = {expected: 1.0}
    _add_vector(expected_coeff, scenario_cost, probabilities, factor=-1.0)
    rows.add(expected_coeff, 0.0, 0.0)

    for scenario_index in range(scenario_count):
        rows.add(
            {
                int(excess[scenario_index]): 1.0,
                int(scenario_cost[scenario_index]): -1.0,
                eta: 1.0,
            },
            0.0,
            np.inf,
        )
    cvar_coeff = {cvar: 1.0, eta: -1.0}
    _add_vector(cvar_coeff, excess, probabilities / (1.0 - ALPHA), factor=-1.0)
    rows.add(cvar_coeff, 0.0, 0.0)

    return ExactModel(
        variables=variables,
        rows=rows,
        actions=real,
        fixed=fixed,
        controlled_jobs=jobs,
        power_kw=power_kw,
        fixed_load_kw=fixed_load_kw,
        x=x,
        scenario_cost=scenario_cost,
        expected=expected,
        cvar=cvar,
        peak_kw=peak_kw,
        action_cost_cap_remaining=remaining,
        crossing_flex_binaries=int(crossing.sum()),
    )


def _solve(
    model: ExactModel,
    objective: np.ndarray,
    time_limit: float,
    mip_rel_gap: float,
    primary_upper: float | None = None,
    peak_upper_kw: float | None = None,
) -> tuple[Any, float]:
    constraint = model.rows.constraint()
    extra_rows: list[tuple[list[int], list[float], float]] = []
    if primary_upper is not None:
        extra_rows.append(([model.expected, model.cvar], [1.0, 1.0], float(primary_upper)))
    if peak_upper_kw is not None:
        extra_rows.append(([model.peak_kw], [1.0], float(peak_upper_kw)))
    if extra_rows:
        extra_row_indices: list[int] = []
        extra_columns: list[int] = []
        extra_values: list[float] = []
        for row_index, (columns, values, _) in enumerate(extra_rows):
            extra_row_indices.extend([row_index] * len(columns))
            extra_columns.extend(columns)
            extra_values.extend(values)
        extra = coo_matrix(
            (extra_values, (extra_row_indices, extra_columns)),
            shape=(len(extra_rows), len(model.variables.lower)),
        ).tocsr()
        matrix = vstack([constraint.A, extra], format="csr")
        lower = np.concatenate([np.asarray(constraint.lb), np.full(len(extra_rows), -np.inf)])
        upper = np.concatenate(
            [np.asarray(constraint.ub), np.asarray([row[2] for row in extra_rows])]
        )
        constraint = LinearConstraint(matrix, lower, upper)
    started = time.perf_counter()
    result = milp(
        c=objective,
        integrality=np.asarray(model.variables.integrality, dtype=np.uint8),
        bounds=Bounds(np.asarray(model.variables.lower), np.asarray(model.variables.upper)),
        constraints=constraint,
        options={
            "disp": False,
            "presolve": True,
            "time_limit": float(time_limit),
            "mip_rel_gap": float(mip_rel_gap),
        },
    )
    return result, time.perf_counter() - started


def _status(result: Any) -> dict[str, Any]:
    labels = {0: "optimal", 1: "limit_reached", 2: "infeasible", 3: "unbounded", 4: "solver_error"}
    return {
        "code": int(result.status),
        "label": labels.get(int(result.status), "unknown"),
        "message": str(result.message),
        "success": bool(result.success),
        "objective": None if result.fun is None else float(result.fun),
        "mip_gap": None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
        "mip_dual_bound": None
        if getattr(result, "mip_dual_bound", None) is None
        else float(result.mip_dual_bound),
        "mip_node_count": None
        if getattr(result, "mip_node_count", None) is None
        else int(result.mip_node_count),
    }


def metrics_for_rows(data: FrozenData, rows: pd.DataFrame) -> dict[str, Any]:
    load = _sum_power(rows, len(data.site))
    recourse = data.evaluator.evaluate_recourse(load, data.reference_load_mw, data.site, alpha=ALPHA)
    expected_jobs = set(data.actions["job_id"].astype(str))
    counts = rows["job_id"].astype(str).value_counts().to_dict()
    one_hot_violations = int(sum(abs(int(counts.get(job_id, 0)) - 1) for job_id in expected_jobs))
    return {
        "true_action_cost_usd": float(rows["action_cost"].astype(float).sum()),
        "expected_recourse_cost_usd": float(recourse["expected_recourse_cost_usd"]),
        "cvar95_recourse_cost_usd": float(recourse["cvar_recourse_cost_usd"]),
        "audit_objective_usd": float(
            recourse["expected_recourse_cost_usd"] + recourse["cvar_recourse_cost_usd"]
        ),
        "peak_it_mw": float(load.max()),
        "dummy_actions": int(rows["is_dummy"].astype(bool).sum()),
        "one_hot_violations": one_hot_violations,
        "scenario_costs_usd": {
            str(row["name"]): float(row["total_recourse_cost_usd"])
            for row in recourse["scenarios"]
        },
    }


def decode_rows(model: ExactModel, vector: np.ndarray) -> pd.DataFrame:
    selected = model.actions.loc[np.asarray(vector[model.x]) > 0.5].copy()
    counts = selected.groupby("job_id").size().to_dict()
    invalid = [job_id for job_id in model.controlled_jobs if counts.get(job_id, 0) != 1]
    if invalid:
        raise RuntimeError(f"MILP result violates one-hot decoding for {invalid[:10]}")
    return pd.concat([model.fixed, selected], ignore_index=True)


def solve_controlled_case(
    data: FrozenData,
    controlled_jobs: list[str],
    time_limit: float,
    mip_rel_gap: float,
    lexicographic_tolerance: float,
) -> dict[str, Any]:
    validate_solver_controls(time_limit, mip_rel_gap, lexicographic_tolerance)
    model = build_exact_model(data, controlled_jobs)
    variable_count = len(model.variables.lower)
    primary_objective = np.zeros(variable_count, dtype=float)
    primary_objective[model.expected] = 1.0
    primary_objective[model.cvar] = 1.0
    primary, primary_seconds = _solve(model, primary_objective, time_limit, mip_rel_gap)
    primary_info = _status(primary)
    primary_info["walltime_seconds"] = primary_seconds
    if primary.x is None:
        return {
            "controlled_jobs": len(controlled_jobs),
            "candidate_actions": len(model.actions),
            "variables": variable_count,
            "integer_variables": int(np.count_nonzero(model.variables.integrality)),
            "free_flex_indicator_binaries": model.crossing_flex_binaries,
            "constraints": len(model.rows.lower),
            "primary": primary_info,
            "peak_tiebreak": None,
            "selected_action_ids": [],
            "metrics": None,
            "milp_recourse_crosscheck": None,
        }

    primary_rows = decode_rows(model, np.asarray(primary.x))
    primary_exact = metrics_for_rows(data, primary_rows)
    primary_upper = float(primary.fun) + max(
        float(lexicographic_tolerance), abs(float(primary.fun)) * 1e-10
    )
    peak_objective = np.zeros(variable_count, dtype=float)
    peak_objective[model.peak_kw] = 1.0
    peak, peak_seconds = _solve(
        model,
        peak_objective,
        time_limit,
        mip_rel_gap,
        primary_upper=primary_upper,
    )
    peak_info = _status(peak)
    peak_info["walltime_seconds"] = peak_seconds

    guarded, guarded_seconds = _solve(
        model,
        primary_objective,
        time_limit,
        mip_rel_gap,
        peak_upper_kw=float(data.incumbent_load_mw.max() * POWER_SCALE + 1e-9),
    )
    guarded_info = _status(guarded)
    guarded_info["walltime_seconds"] = guarded_seconds
    chosen = primary
    chosen_phase = "primary"
    if peak.x is not None:
        candidate_rows = decode_rows(model, np.asarray(peak.x))
        candidate_exact = metrics_for_rows(data, candidate_rows)
        if candidate_exact["audit_objective_usd"] <= primary_exact["audit_objective_usd"] + max(
            5e-7, lexicographic_tolerance * 5.0
        ):
            chosen = peak
            chosen_phase = "peak_tiebreak"

    rows = decode_rows(model, np.asarray(chosen.x))
    exact = metrics_for_rows(data, rows)
    selected = rows.loc[rows["job_id"].isin(controlled_jobs)].sort_values("job_id")
    incumbent_by_job = data.incumbent.set_index("job_id")["action_id"].astype(str).to_dict()
    changed_jobs = [
        str(row.job_id)
        for row in selected.itertuples(index=False)
        if str(row.action_id) != incumbent_by_job[str(row.job_id)]
    ]
    milp_expected = float(chosen.x[model.expected])
    milp_cvar = float(chosen.x[model.cvar])
    crosscheck = {
        "milp_expected_usd": milp_expected,
        "evaluator_expected_usd": exact["expected_recourse_cost_usd"],
        "absolute_expected_error_usd": abs(milp_expected - exact["expected_recourse_cost_usd"]),
        "milp_cvar95_usd": milp_cvar,
        "evaluator_cvar95_usd": exact["cvar95_recourse_cost_usd"],
        "absolute_cvar95_error_usd": abs(milp_cvar - exact["cvar95_recourse_cost_usd"]),
    }
    if max(crosscheck["absolute_expected_error_usd"], crosscheck["absolute_cvar95_error_usd"]) > 2e-5:
        raise RuntimeError(f"MILP/evaluator recourse mismatch: {crosscheck}")

    guarded_metrics = None
    guarded_crosscheck = None
    guarded_action_ids: list[str] = []
    guarded_changed_jobs: list[str] = []
    if guarded.x is not None:
        guarded_rows = decode_rows(model, np.asarray(guarded.x))
        guarded_metrics = metrics_for_rows(data, guarded_rows)
        guarded_selected = guarded_rows.loc[
            guarded_rows["job_id"].isin(controlled_jobs)
        ].sort_values("job_id")
        guarded_action_ids = guarded_selected["action_id"].astype(str).tolist()
        guarded_changed_jobs = [
            str(row.job_id)
            for row in guarded_selected.itertuples(index=False)
            if str(row.action_id) != incumbent_by_job[str(row.job_id)]
        ]
        guarded_peak_limit_mw = float(data.incumbent_load_mw.max() + 1e-12)
        if guarded_metrics["peak_it_mw"] > guarded_peak_limit_mw + 5e-10:
            raise RuntimeError("Peak-guarded MILP result violates the current-final peak limit.")
        guarded_crosscheck = {
            "milp_expected_usd": float(guarded.x[model.expected]),
            "evaluator_expected_usd": guarded_metrics["expected_recourse_cost_usd"],
            "absolute_expected_error_usd": abs(
                float(guarded.x[model.expected])
                - guarded_metrics["expected_recourse_cost_usd"]
            ),
            "milp_cvar95_usd": float(guarded.x[model.cvar]),
            "evaluator_cvar95_usd": guarded_metrics["cvar95_recourse_cost_usd"],
            "absolute_cvar95_error_usd": abs(
                float(guarded.x[model.cvar])
                - guarded_metrics["cvar95_recourse_cost_usd"]
            ),
        }
        if max(
            guarded_crosscheck["absolute_expected_error_usd"],
            guarded_crosscheck["absolute_cvar95_error_usd"],
        ) > 2e-5:
            raise RuntimeError(
                f"Peak-guarded MILP/evaluator recourse mismatch: {guarded_crosscheck}"
            )

    return {
        "controlled_jobs": len(controlled_jobs),
        "candidate_actions": len(model.actions),
        "variables": variable_count,
        "integer_variables": int(np.count_nonzero(model.variables.integrality)),
        "free_flex_indicator_binaries": model.crossing_flex_binaries,
        "constraints": len(model.rows.lower),
        "action_cost_cap_remaining_usd": model.action_cost_cap_remaining,
        "primary": primary_info,
        "peak_tiebreak": peak_info,
        "peak_guarded_primary": guarded_info,
        "chosen_phase": chosen_phase,
        "selected_action_ids": selected["action_id"].astype(str).tolist(),
        "changed_job_ids": changed_jobs,
        "changed_jobs": len(changed_jobs),
        "metrics": exact,
        "peak_guarded_selected_action_ids": guarded_action_ids,
        "peak_guarded_changed_job_ids": guarded_changed_jobs,
        "peak_guarded_changed_jobs": len(guarded_changed_jobs),
        "peak_guarded_metrics": guarded_metrics,
        "peak_guarded_milp_recourse_crosscheck": guarded_crosscheck,
        "milp_recourse_crosscheck": crosscheck,
    }


def choose_nested_jobs(data: FrozenData, sizes: list[int], seed: int) -> dict[int, list[str]]:
    chosen = {
        size: data.evaluator.choose_jobs(
            data.actions,
            data.selection_site,
            max_jobs=size,
            seed=seed,
        )
        for size in sizes
    }
    ordered = sorted(sizes)
    for smaller, larger in zip(ordered, ordered[1:]):
        if not set(chosen[smaller]) <= set(chosen[larger]):
            raise RuntimeError("Same-seed controlled neighborhoods are not nested.")
    return chosen


def csv_rows(cases: list[dict[str, Any]], incumbent_metrics: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        common = {
            "controlled_jobs": case["controlled_jobs"],
            "candidate_actions": case["candidate_actions"],
            "variables": case["variables"],
            "integer_variables": case["integer_variables"],
            "free_flex_indicator_binaries": case["free_flex_indicator_binaries"],
            "constraints": case["constraints"],
            "seed": seed,
            "selection_rule": SELECTION_RULE,
            "solver": f"scipy.optimize.milp/HiGHS (SciPy {scipy.__version__})",
        }
        solution_rows = [
            ("current_final", incumbent_metrics, 0),
            ("exact_milp", case["metrics"], case.get("changed_jobs")),
            (
                "exact_milp_peak_guarded",
                case.get("peak_guarded_metrics"),
                case.get("peak_guarded_changed_jobs"),
            ),
        ]
        for solution, metrics, changed_jobs in solution_rows:
            row = dict(common)
            row["solution"] = solution
            row["changed_jobs"] = changed_jobs
            if metrics is None:
                row.update(
                    {
                        "solver_primary_status": case["primary"]["label"],
                        "solver_primary_mip_gap": case["primary"]["mip_gap"],
                        "solver_primary_dual_bound": case["primary"]["mip_dual_bound"],
                        "solver_primary_walltime_seconds": case["primary"]["walltime_seconds"],
                        "solver_peak_phase_status": "not_run",
                        "solver_peak_phase_walltime_seconds": None,
                        "solution_is_primary_optimal": False,
                    }
                )
            else:
                primary = (
                    case.get("peak_guarded_primary", case["primary"])
                    if solution == "exact_milp_peak_guarded"
                    else case["primary"]
                )
                peak = (
                    {}
                    if solution == "exact_milp_peak_guarded"
                    else (case.get("peak_tiebreak") or {})
                )
                reference_metrics = (
                    case.get("peak_guarded_metrics")
                    if solution == "exact_milp_peak_guarded"
                    else case["metrics"]
                )
                current_row = solution == "current_final"
                row.update(
                    {
                        "solver_primary_status": "not_applicable" if current_row else primary["label"],
                        "solver_primary_mip_gap": None if current_row else primary["mip_gap"],
                        "solver_primary_dual_bound": None if current_row else primary["mip_dual_bound"],
                        "solver_primary_walltime_seconds": None
                        if current_row
                        else primary["walltime_seconds"],
                        "solver_peak_phase_status": "not_applicable"
                        if current_row
                        else peak.get(
                            "label",
                            "enforced_not_above_current"
                            if solution == "exact_milp_peak_guarded"
                            else "not_run",
                        ),
                        "solver_peak_phase_walltime_seconds": None
                        if current_row
                        else peak.get("walltime_seconds"),
                        "solution_is_primary_optimal": bool(
                            primary["code"] == 0
                            and reference_metrics is not None
                            and abs(
                                metrics["audit_objective_usd"]
                                - reference_metrics["audit_objective_usd"]
                            )
                            <= 5e-7
                        ),
                        "true_action_cost_usd": metrics["true_action_cost_usd"],
                        "expected_recourse_cost_usd": metrics["expected_recourse_cost_usd"],
                        "cvar95_recourse_cost_usd": metrics["cvar95_recourse_cost_usd"],
                        "audit_objective_usd": metrics["audit_objective_usd"],
                        "peak_it_mw": metrics["peak_it_mw"],
                        "dummy_actions": metrics["dummy_actions"],
                        "one_hot_violations": metrics["one_hot_violations"],
                        "audit_delta_vs_current_usd": metrics["audit_objective_usd"]
                        - incumbent_metrics["audit_objective_usd"],
                        "audit_improvement_vs_current_pct": 100.0
                        * (incumbent_metrics["audit_objective_usd"] - metrics["audit_objective_usd"])
                        / incumbent_metrics["audit_objective_usd"],
                        "peak_delta_vs_current_mw": metrics["peak_it_mw"] - incumbent_metrics["peak_it_mw"],
                        "audit_gap_to_exact_usd": None
                        if case["metrics"] is None
                        else metrics["audit_objective_usd"]
                        - case["metrics"]["audit_objective_usd"],
                        "audit_gap_to_exact_pct": None
                        if case["metrics"] is None
                        else 100.0
                        * (
                            metrics["audit_objective_usd"]
                            - case["metrics"]["audit_objective_usd"]
                        )
                        / case["metrics"]["audit_objective_usd"],
                        "audit_gap_to_peak_guarded_exact_usd": None
                        if case.get("peak_guarded_metrics") is None
                        else metrics["audit_objective_usd"]
                        - case["peak_guarded_metrics"]["audit_objective_usd"],
                        "peak_guarded": solution == "exact_milp_peak_guarded",
                    }
                )
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[30, 60, 100])
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--time-limit", type=float, default=120.0, help="Seconds per MILP phase and size.")
    parser.add_argument("--mip-rel-gap", type=float, default=0.0)
    parser.add_argument("--lexicographic-tolerance", type=float, default=1e-7)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_solver_controls(
        args.time_limit,
        args.mip_rel_gap,
        args.lexicographic_tolerance,
    )
    (output_csv, output_csv_record), (output_json, output_json_record) = (
        validate_output_paths(args.output_csv, args.output_json)
    )
    sizes = sorted(set(int(size) for size in args.sizes))
    if not sizes or sizes[0] <= 0 or sizes[-1] > 588:
        raise ValueError("Neighborhood sizes must be between 1 and 588.")
    prepare_output_directories(output_csv, output_json)
    data = load_frozen_data()
    incumbent_metrics = metrics_for_rows(data, data.incumbent)
    neighborhoods = choose_nested_jobs(data, sizes, args.seed)

    cases: list[dict[str, Any]] = []
    for size in sizes:
        print(f"Solving exact controlled baseline for {size} jobs...", flush=True)
        case = solve_controlled_case(
            data,
            neighborhoods[size],
            time_limit=args.time_limit,
            mip_rel_gap=args.mip_rel_gap,
            lexicographic_tolerance=args.lexicographic_tolerance,
        )
        case["controlled_job_ids"] = neighborhoods[size]
        case["controlled_job_ids_sha256"] = sha256_lines(neighborhoods[size])
        cases.append(case)
        print(
            json.dumps(
                {
                    "size": size,
                    "status": case["primary"]["label"],
                    "gap": case["primary"]["mip_gap"],
                    "audit_objective": None if case["metrics"] is None else case["metrics"]["audit_objective_usd"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    rows = csv_rows(cases, incumbent_metrics, args.seed)
    unconstrained_certified = all(case["primary"]["code"] == 0 for case in cases)
    peak_tiebreak_certified = all(
        case.get("peak_tiebreak") is not None and case["peak_tiebreak"]["code"] == 0
        for case in cases
    )
    peak_guarded_certified = all(
        case.get("peak_guarded_primary") is not None
        and case["peak_guarded_primary"]["code"] == 0
        for case in cases
    )
    complete_metrics = all(
        case["metrics"] is not None and case.get("peak_guarded_metrics") is not None
        for case in cases
    )
    payload = {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "status": "pass"
        if complete_metrics
        and unconstrained_certified
        and peak_tiebreak_certified
        and peak_guarded_certified
        else "partial",
        "optimality_certified_all": bool(
            unconstrained_certified and peak_tiebreak_certified and peak_guarded_certified
        ),
        "unconstrained_primary_optimality_certified_all": unconstrained_certified,
        "secondary_peak_optimality_certified_all": peak_tiebreak_certified,
        "peak_guarded_primary_optimality_certified_all": peak_guarded_certified,
        "formulation": {
            "controlled_design": "all non-neighborhood jobs fixed to current final; one real action per controlled job",
            "primary_objective": "min expected_recourse_cost_usd + discrete_CVaR95_recourse_cost_usd",
            "secondary_objective": "min peak_it_mw subject to primary optimum plus numerical tolerance",
            "same_peak_comparator": "separate primary solve with peak_it_mw constrained not above current final",
            "exact_recourse": "grid import, renewable curtailment, capped flex delivery and shortfall linearized per scenario and slot",
            "cvar": "Rockafellar-Uryasev epigraph over the evaluator's five fixed scenarios and probabilities",
            "guards": {
                "dummy_actions": 0,
                "one_hot": "equality for every controlled job",
                "full_schedule_action_cost_cap_usd": data.action_cost_guard_usd,
                "cap_rule": "sum of each job's minimum-cost real action + USD 1.00",
            },
            "alpha": ALPHA,
            "power_internal_unit": "kW",
            "selection_rule": SELECTION_RULE,
            "seed": args.seed,
            "requested_sizes": sizes,
            "time_limit_seconds_per_phase": args.time_limit,
            "requested_mip_rel_gap": args.mip_rel_gap,
            "lexicographic_primary_tolerance_usd": args.lexicographic_tolerance,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "solver_backend": "HiGHS bundled through scipy.optimize.milp",
        },
        "inputs": {
            str(path.relative_to(ROOT)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (
                ACTIONS_PATH,
                SITE_PATH,
                SELECTION_SITE_PATH,
                INCUMBENT_PATH,
                METRICS_PATH,
                SOLVER_PATH,
                BASELINE_SCRIPT,
            )
        },
        "current_final_metrics": incumbent_metrics,
        "cases": cases,
        "outputs": {
            "csv": output_csv_record,
            "json": output_json_record,
        },
    }
    publish_output_pair(pd.DataFrame(rows), payload, output_csv, output_json)
    print(f"Wrote {output_csv_record}")
    print(f"Wrote {output_json_record}")
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
