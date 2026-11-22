"""Parse QBoson PDF reports and verify every returned vector against a frozen model.

Report rows are result candidates, not independent runs. A provider PDF does not
embed the submitted matrix or the credit ledger; those checks remain distinct.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import re
import numpy as np
from .qubo import PackingInstance, read_matrix, decode_spins, packing_qubo
from .results import inspect_solutions
from .solvers import solve_milp

_DATE = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_ROW = re.compile(r"结果\s*(\d+)\s*[：:]\s*Hamiltonian\s*=\s*(" + _NUMBER + r")\s*\[([\s\d,\-+]+)\]")


def _field(text: str, label: str) -> str:
    match = re.search(re.escape(label) + r"\s*[：:]\s*([^\n]+)", text)
    if match is None:
        raise ValueError(f"Missing report field: {label}")
    return match.group(1).strip()


def parse_report_text(text: str) -> dict:
    """Read the provider's observed Chinese PDF text layout; fail on ambiguity."""
    timestamps = re.findall(_DATE, text)
    if len(timestamps) != 3 or timestamps[1] != timestamps[2]:
        raise ValueError("Expected creation time and two agreeing completion times.")
    created, completed = [datetime.fromisoformat(t) for t in timestamps[:2]]
    elapsed = (completed - created).total_seconds()
    if elapsed < 0:
        raise ValueError("Completion precedes creation.")
    status = re.search(r"结\s*束\s*时\s*间\s*(成功|失败|取消)", text)
    if status is None:
        raise ValueError("Cannot read the report status table.")
    runtime = _field(text, "运行时长")
    duration = re.fullmatch(r"(" + _NUMBER + r")\s*ms", runtime)
    if duration is None or float(duration.group(1)) < 0:
        raise ValueError("Expected a nonnegative runtime explicitly in milliseconds.")
    labels = re.findall(r"结果\s*(\d+)\s*[：:]\s*Hamiltonian", text)
    groups = list(_ROW.finditer(text))
    if not groups or len(groups) != len(labels):
        raise ValueError("Incomplete or malformed solution vector(s).")
    rows = []
    for expected, match in enumerate(groups, 1):
        index, energy, vector = match.groups()
        if int(index) != expected:
            raise ValueError("Result numbering must be consecutive and unique.")
        tokens = [t.strip() for t in vector.split(",")]
        if any(not re.fullmatch(r"[+-]?\d+", t) for t in tokens):
            raise ValueError("Invalid spin token.")
        spins = [int(t) for t in tokens]
        if not all(s in (-1, 1) for s in spins):
            raise ValueError("A spin must equal -1 or +1.")
        rows.append({"result_index": expected, "reported_H": float(energy), "spins": spins})
    filename = _field(text, "文件名称")
    if not re.fullmatch(r"[A-Za-z0-9_-]+_ising\.csv", filename):
        raise ValueError("Unexpected input filename; paths are not accepted.")
    return {
        "task_name": _field(text, "任务名称"), "task_id": _field(text, "任务ID"),
        "status": status.group(1), "machine": _field(text, "量子计算机"),
        "matrix_type": _field(text, "矩阵类型"), "mode": _field(text, "计算模式"),
        "sampling_count_as_reported": _field(text, "采样次数"), "input_filename": filename,
        "created_at_as_reported": timestamps[0], "completed_at_as_reported": timestamps[1],
        "timestamp_timezone": "not specified in the PDF",
        "reported_runtime_ms": float(duration.group(1)), "created_to_completed_s": elapsed,
        "runtime_definition": "provider field 运行时长; not assumed to be full task latency",
        "candidate_rows_are_not_independent_runs": True, "rows": rows,
    }


def read_provider_pdf(path: str | Path) -> dict:
    import fitz
    path = Path(path)
    with fitz.open(path) as pdf:
        text = "\n".join(page.get_text() for page in pdf)
        count = len(pdf)
    result = parse_report_text(text)
    result.update({"report_filename": path.name, "report_pages": count})
    return result


def select_raw_candidate(rows: list[dict]) -> dict | None:
    """Filter invalid assignments, then minimize H and use original utility on ties.

    This selection uses no optimum oracle or repaired solution. A row index is
    the final deterministic tie break. The reported raw rows are never changed.
    """
    feasible = [r for r in rows if r["raw_feasible"] and r["lifted_raw_feasible"]]
    if not feasible:
        return None
    return min(feasible, key=lambda r: (r["ising_energy"], -r["lifted_raw_original_utility"], r["row"]))


def dispatch_gate(instance: PackingInstance, seed, proposed, *, submitted_state: str,
                  current_state: str, elapsed_s: float, deadline_s: float) -> tuple[np.ndarray, dict]:
    """Retrospective/action gate. Never treat a late or stale vector as executable.

    elapsed_s must come from a monotonic request-to-receipt clock online. The
    report's server timestamps are only a partial retrospective latency measure.
    """
    seed = np.asarray(seed)
    if not instance.evaluate(seed)["feasible"]:
        raise ValueError("The fallback seed is infeasible.")
    if not np.isfinite(elapsed_s) or not np.isfinite(deadline_s) or elapsed_s < 0 or deadline_s <= 0:
        raise ValueError("Invalid deadline/elapsed time.")
    if elapsed_s > deadline_s:
        reason = "deadline_exceeded"
    elif submitted_state != current_state:
        reason = "state_changed"
    elif proposed is None:
        reason = "no_feasible_returned_candidate"
    else:
        proposed = np.asarray(proposed)
        # Malformed vectors are rejected explicitly, not silently coerced.
        if not instance.evaluate(proposed)["feasible"]:
            reason = "infeasible_returned_candidate"
        elif float(instance.weights @ proposed) <= float(instance.weights @ seed) + 1e-10:
            reason = "no_strict_original_utility_improvement"
        else:
            return proposed.astype(int).copy(), {"accepted": True, "reason": "strict_feasible_improvement"}
    return seed.astype(int).copy(), {"accepted": False, "reason": reason}


def verify_provider_report(pdf_path: str | Path, model_dir: str | Path,
                           *, reference_time_limit: float = 20.) -> dict:
    provider = read_provider_pdf(pdf_path)
    if provider["status"] != "成功" or provider["machine"] != "SPQC-550":
        raise ValueError("Expected a successful SPQC-550 task.")
    if provider["matrix_type"] != "Ising" or provider["mode"] != "优化模式":
        raise ValueError("Unexpected matrix type or calculation mode.")
    name = provider["input_filename"].removesuffix("_ising.csv")
    model_dir = Path(model_dir)
    meta_path = model_dir / (name + ".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["name"] != name:
        raise ValueError("Metadata name mismatch.")
    expected_n = int(meta["n_binary"]) + 1
    if any(len(r["spins"]) != expected_n for r in provider["rows"]):
        raise ValueError("A returned vector has the wrong dimension.")
    J = read_matrix(model_dir / provider["input_filename"])
    if J.shape != (expected_n, expected_n):
        raise ValueError("Unexpected matrix dimension.")
    for row in provider["rows"]:
        s = np.asarray(row["spins"], dtype=np.int64)
        if not np.isclose(float(-s @ J @ s), row["reported_H"], rtol=0, atol=1e-8):
            raise ValueError(f"Energy mismatch in {name}, result {row['result_index']}.")
    result = inspect_solutions(meta_path, np.asarray([r["spins"] for r in provider["rows"]]),
                               source="user_supplied_QBoson_PDF_report")
    # Use explicit evidence fields instead of an unqualified authenticity claim.
    result.pop("physical_run_verified", None)
    result.pop("physical_verification_requirement", None)
    result.update({"name": name, "provider": provider, "provider_report_attached": True,
                   "all_reported_energies_match_expected_matrix": True,
                   "input_match_basis": "filename, vector dimensions and every vector/energy pair",
                   "input_matrix_embedded_in_report": False, "credit_record_embedded_in_report": False,
                   "scope": meta["scope"], "binary_variables": expected_n - 1,
                   "ising_spins": expected_n, "full_candidates": meta["original_n"],
                   "source_epoch": meta["source_epoch"], "epoch_seconds": meta["epoch_seconds"]})
    quant = PackingInstance.from_dict(meta)
    full = PackingInstance.from_dict(meta["lifting"]["full_instance"])
    original = PackingInstance(meta["original_weights"], quant.edges, quant.labels)
    refs = {}
    for key, inst in (("quantized", quant), ("full_original", full), ("conditional_original_core", original)):
        x, info = solve_milp(inst, reference_time_limit)
        if not info["optimal"]:
            raise ValueError(f"Reference optimality not established: {key}.")
        refs[key] = {"objective": float(inst.weights @ x), "bits": x.tolist(), **info}
    qvalue = refs["quantized"]["objective"]
    sp = np.r_[2*np.asarray(refs["quantized"]["bits"])-1, 1]
    Hstar = float(-sp @ J @ sp)
    conditional = refs["conditional_original_core"]["objective"] + float(meta["fixed_original_utility"])
    if not np.isclose(Hstar, meta["exact_ising_minimum"]) or not np.isclose(qvalue, meta["exact_quantized_optimum"]):
        raise ValueError("Recomputed optimum disagrees with the pre-run reference.")
    if not np.isclose(conditional, meta["conditional_original_reference_utility"]) or not np.isclose(refs["full_original"]["objective"], meta["full_original_reference_utility"]):
        raise ValueError("Original-utility optimum disagrees with the pre-run reference.")
    result["reference_recheck"] = refs
    for raw, reported in zip(result["rows"], provider["rows"]):
        raw.update({"result_index": reported["result_index"], "reported_H": reported["reported_H"],
                    "spins": reported["spins"], "energy_residual": raw["ising_energy"]-reported["reported_H"],
                    "hits_known_quantized_optimum": bool(raw["raw_feasible"] and np.isclose(raw["ising_energy"],Hstar,rtol=0,atol=1e-8))})
        # State-level route timing is not available in a frozen assignment model.
        if raw["lifted_raw_feasible"]:
            ix = np.flatnonzero(raw["lifted_full_bits"])
            labels = [full.labels[i] for i in ix]
            vehicles = [d["vehicle"] for d in labels]
            requests = [r for d in labels for r in d["requests"]]
            if len(vehicles)!=len(set(vehicles)) or len(requests)!=len(set(requests)):
                raise ValueError("Lifted assignment duplicates a vehicle or request.")
            raw["new_request_count"] = len(requests)
            raw["updated_vehicle_count"] = len(vehicles)
            raw["new_request_ids"] = sorted(requests)
    selected = select_raw_candidate(result["rows"])
    if selected is None:
        raise ValueError("No raw feasible candidate returned; explicit fallback is required.")
    polished = selected.get("tie_polished_full_original_utility", selected["lifted_raw_original_utility"])
    corebits = np.asarray(selected.get("tie_polished_bits",selected["bits"]),int)
    lifted = np.zeros(len(full.weights),int)
    lifted[meta["lifting"]["fixed"]] = 1
    lifted[meta["lifting"]["kept"]] = corebits
    server_gate_bits, server_gate_info = dispatch_gate(full, meta["lifting"]["seed"], lifted,
        submitted_state="frozen_snapshot", current_state="frozen_snapshot",
        elapsed_s=provider["created_to_completed_s"], deadline_s=meta["epoch_seconds"])
    # Report only the server-span check; no fabricated client receipt timestamps.
    result["selected_result_index"] = selected["result_index"]
    result["selection_rule"] = "feasible first; minimum H; original-utility tie break; row order"
    result["summary"] = {
        "name": name, "task_name": provider["task_name"], "task_id": provider["task_id"],
        "scope": meta["scope"], "full_candidates": meta["original_n"], "ising_spins": expected_n,
        "returned_rows": len(result["rows"]), "raw_feasible_rows": sum(r["raw_feasible"] for r in result["rows"]),
        "optimal_rows": sum(r["hits_known_quantized_optimum"] for r in result["rows"]),
        "energy_checks_passed": len(result["rows"]), "known_minimum_H": Hstar,
        "best_raw_H": selected["ising_energy"], "absolute_H_gap": selected["ising_energy"]-Hstar,
        "reported_runtime_ms": provider["reported_runtime_ms"], "created_to_completed_s": provider["created_to_completed_s"],
        "epoch_seconds": meta["epoch_seconds"], "server_span_within_epoch": provider["created_to_completed_s"]<=meta["epoch_seconds"],
        "first_raw_original_utility": result["rows"][0]["lifted_raw_original_utility"],
        "selected_raw_result_index": selected["result_index"],
        "selected_raw_original_utility": selected["lifted_raw_original_utility"],
        "selected_new_request_count": selected["new_request_count"],
        "polished_original_utility": polished,
        "classical_polish_elapsed_s": selected.get("tie_polish",{}).get("elapsed_s",0),
        "classical_polish_exchanges": selected.get("tie_polish",{}).get("n_exchanges",0),
        "conditional_original_optimum": conditional,
        "full_original_optimum": refs["full_original"]["objective"],
        "seed_original_utility": float(full.weights @ np.asarray(meta["lifting"]["seed"])),
        "conditional_original_gap": conditional-polished,
        "full_original_gap": refs["full_original"]["objective"]-polished,
        "retrospective_gate_reason": server_gate_info["reason"],
        "is_full_day_hardware_replay": False,
    }
    result["polished_lifted_bits"] = lifted.tolist()
    result["retrospective_gate"] = {"basis":"server creation-to-completion interval only; client receipt unmeasured",
        **server_gate_info,"returned_bits":server_gate_bits.tolist()}
    return result
