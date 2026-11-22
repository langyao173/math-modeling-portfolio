from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from upper_bound_common import (
    ROOT,
    affine_model,
    affine_rows_from_columns,
    anf_degree,
    anf_terms_for_outputs,
    exact_cnot_synthesis_mitm,
    format_anf_terms,
    learn_basis_truth_table,
    load_public_samples,
)


TARGETS = ("track_a_002", "track_a_007", "track_a_008")
DEFAULT_JSON = ROOT / "experiments" / "exact_synthesis_candidates.json"
DEFAULT_MD = ROOT / "experiments" / "track_a_truth_table_anf_report.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Crack Track A generators via truth table, ANF, and bounded exact synthesis probes.")
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--max-cnot", type=int, default=11)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    target_names = tuple(item.strip() for item in args.targets.split(",") if item.strip())
    payload: dict[str, Any] = {
        "variant": "track_a_generator_cracking_probe_dev_only",
        "max_cnot": args.max_cnot,
        "samples": {},
    }
    for loaded in load_public_samples(ROOT, targets=target_names):
        payload["samples"][loaded.name] = analyze_sample(loaded.benchmark, max_cnot=args.max_cnot)
    payload["decision"] = decision(payload)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


def analyze_sample(benchmark, *, max_cnot: int) -> dict[str, Any]:
    outputs, deterministic = learn_basis_truth_table(benchmark.circuit)
    width = len(benchmark.logical_qubits)
    terms = anf_terms_for_outputs(outputs, width)
    degrees = [anf_degree(bit_terms) for bit_terms in terms]
    is_affine, constant, columns = affine_model(outputs, width)
    is_permutation = len(set(outputs)) == len(outputs)
    candidates: list[dict[str, Any]] = []
    if is_affine and columns:
        rows = affine_rows_from_columns(columns, width)
        sequence_le_6 = exact_cnot_synthesis_mitm(rows, width, min(6, max_cnot))
        diagnostic_limit = min(8, max_cnot)
        sequence_diagnostic = exact_cnot_synthesis_mitm(rows, width, diagnostic_limit)
        candidates.append(
            {
                "name": f"{benchmark.name}_exact_affine_cnot_le_6",
                "kind": "affine_cnot_exact_mitm",
                "found": sequence_le_6 is not None,
                "max_cnot": 6,
                "cnot_count": len(sequence_le_6) if sequence_le_6 is not None else None,
                "sequence": list(sequence_le_6) if sequence_le_6 is not None else [],
                "target": "A002 <= 6 gate" if benchmark.name == "track_a_002" else "affine baseline",
            }
        )
        candidates.append(
            {
                "name": f"{benchmark.name}_exact_affine_cnot_le_{diagnostic_limit}",
                "kind": "affine_cnot_exact_mitm",
                "found": sequence_diagnostic is not None,
                "max_cnot": diagnostic_limit,
                "cnot_count": len(sequence_diagnostic) if sequence_diagnostic is not None else None,
                "sequence": list(sequence_diagnostic) if sequence_diagnostic is not None else [],
                "target": "diagnostic only; exact MITM above 8 CNOT is intentionally not run",
            }
        )
    else:
        candidates.append(
            {
                "name": f"{benchmark.name}_cnot_only_impossible",
                "kind": "lower_bound",
                "found": False,
                "reason": "non_affine_truth_table_requires_non_CNOT_semantics",
            }
        )
    structure = classify_structure(width, terms, degrees, is_affine, is_permutation)
    lower_bound = lower_bound_summary(width, terms, degrees, is_affine)
    return {
        "family": benchmark.family,
        "width": width,
        "deterministic_basis_map": deterministic,
        "is_permutation": is_permutation,
        "is_affine": is_affine,
        "affine_constant": constant,
        "affine_columns": list(columns),
        "anf_degrees": degrees,
        "max_degree": max(degrees) if degrees else 0,
        "structure": structure,
        "lower_bound": lower_bound,
        "anf": [
            {
                "output_bit": index,
                "degree": degrees[index],
                "term_count": len(terms[index]),
                "terms": [list(term) for term in terms[index]],
                "expression": format_anf_terms(terms[index]),
                "dependencies": sorted({variable for term in terms[index] for variable in term}),
            }
            for index in range(width)
        ],
        "exact_synthesis_candidates": candidates,
    }


def classify_structure(width: int, terms: list[list[tuple[int, ...]]], degrees: list[int], is_affine: bool, is_permutation: bool) -> dict[str, Any]:
    nonlinear_bits = [index for index, degree in enumerate(degrees) if degree >= 2]
    quadratic_bits = [index for index, degree in enumerate(degrees) if degree == 2]
    cubic_bits = [index for index, degree in enumerate(degrees) if degree >= 3]
    dependency_sets = [sorted({variable for term in bit_terms for variable in term}) for bit_terms in terms]
    nested = all(set(dependency_sets[index]).issubset(set(dependency_sets[index + 1]) | {index}) for index in range(width - 1))
    return {
        "likely_classes": likely_classes(is_affine, is_permutation, degrees, dependency_sets, nested),
        "nonlinear_bits": nonlinear_bits,
        "quadratic_bits": quadratic_bits,
        "cubic_or_higher_bits": cubic_bits,
        "dependency_sets": dependency_sets,
        "nested_dependency_heuristic": nested,
        "carry_chain_evidence": bool(is_permutation and nonlinear_bits and (nested or max(degrees) >= 3)),
    }


def likely_classes(
    is_affine: bool,
    is_permutation: bool,
    degrees: list[int],
    dependency_sets: list[list[int]],
    nested: bool,
) -> list[str]:
    if is_affine:
        return ["affine_linear_reversible"]
    classes: list[str] = []
    if is_permutation and max(degrees, default=0) <= 2:
        classes.append("quadratic_reversible")
        classes.append("controlled_increment_or_quadratic_boundary_candidate")
    if is_permutation and max(degrees, default=0) >= 3:
        classes.append("nonlinear_reversible")
        classes.append("carry_or_comparator_candidate")
    if nested:
        classes.append("ripple_carry_like_nested_dependencies")
    if any(len(deps) <= 3 and degree >= 2 for deps, degree in zip(dependency_sets, degrees)):
        classes.append("Peres_MAJ_UMA_local_candidate")
    return classes or ["unclassified"]


def lower_bound_summary(width: int, terms: list[list[tuple[int, ...]]], degrees: list[int], is_affine: bool) -> dict[str, Any]:
    nonlinear_term_count = sum(1 for bit_terms in terms for term in bit_terms if len(term) >= 2)
    max_degree = max(degrees) if degrees else 0
    if is_affine:
        return {
            "cnot_only_possible": True,
            "nonlinear_gate_required": False,
            "reason": "affine truth table",
        }
    return {
        "cnot_only_possible": False,
        "nonlinear_gate_required": True,
        "max_degree": max_degree,
        "nonlinear_term_count": nonlinear_term_count,
        "optimistic_relative_phase_toffoli_cx_floor": 3 * max(1, len({tuple(term) for bit_terms in terms for term in bit_terms if len(term) >= 2})),
        "reason": "CNOT networks can only realize affine maps; nonlinear ANF terms require Toffoli/phase/H-boundary resources.",
    }


def decision(payload: dict[str, Any]) -> dict[str, Any]:
    found_a002_le6 = any(
        candidate["found"] and candidate.get("cnot_count", 999) <= 6
        for candidate in payload["samples"].get("track_a_002", {}).get("exact_synthesis_candidates", [])
    )
    found_a007_a008_le11 = False
    for sample in ("track_a_007", "track_a_008"):
        for candidate in payload["samples"].get(sample, {}).get("exact_synthesis_candidates", []):
            if candidate.get("found") and candidate.get("cnot_count", 999) <= 11:
                found_a007_a008_le11 = True
    return {
        "a002_cx_le_6_found": found_a002_le6,
        "a007_or_a008_cx_le_11_found": found_a007_a008_le11,
        "recommendation": "Continue Track A exact synthesis only if a concrete <=11 nonlinear candidate is found." if found_a007_a008_le11 or found_a002_le6 else "No <=11/<=6 candidate found by this generator probe; keep root frozen unless a stronger global synthesizer is introduced.",
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Track A Truth Table And ANF Generator-Cracking Probe",
        "",
        "This probe asks what A002/A007/A008 compute, rather than searching for local deletions.",
        "",
        f"- Decision: `{payload['decision']['recommendation']}`",
        "",
        "## Summary",
        "",
        "| Sample | Family | Width | Permutation | Affine | Max degree | Likely classes | Candidate found |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for name, row in payload["samples"].items():
        candidate_found = any(candidate.get("found") for candidate in row["exact_synthesis_candidates"])
        lines.append(
            f"| `{name}` | `{row['family']}` | {row['width']} | {row['is_permutation']} | {row['is_affine']} | {row['max_degree']} | `{', '.join(row['structure']['likely_classes'])}` | {candidate_found} |"
        )
    for name, row in payload["samples"].items():
        lines.extend(
            [
                "",
                f"## {name}",
                "",
                f"- Structure: `{', '.join(row['structure']['likely_classes'])}`.",
                f"- Lower bound note: `{row['lower_bound']['reason']}`.",
                "",
                "| Output bit | Degree | Terms | Dependencies |",
                "|---:|---:|---|---|",
            ]
        )
        for bit in row["anf"]:
            lines.append(
                f"| y{bit['output_bit']} | {bit['degree']} | `{bit['expression']}` | `{bit['dependencies']}` |"
            )
        lines.extend(["", "### Exact Synthesis Candidates", "", "| Candidate | Kind | Found | Max CX | CX | Reason/Target |", "|---|---|---:|---:|---:|---|"])
        for candidate in row["exact_synthesis_candidates"]:
            lines.append(
                f"| `{candidate['name']}` | `{candidate['kind']}` | {candidate.get('found')} | {candidate.get('max_cnot')} | {candidate.get('cnot_count')} | `{candidate.get('reason', candidate.get('target', ''))}` |"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
