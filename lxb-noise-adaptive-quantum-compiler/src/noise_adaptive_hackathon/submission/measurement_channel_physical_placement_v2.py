from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import itertools
import math
import time

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import count_two_qubit_gates
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .measurement_channel import (
    AffineParityStar,
    _build_physical_star_circuit,
    detect_affine_parity_star,
)
from .measurement_channel_general import measured_affine_channels_equivalent
from .selection import CompilationCandidate, candidate_metrics
from .track_b import is_clifford_only
from .validation import validate_or_raise


@dataclass(frozen=True)
class _HubPlacementOption:
    cost: float
    center: int
    assignments: tuple[tuple[int, int], ...]


def measurement_channel_physical_placement_v2_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    candidates: list[CompilationCandidate],
    *,
    limit: int = 1,
    max_seconds: float = 3.0,
) -> list[CompilationCandidate]:
    """Generate same-semantics physical star placements for affine Track B circuits.

    The pass is intentionally opt-in. It only applies to circuits whose source
    structure is detected as an affine parity-star measurement channel, and every
    emitted circuit must preserve the measured affine channel. It does not route
    or change CX count; it enumerates legal physical stars and lets the existing
    selector/lowshot infrastructure compare them.
    """
    started = time.perf_counter()
    if not is_clifford_only(source) or not candidates:
        return []
    star = detect_affine_parity_star(source)
    if star is None:
        return []

    base = _best_affine_star_candidate(source, backend, candidates)
    if base is None:
        return []
    base_metrics = candidate_metrics(base, backend, source)
    base_cx = count_two_qubit_gates(base.circuit)

    accepted: list[CompilationCandidate] = []
    seen = {
        tuple(str(op) for op in candidate.circuit.all_operations())
        for candidate in candidates
    }
    for index, layout in enumerate(
        _enumerate_star_layouts(
            star,
            backend,
            max_seconds=max(0.2, max_seconds - (time.perf_counter() - started)),
            limit=max(limit * 3, 4),
        )
    ):
        if time.perf_counter() - started >= max_seconds:
            break
        compiled = _build_physical_star_circuit(star, layout)
        signature = tuple(str(op) for op in compiled.all_operations())
        if signature in seen:
            continue
        seen.add(signature)
        try:
            validate_or_raise(source, compiled, backend)
        except Exception:
            continue
        if not measured_affine_channels_equivalent(source, compiled, layout):
            continue
        candidate = CompilationCandidate(
            f"v20_mc_physical_placement_v2_{index:02d}",
            compiled,
            dict(layout),
            time.perf_counter() - started,
        )
        if not _safe_to_compare(candidate, base, base_metrics, base_cx, backend, source):
            continue
        accepted.append(candidate)
        if len(accepted) >= limit:
            break
    return accepted


def _best_affine_star_candidate(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    candidates: list[CompilationCandidate],
) -> CompilationCandidate | None:
    affine: list[CompilationCandidate] = []
    for candidate in candidates:
        if not _looks_like_affine_star(candidate.name):
            continue
        if not measured_affine_channels_equivalent(source, candidate.circuit, candidate.layout):
            continue
        affine.append(candidate)
    if not affine:
        return None
    return min(
        affine,
        key=lambda candidate: (
            candidate_metrics(candidate, backend, source).two_qubit_count,
            candidate_metrics(candidate, backend, source).depth,
            -candidate_metrics(candidate, backend, source).log_reliability,
            candidate_metrics(candidate, backend, source).readout_cost,
            candidate.name,
        ),
    )


def _looks_like_affine_star(name: str) -> bool:
    return (
        name.startswith("v13c_measurement_parity_star")
        or name.startswith("v13_1_disjoint_star")
        or name.startswith("v20_mc_physical_placement_v2")
        or "measurement_parity" in name
        or "affine" in name
    )


def _enumerate_star_layouts(
    star: AffineParityStar,
    backend: CompetitionBackend,
    *,
    max_seconds: float,
    limit: int,
    per_hub_option_limit: int = 128,
    beam_width: int = 1024,
) -> list[dict[cirq.LineQubit, int]]:
    deadline = time.perf_counter() + max_seconds
    option_groups = _hub_option_groups(star, backend, per_hub_option_limit=per_hub_option_limit)
    if not option_groups:
        return []

    beams: list[tuple[float, frozenset[int], dict[cirq.LineQubit, int]]] = [(0.0, frozenset(), {})]
    for hub, options in option_groups:
        if time.perf_counter() >= deadline:
            return []
        logical_hub = star.logical_qubits[hub]
        next_beams: list[tuple[float, frozenset[int], dict[cirq.LineQubit, int]]] = []
        for current_cost, used, current_layout in beams:
            for option in options:
                nodes = frozenset((option.center, *(physical for _, physical in option.assignments)))
                if used & nodes:
                    continue
                layout = dict(current_layout)
                layout[logical_hub] = option.center
                for leaf, physical_leaf in option.assignments:
                    layout[star.logical_qubits[leaf]] = physical_leaf
                next_beams.append((current_cost + option.cost, used | nodes, layout))
        if not next_beams:
            return []
        beams = sorted(
            next_beams,
            key=lambda item: (item[0], _layout_tiebreak(item[2])),
        )[:beam_width]

    ranked = sorted(
        beams,
        key=lambda item: (
            _layout_elr_cost(star, item[2], backend),
            item[0],
            _layout_tiebreak(item[2]),
        ),
    )
    layouts: list[dict[cirq.LineQubit, int]] = []
    seen: set[tuple[int, ...]] = set()
    for _, _, layout in ranked:
        key = tuple(layout[logical] for logical in star.logical_qubits)
        if key in seen:
            continue
        seen.add(key)
        layouts.append(dict(layout))
        if len(layouts) >= limit:
            break
    return layouts


def _hub_option_groups(
    star: AffineParityStar,
    backend: CompetitionBackend,
    *,
    per_hub_option_limit: int,
) -> list[tuple[int, list[_HubPlacementOption]]]:
    leaves_by_hub: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for hub, leaf, flip in star.leaf_assignments:
        leaves_by_hub[hub].append((leaf, flip))

    neighbors: dict[int, list[int]] = defaultdict(list)
    for left, right in backend.coupling_map:
        neighbors[left].append(right)
        neighbors[right].append(left)
    for node in neighbors:
        neighbors[node] = sorted(neighbors[node])

    groups: list[tuple[int, list[_HubPlacementOption]]] = []
    for hub in sorted(star.hubs, key=lambda item: (-len(leaves_by_hub[item]), item)):
        leaves = tuple(sorted(leaves_by_hub[hub]))
        options: list[_HubPlacementOption] = []
        for center in range(backend.num_qubits):
            if len(neighbors[center]) < len(leaves):
                continue
            nearby = sorted(
                neighbors[center],
                key=lambda node: (
                    _edge_log_cost(backend, center, node),
                    backend.readout(node),
                    -len(neighbors[node]),
                    node,
                ),
            )[: max(len(leaves), 8)]
            for physical_leaves in itertools.combinations(nearby, len(leaves)):
                option = _best_leaf_assignment(hub, center, leaves, physical_leaves, backend)
                if option is not None:
                    options.append(option)
        if not options:
            return []
        groups.append((hub, sorted(options, key=lambda item: (item.cost, item.center, item.assignments))[:per_hub_option_limit]))
    return groups


def _best_leaf_assignment(
    hub: int,
    center: int,
    leaves: tuple[tuple[int, int], ...],
    physical_leaves: tuple[int, ...],
    backend: CompetitionBackend,
) -> _HubPlacementOption | None:
    best: _HubPlacementOption | None = None
    for permutation in itertools.permutations(physical_leaves):
        assignments = tuple((leaf, physical) for (leaf, _), physical in zip(leaves, permutation))
        cost = _hub_assignment_cost(hub, center, leaves, assignments, backend)
        option = _HubPlacementOption(cost=cost, center=center, assignments=assignments)
        if best is None or (option.cost, option.assignments) < (best.cost, best.assignments):
            best = option
    return best


def _hub_assignment_cost(
    hub: int,
    center: int,
    leaves: tuple[tuple[int, int], ...],
    assignments: tuple[tuple[int, int], ...],
    backend: CompetitionBackend,
) -> float:
    leaf_count = len(leaves)
    hub_readout_weight = 1.0 + 0.35 * leaf_count
    cost = 0.22 * hub_readout_weight * _readout_log_cost(backend, center)
    for (leaf, flip), (_, physical_leaf) in zip(leaves, assignments):
        edge = _edge_log_cost(backend, center, physical_leaf)
        edge_sensitivity = backend.edge_error(center, physical_leaf) * (1.0 + 0.15 * leaf_count)
        readout = _readout_log_cost(backend, physical_leaf)
        flip_penalty = 0.0005 if flip else 0.0
        tie = 1e-6 * (hub * 37 + leaf * 11 + physical_leaf)
        cost += edge + 0.65 * edge_sensitivity + 0.16 * readout + flip_penalty + tie
    degree_bonus = -0.0004 * leaf_count
    return cost + degree_bonus


def _safe_to_compare(
    candidate: CompilationCandidate,
    base: CompilationCandidate,
    base_metrics,
    base_cx: int,
    backend: CompetitionBackend,
    source: cirq.Circuit,
) -> bool:
    metrics = candidate_metrics(candidate, backend, source)
    if metrics.compile_time_sec > 12.0:
        return False
    if metrics.two_qubit_count != base_cx:
        return False
    if metrics.depth > base_metrics.depth:
        return False
    if count_two_qubit_gates(candidate.circuit) != count_two_qubit_gates(base.circuit):
        return False
    if metrics.log_reliability < base_metrics.log_reliability - 0.08:
        return False
    if metrics.readout_cost > base_metrics.readout_cost + 0.03:
        return False
    if metrics.clifford_sensitivity_cost > base_metrics.clifford_sensitivity_cost + 1.5:
        return False
    return True


def _layout_elr_cost(
    star: AffineParityStar,
    layout: dict[cirq.LineQubit, int],
    backend: CompetitionBackend,
) -> float:
    cost = sum(_readout_log_cost(backend, physical) for physical in layout.values())
    for hub, leaf, _ in star.leaf_assignments:
        center = layout[star.logical_qubits[hub]]
        physical_leaf = layout[star.logical_qubits[leaf]]
        cost += _edge_log_cost(backend, center, physical_leaf)
    return cost


def _edge_log_cost(backend: CompetitionBackend, left: int, right: int) -> float:
    return -math.log(max(1.0 - backend.edge_error(left, right), 1e-12))


def _readout_log_cost(backend: CompetitionBackend, qubit: int) -> float:
    return -math.log(max(1.0 - backend.readout(qubit), 1e-12))


def _layout_tiebreak(layout: dict[cirq.LineQubit, int]) -> tuple[int, ...]:
    return tuple(physical for _, physical in sorted((logical.x, physical) for logical, physical in layout.items()))
