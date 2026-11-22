from __future__ import annotations

import json
import math
import random
from pathlib import Path

from noise_adaptive_hackathon.formats.backend import CompetitionBackend


JITTER_VARIANTS = ("original", "twoq_lognormal", "readout_lognormal", "mixed_lognormal", "worst_edge_relocated")


def jitter_backend_payload(
    payload: dict[str, object],
    variant: str,
    *,
    rng: random.Random,
) -> dict[str, object]:
    mutated = json.loads(json.dumps(payload))
    if variant == "original":
        return mutated
    if variant in {"twoq_lognormal", "mixed_lognormal"}:
        mutated["two_qubit_error"] = {
            edge: _clamp_error(float(error) * math.exp(rng.gauss(0.0, 0.25)))
            for edge, error in dict(mutated["two_qubit_error"]).items()
        }
    if variant in {"readout_lognormal", "mixed_lognormal"}:
        mutated["readout_error"] = {
            qubit: _clamp_error(float(error) * math.exp(rng.gauss(0.0, 0.30)))
            for qubit, error in dict(mutated["readout_error"]).items()
        }
    if variant == "worst_edge_relocated":
        errors = dict(mutated["two_qubit_error"])
        worst = max(errors, key=lambda edge: errors[edge])
        best = min(errors, key=lambda edge: errors[edge])
        errors[worst], errors[best] = errors[best], errors[worst]
        mutated["two_qubit_error"] = errors
    if variant not in JITTER_VARIANTS:
        raise ValueError(f"unknown backend jitter variant: {variant}")
    return mutated


def write_jittered_backends(
    source_root: Path,
    out_root: Path,
    *,
    seed: int,
    size_classes: tuple[str, ...] = ("medium", "large"),
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    records: list[dict[str, object]] = []
    for size_class in size_classes:
        for source_path in sorted((source_root / "backends" / size_class).glob("*.json")):
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            for variant in JITTER_VARIANTS:
                mutated = jitter_backend_payload(payload, variant, rng=rng)
                snapshot_id = f"{payload['snapshot_id']}_{variant}"
                mutated["snapshot_id"] = snapshot_id
                backend = CompetitionBackend.parse_obj(mutated)
                target = out_root / "backends" / size_class / f"{snapshot_id}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(backend.json(indent=2), encoding="utf-8")
                records.append(
                    {
                        "size_class": size_class,
                        "source_snapshot_id": payload["snapshot_id"],
                        "snapshot_id": snapshot_id,
                        "variant": variant,
                        "num_qubits": backend.num_qubits,
                    }
                )
    return records


def _clamp_error(value: float) -> float:
    return min(max(value, 1e-5), 0.20)
