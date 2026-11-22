from __future__ import annotations

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import validate_compiled_circuit
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .track_b import is_clifford_only


def validate_or_raise(source: cirq.Circuit, compiled: cirq.Circuit, backend: CompetitionBackend) -> None:
    track = "track_b" if is_clifford_only(source) else "track_a"
    validation = validate_compiled_circuit(compiled, backend, track)
    if not validation.valid:
        raise ValueError("compiled circuit failed local validation: " + "; ".join(validation.errors))
