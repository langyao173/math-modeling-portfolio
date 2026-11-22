"""Prediction input and output contracts."""

from .input_schema import discover_query_frame, normalize_query_frame
from .output_schema import build_minimal_submission, write_atomic_submission

__all__ = [
    "build_minimal_submission",
    "discover_query_frame",
    "normalize_query_frame",
    "write_atomic_submission",
]
