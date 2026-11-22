# Team review and independent work must be documented truthfully.
"""Data location shared by the portable research entry points.

The package layout is ``data/A`` beside ``A_work``.  An explicit CLI path or
MATHORCUP_DATA_DIR takes precedence; a missing explicit path is deliberately
not replaced with another dataset.  The last fallback supports the original
development workspace without depending on its absolute location.
"""
import os
from pathlib import Path


WORK = Path(__file__).resolve().parents[1]
DATA_HELP = (
    "CSV directory; defaults to MATHORCUP_DATA_DIR, then project data/A, "
    "then the legacy workspace data directory."
)


def resolve_data_dir(explicit=None):
    """Return the input directory as an absolute path, without reading data."""
    selected = explicit if explicit is not None else os.environ.get("MATHORCUP_DATA_DIR")
    if selected:
        return Path(selected).expanduser().resolve()
    candidates = [WORK.parent / "data/A", WORK / "package/MathorCup_A/data/A"]
    return next((path for path in candidates if (path / "1.csv").is_file()), candidates[0])
