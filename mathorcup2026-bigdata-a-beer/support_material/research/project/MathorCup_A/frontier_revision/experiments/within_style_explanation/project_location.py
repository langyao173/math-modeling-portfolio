"""Locate either the working checkout or a portable frontier_revision package."""
from pathlib import Path
import os


def find_project():
    configured = os.environ.get("MATHORCUP_PROJECT_DIR")
    if configured:
        location = Path(configured).expanduser().resolve()
        if not (location/"data/A/1.csv").is_file():
            raise FileNotFoundError("MATHORCUP_PROJECT_DIR must contain data/A/1.csv")
        return location
    for parent in Path(__file__).resolve().parents:
        for location in (parent, parent/"project/MathorCup_A"):
            if (location/"data/A/1.csv").is_file():
                return location
    raise FileNotFoundError("Cannot find data/A/1.csv in ancestor projects; set MATHORCUP_PROJECT_DIR.")


PROJECT = find_project()
