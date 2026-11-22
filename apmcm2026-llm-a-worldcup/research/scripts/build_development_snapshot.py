"""Bootstrap the disclosed, development-only formal pre-audit input snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from worldcup_forecasting.application.runtime import write_csv, write_json
from worldcup_forecasting.data import CanonicalDataBundle, build_canonical_data
from worldcup_forecasting.experiments._nested_selection_types import frame_content_hash
from worldcup_forecasting.experiments.development_snapshot import build_snapshot_manifest
from worldcup_forecasting.experiments.research_features import build_research_feature_bundle

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "data" / "development_snapshot"


def _development_bundle(bundle: CanonicalDataBundle) -> CanonicalDataBundle:
    matches = bundle.matches.loc[bundle.matches["Year"].le(2010)].copy()
    match_ids = set(matches["MatchID"])
    return CanonicalDataBundle(
        tournaments=bundle.tournaments.loc[bundle.tournaments["Year"].le(2010)].copy(),
        matches=matches,
        players=bundle.players.loc[bundle.players["MatchID"].isin(match_ids)].copy(),
        reconciliation=bundle.reconciliation.loc[
            bundle.reconciliation["Year"].le(2010)
        ].copy(),
        raw_hashes=dict(bundle.raw_hashes),
        schema_report=dict(bundle.schema_report),
        duplicate_report=bundle.duplicate_report.copy(),
        team_name_registry=bundle.team_name_registry.loc[
            bundle.team_name_registry["first_year"].le(2010)
        ].copy(),
        stage_registry=bundle.stage_registry.loc[
            bundle.stage_registry["first_year"].le(2010)
        ].copy(),
        data_lineage=dict(bundle.data_lineage),
    )


def _dump_atomic(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)


def main() -> None:
    """Create and immediately verify the formal development snapshot."""
    canonical = build_canonical_data(ROOT / "data" / "raw")
    development_bundle = _development_bundle(canonical)
    feature_bundle = build_research_feature_bundle(development_bundle)
    features = feature_bundle.pre_match.copy()
    if int(features["Year"].min()) != 1930 or int(features["Year"].max()) != 2010:
        raise RuntimeError("Development snapshot boundary is not exactly 1930-2010")
    if len(features) != 772 or features["MatchID"].duplicated().any():
        raise RuntimeError("Development snapshot must contain 772 unique matches")
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    _dump_atomic(features, SNAPSHOT_DIR / "development_features.joblib")
    write_csv(
        development_bundle.tournaments.sort_values("Year").reset_index(drop=True),
        SNAPSHOT_DIR / "tournaments_through_2010.csv",
    )
    manifest = build_snapshot_manifest(
        SNAPSHOT_DIR,
        feature_schema_hash=feature_bundle.schema_hash,
        development_frame_hash=frame_content_hash(features),
        source_raw_hashes=canonical.raw_hashes,
        development_rows=len(features),
    )
    write_json(manifest, SNAPSHOT_DIR / "snapshot_manifest.json")
    print(
        json.dumps(
            {
                "status": "development_snapshot_bootstrapped",
                "directory": str(SNAPSHOT_DIR),
                "rows": len(features),
                "feature_schema_hash": feature_bundle.schema_hash,
                "development_frame_hash": manifest["development_frame_hash"],
                "physical_raw_no_read_claim": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
