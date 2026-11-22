"""Read-only independent checks of the frozen sparse-transfer pilot.

No model is fitted, no optimizer is run, and no experiment file is modified.
Outputs document the exact reviewed snapshot, source chemistry, and decoder.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
UPGRADE = HERE.parent
SPARSE = UPGRADE / "experiments/sparse_transfer"
sys.path.insert(0, str(SPARSE))
from control_problem import FixedProblem, PROJECT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # The historical pilot used epsilon=1e-7. Current joint-search certificates
    # use 1e-5 and must not be combined with these old exploratory lower bounds.
    p = FixedProblem(policy="strict", style_epsilon=1e-7)
    project = p.df
    root = PROJECT
    raw = pd.read_excel(root / "data/A/8.xlsx", sheet_name="Calibrated chemical dataset")
    raw = raw.set_index("beer")
    field_specs = {
        "DMS": (19., "ug/L", "log10"),
        "X2.3.butanedione": (.09, "mg/L", "identity"),
        "X4VG": (.2, "mg/L", "log10"),
        "X.E..2.nonenal": (.0001, "mg/L", "identity"),
    }
    records = []
    for beer, c in p.contexts.items():
        for col, (t, unit, transform) in field_specs.items():
            value = float(c["src"][col])
            representative = 10.**value if transform == "log10" else value
            calibrated = float(raw.loc[beer, col])
            strict_cap = representative if np.isfinite(value) else t
            threshold_cap = max(representative, t) if np.isfinite(value) else t
            positives = raw.loc[raw[col] > 0, col]
            floor = float(positives.min()/2) if transform == "log10" else None
            records.append(dict(
                beer=beer, field=col, unit=unit, transform=transform,
                attachment_encoded_value=value,
                attachment_backtransformed_representative=representative,
                original_calibrated_value=calibrated, threshold=t,
                calibrated_zero=bool(calibrated == 0),
                half_minimum_positive=floor,
                source_is_preprocessing_floor=bool(
                    floor is not None and np.isclose(representative, floor, rtol=1e-8, atol=0)),
                strict_cap_concentration=strict_cap,
                exceedance_cap_concentration=threshold_cap,
                max_cap_relaxes_strict=bool(threshold_cap > strict_cap),
                source_missing=bool(not np.isfinite(value)),
            ))
    chemistry = pd.DataFrame(records)
    chemistry.to_csv(HERE / "sparse_source_chemistry.csv", index=False)

    zero = []
    bound_issues = []
    for beer, c in p.contexts.items():
        row, zz = p.decode(beer, np.zeros(p.p), [])
        zero.append(dict(beer=beer,
            max_rank_drift=float(np.max(np.abs(zz-c["z0"]))),
            kcal_encoded_delta=float(row["kcalperc"]-c["src"].kcalperc),
            mandatory_missing_fields=[p.features[j] for j in c["mandatory"]],
            negative_upper_coordinates=[p.features[j] for j in np.flatnonzero(c["hi"] < -1e-10)],
            source_outside_lower=int(np.sum(c["lo"] > 1e-10)),
            source_outside_upper=int(np.sum(c["hi"] < -1e-10))))
        neg = set(np.flatnonzero(c["hi"] < -1e-10))
        outside = set(np.flatnonzero(c["lo"] > 1e-10)) | neg
        unexpected = outside.difference(c["mandatory"])
        if unexpected:
            bound_issues.append(dict(beer=beer, unexpected=[p.features[j] for j in unexpected]))

    pilot = pd.read_csv(SPARSE / "pilot_profiles.csv")
    saved = pd.read_csv(SPARSE / "pilot_checks.csv")
    checks, _ = p.check_rows(pilot.to_dict("records"))
    checks.to_csv(HERE / "sparse_pilot_recheck.csv", index=False)
    differences = {}
    for col in checks.select_dtypes(include="number"):
        if col not in saved:
            continue
        differences[col] = float(np.nanmax(np.abs(checks[col].to_numpy()-saved[col].to_numpy())))

    per_beer = saved[saved.margin == 0].copy()
    trans = pd.read_csv(root / "A_work/audit/transformation_evidence.csv")
    vg = trans[trans.field.eq("X4VG")].iloc[0]
    summary = dict(
        review_type="read-only independent snapshot audit; no fitting or optimization",
        historical_pilot_policy="strict",
        historical_pilot_style_epsilon=1e-7,
        current_joint_search_style_epsilon=1e-5,
        historical_bounds_are_not_joint_search_certificates=True,
        snapshot_sha256={("source_project/"+str(path.relative_to(root))
            if path.is_relative_to(root) else str(path.relative_to(UPGRADE))):sha(path) for path in [
            SPARSE / "control_problem.py", SPARSE / "pilot_minimum.py",
            SPARSE / "pilot_profiles.csv", SPARSE / "pilot_checks.csv",
            SPARSE / "pilot_linear_bounds.json",
            root / "data/A/1.csv", root / "data/A/8.xlsx",
        ]},
        source_count=len(p.contexts), editable_base_fields=p.p,
        precision_matrix_min_eigenvalue=float(np.linalg.eigvalsh(p.P).min()),
        precision_matrix_symmetry_error=float(np.max(np.abs(p.P-p.P.T))),
        unexpected_nonmandatory_source_bound_violations=bound_issues,
        zero_decode=zero,
        pilot_candidate_count=len(saved), pilot_all_accepted=int(saved.accepted.sum()),
        margin_zero_candidate_count=len(per_beer),
        margin_zero_distance_reduction_range=[
            float(per_beer.target_distance_reduction.min()), float(per_beer.target_distance_reduction.max())],
        margin_zero_ensemble_stale_worsening_count=int((per_beer.ensemble_stale_delta > 1e-8).sum()),
        recheck_max_absolute_difference_by_column=differences,
        recheck_class_and_acceptance_match=bool(
            (checks.predicted_style.to_numpy()==saved.predicted_style.to_numpy()).all()
            and (checks.accepted.to_numpy()==saved.accepted.to_numpy()).all()),
        threshold_policy_relaxes_source_field_count=int(chemistry.max_cap_relaxes_strict.sum()),
        x4vg=dict(
            full_dataset_calibrated_zeros=int((raw.X4VG == 0).sum()),
            full_dataset_min_positive=float(raw.loc[raw.X4VG > 0, "X4VG"].min()),
            half_min_positive=float(vg.half_min_positive),
            preprocessing_floor_source_beer_count=int(chemistry[
                chemistry.field.eq("X4VG")].source_is_preprocessing_floor.sum()),
            floor_is_known_instrument_lod=False,
        ),
    )
    (HERE / "sparse_independent_review.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items()
        if k not in {"snapshot_sha256", "recheck_max_absolute_difference_by_column"}},
        indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
