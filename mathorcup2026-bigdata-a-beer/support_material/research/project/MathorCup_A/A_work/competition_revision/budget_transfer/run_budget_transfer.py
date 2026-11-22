"""Prespecified expanded-budget chemical-profile transfer with unchanged nonselected coordinates.

This extension reuses the existing Q3 objective, rank map and fitted models.
Selection/optimization is exploratory design, not an independent tasting test.
Author-seven controls use the compounds actually listed for spiking in the
published validation table, not a similarly named but different ester.
"""
from pathlib import Path
import os
for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import sys, json, argparse, time
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
AWORK = HERE.parents[1]
sys.path.insert(0, str(AWORK / "q3"))
sys.path.insert(0, str(AWORK / "q1"))
from conditional_model import DERIVED, TECHNICAL
from transfer_profiles import DEFECTS, as_raw
from sensory_model import predict_bundle
from forward_support import forward_supports

AUTHOR_SEVEN = ["ethyl_acetate", "ethyl_hexanoate", "isopentyl_acetate",
                "phenethyl_acetate", "ethanol..v.v.", "glycerol", "lactic_acid.mg.L."]
MARGINS = [0., .025, .05, .1, .2, .4, .8]
ALPHAS = np.unique(np.r_[.01, .025, np.arange(1, 21) / 20.])
BUDGETS = [3, 5, 10, 20, 40, 60, 80, 100, 140, 180]
LAMBDA = .2
TOL = 1e-8


def main():
    ap = argparse.ArgumentParser()
    data_candidates=[AWORK.parent/"data/A", AWORK/"package/MathorCup_A/data/A"]
    default_data=Path(os.environ["MATHORCUP_DATA_DIR"]).expanduser() if os.environ.get("MATHORCUP_DATA_DIR") else next((p for p in data_candidates if (p/"1.csv").exists()), AWORK.parent/"data/A")
    ap.add_argument("--data", type=Path, default=default_data)
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    df = pd.read_csv(args.data / "1.csv")
    original = df.set_index("beer")
    calibrated = pd.read_excel(args.data / "8.xlsx", sheet_name="Calibrated chemical dataset").set_index("beer")
    pack = joblib.load(AWORK / "q3/transfer_and_augmentation_models.joblib")
    control, features, proxy = pack["model"], pack["features"], pack["stale_proxy"]
    summaries = pack["derived_predictors"]
    bundle = joblib.load(AWORK / "q1/sensory_ensemble.joblib")
    p = len(features)
    x = df[features].to_numpy(float)
    z = control.map_.transform(x)
    mu = control.mean_["Tripel"]
    precision = np.linalg.inv(control.cov_["Tripel"])
    hessian = precision + LAMBDA * np.eye(p)
    a = proxy.coef_
    source_indices = np.flatnonzero(df.tasting_category_fine.to_numpy() == "Pils/Lager")
    original_final = pd.read_csv(AWORK / "q3/transfer_final_profiles.csv").set_index("beer")
    original_final_diagnostics = pd.read_csv(AWORK / "q3/transfer_final_diagnostics.csv")
    clf = LogisticRegression(C=.1, max_iter=3000, class_weight="balanced").fit(z, df.tasting_category_fine)
    ti = list(clf.classes_).index("Tripel")
    base_predictions = predict_bundle(bundle, df)
    base_predictions.index = df.beer
    author_idx = np.array([features.index(c) for c in AUTHOR_SEVEN], dtype=int)
    pred0_summary = {c: reg.predict(z[source_indices]) for c, reg in summaries.items()}

    paths, path_z, path_records, qp_records = [], [], [], []
    source_context = {}
    support_rows, forward_records = [], []
    for src_no, idx in enumerate(source_indices):
        src = df.iloc[idx]
        beer = int(src.beer)
        x0 = x[idx]
        z0 = z[idx]
        lower = control.map_.lower_.copy()
        upper = control.map_.upper_.copy()
        caps = {}
        for col, info in DEFECTS.items():
            j = features.index(col)
            cap = x0[j] if np.isfinite(x0[j]) else (np.log10(info["threshold"]) if info["transform"] == "log10" else info["threshold"])
            caps[j] = cap
            caprow = control.map_.impute(x0[None])[0]
            caprow[j] = cap
            upper[j] = min(upper[j], control.map_.transform(caprow[None])[0, j])
        mandatory = np.flatnonzero(z0 > upper + TOL)
        b = precision @ (mu - z0)
        step = np.clip(b / np.diag(hessian), lower-z0, upper-z0)
        benefit = b * step - .5*np.diag(hessian)*step**2
        gradient_rank = np.argsort(-benefit, kind="stable")
        full_z = control.map_.transform(original_final.loc[[beer], features])[0]
        full_rank = np.argsort(-np.abs(full_z-z0), kind="stable")
        methods = {}
        for rank_name, ranking in [("gradient", gradient_rank), ("full_change", full_rank)]:
            priority = list(mandatory) + [int(j) for j in ranking if j not in mandatory]
            for k in BUDGETS:
                methods[f"{rank_name}_k{k}"] = (np.array(priority[:k], dtype=int), False, k, rank_name)
        adaptive_supports, adaptive_trace = forward_supports(hessian,b,a,lower-z0,upper-z0,mandatory,budgets=BUDGETS)
        for k, selected in adaptive_supports.items():
            methods[f"forward_schur_k{k}"]=(selected,False,k,"conditional_schur_forward")
        for rec in adaptive_trace:
            rec.update(beer=beer,selected_field=features[rec["selected_index"]])
            forward_records.append(rec)
        methods["all_225"] = (np.arange(p), False, p, "all")
        methods["author7_free"] = (author_idx, False, 7, "author_actual_spiking_compounds")
        methods["author7_add_only"] = (author_idx, True, 7, "author_actual_spiking_compounds")
        author_repaired = np.unique(np.r_[author_idx, mandatory]).astype(int)
        methods["author7_plus_required"] = (author_repaired, False, len(author_repaired), "author_plus_missing_defect_repair")
        d0 = np.sqrt((z0-mu) @ precision @ (z0-mu)/p)
        source_context[beer] = dict(idx=int(idx), z0=z0, x0=x0, caps=caps,
                                    d0=d0, base_stale=float(base_predictions.loc[beer, "stale_hops"]),
                                    src_no=src_no)
        for method, (selected, add_only, k, rank_name) in methods.items():
            frozen = np.ones(p, dtype=bool)
            frozen[selected] = False
            support_rows.append(dict(beer=beer, method=method, allowed_coordinates=k,
                                     support_rule=rank_name, add_only=add_only,
                                     mandatory_missing_defect_coordinates=json.dumps([features[j] for j in mandatory]),
                                     selected_fields=json.dumps([features[j] for j in selected])))
            # The zero-change measured source is retained even when its missing
            # defect cannot be certified under the strict generated-output rule.
            zero = {c: src[c] for c in df.columns}
            zero.update(method=method, beer=beer, margin=0., alpha=0., allowed_coordinates=k,
                        source_style="Pils/Lager", support_rule=rank_name,
                        is_generated=False, qp_success=True)
            paths.append(zero); path_z.append(z0.copy())
            path_records.append(dict(beer=beer, method=method, margin=0., alpha=0., allowed_coordinates=k,
                                     qp_success=True, qp_status="source_returned_no_intervention"))
            frozen_violation = bool(np.any(z0[frozen] > upper[frozen] + TOL))
            if frozen_violation:
                qp_records.append(dict(beer=beer, method=method, margin=np.nan, allowed_coordinates=k,
                    success=False, message="Frozen missing-defect imputation exceeds output threshold; exact support infeasible",
                    iterations=0, seconds=0., linear_delta=np.nan))
                continue
            sub_h = hessian[np.ix_(selected, selected)]
            sub_b = b[selected]
            dlo, dhi = lower[selected]-z0[selected], upper[selected]-z0[selected]
            if add_only:
                dlo = np.maximum(dlo, 0.)
            for margin in MARGINS:
                start = np.clip(np.zeros(len(selected)), dlo, dhi)
                ct = time.perf_counter()
                ans = minimize(lambda d: float(.5*d@sub_h@d-sub_b@d), start,
                    jac=lambda d: sub_h@d-sub_b, method="SLSQP", bounds=Bounds(dlo, dhi),
                    constraints=[LinearConstraint(a[selected], -np.inf, -margin)],
                    options={"maxiter": 600, "ftol": 1e-10})
                elapsed = time.perf_counter()-ct
                delta = np.zeros(p); delta[selected] = ans.x
                valid = bool(ans.success and np.max(dlo-ans.x)<=TOL and np.max(ans.x-dhi)<=TOL and a@delta<=-margin+TOL)
                qp_records.append(dict(beer=beer, method=method, margin=margin, allowed_coordinates=k,
                    success=valid, message=str(ans.message), iterations=int(ans.nit), seconds=elapsed,
                    linear_delta=float(a@delta), objective_delta=float(.5*delta@hessian@delta-b@delta)))
                if not valid:
                    continue
                for alpha in ALPHAS:
                    zn = z0 + alpha*delta
                    # A missing-defect cap may require the full correction. Do
                    # not clip after interpolation: that would change support or
                    # falsely label an infeasible partial path as a solution.
                    if np.max(zn-upper)>TOL or np.max(lower-zn)>TOL:
                        continue
                    xn = control.map_.inverse_transform(zn[None])[0]
                    # Frozen known fields are bit-for-bit original, and frozen
                    # missing fields remain missing in the exported profile.
                    xn[frozen] = x0[frozen]
                    for j, cap in caps.items():
                        if j in selected:
                            xn[j] = min(xn[j], cap)
                    zn = control.map_.transform(xn[None])[0]
                    item = {c: src[c] for c in df.columns}
                    item.update(dict(zip(features, xn)))
                    item.update(method=method, beer=beer, margin=margin, alpha=float(alpha),
                                allowed_coordinates=k, source_style="Pils/Lager", support_rule=rank_name,
                                is_generated=True, qp_success=True)
                    paths.append(item); path_z.append(zn)
                    path_records.append(dict(beer=beer, method=method, margin=margin, alpha=float(alpha),
                                             allowed_coordinates=k, qp_success=True, qp_status="optimized_path_candidate"))
        print(f"beer {beer}: {len(paths)} path candidates; {len(qp_records)} QP attempts", flush=True)

    candidates = pd.DataFrame(paths)
    all_z = np.asarray(path_z)
    is_generated = candidates.is_generated.to_numpy(bool)
    # Baseline-anchored auxiliary reconstruction ensures that leaving direct
    # coordinates unchanged cannot change four undefined summary fields.
    for col, reg in summaries.items():
        new = reg.predict(all_z)
        base = np.array([pred0_summary[col][source_context[int(b)]["src_no"]] for b in candidates.beer])
        old = np.array([original.loc[int(b), col] for b in candidates.beer], dtype=float)
        v = np.where(np.isfinite(old), old + new-base, new)
        lo, hi = np.nanmin(df[col]), np.nanmax(df[col])
        v = np.clip(v, lo, hi)
        candidates.loc[is_generated, col] = v[is_generated]
    raw = {c: np.power(10., candidates[c].to_numpy(float)) for c in
           ["ethanol..v.v.", "GFS", "unfermentables", "protein.g.L.", "lactic_acid.mg.L."]}
    kcal = np.log10(5.523*raw["ethanol..v.v."] + .4*(raw["GFS"]+raw["unfermentables"]+raw["protein.g.L."]) + .0003*raw["lactic_acid.mg.L."])
    candidates.loc[is_generated, "kcalperc"] = kcal[is_generated]
    # Evaluate in modest batches, keeping memory bounded for kernel models.
    pp = []
    for start in range(0, len(candidates), 512):
        pp.append(predict_bundle(bundle, candidates.iloc[start:start+512]))
    predictions = pd.concat(pp, ignore_index=True)
    all_style = clf.predict(all_z)
    all_prob = clf.predict_proba(all_z)[:, ti]
    checks = []
    for i, rec in enumerate(path_records):
        beer = rec["beer"]; ctx = source_context[beer]
        zn = all_z[i]; change = zn-ctx["z0"]
        chemical_violation = 0.
        unknown_defect_unresolved = 0
        for j, cap in ctx["caps"].items():
            v = candidates.iloc[i][features[j]]
            if not np.isfinite(v):
                unknown_defect_unresolved += 1
            else:
                chemical_violation = max(chemical_violation, float(v-cap))
        linear_delta = float(a@change)
        nonlinear_delta = float(predictions.loc[i, "stale_hops"]-ctx["base_stale"])
        dn = np.sqrt((zn-mu)@precision@(zn-mu)/p)
        obj = .5*(zn-mu)@precision@(zn-mu)+LAMBDA/2*(change@change)
        is_moving = bool(np.linalg.norm(change)>1e-8)
        accepted = bool(chemical_violation<=TOL and unknown_defect_unresolved==0 and linear_delta<=TOL and nonlinear_delta<=TOL)
        # An unchanged measured source is always retained as a fallback action,
        # but is not a certified generated output if a defect remains unknown.
        eligible = bool(accepted or rec["alpha"]==0.)
        checks.append(dict(**rec, candidate_index=i, direct_coordinates_changed=int(np.sum(np.abs(change)>1e-8)),
            nontrivial=is_moving, source_target_distance=float(ctx["d0"]), target_distance=float(dn),
            target_distance_reduction=float(1-dn/ctx["d0"]), intervention_rms=float(np.linalg.norm(change)/np.sqrt(p)),
            quadratic_objective=float(obj), max_field_scale_chemical_violation=chemical_violation,
            unresolved_missing_defects=unknown_defect_unresolved, linear_stale_delta=linear_delta,
            ensemble_stale_delta=nonlinear_delta, accepted_generated_profile=accepted and is_moving,
            eligible_for_action=eligible, predicted_style=str(all_style[i]), target_probability=float(all_prob[i])))
    checkdf = pd.DataFrame(checks)
    checkdf.to_csv(args.out / "path_checks.csv", index=False)
    pd.DataFrame(qp_records).to_csv(args.out / "qp_attempts.csv", index=False)
    pd.DataFrame(forward_records).to_csv(args.out / "forward_support_trace.csv", index=False)
    pd.DataFrame(support_rows).to_csv(args.out / "selected_supports.csv", index=False)
    selected_indices, selected_records, selected_profile_rows = [], [], []
    support_by_case = {(r["beer"], r["method"]): r for r in support_rows}
    for (beer, method), group in checkdf.groupby(["beer", "method"], sort=False):
        # A budget is an upper bound, so each larger nested support also keeps
        # all candidates already found for smaller budgets. A finite path grid
        # must not discard an available feasible lower-budget design.
        if "_k" in method:
            family, kval = method.rsplit("_k", 1)
            nested_methods = [f"{family}_k{k}" for k in BUDGETS if k<=int(kval)]
            group = checkdf[(checkdf.beer==beer) & checkdf.method.isin(nested_methods)]
        valid = group[group.eligible_for_action]
        best = valid.sort_values(["quadratic_objective", "direct_coordinates_changed", "margin", "alpha"], kind="stable").iloc[0]
        ii = int(best.candidate_index)
        selected_indices.append(ii)
        chosen = best.to_dict()
        chosen["selection_origin_method"] = chosen["method"]
        chosen["method"] = method
        chosen["allowed_coordinates"] = support_by_case[beer, method]["allowed_coordinates"]
        selected_records.append(chosen)
        chosen_profile = candidates.iloc[ii].to_dict()
        chosen_profile["selection_origin_method"] = chosen_profile["method"]
        chosen_profile["method"] = method
        chosen_profile["allowed_coordinates"] = chosen["allowed_coordinates"]
        selected_profile_rows.append(chosen_profile)
    selected = pd.DataFrame(selected_records)
    selected_profiles = pd.DataFrame(selected_profile_rows)
    # Metadata only: the supplied category describes the observed source.
    # A requested destination and a fitted classifier are not new tasting labels.
    selected_profiles["source_tasting_category_fine"] = selected_profiles.beer.map(original.tasting_category_fine)
    selected_profiles["tasting_category_fine"] = selected_profiles.source_tasting_category_fine
    selected_profiles["requested_target_style"] = "Tripel"
    selected_profiles["diagnostic_predicted_style"] = selected.predicted_style.to_numpy()
    selected_profiles["profile_origin"] = np.where(selected.nontrivial, "virtual_transfer", "observed_source_fallback")
    selected_profiles["tasting_category_fine_origin"] = "observed_source_label; transfer_not_remeasured"
    selected_predictions = predictions.iloc[selected_indices].reset_index(drop=True)
    selected.to_csv(args.out / "selected_diagnostics.csv", index=False)
    selected_profiles.to_csv(args.out / "selected_profiles.csv", index=False)
    pd.concat([selected_profiles[["beer", "method"]], selected_predictions], axis=1).to_csv(args.out / "selected_sensory_predictions.csv", index=False)

    certificates, changes = [], []
    support_lookup = {(r["beer"], r["method"]): set(json.loads(r["selected_fields"])) for r in support_rows}
    exact_frozen_ok = True
    for i, row in selected_profiles.iterrows():
        beer = int(row.beer); src = original.loc[beer]
        support = support_lookup[beer, row.method]
        for col in features:
            before, after = src[col], row[col]
            frozen = col not in support
            same = bool((pd.isna(before) and pd.isna(after)) or before==after)
            if frozen and not same:
                exact_frozen_ok = False
            if not same:
                changes.append(dict(beer=beer, method=row.method, chemical=col, before=before, after=after,
                                    selected=col in support, observed_baseline=bool(np.isfinite(before))))
        for col, info in DEFECTS.items():
            before, after = as_raw(src[col], info), as_raw(row[col], info)
            original_cal = calibrated.loc[beer, col] if col in calibrated else np.nan
            if not np.isfinite(src[col]):
                status="missing"
            elif info["transform"]=="log10" and (not np.isfinite(original_cal) or original_cal<=0):
                status="zero_or_nonpositive_replacement"
            else:
                status="positive_reported_measurement"
            certificates.append(dict(beer=beer, method=row.method, chemical=col, unit=info["unit"],
                source_status=status, source_model_representative=before, transferred_concentration=after,
                threshold=info["threshold"], nonincrease=(bool(after<=before+TOL) if np.isfinite(before) and np.isfinite(after) else None),
                output_below_threshold=(bool(after<=info["threshold"]+TOL) if np.isfinite(after) else None)))
    pd.DataFrame(certificates).to_csv(args.out / "chemical_certificate.csv", index=False)
    pd.DataFrame(changes).to_csv(args.out / "changed_direct_fields.csv", index=False)
    assert exact_frozen_ok, "A supposedly frozen direct field changed."
    assert np.all(selected.direct_coordinates_changed<=selected.allowed_coordinates)
    moving = selected[selected.nontrivial]
    assert np.all(moving.accepted_generated_profile), "Selected movement failed acceptance checks"
    for tech in TECHNICAL:
        assert all((pd.isna(original.loc[int(r.beer), tech]) and pd.isna(r[tech])) or original.loc[int(r.beer), tech]==r[tech]
                   for _, r in selected_profiles.iterrows())
    summary = selected.groupby("method", sort=False).agg(
        n_sources=("beer", "size"), n_nontrivial=("nontrivial", "sum"),
        allowed_coordinates_mean=("allowed_coordinates", "mean"), actual_changed_mean=("direct_coordinates_changed", "mean"),
        target_distance_reduction_mean=("target_distance_reduction", "mean"),
        target_distance_reduction_min=("target_distance_reduction", "min"),
        intervention_rms_mean=("intervention_rms", "mean"), target_probability_mean=("target_probability", "mean"),
        tripel_classified=("predicted_style", lambda s: int((s=="Tripel").sum())),
        ensemble_stale_max_delta=("ensemble_stale_delta", "max"), linear_stale_max_delta=("linear_stale_delta", "max"),
        unresolved_missing_defects=("unresolved_missing_defects", "sum"),
        accepted_generated_profiles=("accepted_generated_profile", "sum"),
        objective_mean=("quadratic_objective", "mean")).reset_index()
    summary.to_csv(args.out / "pareto_summary.csv", index=False)
    original_final_diagnostics.to_csv(args.out / "previous_full_transfer_reference.csv", index=False)
    metadata = dict(n_sources=len(source_indices), n_methods=len(summary), prespecified_budgets=BUDGETS, n_qp_attempts=len(qp_records),
        n_forward_candidate_qp_checks=int(sum(r["evaluated_candidates"] for r in forward_records)),
        n_qp_success=int(pd.DataFrame(qp_records).success.sum()), n_path_candidates=len(candidates),
        seconds=time.perf_counter()-started, lambda_preservation=LAMBDA, margins=MARGINS, alphas=ALPHAS.tolist(),
        support_budget_counts="225 direct model coordinates; excludes five derived fields and one internal standard",
        support_rules={"gradient":"per-coordinate bound-aware quadratic improvement at source, mandatory missing-defect repair first",
                       "full_change":"absolute rank-coordinate change in the previously generated complete solution, mandatory repair first",
                       "forward_schur":"successively recomputed residual gradient and conditional curvature; top six proposals checked by the original bounded QP; selected support grows to 180"},
        author_seven_fields=AUTHOR_SEVEN,
        author_mapping_note="Phenylethyl acetate in validation concentrations maps to phenethyl_acetate, not the distinct ethyl.phenylacetate.",
        auxiliary_reconstruction="source-anchored reg(new)-reg(source), clipped to observed range; kcal identity exact; frozen direct NaNs preserved",
        selection="minimum original quadratic objective among accepted path candidates, including source fallback and all smaller-budget candidates in each nested support family",
        frozen_direct_fields_exactly_preserved=exact_frozen_ok,
        all_accepted_movements_chemical_and_two_proxy_checks=True,
        previous_full_reference_distance_reduction=float(original_final_diagnostics.target_distance_reduction.mean()),
        evidence_scope="Model-based exploratory optimization on fitted data; no causal chemical response or new physical beer experiment.")
    (args.out / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
