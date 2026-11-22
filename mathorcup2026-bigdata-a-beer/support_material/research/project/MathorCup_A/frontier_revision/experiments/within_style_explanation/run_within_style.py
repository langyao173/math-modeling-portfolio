"""Reproducible style-confounding sensitivity, nested CV, and 400 bootstraps.

The source attachments and prior Q2 package are read only. One CPU BLAS thread.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
from scipy.linalg import solve
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits, threadpool_info
from within_style_model import (
    ALPHAS, prepare, ridge_fit, predict, choose_alphas, lookup_centers,
    stratified_bootstrap_indices, top_indices,
)

HERE = Path(__file__).resolve().parent
UPGRADE = HERE.parents[1]
from project_location import PROJECT
SEED = 20261024


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jdump(value, path):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda v:v.item() if isinstance(v, np.generic) else str(v)), encoding="utf-8")


def progress(**values):
    print(json.dumps(values, ensure_ascii=False, default=lambda v:float(v)), flush=True)


def covariance_evidence(prep):
    u, v, ur, vr = [prep[k] for k in ("u", "v", "ur", "vr")]
    total = u.T@v/len(u)
    within = ur.T@vr/len(u)
    between = prep["u_style"].T@prep["v_style"]/len(u)
    den = np.sqrt(np.mean(ur**2, axis=0)[:, None]*np.mean(vr**2, axis=0)[None, :])
    correlation_within = np.divide(within, den, out=np.zeros_like(within), where=den > 1e-14)
    eta_x = np.mean(prep["u_style"]**2, axis=0)/np.mean(u**2, axis=0)
    eta_y = np.mean(prep["v_style"]**2, axis=0)/np.mean(v**2, axis=0)
    return dict(correlation_total=total, correlation_within=correlation_within,
        covariance_total=total, covariance_within=within, covariance_between=between,
        style_variance_share_x=eta_x, style_variance_share_y=eta_y,
        decomposition_max_error=float(np.max(np.abs(total-within-between))))


def metric_rows(y, prediction, targets, model, baseline=None):
    rows = []
    for j, target in enumerate(targets):
        residual = y[:, j]-prediction[:, j]
        row = dict(model=model, target=target, R2=float(r2_score(y[:, j], prediction[:, j])),
            RMSE=float(np.sqrt(np.mean(residual**2))), MAE=float(np.mean(np.abs(residual))),
            Spearman=float(spearmanr(y[:, j], prediction[:, j]).statistic))
        if baseline is not None:
            denominator = float(np.sum((y[:, j]-baseline[:, j])**2))
            row["incremental_R2_over_style_mean"] = float(1-np.sum(residual**2)/denominator)
        rows.append(row)
    return rows


def bootstrap_metrics(y, predictions, styles, targets, replicates):
    pairs = [
        ("within_rank", "style_mean_rank"),
        ("within_rank", "style_mean_raw"),
        ("within_rank", "total_rank"),
        ("within_linear", "style_mean_raw"),
    ]
    var = np.var(y, axis=0)
    loss_gain = np.stack([((y-predictions[b])**2-(y-predictions[a])**2)/var for a,b in pairs])
    draws = np.empty((replicates, len(pairs), len(targets)))
    rng = np.random.default_rng(SEED+80000)
    for b in range(replicates):
        idx = stratified_bootstrap_indices(styles, rng)
        draws[b] = loss_gain[:, idx].mean(axis=1)
    rows = []
    for k, (model, baseline) in enumerate(pairs):
        for j, target in enumerate(list(targets)+["__macro__"]):
            point = loss_gain[k, :, j].mean() if j < len(targets) else loss_gain[k].mean()
            samples = draws[:, k, j] if j < len(targets) else draws[:, k].mean(axis=1)
            ci = np.quantile(samples, [.025, .975])
            rows.append(dict(model=model, baseline=baseline, target=target,
                delta_R2=float(point), paired_bootstrap_025=float(ci[0]),
                paired_bootstrap_975=float(ci[1]), bootstrap_positive_frequency=float(np.mean(samples > 0)),
                replicates=replicates))
    pd.DataFrame(rows).to_csv(HERE / "oof_paired_differences.csv", index=False)


def independent_fwl_check(prep, styles, alpha):
    # Independent block normal equations, with unpenalized style dummies.
    categories, code = np.unique(styles, return_inverse=True)
    d = np.eye(len(categories))[code]
    u, v = prep["u"], prep["v"]
    joint = np.column_stack([d, u])
    penalty = np.r_[np.zeros(d.shape[1]), np.full(u.shape[1], len(u)*alpha)]
    theta = solve(joint.T@joint+np.diag(penalty), joint.T@v, assume_a="sym")
    b = ridge_fit(prep, True, alpha)
    predicted = prep["v_style"]+prep["ur"]@b
    return dict(
        residualized_vs_joint_block_coefficient_max_error=float(np.max(np.abs(theta[d.shape[1]:]-b))),
        residualized_vs_joint_block_prediction_max_error=float(np.max(np.abs(joint@theta-predicted))),
        max_within_style_x_mean=float(max(np.max(np.abs(prep["ur"][styles == s].mean(axis=0))) for s in categories)),
        max_within_style_y_mean=float(max(np.max(np.abs(prep["vr"][styles == s].mean(axis=0))) for s in categories)),
    )


def main(args):
    start = time.perf_counter()
    HERE.mkdir(parents=True, exist_ok=True)
    files = dict(chemistry=PROJECT / "data/A/1.csv", sensory=PROJECT / "data/A/4.csv",
        frozen_folds=PROJECT / "A_work/q1/fold_assignments.csv",
        old_top3=PROJECT / "A_work/q2/top3_all_50_sensory_dimensions.csv",
        chemical_dictionary=PROJECT / "A_work/q2/chemical_dictionary.csv",
        graph_nodes=PROJECT / "A_work/q2/graph_nodes.csv")
    source_hash = {k:sha(v) for k,v in files.items()}
    xd = pd.read_csv(files["chemistry"]).sort_values("beer").reset_index(drop=True)
    yd = pd.read_csv(files["sensory"]).set_index("beer").loc[xd.beer].reset_index()
    assert xd.beer.is_unique and yd.beer.is_unique and len(xd) == len(yd) == 250
    assert np.array_equal(xd.tasting_category_fine, yd.tasting_category_fine)
    features, targets = xd.columns[3:].to_list(), yd.columns[3:].to_list()
    x, y = xd[features].to_numpy(float), yd[targets].to_numpy(float)
    styles, ids = xd.tasting_category_fine.to_numpy(), xd.beer.to_numpy(int)
    assert np.isfinite(y).all()
    fold = pd.read_csv(files["frozen_folds"]).set_index("beer").loc[ids].fold.to_numpy(int)
    dictionary = pd.read_csv(files["chemical_dictionary"]).set_index("field").loc[features]
    eligible = np.flatnonzero(dictionary.single_compound_candidate.to_numpy(bool))
    assert len(features) == 231 and len(targets) == 50 and len(eligible) == 218
    nodes = pd.read_csv(files["graph_nodes"]).set_index("field").loc[features]
    old = pd.read_csv(files["old_top3"])
    pd.DataFrame(dict(beer=ids, style=styles, fold=fold)).to_csv(HERE / "fold_assignments.csv", index=False)
    pd.Series(styles).value_counts().rename_axis("style").rename("n").to_csv(HERE / "style_counts.csv")
    protocol = dict(
        question="Which total chemical–sensory associations persist after controlling observed beer style?",
        source_sha256=source_hash, n_samples=len(x), n_predictors=len(features), n_targets=len(targets),
        candidate_compounds=len(eligible), features=features, targets=targets,
        feature_policy="All 231 prior-Q2 predictors retained in both models; candidate list excludes the same 13 non-single-compound/internal-standard fields.",
        primary_transform="Training-fitted median imputation, midpoint normal-score maps and global training standardization for both X and Y.",
        primary_model="Unpenalized style fixed effects plus ridge on within-style centered normal scores; fitted style means only from training rows.",
        objective="mean squared residual normal-score loss + alpha * Frobenius norm squared of coefficient matrix",
        primary_comparator="Same transformations and ridge penalty family, without style centering.",
        sensitivity="Standardized original Y instead of rank-normal Y, with the same outer folds and independent inner alpha selection.",
        baselines=["training style mean in original Y", "inverse-transformed training style mean in rank-normal Y", "global training mean in original Y"],
        outer_folds="Frozen existing Q1/Q2 five folds, reused unchanged; every outer training set contains all 22 styles.",
        inner_folds="Three style-balanced folds of unique beer IDs; bootstrap copies never cross a training/validation split.",
        unknown_style_rule="Training grand means; only relevant to rare singleton-style bootstrap inner validation.",
        alpha_grid=ALPHAS.tolist(), alpha_selection="One alpha per multi-output model; minimum inner mean squared normalized-target error.",
        bootstrap_replicates=args.bootstrap, bootstrap_scheme="Sample beers with replacement within each of 22 styles, preserving each style count.",
        bootstrap_refit="Every replicate refits imputation, X/Y rank maps, style means, and ridge; alpha reselected using grouped inner CV.",
        bootstrap_target="Descriptive stability conditional on the observed style mix, including hyperparameter selection; not a causal effect interval or multiplicity-adjusted significance test.",
        top3_policy="Per replicate top 3 absolute coefficients among the same 218 chemical candidates. Final Top-3 by inclusion frequency, ties by median absolute bootstrap coefficient.",
        paired_oof_bootstrap=args.paired_bootstrap,
        oof_uncertainty="Paired within-style resampling of frozen OOF per-beer losses with original-Y global variance denominator; conditional on the five fitted fold models.",
        seed=SEED, cpu_threads=1, style_input_scope="Known style is used only for this Q2 sensitivity, never relabeled as chemical-only Q1 performance.",
        code_sha256={p.name:sha(p) for p in [HERE / "within_style_model.py", Path(__file__)]},
    )
    jdump(protocol, HERE / "protocol.json")

    names = ["total_rank", "within_rank", "total_linear", "within_linear", "style_mean_rank", "style_mean_raw", "global_mean"]
    oof = {name:np.zeros_like(y) for name in names}
    params, searches, fold_metrics, checks = [], [], [], []
    for k in np.unique(fold):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        assert not (set(ids[tr]) & set(ids[te]))
        assert set(styles[tr]) == set(styles)
        for mode in ["rank", "linear"]:
            chosen, losses, validation = choose_alphas(x[tr], y[tr], styles[tr], ids[tr], seed=SEED+100+int(k), target_mode=mode)
            prep = prepare(x[tr], y[tr], styles[tr], target_mode=mode)
            for b, within in enumerate([False, True]):
                label = ("within_" if within else "total_")+mode
                alpha = chosen["within" if within else "total"]
                coefficient = ridge_fit(prep, within, alpha)
                oof[label][te], _ = predict(prep, coefficient, x[te], styles[te], within)
                params.append(dict(fold=int(k), model=label, alpha=alpha, **validation))
                for a, loss in zip(ALPHAS, losses[b]):
                    searches.append(dict(fold=str(k), model=label, alpha=float(a), inner_MSE=float(loss)))
            if mode == "rank":
                offset = lookup_centers(prep["vc"], styles[te], prep["vm"])
                oof["style_mean_rank"][te] = prep["ymap"].inverse_transform(offset)
                oof["style_mean_raw"][te] = lookup_centers(prep["raw_y_centers"], styles[te], prep["raw_y_mean"])
                oof["global_mean"][te] = prep["raw_y_mean"]
        for name in names:
            rr = metric_rows(y[te], oof[name][te], targets, name, oof["style_mean_raw"][te])
            for row in rr:
                row["fold"] = int(k)
            fold_metrics += rr
        progress(stage="outer_fold", fold=int(k), elapsed_seconds=time.perf_counter()-start)

    metrics = []
    for name in names:
        prediction = oof[name]
        frame = pd.DataFrame(prediction, columns=targets)
        frame.insert(0, "fold", fold); frame.insert(0, "beer", ids)
        frame.to_csv(HERE / f"oof_{name}.csv", index=False)
        metrics += metric_rows(y, prediction, targets, name, oof["style_mean_raw"])
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(HERE / "prediction_metrics.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(HERE / "fold_metrics.csv", index=False)
    summary = metrics.groupby("model")[["R2", "RMSE", "Spearman", "incremental_R2_over_style_mean"]].mean()
    summary.to_csv(HERE / "prediction_summary.csv")
    bootstrap_metrics(y, oof, styles, targets, args.paired_bootstrap)
    progress(stage="prediction_complete", metrics=summary.R2.to_dict(), elapsed_seconds=time.perf_counter()-start)

    full, coefficients, selected = {}, {}, {}
    for mode in ["rank", "linear"]:
        chosen, losses, validation = choose_alphas(x, y, styles, ids, seed=SEED+1000, target_mode=mode)
        full[mode] = prepare(x, y, styles, mode)
        for b, within in enumerate([False, True]):
            label = ("within_" if within else "total_")+mode
            alpha = chosen["within" if within else "total"]
            selected[label] = alpha
            coefficients[label] = ridge_fit(full[mode], within, alpha)
            params.append(dict(fold="full", model=label, alpha=alpha, **validation))
            for a, loss in zip(ALPHAS, losses[b]):
                searches.append(dict(fold="full", model=label, alpha=float(a), inner_MSE=float(loss)))
            pd.DataFrame(coefficients[label], index=features, columns=targets).rename_axis("field").to_csv(HERE / f"coefficients_{label}.csv")
    pd.DataFrame(params).to_csv(HERE / "nested_cv_parameters.csv", index=False)
    pd.DataFrame(searches).to_csv(HERE / "nested_cv_search.csv", index=False)
    cov = covariance_evidence(full["rank"])
    check = independent_fwl_check(full["rank"], styles, selected["within_rank"])
    check.update(covariance_decomposition_max_error=cov["decomposition_max_error"], all_outer_styles_seen=True,
        outer_unique_ids=len(np.unique(ids)), immutable_source_hashes_match=all(sha(v) == source_hash[k] for k,v in files.items()))
    assert check["residualized_vs_joint_block_coefficient_max_error"] < 1e-9
    assert check["residualized_vs_joint_block_prediction_max_error"] < 1e-9
    assert cov["decomposition_max_error"] < 1e-12
    jdump(check, HERE / "independent_checks.json")

    # Jointly resampled total/within models: differences share identical draws.
    p, m = x.shape[1], y.shape[1]
    bootstrap = np.zeros((2, args.bootstrap, p, m), dtype=np.float32)
    btop = np.zeros((2, args.bootstrap, 3, m), dtype=np.int32)
    bindices = np.zeros((args.bootstrap, len(x)), dtype=np.int32)
    bootstrap_rows = []
    checkpoint = HERE / "bootstrap_checkpoint.npz"
    completed = 0
    if checkpoint.exists():
        cp = np.load(checkpoint, allow_pickle=False)
        if str(cp["protocol_code_hash"]) == protocol["code_sha256"][Path(__file__).name]:
            completed = min(int(cp["completed"]), args.bootstrap)
            bootstrap[:, :completed] = cp["coefficient"][:, :completed]
            btop[:, :completed] = cp["top3"][:, :completed]
            bindices[:completed] = cp["sample_indices"][:completed]
            if (HERE / "bootstrap_parameters.csv").exists():
                bootstrap_rows = pd.read_csv(HERE / "bootstrap_parameters.csv").to_dict("records")[:2*completed]
    progress(stage="bootstrap_start", completed=completed, requested=args.bootstrap, full_alpha=selected)
    for b in range(completed, args.bootstrap):
        ix = stratified_bootstrap_indices(styles, np.random.default_rng(SEED+5000+b))
        bindices[b] = ix
        chosen, losses, validation = choose_alphas(x[ix], y[ix], styles[ix], ids[ix], seed=SEED+10000+b, target_mode="rank")
        prep = prepare(x[ix], y[ix], styles[ix], "rank")
        for model, within in enumerate([False, True]):
            label = "within_rank" if within else "total_rank"
            alpha = chosen["within" if within else "total"]
            coef = ridge_fit(prep, within, alpha)
            bootstrap[model, b] = coef
            btop[model, b] = top_indices(coef, eligible)
            bootstrap_rows.append(dict(replicate=b, model=label, alpha=alpha,
                minimum_inner_MSE=float(np.min(losses[model])), **validation))
        if (b+1) % 20 == 0 or b+1 == args.bootstrap:
            pd.DataFrame(bootstrap_rows).to_csv(HERE / "bootstrap_parameters.csv", index=False)
            np.savez_compressed(checkpoint, completed=b+1,
                protocol_code_hash=protocol["code_sha256"][Path(__file__).name],
                coefficient=bootstrap[:, :b+1], top3=btop[:, :b+1], sample_indices=bindices[:b+1])
            progress(stage="bootstrap", completed=b+1, requested=args.bootstrap, elapsed_seconds=time.perf_counter()-start)
    np.savez_compressed(HERE / "bootstrap_coefficients.npz", coefficient=bootstrap, top3=btop,
        sample_indices=bindices, features=np.array(features), targets=np.array(targets),
        models=np.array(["total_rank", "within_rank"]))

    evidence = pd.DataFrame(dict(target=np.repeat(targets, p), field=np.tile(features, m)))
    for col in ["graph_name", "origin", "single_compound_candidate", "unit"]:
        evidence[col] = evidence.field.map(dictionary[col])
    evidence["graph_module"] = evidence.field.map(nodes.module)
    for name, value in coefficients.items():
        evidence["coefficient_"+name] = value.ravel(order="F")
    for name in ["correlation_total", "correlation_within", "covariance_total", "covariance_within", "covariance_between"]:
        evidence[name] = cov[name].ravel(order="F")
    evidence["style_variance_share_x"] = np.tile(cov["style_variance_share_x"], m)
    evidence["style_variance_share_y"] = np.repeat(cov["style_variance_share_y"], p)
    evidence["total_coefficient_at_within_alpha"] = ridge_fit(full["rank"], False, selected["within_rank"]).ravel(order="F")
    evidence["within_coefficient_at_total_alpha"] = ridge_fit(full["rank"], True, selected["total_rank"]).ravel(order="F")
    frequency, top_tables = [], {}
    for model, label in enumerate(["total", "within"]):
        coef = bootstrap[model].astype(float)
        freq = np.zeros((p, m))
        for b in range(args.bootstrap):
            np.add.at(freq, (btop[model, b].ravel(), np.tile(np.arange(m), 3)), 1.)
        freq /= args.bootstrap
        frequency.append(freq)
        sign = np.mean(np.sign(coef) == np.sign(coefficients[label+"_rank"])[None], axis=0)
        lo, hi = np.quantile(coef, [.025, .975], axis=0)
        median_abs = np.median(np.abs(coef), axis=0)
        for name, value in dict(top3_frequency=freq, sign_agreement=sign, coefficient_025=lo,
                                coefficient_975=hi, median_absolute_coefficient=median_abs).items():
            evidence[label+"_"+name] = value.ravel(order="F")
        within_rank = np.full((p, m), np.nan)
        top_rows = []
        for j, target in enumerate(targets):
            order = np.lexsort((-median_abs[eligible, j], -freq[eligible, j]))
            ordered = eligible[order]
            within_rank[ordered, j] = np.arange(1, len(ordered)+1)
            for r, index in enumerate(ordered[:3]):
                top_rows.append(dict(target=target, field=features[index], rank=r+1))
        evidence[label+"_stability_rank"] = within_rank.ravel(order="F")
        top_tables[label] = pd.DataFrame(top_rows)
    evidence["same_sign_total_within"] = np.sign(evidence.coefficient_total_rank) == np.sign(evidence.coefficient_within_rank)
    evidence["within_bootstrap_interval_excludes_zero"] = (evidence.within_coefficient_025 > 0) | (evidence.within_coefficient_975 < 0)
    evidence["within_top3_frequency_drop"] = evidence.total_top3_frequency-evidence.within_top3_frequency
    evidence["strong_total_weak_within_correlation"] = (evidence.correlation_total.abs() >= .3) & (evidence.correlation_within.abs() < .1)
    evidence.to_csv(HERE / "full_data_pair_evidence.csv", index=False)
    for label, table in top_tables.items():
        table = table.merge(evidence, on=["target", "field"], validate="one_to_one")
        table.to_csv(HERE / f"top3_{label}.csv", index=False)
    old_columns = [c for c in old if c not in ["target", "field"]]
    aligned = old.rename(columns={c:"old_"+c for c in old_columns}).merge(evidence,
        on=["target", "field"], how="left", validate="one_to_one")
    assert len(aligned) == 150 and aligned.within_top3_frequency.notna().all()
    aligned.to_csv(HERE / "old_top3_alignment.csv", index=False)

    module_rows = []
    module_ids = nodes.module.to_numpy(int)
    for model, label in enumerate(["total", "within"]):
        top_module = module_ids[btop[model]]
        for module in np.unique(module_ids):
            probability = np.any(top_module == module, axis=1).mean(axis=0)
            for j, target in enumerate(targets):
                module_rows.append(dict(model=label, graph_module=int(module), target=target,
                    any_member_top3_frequency=float(probability[j]),
                    eligible_member_count=int(np.sum(module_ids[eligible] == module))))
    pd.DataFrame(module_rows).to_csv(HERE / "module_stability.csv", index=False)

    focused_targets = [t for t in targets if t.startswith(("A_malt", "F_malt")) or
        t in ["acidity", "lactic", "acetic", "alcohol", "body", "X4vg", "diacetyl", "dms", "stale_hops", "t2n"]]
    focus_keys = set(zip(aligned[aligned.target.isin(focused_targets)].target, aligned[aligned.target.isin(focused_targets)].field))
    for table in top_tables.values():
        focus_keys |= set(zip(table[table.target.isin(focused_targets)].target, table[table.target.isin(focused_targets)].field))
    anchors = {"lactic":"lactic_acid.mg.L.", "acidity":"lactic_acid.mg.L.", "acetic":"acetic_acid.g.L.",
        "alcohol":"ethanol..v.v.", "X4vg":"X4VG", "diacetyl":"X2.3.butanedione", "dms":"DMS", "t2n":"X.E..2.nonenal"}
    focus_keys |= set(anchors.items())
    keymask = [(t, f) in focus_keys for t,f in zip(evidence.target, evidence.field)]
    evidence.loc[keymask].to_csv(HERE / "focus_evidence.csv", index=False)
    evidence.loc[[(t,f) in set(anchors.items()) for t,f in zip(evidence.target,evidence.field)]].to_csv(HERE / "chemical_anchor_evidence.csv", index=False)

    counts = dict(
        old_top3_entries=len(aligned),
        old_entries_still_in_within_stability_top3=int((aligned.within_stability_rank <= 3).sum()),
        old_entries_within_frequency_at_least_half=int((aligned.within_top3_frequency >= .5).sum()),
        old_entries_within_stable_direction=int((aligned.within_sign_agreement >= .95).sum()),
        old_entries_within_interval_excludes_zero=int(aligned.within_bootstrap_interval_excludes_zero.sum()),
        old_entries_strong_total_weak_within_correlation=int(aligned.strong_total_weak_within_correlation.sum()),
        full_alpha=selected,
        nested_mean_R2=summary.R2.to_dict(),
        coefficient_bootstrap_replicates=args.bootstrap, paired_oof_bootstrap_replicates=args.paired_bootstrap,
        wall_seconds=time.perf_counter()-start,
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
        threadpools=threadpool_info(),
    )
    assert all(sha(v) == source_hash[k] for k,v in files.items()), "A source file changed during analysis"
    jdump(counts, HERE / "summary.json")
    progress(stage="complete", **{k:v for k,v in counts.items() if k != "threadpools"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--paired-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args)
