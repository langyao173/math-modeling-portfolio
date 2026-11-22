"""Independent read-only numerical checks of saved generation refinements.

Does not fit or resample any beer generator. Small synthetic arrays verify the
conditional-Gaussian and full-second-moment identities independently.
"""
import os
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from sklearn.model_selection import StratifiedKFold
import joblib

HERE = Path(__file__).resolve().parent
UPGRADE = HERE.parent
EXP = UPGRADE / "experiments/generation_moments"
sys.path.insert(0, str(EXP))
import conditional_moments  # Needed only to deserialize parent-generated models.
from control_problem import PROJECT


def main():
    rng = np.random.default_rng(20261024)
    p = 5
    a = rng.normal(size=(p, p))
    covariance = a@a.T + .7*np.eye(p)
    precision = np.linalg.inv(covariance)
    mean, z = rng.normal(size=p), rng.normal(size=p)
    residual = precision@(z-mean)
    mean_errors, variance_errors, residual_errors = [], [], []
    for j in rng.permutation(p):
        keep = np.arange(p) != j
        regression = np.linalg.solve(covariance[np.ix_(keep, keep)], covariance[keep, j])
        conditional_mean = mean[j] + regression@(z[keep]-mean[keep])
        conditional_variance = covariance[j, j]-covariance[j, keep]@regression
        mean_errors.append(abs(conditional_mean-(z[j]-residual[j]/precision[j, j])))
        variance_errors.append(abs(conditional_variance-1/precision[j, j]))
        new = float(rng.normal())
        delta = new-z[j]
        z[j] = new
        residual = residual+delta*precision[:, j]
        residual_errors.append(float(np.max(abs(residual-precision@(z-mean)))))
    draws = rng.normal(size=(9, 7, p)) @ np.linalg.cholesky(covariance).T
    posterior_mean = draws.mean(axis=0)
    center = posterior_mean.mean(axis=0)
    second_sum = np.einsum("rip,riq->pq", draws, draws)/len(draws)
    scatter = second_sum-draws.shape[1]*np.outer(center, center)
    centered = draws-center
    direct = np.einsum("rip,riq->pq", centered, centered)/len(draws)
    plugin = (posterior_mean-center).T@(posterior_mean-center)
    plugin_difference = scatter-plugin
    shrunk = .8*center
    target = draws-shrunk
    actual_mean_scatter = np.einsum("rip,riq->pq", target, target)/len(draws)
    correction = draws.shape[1]*np.outer(center-shrunk, center-shrunk)
    algebra = dict(
        conditional_mean_max_error=float(max(mean_errors)),
        conditional_variance_max_error=float(max(variance_errors)),
        precision_residual_update_max_error=float(max(residual_errors)),
        full_second_moment_scatter_max_error=float(np.max(abs(scatter-direct))),
        full_minus_plugin_min_eigenvalue=float(np.linalg.eigvalsh(plugin_difference)[0]),
        full_minus_plugin_off_diagonal_norm=float(np.linalg.norm(plugin_difference-np.diag(np.diag(plugin_difference)))),
        shrunken_mean_scatter_correction_max_error=float(np.max(abs(actual_mean_scatter-scatter-correction))),
    )
    data = pd.read_csv(PROJECT / "data/A/1.csv")
    derived = ["kcalperc", "esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum", "X3.methyl.2.butenal"]
    features = [c for c in data.columns[3:] if c not in derived]
    x, styles = data[features].to_numpy(float), data.tasting_category_fine.to_numpy(str)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=20261023+300).split(x, styles))
    cache_rows, mass_rows, anchor_rows = [], [], []
    for fold, (tr, va) in enumerate(folds):
        for name in ["interval_reference", "conditional_moments", "conditional_moments_missing"]:
            model = joblib.load(EXP / f"fold_{fold}_{name}.joblib")
            missing_aware = name.endswith("_missing")
            mu = np.stack([model.mean_[s] for s in model.styles_])
            centers = np.stack([model.center_[s] for s in model.styles_])
            ns = np.array([model.n_[s] for s in model.styles_])
            grand = np.average(centers, axis=0, weights=ns)
            expected_means = (ns[:, None]*centers+model.kappa*grand)/(ns[:, None]+model.kappa)
            covariance = np.stack([model.cov_[s] for s in model.styles_])
            sd = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
            total_tv, cond_tv, interval_errors = [], [], []
            all_values_exact = True
            n_observed = []
            for j, (values, low, high) in enumerate(model.intervals_):
                raw = x[tr, j]
                observed = np.isfinite(raw)
                n_observed.append(int(observed.sum()))
                fitted = raw[observed] if missing_aware else np.where(observed, raw, model.map_.medians_[j])
                vv, cc = np.unique(fitted, return_counts=True)
                all_values_exact = all_values_exact and np.array_equal(values, vv)
                empirical = cc/len(fitted)
                a, b = ndtri(low), ndtri(high)
                mass = ndtr((b[None, :]-mu[:, j, None])/sd[:, j, None])-ndtr((a[None, :]-mu[:, j, None])/sd[:, j, None])
                mixture = np.average(mass, axis=0, weights=ns)
                total_tv.append(.5*np.sum(np.abs(mixture-empirical)))
                ctv = []
                for k, style in enumerate(model.styles_):
                    group = styles[tr] == style
                    sample = raw[group & observed] if missing_aware else np.where(observed, raw, model.map_.medians_[j])[group]
                    counts = np.bincount(np.searchsorted(values, sample), minlength=len(values))
                    ctv.append(.5*np.abs(mass[k]-counts/len(sample)).sum() if len(sample) else np.nan)
                ctv = np.asarray(ctv)
                cond_tv.append(float(np.average(ctv[np.isfinite(ctv)], weights=ns[np.isfinite(ctv)])))
                ix = np.searchsorted(values, raw[observed])
                latent = model.latent_[observed, j]
                interval_errors.extend([float(np.max(np.maximum(a[ix]-latent, 0))), float(np.max(np.maximum(latent-b[ix], 0)))])
                if features[j] in ["X4VG", "DMS", "X2.3.butanedione"]:
                    anchor_rows.append(dict(fold=fold, model=name, field=features[j],
                        minimum_recorded_value=float(values[0]), empirical_minimum_mass=float(empirical[0]),
                        analytic_generated_minimum_mass=float(mixture[0]),
                        global_training_marginal_total_variation=float(total_tv[-1]),
                        training_n_observed=int(observed.sum())))
            cache_rows.append(dict(fold=fold, model=name, n_training=len(tr), n_validation=len(va),
                n_styles=len(model.styles_), n_features=len(features),
                source_training_values_match=all_values_exact,
                mean_shrinkage_identity_max_error=float(np.max(abs(mu-expected_means))),
                covariance_symmetry_max_error=float(np.max(abs(covariance-covariance.transpose(0, 2, 1)))),
                minimum_covariance_eigenvalue=float(min(np.linalg.eigvalsh(c)[0] for c in covariance)),
                posterior_mean_interval_max_violation=float(max(interval_errors)),
                minimum_observed_entries_in_column=int(min(n_observed)),
                iterations=getattr(model, "iterations", 0), burn_sweeps=getattr(model, "burn_sweeps", 0),
                retained_sweeps=getattr(model, "retained_sweeps", 0),
                missing_intervals=getattr(model, "missing_intervals", False),
                seed=getattr(model, "seed", None),
                kappa=model.kappa, nu=model.nu, pooled_shrinkage=float(model.pooled_shrinkage_)))
            mass_rows.append(dict(fold=fold, model=name,
                mean_global_training_marginal_total_variation=float(np.mean(total_tv)),
                maximum_global_training_marginal_total_variation=float(np.max(total_tv)),
                mean_style_conditional_training_marginal_total_variation=float(np.mean(cond_tv)),
                scope="Analytic probabilities from saved Gaussian parameters; in-sample identity audit only, not validation performance"))
    cache = pd.DataFrame(cache_rows)
    cache.to_csv(HERE / "generation_saved_model_checks.csv", index=False)
    pd.DataFrame(mass_rows).to_csv(HERE / "generation_atom_probability_checks.csv", index=False)
    pd.DataFrame(anchor_rows).to_csv(HERE / "generation_anchor_atoms.csv", index=False)
    outer = pd.read_csv(EXP / "outer_metrics.csv")
    old = pd.read_csv(PROJECT / "A_work/q3/extension_outer_comparison.csv")
    reference_errors = []
    for row in outer[outer.model == "interval_reference"].itertuples():
        ref = old[(old.model == "interval_gaussian") & (old.outer_fold == row.fold)].iloc[0]
        reference_errors.extend(abs(getattr(row, col)-ref[col]) for col in ["energy", "variogram", "c2st_auc", "nn_ratio"])
    scores = pd.read_csv(EXP / "scores_by_beer.csv")
    idcheck = {name:bool(len(group) == 250 and group.beer.nunique() == 250) for name, group in scores.groupby("model")}
    errors = []
    summary = pd.read_csv(EXP / "summary.csv").set_index("model")
    for name, group in scores.groupby("model"):
        errors.append(abs(group.energy.mean()-summary.loc[name, "beer_weighted_energy"]))
    trace = pd.concat([pd.read_csv(f).assign(source=f.name) for f in EXP.glob("*_trace.csv")], ignore_index=True)
    report = dict(review_scope="Read-only review of source, frozen outputs, and independent algebra; no generator was refitted.",
        algebra=algebra,
        source_code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [EXP/"conditional_moments.py", EXP/"run_refinement.py"]},
        source_data_sha256=hashlib.sha256((PROJECT/"data/A/1.csv").read_bytes()).hexdigest(),
        reference_reproduction_max_error=float(max(reference_errors)),
        each_model_has_all_250_unique_validation_beers=idcheck,
        weighted_energy_recalculation_max_error=float(max(errors)),
        maximum_posterior_mean_interval_violation=float(cache.posterior_mean_interval_max_violation.max()),
        minimum_saved_covariance_eigenvalue=float(cache.minimum_covariance_eigenvalue.min()),
        maximum_logged_last_sample_interval_violation=float(trace.maximum_interval_violation.max()),
        iteration8_relative_covariance_change=dict(zip(trace[trace.iteration == 8].source, trace[trace.iteration == 8].relative_pooled_change)),
        supported_claim="Finite-MCMC regularized conditional first/full-second-moment refinement with observed-value support.",
        unsupported_claims=["Exact or converged EM/maximum likelihood", "Exact conditional empirical point-mass probabilities", "New independent test", "Overall generation improvement from these results", "Monte Carlo draws increase number of observed beers"],
    )
    (HERE / "generation_moments_independent_review.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(pd.DataFrame(mass_rows).to_string(index=False))


if __name__ == "__main__":
    main()
