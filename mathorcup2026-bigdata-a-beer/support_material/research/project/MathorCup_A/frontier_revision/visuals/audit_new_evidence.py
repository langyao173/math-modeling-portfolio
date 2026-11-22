#!/usr/bin/env python3
"""Independent algebraic checks of plotted semantic and geometry evidence.

This script imports no experiment solver or scoring function. It recomputes
R² and paired bootstrap intervals directly from saved OOF arrays, and checks
convex primal/dual and ellipsoid support values from the published witnesses.
It does not turn floating-point certificates into exact-arithmetic proofs.
"""
from pathlib import Path
import hashlib
import json
import os
for _thread_option in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_thread_option]="1"
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
UP = HERE.parent
BASE = UP / "project/MathorCup_A"
if not (BASE / "data/A/1.csv").is_file():
    BASE = next((parent for parent in HERE.parents
                 if (parent / "data/A/1.csv").is_file()),None)
    if BASE is None:
        raise FileNotFoundError('Locate this audit in the research package with data/A/1.csv in an ancestor directory.')
SEM = UP / "experiments/semantic_auxiliary"
GEO = UP / "experiments/sparse_transfer"
WITHIN = UP / "experiments/within_style_explanation"
sources = {}


def record(path):
    path = Path(path)
    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def semantic_check():
    branch = np.load(record(SEM / "oof_predictions.npz"))
    blend = np.load(record(SEM / "blend_oof.npz"))
    old = np.load(record(BASE / "A_work/q1/oof_arrays.npz"))
    old_folds = pd.read_csv(record(BASE / "A_work/q1/fold_assignments.csv"))
    now_folds = pd.read_csv(record(SEM / "fold_assignments.csv"))
    np.testing.assert_array_equal(old_folds.to_numpy(), now_folds.to_numpy())
    np.testing.assert_array_equal(branch["fold"], old_folds.fold.to_numpy())
    np.testing.assert_array_equal(branch["beer"], old_folds.beer.to_numpy())
    np.testing.assert_array_equal(blend["fold"], branch["fold"])
    np.testing.assert_allclose(branch["Y"], old["Y"], atol=0, rtol=0)
    np.testing.assert_allclose(blend["Y"], old["Y"], atol=0, rtol=0)
    raw_y = pd.read_csv(record(BASE / "data/A/4.csv")).set_index("beer").loc[branch["beer"]]
    targets = raw_y.columns[2:]
    core = np.flatnonzero([str(t).startswith(("A_", "F_")) for t in targets])
    assert len(core) == 30
    np.testing.assert_allclose(raw_y[targets].to_numpy(float), old["Y"], atol=0, rtol=0)
    y = old["Y"]
    denom = np.sum((y-y.mean(0))**2, axis=0)

    def score(pred):
        r = 1 - np.sum((y-pred)**2, axis=0)/denom
        return np.array([r.mean(), r[core].mean()])

    baseline = old["nested_ensemble"]
    baseline_r2 = score(baseline)
    # A weighted sum form, algebraically equivalent to sampling beer rows,
    # permits a separate implementation of the paired bootstrap statistic.
    sampled = np.random.default_rng(10041).integers(0, len(y), (2000, len(y)))
    weights = np.zeros((len(sampled), len(y)), dtype=float)
    np.add.at(weights, (np.arange(len(sampled))[:,None], sampled), 1)
    weighted_total = weights @ y
    bootstrap_den = weights @ (y*y) - weighted_total*weighted_total / len(y)
    baseline_sse = (y-baseline)**2
    rows = []
    max_score_error = 0.
    max_ci_error = 0.
    for stage, arrays, filename in [("RRR", branch, "model_comparison.csv"),
                                    ("blend", blend, "blend_comparison.csv")]:
        table = pd.read_csv(record(SEM / filename))
        for row in table.itertuples():
            key = row.model.removeprefix("nested_blend_")
            pred = arrays[key]
            r2 = score(pred)
            stated = np.array([row.all50_r2, row.core30_r2])
            error = float(np.max(abs(r2-stated)))
            assert error < 1e-12
            max_score_error = max(max_score_error, error)
            item = dict(stage=stage, model=row.model,
                        all50_r2=float(r2[0]), core30_r2=float(r2[1]),
                        max_reported_r2_error=error)
            if stage == "blend":
                per_target_delta = (weights @ (baseline_sse-(y-pred)**2)) / np.maximum(bootstrap_den, 1e-20)
                ci50 = np.quantile(per_target_delta.mean(1), [.025,.975])
                ci30 = np.quantile(per_target_delta[:,core].mean(1), [.025,.975])
                err50 = np.max(abs(ci50-np.array(json.loads(row.ci_vs_frozen_all50))))
                err30 = np.max(abs(ci30-np.array(json.loads(row.ci_vs_frozen_core30))))
                max_ci_error = max(max_ci_error, float(err50), float(err30))
                assert max(err50,err30) < 1e-12
                item.update(delta_all50=float(r2[0]-baseline_r2[0]),
                            delta_core30=float(r2[1]-baseline_r2[1]),
                            ci95_all50=ci50.tolist(), ci95_core30=ci30.tolist())
            rows.append(item)
    return dict(same_outer_folds=True, same_targets=True, n=250, target_dimensions=50,
                core_target_dimensions=len(core), baseline_r2=baseline_r2.tolist(),
                bootstrap_replicates=2000, bootstrap_seed=10041,
                bootstrap_scope="paired beer resampling of fixed OOF predictions; no model refitting",
                max_score_error=max_score_error, max_ci_error=max_ci_error, rows=rows)


def geometry_check():
    table = pd.read_csv(record(GEO / "classification_geometry_certificates.csv")).set_index("beer")
    payload = json.loads(record(GEO / "classification_geometry_witnesses.json").read_text())
    rows = []
    for beer, witness in payload.items():
        row = table.loc[int(beer)]
        P = np.array(witness["P"])
        G = np.array(witness["G"])
        h = np.array(witness["h"])
        g = np.array(witness["g"])
        nu = np.array(witness["multipliers"])
        d = np.array(witness["primal"])
        f0 = float(witness["F0"])
        np.testing.assert_allclose(P, P.T, atol=1e-13, rtol=0)
        mineig = float(np.linalg.eigvalsh(P).min())
        assert mineig > 0 and np.min(nu) >= 0
        lag_grad = g + G.T @ nu
        dual = f0 - nu @ h - .5*lag_grad @ np.linalg.solve(P,lag_grad)
        primal = f0 + g @ d + .5*d @ P @ d
        stationarity = np.max(abs(P@d+lag_grad))
        violation = np.max(G@d-h)
        increase = np.sqrt(dual/f0)-1
        # G begins with one defect constraint, followed by the style
        # competitors, then the 2k coordinate bounds. Recover each style
        # margin and evaluate its analytic ellipsoid support function.
        k = len(d)
        competitors = len(h)-2*k-1
        assert competitors == 21
        w = -G[1:1+competitors]
        base_margin = h[1:1+competitors]+row.style_epsilon
        center = -np.linalg.solve(P,g)
        radius_squared = g @ np.linalg.solve(P,g)
        support_norm_sq = np.einsum("ij,ji->i",w,np.linalg.solve(P,w.T))
        max_margins = base_margin+w@center+np.sqrt(np.maximum(radius_squared*support_norm_sq,0))
        min_margin = float(max_margins.min())
        np.testing.assert_allclose(
            [dual,primal,increase,min_margin],
            [row.distance_objective_dual,row.distance_objective_primal,
             row.unavoidable_distance_increase_lower_bound,
             row.ellipsoid_single_halfspace_min_upper_margin],
            atol=1e-10,rtol=1e-12)
        assert dual > f0 and min_margin < row.style_epsilon
        assert abs(primal-dual) < 1e-9 and violation < 1e-9 and stationarity < 1e-9
        rows.append(dict(beer=int(beer),support_size=k,minimum_hessian_eigenvalue=mineig,
                         minimum_multiplier=float(nu.min()),primal=float(primal),dual=float(dual),
                         primal_dual_gap=float(primal-dual),primal_violation=float(violation),
                         stationarity_residual=float(stationarity),
                         unavoidable_distance_increase_lower_bound=float(increase),
                         ellipsoid_min_upper_margin=min_margin,
                         all_continuous_candidates_on_this_support_excluded=True))
    assert len(rows) == 11
    return dict(n_supports=11, scope="the returned fixed supports only; not every support of the same cardinality",
                arithmetic="double precision; independently recomputed primal/dual and analytic support values",
                max_absolute_primal_dual_gap=max(abs(r["primal_dual_gap"]) for r in rows),
                all_dual_bounds_above_source=True, all_single_halfspace_exclusions=True, rows=rows)


def semantic_style_control_check():
    """Reconstruct controls, blend identities and all paired statistics independently."""
    from sklearn.model_selection import KFold
    path=SEM/"style_controls"
    complete=json.loads(record(path/"COMPLETE.json").read_text())
    assert complete["complete"] and complete["fitted_outer_models"]==60
    protocol=json.loads(record(path/"protocol.json").read_text())
    assert protocol["controls_not_ensembled"] and protocol["outer_folds"]==5
    original={"rrr":np.load(record(SEM/"oof_predictions.npz")),
              "blend":np.load(record(SEM/"blend_oof.npz"))}
    arrays={"rrr":np.load(record(path/"oof_predictions.npz")),
            "blend":np.load(record(path/"blend_oof.npz"))}
    rawx=pd.read_csv(record(BASE/"data/A/1.csv")).sort_values("beer")
    rawy=pd.read_csv(record(BASE/"data/A/4.csv")).set_index("beer").loc[rawx.beer]
    targets=rawy.columns[2:];y=rawy[targets].to_numpy(float)
    core=np.flatnonzero([str(t).startswith(("A_","F_")) for t in targets])
    assert len(core)==30
    for stage in arrays:
        for key in ["Y","fold","beer"]:
            np.testing.assert_array_equal(arrays[stage][key],original[stage][key])
        np.testing.assert_array_equal(arrays[stage]["Y"],y)
    denominator=np.sum((y-y.mean(0))**2,axis=0)
    sampled=np.random.default_rng(10041).integers(0,len(y),(2000,len(y)))
    weights=np.zeros((2000,len(y)))
    np.add.at(weights,(np.arange(2000)[:,None],sampled),1)
    totals=weights@y
    bootstrap_den=weights@(y*y)-totals*totals/len(y)
    contrasts=json.loads(record(path/"contrasts.json").read_text())
    table=pd.read_csv(record(path/"model_comparison.csv")).set_index(["stage","model"])
    score_error=0.;ci_error=0.;difference_error=0.
    for row in contrasts:
        stage=row["stage"];key=row["control"];selection=key.rsplit("_",1)[-1]
        pred=arrays[stage][key];reference=original[stage]["aligned_aux_"+selection]
        r2=1-np.sum((y-pred)**2,axis=0)/denominator
        stated=table.loc[(stage,key)]
        score_error=max(score_error,float(np.max(np.abs([r2.mean()-stated.all50_r2,r2[core].mean()-stated.core30_r2]))))
        gain=((y-pred)**2-(y-reference)**2)
        delta=gain.sum(0)/denominator
        samples=(weights@gain)/np.maximum(bootstrap_den,1e-20)
        for group,idx in [("all50",np.arange(50)),("core30",core)]:
            difference_error=max(difference_error,abs(float(delta[idx].mean())-row[group+"_delta"]))
            interval=np.quantile(samples[:,idx].mean(1),[.025,.975])
            ci_error=max(ci_error,float(np.max(abs(interval-np.array(row[group+"_ci"])))))
    assert max(score_error,ci_error,difference_error)<1e-12
    folds=arrays["rrr"]["fold"]
    blend_weights=pd.read_csv(record(path/"blend_weights.csv"))
    old=np.load(record(BASE/"A_work/q1/oof_arrays.npz"))["nested_ensemble"]
    blend_error=0.
    for row in blend_weights.itertuples():
        ix=folds==row.fold
        expected=(1-row.weight)*old[ix]+row.weight*arrays["rrr"][row.model][ix]
        blend_error=max(blend_error,float(np.max(abs(expected-arrays["blend"][row.model][ix]))))
    assert blend_error<1e-12 and len(blend_weights)==60
    records=pd.read_csv(record(path/"permutation_diagnostics.csv"))
    style=rawx.tasting_category_fine.astype(str).to_numpy()
    checked=0;permutations_checked=0;style_partitions_checked=0
    for fold in range(5):
        tr=np.flatnonzero(folds!=fold)
        inner=list(KFold(3,shuffle=True,random_state=protocol["seed"]+fold).split(tr))
        for name,seed in protocol["variants"]:
            for inner_id in [-1,0,1,2]:
                ids=tr if inner_id==-1 else tr[inner[inner_id][0]]
                ss=style[ids]
                rec=records[(records.fold==fold)&(records.variant==name)&(records.inner==inner_id)].iloc[0]
                assert rec.n==len(ids) and rec.active_styles==len(np.unique(ss))
                if name!="style_only":
                    rng=np.random.default_rng(seed+100003+fold if inner_id==-1 else seed+1009*fold+137*inner_id)
                    perm=np.arange(len(ids));singletons=0
                    for label in np.unique(ss):
                        ix=np.flatnonzero(ss==label);perm[ix]=rng.permutation(ix);singletons+=len(ix)==1
                    np.testing.assert_array_equal(ss[perm],ss)
                    assert rec.moved_rows==int(np.sum(perm!=np.arange(len(ids))))
                    assert rec.singleton_rows==singletons
                    assert rec.permutation_sha256==hashlib.sha256(perm.tobytes()).hexdigest()
                    permutations_checked+=1
                else:
                    assert rec.moved_rows==rec.singleton_rows==0
                    style_partitions_checked+=1
                checked+=1
    assert checked==len(records)==120
    assert permutations_checked==100 and style_partitions_checked==20
    return dict(complete=True,models=24,paired_intervals=48,outer_blend_identities=60,
                independently_reconstructed_training_partitions=checked,
                within_style_permutation_hashes_verified=permutations_checked,
                style_only_partitions_checked=style_partitions_checked,
                score_max_error=score_error,paired_delta_max_error=difference_error,
                paired_ci_max_error=ci_error,blend_identity_max_error=blend_error,
                controls_not_ensembled=True,bootstrap_seed=10041,bootstrap_replicates=2000,
                scope="Fixed OOF predictions; conditional exploratory intervals without multiple-comparison correction. Style-only has a different auxiliary space.")


def within_style_check():
    from scipy.special import ndtri
    from scipy.stats import rankdata
    raw_x=pd.read_csv(record(BASE/"data/A/1.csv")).sort_values("beer")
    raw_y=pd.read_csv(record(BASE/"data/A/4.csv")).set_index("beer").loc[raw_x.beer]
    features=raw_x.columns[3:].tolist();targets=raw_y.columns[2:].tolist()
    x=raw_x[features].to_numpy(float);y=raw_y[targets].to_numpy(float)
    styles=raw_x.tasting_category_fine.to_numpy()
    # Equivalent average-rank construction, independent of the experiment's
    # searchsorted implementation and its residualization helper.
    def normal_scores(values):
        filled=np.where(np.isfinite(values),values,np.nanmedian(values,axis=0))
        scores=ndtri(rankdata(filled,axis=0,method="average")/(len(filled)+1))
        sd=scores.std(0);sd[sd<1e-12]=1
        return (scores-scores.mean(0))/sd
    u=normal_scores(x);v=normal_scores(y)
    uc=np.empty_like(u);vc=np.empty_like(v)
    for style in np.unique(styles):
        j=styles==style;uc[j]=u[j].mean(0);vc[j]=v[j].mean(0)
    ur=u-uc;vr=v-vc
    total=u.T@v/len(y);within=ur.T@vr/len(y);between=uc.T@vc/len(y)
    den=np.sqrt(np.mean(ur*ur,0)[:,None]*np.mean(vr*vr,0)[None,:])
    corr=np.divide(within,den,out=np.zeros_like(within),where=den>1e-14)
    pairs=pd.read_csv(record(WITHIN/"full_data_pair_evidence.csv"))
    fi=np.array([features.index(f) for f in pairs.field]);ti=np.array([targets.index(t) for t in pairs.target])
    errors={}
    for name,matrix in [("correlation_total",total),("correlation_within",corr),
                        ("covariance_total",total),("covariance_within",within),("covariance_between",between)]:
        errors[name]=float(np.max(abs(matrix[fi,ti]-pairs[name].to_numpy())))
        assert errors[name]<1e-12
    fixed=pd.read_csv(record(WITHIN/"fixed_alpha_pair_evidence.csv"))
    data=np.load(record(WITHIN/"bootstrap_fixed_alpha_coefficients.npz"))
    np.testing.assert_array_equal(data["features"],features)
    np.testing.assert_array_equal(data["targets"],targets)
    fi=np.array([features.index(f) for f in fixed.field]);ti=np.array([targets.index(t) for t in fixed.target])
    frequency_error=0.
    for model_index,model in enumerate(data["models"]):
        counts=np.zeros((len(features),len(targets)))
        top=data["top3"][model_index]
        for j in range(len(targets)):
            counts[:,j]=np.bincount(top[:,:,j].ravel(),minlength=len(features))/top.shape[0]
        np.testing.assert_allclose(counts.sum(0),3,atol=1e-14,rtol=0)
        error=float(np.max(abs(counts[fi,ti]-fixed[str(model)+"_top3_frequency"].to_numpy())))
        frequency_error=max(frequency_error,error)
        assert error<1e-12
    pred=pd.read_csv(record(WITHIN/"oof_within_linear.csv"))
    baseline=pd.read_csv(record(WITHIN/"oof_style_mean_raw.csv"))
    folds=pd.read_csv(record(BASE/"A_work/q1/fold_assignments.csv"))
    np.testing.assert_array_equal(pred[["beer","fold"]].to_numpy(),folds.to_numpy())
    np.testing.assert_array_equal(baseline[["beer","fold"]].to_numpy(),folds.to_numpy())
    pp=pred[targets].to_numpy();bb=baseline[targets].to_numpy()
    gain=((y-bb)**2-(y-pp)**2)/np.var(y,axis=0)
    estimate=gain.mean(0)
    rng=np.random.default_rng(20261024+80000)
    groups=[np.flatnonzero(styles==s) for s in np.unique(styles)]
    weights=np.zeros((2000,len(y)))
    for i in range(2000):
        sampled=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        weights[i]=np.bincount(sampled,minlength=len(y))
    samples=weights@gain/len(y)
    all_estimates=np.r_[estimate,estimate.mean()]
    all_samples=np.column_stack([samples,samples.mean(1)])
    intervals=np.quantile(all_samples,[.025,.975],axis=0)
    saved=pd.read_csv(record(WITHIN/"oof_paired_differences.csv"))
    saved=saved[(saved.model=="within_linear")&(saved.baseline=="style_mean_raw")].set_index("target").loc[targets+["__macro__"]]
    error=max(float(np.max(abs(all_estimates-saved.delta_R2))),
              float(np.max(abs(intervals[0]-saved.paired_bootstrap_025))),
              float(np.max(abs(intervals[1]-saved.paired_bootstrap_975))))
    assert error<1e-12
    return dict(n_pairs=len(pairs),n_fixed_alpha_pairs=len(fixed),same_outer_folds=True,
                covariance_and_correlation_max_errors=errors,
                covariance_identity_max_error=float(np.max(abs(total-within-between))),
                all_four_frequency_tables_max_error=frequency_error,
                oof_gain_and_interval_max_error=error,
                macro_gain=float(all_estimates[-1]),macro_ci95=intervals[:,-1].tolist(),
                scope="descriptive full-sample association, fixed-alpha ranking frequencies; paired fixed-loss OOF uncertainty")


def main():
    report = dict(semantic=semantic_check(),fixed_support_geometry=geometry_check(),source_sha256=sources)
    if (SEM/"style_controls/COMPLETE.json").is_file():
        report["semantic_style_controls"]=semantic_style_control_check()
    if WITHIN.is_dir():report["within_style"]=within_style_check()
    out = HERE / "qa/new_evidence_numeric_audit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(dict(output=str(out),semantic_max_score_error=report["semantic"]["max_score_error"],
                         semantic_max_ci_error=report["semantic"]["max_ci_error"],
                         geometry_max_primal_dual_gap=report["fixed_support_geometry"]["max_absolute_primal_dual_gap"],
                         geometry_fixed_supports=11,
                         semantic_style_controls=report.get("semantic_style_controls"),
                         within_style=report.get("within_style")),indent=2))


if __name__ == "__main__":
    main()
