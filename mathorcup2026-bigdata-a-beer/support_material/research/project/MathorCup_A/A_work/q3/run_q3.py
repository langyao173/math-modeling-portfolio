"""Reproduce question 3 conditional generation and held-out evaluation.

Original competition datasets remain read-only. A fixed test set is not used
for hyperparameter selection. Generated profiles are computational candidates.
"""
from pathlib import Path
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mathorcup-q3-mpl")
import argparse, json, sys, warnings
import numpy as np
import pandas as pd
import joblib
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, r2_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from conditional_model import ConditionalCopula, DERIVED, energy_scores, variogram_scores


DEFAULT_DATA = Path(os.environ.get("MATHORCUP_DATA_DIR", "/workspace/scratch/a7d3fe1f9c53/contest_2026/A/附件"))
SEED = 20261023
PARAMS = [
    dict(name="empirical_bootstrap", kind="bootstrap", kappa=0, nu=20),
    dict(name="independent_marginals", kind="independent", kappa=2, nu=20),
    dict(name="pooled_copula", kind="pooled", kappa=2, nu=20),
    dict(name="hierarchical_k0_n20", kind="hierarchical", kappa=0, nu=20),
    dict(name="hierarchical_k2_n20", kind="hierarchical", kappa=2, nu=20),
    dict(name="hierarchical_k2_n5", kind="hierarchical", kappa=2, nu=5),
    dict(name="hierarchical_k5_n20", kind="hierarchical", kappa=5, nu=20),
]


def fit_model(param, x, style):
    return ConditionalCopula(**{k:v for k,v in param.items() if k != "name"}).fit(x, style)


def score_model(model, x, styles, n_sample=256, seed=SEED):
    z = model.map_.transform(x)
    styles = np.asarray(styles)
    rng = np.random.default_rng(seed)
    p = z.shape[1]
    pairs = rng.integers(0, p, size=(1000, 2))
    pairs = pairs[pairs[:,0] != pairs[:,1]]
    es, vs = np.zeros(len(x)), np.zeros(len(x))
    generated, synthetic_styles = [], []
    for s in np.unique(styles):
        g, gz = model.sample(s, n_sample, rng)
        ix = np.flatnonzero(styles == s)
        es[ix] = energy_scores(z[ix], gz)
        vs[ix] = variogram_scores(z[ix], gz, pairs)
        # Composition matches held-out labels exactly for pooled diagnostics.
        chosen = rng.choice(len(gz), len(ix), replace=False)
        generated.append(gz[chosen]);synthetic_styles.extend([s] * len(chosen))
    return es, vs, z, np.vstack(generated), np.array(synthetic_styles)


def paired_ci(a, b, seed=SEED, nboot=3000):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (nboot, len(d)))].mean(axis=1)
    return float(d.mean()), *np.quantile(bs, [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    df = pd.read_csv(args.data / "1.csv")
    features = [c for c in df.columns[3:] if c not in DERIVED]
    x = df[features].to_numpy(float);style = df.tasting_category_fine.to_numpy(str)
    tr, te = train_test_split(np.arange(len(df)), test_size=.25, stratify=style, random_state=SEED)
    split = pd.DataFrame({"beer":df.beer,"style":style,"set":"development"})
    split.loc[te,"set"] = "locked_test"
    split.to_csv(out/"evaluation_split.csv",index=False)
    pd.Series(features).to_csv(out/"chemical_features.csv",header=["field_name_R"],index=False)
    cv = RepeatedStratifiedKFold(n_splits=2,n_repeats=3,random_state=SEED+1)
    cv_rows=[]
    if not args.skip_evaluation:
        for fold,(a,b) in enumerate(cv.split(x[tr],style[tr])):
            a,b=tr[a],tr[b]
            for param in PARAMS:
                model=fit_model(param,x[a],style[a])
                es,vs,*_=score_model(model,x[b],style[b],128,SEED+fold)
                cv_rows.append(dict(model=param["name"],fold=fold,energy=es.mean(),variogram=vs.mean()))
            print("CV fold",fold,"completed",flush=True)
        cv_frame=pd.DataFrame(cv_rows)
        cv_frame.to_csv(out/"development_cv.csv",index=False)
        summary=cv_frame.groupby("model").agg(energy=("energy","mean"),energy_fold_sd=("energy","std"),variogram=("variogram","mean")).sort_values("energy")
        summary.to_csv(out/"development_model_comparison.csv")
        # A resampling reference is not a novel generator; choose among novel models.
        eligible=summary.drop(index="empirical_bootstrap")
        selected=eligible.index[0]
        selected_param=next(p for p in PARAMS if p["name"]==selected)
        (out/"selected_model.json").write_text(json.dumps({**selected_param,"selection":"minimum development 6-fold mean energy among non-resampling models","test_used_for_selection":False},ensure_ascii=False,indent=2))
        print("Selected before test:",selected,flush=True)
        test_rows=[];bybeer=[];energy_by_model={};zreal=None
        evaluated=[p for p in PARAMS if p["name"] in ["empirical_bootstrap","independent_marginals","pooled_copula",selected]]
        for param in evaluated:
            model=fit_model(param,x[tr],style[tr])
            es,vs,zreal,zgen,sgen=score_model(model,x[te],style[te],512,SEED+100)
            energy_by_model[param["name"]]=es
            clf=LogisticRegression(C=.1,max_iter=3000,class_weight="balanced")
            clf.fit(model.z_,style[tr])
            predreal=clf.predict(zreal);predgen=clf.predict(zgen)
            real_nn=cdist(zreal,model.z_).min(axis=1)/np.sqrt(len(features))
            gen_nn=cdist(zgen,model.z_).min(axis=1)/np.sqrt(len(features))
            combined=np.vstack([zreal,zgen]);isgen=np.r_[np.zeros(len(zreal)),np.ones(len(zgen))]
            disc=RandomForestClassifier(n_estimators=200,max_depth=4,min_samples_leaf=4,n_jobs=1,random_state=SEED)
            prob=cross_val_predict(disc,combined,isgen,method="predict_proba",cv=StratifiedKFold(5,shuffle=True,random_state=SEED))[:,1]
            corr_real=np.corrcoef(zreal,rowvar=False);corr_gen=np.corrcoef(zgen,rowvar=False)
            mask=np.triu(np.ones_like(corr_real,dtype=bool),1)
            corr_mae=np.nanmean(np.abs(corr_real-corr_gen)[mask])
            test_rows.append(dict(model=param["name"],n_test=len(te),energy=es.mean(),variogram=vs.mean(),mean_marginal_wasserstein=np.mean([wasserstein_distance(zreal[:,j],zgen[:,j]) for j in range(len(features))]),pooled_correlation_mae=corr_mae,real_style_accuracy=accuracy_score(style[te],predreal),real_style_balanced_accuracy=balanced_accuracy_score(style[te],predreal),generated_style_accuracy=accuracy_score(sgen,predgen),generated_style_balanced_accuracy=balanced_accuracy_score(sgen,predgen),c2st_auc=roc_auc_score(isgen,prob),real_nearest_train_median=np.median(real_nn),generated_nearest_train_median=np.median(gen_nn),exact_train_copy_rate=np.mean(gen_nn<1e-8)))
            for i,j in enumerate(te):
                bybeer.append(dict(model=param["name"],beer=int(df.beer.iloc[j]),style=style[j],energy=es[i],variogram=vs[i]))
            if param["name"]==selected:
                pd.DataFrame({"beer":df.beer.iloc[te].values,"true_style":style[te],"predicted_style":predreal,"nearest_training_distance":real_nn}).to_csv(out/"test_style_classifier.csv",index=False)
            print("Test evaluated",param["name"],flush=True)
        pd.DataFrame(test_rows).to_csv(out/"locked_test_comparison.csv",index=False)
        pd.DataFrame(bybeer).to_csv(out/"locked_test_scores_by_beer.csv",index=False)
        ci=[]
        for name,es in energy_by_model.items():
            if name!=selected:
                mean,lo,hi=paired_ci(es,energy_by_model[selected])
                ci.append(dict(reference=name,selected=selected,energy_reduction=mean,ci95_low=lo,ci95_high=hi))
        pd.DataFrame(ci).to_csv(out/"test_paired_comparisons.csv",index=False)
    else:
        selected_param=json.loads((out/"selected_model.json").read_text())
        selected_param={k:v for k,v in selected_param.items() if k in PARAMS[0]}
    final=fit_model(selected_param,x,style)
    joblib.dump({"model":final,"features":features,"all_features":list(df.columns[3:]),"selected_parameter":selected_param},out/"conditional_generator.joblib")
    all_rows=[];prototypes=[];rng=np.random.default_rng(SEED+200)
    for s in final.styles_:
        g,gz=final.sample(s,100,rng)
        frame=pd.DataFrame(g,columns=features)
        frame.insert(0,"tasting_category_fine",s)
        frame.insert(0,"synthetic_id",[s.replace('/','_').replace(' ','_')+f"_{j+1:03d}" for j in range(len(g))])
        all_rows.append(frame)
        row={"tasting_category_fine":s,**dict(zip(features,final.prototype(s)))}
        prototypes.append(row)
    pd.concat(all_rows,ignore_index=True).to_csv(out/"generated_chemical_profiles.csv",index=False)
    pd.DataFrame(prototypes).to_csv(out/"style_prototypes.csv",index=False)
    # These auxiliary summaries are explicitly predictions, not definitions.
    # A subsequent source-grounded deterministic reconstruction replaces them.
    derived_models={};quality=[]
    for col in DERIVED:
        target=df[col].to_numpy(float);ok=np.isfinite(target)
        reg=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=2,max_features=.8,random_state=SEED,n_jobs=1)
        reg.fit(final.z_[ok],target[ok]);derived_models[col]=reg
    joblib.dump(derived_models,out/"derived_summary_predictors.joblib")
    # Stale hops is a sensory target, not a directly measured chemical.
    y=pd.read_csv(args.data/"4.csv").set_index("beer").loc[df.beer,"stale_hops"].to_numpy(float)
    ridge=RidgeCV(alphas=[.1,1.,10.,100.,1000.]).fit(final.z_,y)
    joblib.dump({"model":ridge,"map":final.map_,"features":features},out/"stale_hops_linear_constraint.joblib")
    metadata=dict(n_samples=len(df),n_direct_chemical_features=len(features),n_styles=len(final.styles_),styles={s:int(final.n_[s]) for s in final.styles_},n_generated=100*len(final.styles_),generated_units="Each column retains its original 1.csv scale; no blanket exponentiation is applied.",derived_columns_excluded=DERIVED,pooled_shrinkage=final.pooled_shrinkage_,stale_hops_constraint_alpha=float(ridge.alpha_),test_size=len(te),development_size=len(tr))
    (out/"generation_metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    print("Generated",metadata["n_generated"],"profiles",flush=True)


if __name__=="__main__":
    main()
