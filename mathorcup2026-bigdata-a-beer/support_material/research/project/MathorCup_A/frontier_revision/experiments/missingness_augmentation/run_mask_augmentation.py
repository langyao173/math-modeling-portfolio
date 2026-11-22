"""Missingness augmentation and indicators under identical held-out masks.

This controlled study freezes each fold's legacy Extra Trees hyperparameters.
Training augmentation uses only copies of training beers, never held-out rows.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/'sparse_transfer'),str(HERE.parent/'semantic_auxiliary')]
from control_problem import PROJECT,AW,file_sha
from sensory_model import ChemicalPreprocessor,grades_from_reference
from run_semantic_rrr import paired_ci
SEED=20261023


def main():
    HERE.mkdir(exist_ok=True,parents=True)
    x=pd.read_csv(PROJECT/'data/A/1.csv').sort_values('beer').reset_index(drop=True)
    y=pd.read_csv(PROJECT/'data/A/4.csv').set_index('beer').loc[x.beer]
    fields=x.columns[3:].tolist();targets=y.columns[2:].tolist();X=x[fields].to_numpy(float);Y=y[targets].to_numpy(float)
    derived=[fields.index(f) for f in ['kcalperc','esters_sum','aroma_hops_sum','acids_sum','sulfur_sum']]
    scenarios=['original','derived_only_missing']+[f'MCAR_{v}_strict' for v in [10,30,50]]+['first37_strict','gcms_strict','high_concentration_strict']
    methods=['ET_base','ET_indicator','ET_mask_aug','ET_mask_aug_indicator']
    pp={m:{s:np.empty_like(Y) for s in scenarios} for m in methods}
    grades={m:{s:np.empty_like(Y,dtype=int) for s in scenarios} for m in methods};truth=np.empty_like(Y,dtype=int)
    selections=json.loads((AW/'q1/nested_selection.json').read_text())
    protocol=dict(outer_seed=SEED,outer_folds=5,actual_beers_per_training_fold=200,augmentation_rows_per_training_fold=800,
        mask_rates=[.1,.3,.5],weights='Each original row weight1; each of three corruptions weight1/3 (original mass=augmented mass)',
        hyperparameters='Frozen old outer-fold-selected Extra Trees specification; no new tuning or outer selection',
        preprocessing='Fit median/raw scaler on original training rows only; targets standardized on training originals',
        indicator='231 missingness binary indicators; every predictor can still only see chemistry and its observed mask',
        missing_derived='All five derived fields masked for training corruptions and strict evaluation',
        same_masks_as='A_work/audit/model_validation/strict_missingness.py',scenarios=scenarios,
        source_sha256=file_sha(PROJECT/'data/A/1.csv'),label_sha256=file_sha(PROJECT/'data/A/4.csv'))
    (HERE/'protocol.json').write_text(json.dumps(protocol,indent=2))
    t0=time.perf_counter();folds=np.empty(len(X),int)
    for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=SEED).split(X)):
        prep=ChemicalPreprocessor('raw').fit(X[tr]);s=selections[fold]['specs']['extra_trees']
        mean=Y[tr].mean(0);sd=Y[tr].std(0,ddof=1);sd[sd<1e-10]=1.;T=(Y[tr]-mean)/sd
        xaug=[X[tr].copy()]
        for level in [10,30,50]:
            a=X[tr].copy();rng=np.random.default_rng(SEED+69000+1000*fold+level)
            a[rng.random(a.shape)<level/100]=np.nan;a[:,derived]=np.nan;xaug.append(a)
        xa=np.vstack(xaug);ya=np.tile(T,(4,1));weight=np.r_[np.ones(len(tr)),np.full(3*len(tr),1/3)]
        cases={'original':X[te].copy(),'derived_only_missing':X[te].copy()};cases['derived_only_missing'][:,derived]=np.nan
        for level in [10,30,50]:
            a=X[te].copy();rng=np.random.default_rng(SEED+1000*fold+level)
            a[rng.random(a.shape)<level/100]=np.nan;a[:,derived]=np.nan;cases[f'MCAR_{level}_strict']=a
        for name in ['first37','gcms','high_concentration']:
            a=X[te].copy()
            if name=='first37':a[:,:37]=np.nan
            elif name=='gcms':a[:,37:-4]=np.nan
            else:a[a>np.nanquantile(X[tr],.9,axis=0)]=np.nan
            a[:,derived]=np.nan;cases[name+'_strict']=a
        for method in methods:
            augmented='aug' in method;indicator='indicator' in method
            data=xa if augmented else X[tr];z=prep.transform(data)
            if indicator:z=np.column_stack([z,np.isnan(data)])
            model=ExtraTreesRegressor(n_estimators=240,max_features=s['max_features'],min_samples_leaf=s['min_samples_leaf'],n_jobs=1,random_state=SEED)
            model.fit(z,ya if augmented else T,sample_weight=weight if augmented else None)
            for scenario,data in cases.items():
                z=prep.transform(data)
                if indicator:z=np.column_stack([z,np.isnan(data)])
                pred=model.predict(z)*sd+mean;pp[method][scenario][te]=pred
                grades[method][scenario][te]=grades_from_reference(pred,Y[tr])
        truth[te]=grades_from_reference(Y[te],Y[tr]);folds[te]=fold
        print(json.dumps(dict(fold=fold,elapsed=time.perf_counter()-t0)),flush=True)
    original=np.load(AW/'q1/oof_arrays.npz')
    err=float(np.max(abs(original['extra_trees']-pp['ET_base']['original'])))
    if err>1e-11:raise AssertionError(f'Legacy ET reproduction mismatch: {err}')
    baselines={kind:np.load(AW/f'audit/model_validation/{filename}') for kind,filename in
               [('median','strict_missingness_oof.npz'),('knn','strict_knn_missingness_oof.npz')]}
    rows=[];dim=[]
    for method in methods:
        for scenario in scenarios:
            pred=pp[method][scenario];r=r2_score(Y,pred,multioutput='raw_values');delta=abs(grades[method][scenario]-truth)
            row=dict(model=method,scenario=scenario,all50_r2=float(r.mean()),core30_r2=float(r[:30].mean()),
                core30_grade_mae=float(delta[:,:30].mean()),core30_within_one=float((delta[:,:30]<=1).mean()))
            if scenario in baselines['knn'].files:
                row['all50_ci_vs_knn']=paired_ci(Y,pred,baselines['knn'][scenario],np.arange(50))
                row['core30_ci_vs_knn']=paired_ci(Y,pred,baselines['knn'][scenario],np.arange(30))
            rows.append(row)
            dim += [dict(model=method,scenario=scenario,target=t,r2=float(r[i])) for i,t in enumerate(targets)]
    pd.DataFrame(rows).to_csv(HERE/'comparison.csv',index=False);pd.DataFrame(dim).to_csv(HERE/'dimension_metrics.csv',index=False)
    np.savez_compressed(HERE/'oof_predictions.npz',Y=Y,beer=x.beer,fold=folds,
                        **{f'{m}__{s}':p for m,d in pp.items() for s,p in d.items()})
    (HERE/'run_summary.json').write_text(json.dumps(dict(seconds=time.perf_counter()-t0,ET_reproduction_max_error=err),indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
