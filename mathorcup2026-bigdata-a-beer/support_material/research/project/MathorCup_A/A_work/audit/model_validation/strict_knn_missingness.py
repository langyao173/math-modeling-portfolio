#!/usr/bin/env python3
"""Strict missing-summary stress with fixed-specification KNN re-imputation.

This sensitivity refits imputers/estimators while retaining each original fold's
model specifications and ensemble weights; it is not a new model selection.
"""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold

HERE=Path(__file__).parent
Q1=HERE.parents[1]/'q1'
sys.path.insert(0,str(Q1))
from sensory_model import SensoryEstimator,predict_bundle
from strict_missingness import assess
SEED=20261023
BASE=Path(os.environ.get('MATHORCUP_DATA_DIR','/workspace/scratch/a7d3fe1f9c53/contest_2026/A/附件'))


def main():
    x=pd.read_csv(BASE/'1.csv').sort_values('beer').reset_index(drop=True)
    y=pd.read_csv(BASE/'4.csv').set_index('beer').loc[x.beer].reset_index()
    fields=x.columns[3:].tolist();targets=y.columns[3:].tolist()
    X=x[fields].to_numpy(float);Y=y[targets].to_numpy(float)
    derived=['kcalperc','esters_sum','aroma_hops_sum','acids_sum','sulfur_sum']
    indices=[fields.index(c) for c in derived]
    names=['original','derived_only_missing']+[f'MCAR_{level}_strict' for level in [10,30,50]]
    preds={n:np.full_like(Y,np.nan) for n in names}
    foldid=np.empty(len(X),int);specs=[]
    for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=SEED).split(X)):
        frozen=joblib.load(Q1/f'fold_{fold}_models.joblib')
        assert frozen['features']==fields and frozen['targets']==targets
        assert np.allclose(frozen['reference_y'],Y[tr])
        models=[]
        for model in frozen['models']:
            fitted=SensoryEstimator(model.spec,imputer='knn').fit(X[tr],Y[tr])
            models.append(fitted)
        bundle=dict(frozen,models=models)
        specs.append(dict(fold=fold,specifications=[m.spec for m in models],weights=list(map(float,bundle['weights']))))
        foldid[te]=fold
        cases={'original':X[te].copy(),'derived_only_missing':X[te].copy()}
        cases['derived_only_missing'][:,indices]=np.nan
        for level in [10,30,50]:
            rng=np.random.default_rng(SEED+1000*fold+level)
            a=X[te].copy();a[rng.random(a.shape)<level/100]=np.nan
            a[:,indices]=np.nan;cases[f'MCAR_{level}_strict']=a
        for name,a in cases.items():preds[name][te]=predict_bundle(bundle,a).to_numpy()
        print(f'Fold {fold+1}/5 completed; same specs and weights, KNN imputation.',flush=True)
    rows=[];dimensions=[]
    median=pd.read_csv(HERE/'strict_missingness_metrics.csv').set_index('scenario')
    for name,p in preds.items():
        m,rr,rm=assess(Y,p)
        rows.append(dict(scenario=name,imputer='knn_7_distance',all_five_derived_hidden=name!='original',
                         fixed_specs_and_weights=True,median_macro_r2=float(median.loc[name,'macro_r2']),
                         delta_macro_r2_vs_median=float(m['macro_r2']-median.loc[name,'macro_r2']),**m))
        for j,target in enumerate(targets):dimensions.append(dict(scenario=name,target=target,r2=float(rr[j]),rmse=float(rm[j])))
    pd.DataFrame(rows).to_csv(HERE/'strict_knn_missingness_metrics.csv',index=False)
    pd.DataFrame(dimensions).to_csv(HERE/'strict_knn_missingness_dimension_metrics.csv',index=False)
    np.savez_compressed(HERE/'strict_knn_missingness_oof.npz',Y=Y,beer=x.beer.to_numpy(),fold=foldid,**preds)
    summary=dict(derived_columns=derived,fold_specs_weights=specs,metrics=rows,
        interpretation='Insertion of training-only KNN imputation with the existing fold-selected model specifications and ensemble weights. No retuning. Same held-out masks as strict_missingness.py; all five derived fields hidden.')
    (HERE/'strict_knn_missingness_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
