#!/usr/bin/env python3
"""Stress frozen fold models after hiding unavailable derived summaries.

Run only after all five current Q1 fold models have been written.
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
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

HERE=Path(__file__).parent
Q1=HERE.parents[1]/'q1'
sys.path.insert(0,str(Q1))
from sensory_model import predict_bundle
SEED=20261023
BASE=Path(os.environ.get('MATHORCUP_DATA_DIR', '/workspace/scratch/a7d3fe1f9c53/contest_2026/A/附件'))


def assess(Y,P):
    r=r2_score(Y,P,multioutput='raw_values')
    rmse=np.sqrt(np.mean((Y-P)**2,axis=0))
    rank=np.array([spearmanr(Y[:,j],P[:,j]).statistic for j in range(Y.shape[1])])
    return dict(macro_r2=float(r.mean()),median_r2=float(np.median(r)),positive_r2=int((r>0).sum()),
                macro_rmse=float(rmse.mean()),macro_spearman=float(np.nanmean(rank))),r,rmse


def main():
    x=pd.read_csv(BASE/'1.csv').sort_values('beer').reset_index(drop=True)
    y=pd.read_csv(BASE/'4.csv').set_index('beer').loc[x.beer].reset_index()
    fields=x.columns[3:].tolist();targets=y.columns[3:].tolist()
    X=x[fields].to_numpy(float);Y=y[targets].to_numpy(float)
    derived=['kcalperc','esters_sum','aroma_hops_sum','acids_sum','sulfur_sum']
    idx=[fields.index(f) for f in derived]
    names=['original','derived_only_missing']
    for level in [10,30,50]:names.extend([f'MCAR_{level}_original',f'MCAR_{level}_strict'])
    for name in ['first37','gcms','high_concentration']:names.extend([name+'_original',name+'_strict'])
    preds={n:np.full_like(Y,np.nan) for n in names};missing={n:0 for n in names}
    folds=list(KFold(5,shuffle=True,random_state=SEED).split(X));foldid=np.empty(len(X),int)
    artifact_times={}
    for fold,(tr,te) in enumerate(folds):
        path=Q1/f'fold_{fold}_models.joblib';artifact_times[path.name]=path.stat().st_mtime
        bundle=joblib.load(path)
        assert bundle['features']==fields and bundle['targets']==targets
        assert np.allclose(bundle['reference_y'],Y[tr])
        foldid[te]=fold
        cases={'original':X[te].copy(),'derived_only_missing':X[te].copy()}
        cases['derived_only_missing'][:,idx]=np.nan
        for level in [10,30,50]:
            rng=np.random.default_rng(SEED+1000*fold+level)
            a=X[te].copy();a[rng.random(a.shape)<level/100]=np.nan
            cases[f'MCAR_{level}_original']=a
            b=a.copy();b[:,idx]=np.nan;cases[f'MCAR_{level}_strict']=b
        for name in ['first37','gcms','high_concentration']:
            a=X[te].copy()
            if name=='first37':a[:,:37]=np.nan
            elif name=='gcms':a[:,37:-4]=np.nan
            else:a[a>np.nanquantile(X[tr],.9,axis=0)]=np.nan
            cases[name+'_original']=a
            b=a.copy();b[:,idx]=np.nan;cases[name+'_strict']=b
        for name,a in cases.items():
            preds[name][te]=predict_bundle(bundle,a).to_numpy()
            missing[name]+=int(np.isnan(a).sum())
    main_oof=pd.read_csv(Q1/'oof_predictions_standardized.csv').sort_values('beer')
    repeat_error=float(np.max(abs(main_oof[targets].to_numpy()-preds['original'])))
    if repeat_error>1e-9:
        raise RuntimeError(f'Current fold models and main OOF do not agree: max error {repeat_error}')
    metrics=[];dimension=[]
    for name,p in preds.items():
        m,rr,rm=assess(Y,p)
        metrics.append({'scenario':name,'all_five_derived_hidden':name.endswith('_strict') or name=='derived_only_missing',
                        'actual_missing_fraction':missing[name]/X.size,**m})
        for j,t in enumerate(targets):dimension.append({'scenario':name,'target':t,'r2':rr[j],'rmse':rm[j]})
    pd.DataFrame(metrics).to_csv(HERE/'strict_missingness_metrics.csv',index=False)
    pd.DataFrame(dimension).to_csv(HERE/'strict_missingness_dimension_metrics.csv',index=False)
    np.savez_compressed(HERE/'strict_missingness_oof.npz',Y=Y,beer=x.beer.to_numpy(),fold=foldid,**preds)
    summary={'source_q1':str(Q1),'derived_columns':derived,'main_oof_reproduction_max_error':repeat_error,
             'fold_model_mtimes':artifact_times,'metrics':metrics,
             'interpretation':'Frozen-model test-time unavailability stress. Derived columns are hidden rather than rebuilt from held-out original measurements.'}
    (HERE/'strict_missingness_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
