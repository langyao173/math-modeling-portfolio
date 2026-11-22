#!/usr/bin/env python3
"""Independent nested-CV HGB challenger and internal-standard sensitivity.

No validation observation is used for preprocessing or model selection.
"""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

SEED = 20261023
BASE = Path(os.environ.get('MATHORCUP_DATA_DIR', '/workspace/scratch/a7d3fe1f9c53/contest_2026/A/附件'))
OUT = Path(__file__).parent


def hgb():
    return HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=7,
        min_samples_leaf=15, max_bins=32, early_stopping=False,
        random_state=SEED)


def summarize(Y, P):
    r2 = r2_score(Y, P, multioutput='raw_values')
    rmse = np.sqrt(np.mean((Y-P)**2, axis=0))
    spearman = np.array([spearmanr(Y[:,j],P[:,j]).statistic for j in range(Y.shape[1])])
    return {'macro_r2':float(r2.mean()), 'median_r2':float(np.median(r2)),
            'positive_r2':int((r2>0).sum()), 'macro_rmse':float(rmse.mean()),
            'macro_spearman':float(np.nanmean(spearman))}, r2, rmse, spearman


def fit_predict_hgb(Xtr,Ytr,Xte,l2):
    prep = SimpleImputer(strategy='median',keep_empty_features=True)
    A=prep.fit_transform(Xtr); B=prep.transform(Xte)
    mu=Ytr.mean(axis=0);sd=Ytr.std(axis=0,ddof=1);sd[sd<1e-10]=1
    prediction=np.empty((len(Xte),Ytr.shape[1]))
    if np.isscalar(l2):l2=np.repeat(float(l2),Ytr.shape[1])
    for j in range(Ytr.shape[1]):
        model=hgb().set_params(l2_regularization=float(l2[j]))
        model.fit(A,(Ytr[:,j]-mu[j])/sd[j])
        prediction[:,j]=model.predict(B)*sd[j]+mu[j]
    return prediction


def fit_predict_rf(Xtr,Ytr,Xte):
    prep=SimpleImputer(strategy='median',keep_empty_features=True)
    A=prep.fit_transform(Xtr);B=prep.transform(Xte)
    mu=Ytr.mean(axis=0);sd=Ytr.std(axis=0,ddof=1);sd[sd<1e-10]=1
    model=RandomForestRegressor(n_estimators=240,max_features=.5,min_samples_leaf=2,
                                n_jobs=1,random_state=SEED)
    model.fit(A,(Ytr-mu)/sd)
    return model.predict(B)*sd+mu


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    x=pd.read_csv(BASE/'1.csv').sort_values('beer').reset_index(drop=True)
    y=pd.read_csv(BASE/'4.csv').set_index('beer').loc[x.beer].reset_index()
    fields=x.columns[3:].tolist();targets=y.columns[3:].tolist()
    X=x[fields].to_numpy(float);Y=y[targets].to_numpy(float)
    no_internal=np.array([f!='X3.methyl.2.butenal' for f in fields])
    assert sum(~no_internal)==1
    predictions={n:np.empty_like(Y) for n in ['hgb_nested','rf_fixed_all','rf_fixed_no_internal']}
    outer=list(KFold(5,shuffle=True,random_state=SEED).split(X))
    selections=[];foldid=np.empty(len(X),int);start=time.time()
    for fold,(train,test) in enumerate(outer):
        print(f'Fold {fold+1}/5 start, {time.time()-start:.1f}s',flush=True)
        foldid[test]=fold
        A=X[train];T=Y[train]
        inner=list(KFold(3,shuffle=True,random_state=SEED+fold).split(A))
        cand={}
        for l2 in [1.,10.]:
            innerpred=np.empty_like(T)
            for itrain,itest in inner:
                innerpred[itest]=fit_predict_hgb(A[itrain],T[itrain],A[itest],l2)
            cand[l2]=np.mean((T-innerpred)**2,axis=0)
            print(f'Fold {fold+1} l2={l2} inner complete, {time.time()-start:.1f}s',flush=True)
        chosen=np.where(cand[1.]<=cand[10.],1.,10.)
        predictions['hgb_nested'][test]=fit_predict_hgb(A,T,X[test],chosen)
        predictions['rf_fixed_all'][test]=fit_predict_rf(A,T,X[test])
        predictions['rf_fixed_no_internal'][test]=fit_predict_rf(A[:,no_internal],T,X[test][:,no_internal])
        for j,t in enumerate(targets):
            selections.append({'fold':fold,'target':t,'selected_l2':float(chosen[j]),
                               'inner_mse_l2_1':float(cand[1.][j]),'inner_mse_l2_10':float(cand[10.][j])})
        print(f'Fold {fold+1} done, {time.time()-start:.1f}s',flush=True)
        # Persist completed folds for interruption-safe diagnosis; no final score until all folds finish.
        np.savez_compressed(OUT/'progress_predictions.npz', completed_fold=fold,Y=Y,**predictions)
    meta=x[['beer','beer_id','tasting_category_fine']].copy();meta['fold']=foldid
    metrics=[];dims=[]
    for model,p in predictions.items():
        metrics0,r2,rmse,rank=summarize(Y,p)
        metrics.append({'model':model,**metrics0})
        for j,t in enumerate(targets):
            dims.append({'model':model,'target':t,'r2':float(r2[j]),'rmse':float(rmse[j]),'spearman':float(rank[j])})
        pd.concat([meta,pd.DataFrame(p,columns=targets)],axis=1).to_csv(OUT/f'{model}_oof.csv',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'model_comparison.csv',index=False)
    pd.DataFrame(dims).to_csv(OUT/'dimension_metrics.csv',index=False)
    pd.DataFrame(selections).to_csv(OUT/'nested_selections.csv',index=False)
    pd.DataFrame({'beer':x.beer,'fold':foldid}).to_csv(OUT/'fold_assignments.csv',index=False)
    np.savez_compressed(OUT/'oof_arrays.npz',Y=Y,**predictions)
    rng=np.random.default_rng(SEED);diff=[]
    for _ in range(1000):
        ii=rng.integers(0,len(Y),len(Y))
        aa=r2_score(Y[ii],predictions['rf_fixed_all'][ii],multioutput='uniform_average')
        bb=r2_score(Y[ii],predictions['rf_fixed_no_internal'][ii],multioutput='uniform_average')
        diff.append(bb-aa)
    result={'seed':SEED,'sample_order':'beer ascending','outer':'5-fold KFold shuffle',
            'inner':'3-fold KFold seed20261023+outer_fold; target-specific l2 selection',
            'preprocessing':'Training-fold median imputation; target scaling fitted on every training fold',
            'hgb_parameters':hgb().get_params(),
            'hgb_fitted_l2_grid':[1.,10.],
            'hgb_l2_note':'hgb_parameters shows constructor defaults only; every fitted estimator overrides l2_regularization using the target-specific inner-CV selection in nested_selections.csv.',
            'metrics':metrics,
            'rf_no_internal_minus_all':{'delta':metrics[2]['macro_r2']-metrics[1]['macro_r2'],
                'paired_beer_bootstrap_95':np.quantile(diff,[.025,.975]).tolist(),
                'interpretation':'fixed OOF bootstrap; does not include model-refitting uncertainty'},
            'seconds':time.time()-start}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
