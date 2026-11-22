"""Frozen, same-outer-fold modern-model benchmark. No outer-label selection.


Every target prediction is checkpointed; complete runs may be resumed.
"""
from __future__ import annotations
from pathlib import Path
import os
THREADS=int(os.environ.get('BEER_MODERN_THREADS','6'))
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[k]=str(THREADS)
import argparse, hashlib, importlib.metadata, json, platform, resource, sys, time
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from tabicl import TabICLRegressor

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]/'project'/'MathorCup_A'
sys.path.insert(0,str(BASE/'A_work/q1'))
from sensory_model import ChemicalPreprocessor
SEED=20261023
VIEWS=('rank_pca64_y16','raw231_y16','raw231_y50')

def atomic_json(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    temp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def assess(y,p):
    r=r2_score(y,p,multioutput='raw_values')
    return {'macro_r2':float(r.mean()),'core30_r2':float(r[:30].mean()),
            'aroma15_r2':float(r[:15].mean()),'flavour15_r2':float(r[15:30].mean()),
            'other20_r2':float(r[30:].mean()),'macro_rmse':float(np.sqrt(((y-p)**2).mean(axis=0)).mean()),
            'positive_r2':int((r>0).sum())}

def data():
    x=pd.read_csv(BASE/'data/A/1.csv').sort_values('beer').reset_index(drop=True)
    y=pd.read_csv(BASE/'data/A/4.csv').set_index('beer').loc[x.beer].reset_index()
    assert np.array_equal(x.beer,y.beer)
    fold=pd.read_csv(BASE/'A_work/q1/fold_assignments.csv').set_index('beer').loc[x.beer,'fold'].to_numpy()
    X=x.iloc[:,3:].to_numpy(float);Y=y.iloc[:,3:].to_numpy(float)
    for k,(_,te) in enumerate(KFold(5,shuffle=True,random_state=SEED).split(X)):
        assert np.array_equal(np.flatnonzero(fold==k),te)
    assert X.shape==(250,231) and Y.shape==(250,50)
    old=np.load(BASE/'A_work/q1/oof_arrays.npz')
    assert np.allclose(Y,old['Y'])
    return x,y,X,Y,fold

def baselines(Y):
    old=np.load(BASE/'A_work/q1/oof_arrays.npz')
    ret={f'old_{n}':old[n] for n in ('training_mean','ridge_rank','rbf','extra_trees','random_forest','nested_ensemble')}
    f=BASE/'A_work/nature_upgrade/fusion/8base_oof.npz'
    if f.exists():
        a=np.load(f);assert np.allclose(Y,a['Y'])
        ret['old_eight_global']=a['global_prediction']
        ret['old_eight_targetwise']=a['target_prediction']
    return ret

def summarize(out,x,y,Y,fold):
    preds=baselines(Y);rows=[];foldrows=[]
    for view in VIEWS:
        files=[out/'checkpoints'/f'{view}_fold{k}_complete.npz' for k in range(5)]
        for k,f in enumerate(files):
            if f.exists():
                a=np.load(f)
                for name in a.files:
                    if name.startswith('prediction_'):
                        model=view+'_'+name.removeprefix('prediction_')
                        foldrows.append({'model':model,'fold':k,**assess(Y[a['test_indices']],a[name])})
        if all(f.exists() for f in files):
            for f in files:
                a=np.load(f)
                for name in a.files:
                    if name.startswith('prediction_'):
                        model=view+'_'+name.removeprefix('prediction_')
                        if model not in preds:preds[model]=np.full_like(Y,np.nan)
                        preds[model][a['test_indices']]=a[name]
    for name,p in preds.items():
        assert np.isfinite(p).all()
        rows.append({'model':name,**assess(Y,p)})
    pd.DataFrame(rows).to_csv(out/'model_comparison.csv',index=False)
    pd.DataFrame(foldrows).to_csv(out/'fold_metrics.csv',index=False)
    np.savez_compressed(out/'oof_arrays.npz',Y=Y,beer=x.beer.to_numpy(),targets=y.columns[3:].to_numpy(str),fold=fold,**preds)
    dimensions=[]
    for name,p in preds.items():
        r=r2_score(Y,p,multioutput='raw_values'); rmse=np.sqrt(((Y-p)**2).mean(axis=0))
        for j,t in enumerate(y.columns[3:]):dimensions.append({'model':name,'target':t,'core30':j<30,'r2':float(r[j]),'rmse':float(rmse[j])})
    pd.DataFrame(dimensions).to_csv(out/'dimension_metrics.csv',index=False)
    atomic_json(out/'status.json',{'completed_models':[k for k in preds if not k.startswith('old_')],
        'completed_model_folds':len(foldrows),'total_model_folds':25,
        'metrics':rows,'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    return preds

def paired_uncertainty(out,Y,preds,repetitions=2000):
    rng=np.random.default_rng(SEED+2026)
    ref=preds['old_nested_ensemble'];rows=[]
    for name,p in preds.items():
        if name.startswith('old_'):continue
        vals=[]
        for b in range(repetitions):
            ids=rng.integers(0,len(Y),len(Y));yy=Y[ids]
            den=((yy-yy.mean(axis=0))**2).sum(axis=0)
            delta=(((yy-ref[ids])**2).sum(axis=0)-((yy-p[ids])**2).sum(axis=0))/den
            vals.append([delta.mean(),delta[:30].mean()])
        vals=np.asarray(vals)
        point=r2_score(Y,p,multioutput='raw_values')-r2_score(Y,ref,multioutput='raw_values')
        for j,group in enumerate(('all50','core30')):
            rows.append({'model':name,'reference':'old_nested_ensemble','targets':group,
                'delta_r2':float(point.mean() if j==0 else point[:30].mean()),
                'ci025':float(np.quantile(vals[:,j],.025)),'ci975':float(np.quantile(vals[:,j],.975)),
                'repetitions':repetitions,'interpretation':'Descriptive paired beer bootstrap of fixed OOF predictions; no refit and no multiplicity correction.'})
    pd.DataFrame(rows).to_csv(out/'paired_bootstrap.csv',index=False)

def main(args):
    out=HERE; (out/'checkpoints').mkdir(exist_ok=True)
    x,y,X,Y,fold=data()
    checkpoint=HERE/'models/tabicl-regressor-v2-20260212.ckpt'
    spec={'frozen_date':'2026-10-24','outer_fold_source':str(BASE/'A_work/q1/fold_assignments.csv'),
        'outer_seed':SEED,'inner_selection':'None: all candidates and parameters fixed and all reported.',
        'views':list(VIEWS),'target_pca_rank':16,'input_pca_rank':64,'orthogonal_residual_ridge_alpha':100.,
        'target_standardization':'Train-fold mean and sample std before PCA; restore original sensory scale.',
        'input_preprocessing':'raw231 passed to official TabICL; rank_pca64 uses train-only ChemicalPreprocessor(mode=rank) then PCA(full).',
        'tabicl_parameters':{'n_estimators':8,'device':'cpu','n_jobs':THREADS,'use_amp':False,'allow_auto_download':False,
            'random_state':SEED,'batch_size':8,'kv_cache':False,'checkpoint_version':checkpoint.name},
        'checkpoint_sha256':digest(checkpoint),'fold_sha256':digest(BASE/'A_work/q1/fold_assignments.csv'),
        'data_sha256':{n:digest(BASE/'data/A'/n) for n in ('1.csv','4.csv')},
        'versions':{p:importlib.metadata.version(p) for p in ('tabicl','torch','numpy','pandas','scikit-learn','scipy')},
        'cpu_count':os.cpu_count(),'platform':platform.platform(),
        'development_caveat':'Internal adaptive research on a previously analyzed dataset; fixed outer predictions support comparison, not external validation.'}
    if (out/'experiment_spec.json').exists():
        prev=json.loads((out/'experiment_spec.json').read_text())
        assert prev==spec,'Frozen experiment specification changed; create a separate experiment instead.'
    else:atomic_json(out/'experiment_spec.json',spec)
    torch.set_num_threads(THREADS); torch.manual_seed(SEED)
    for view in (args.views or VIEWS):
        assert view in VIEWS
        for f in range(5):
            complete=out/'checkpoints'/f'{view}_fold{f}_complete.npz'
            if complete.exists():
                print(f'SKIP complete {view} fold={f}',flush=True);continue
            tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f)
            prep=ChemicalPreprocessor(mode='rank').fit(X[tr])
            xr=prep.transform(X[tr]);xt=prep.transform(X[te])
            if view.startswith('rank_pca'):
                xpca=PCA(n_components=64,svd_solver='full').fit(xr)
                a=xpca.transform(xr);b=xpca.transform(xt)
            else:a=X[tr];b=X[te]
            latent=view.endswith('y16')
            if latent:
                mu=Y[tr].mean(axis=0);sigma=Y[tr].std(axis=0,ddof=1)
                sigma=np.maximum(sigma,1e-12);T=(Y[tr]-mu)/sigma
                ypca=PCA(n_components=16,svd_solver='full').fit(T)
                target=ypca.transform(T)
            else:target=Y[tr]
            partial=out/'checkpoints'/f'{view}_fold{f}_partial.npz'
            pred=np.full((len(te),target.shape[1]),np.nan);timing=[]
            if partial.exists():
                prev=np.load(partial);pred=prev['prediction'];timing=prev['timing'].tolist()
                assert np.array_equal(prev['test_indices'],te)
            for j in range(target.shape[1]):
                if np.isfinite(pred[:,j]).all():continue
                model=TabICLRegressor(n_estimators=8,n_jobs=THREADS,device='cpu',use_amp=False,
                    model_path=checkpoint,allow_auto_download=False,random_state=SEED)
                t=time.perf_counter();model.fit(a,target[:,j]);fit=time.perf_counter()-t
                t=time.perf_counter();pred[:,j]=model.predict(b);infer=time.perf_counter()-t
                assert np.isfinite(pred[:,j]).all()
                timing.append([j,fit,infer]);
                np.savez_compressed(partial,prediction=pred,test_indices=te,train_indices=tr,timing=np.asarray(timing))
                print(json.dumps({'view':view,'fold':f,'target_done':j+1,'target_total':target.shape[1],
                    'fit_s':round(fit,3),'predict_s':round(infer,3),'threads':THREADS,
                    'peak_rss_MB':round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1)}),flush=True)
                del model
            save={'test_indices':te,'train_indices':tr,'timing':np.asarray(timing),'latent_prediction':pred}
            if latent:
                recon=ypca.inverse_transform(pred)
                V=ypca.components_.T;projector=np.eye(50)-V@V.T
                resid=(T-ypca.mean_)@projector
                ridge=Ridge(alpha=100.).fit(xr,resid)
                rhat=ridge.predict(xt)@projector
                save.update(prediction_no_residual=recon*sigma+mu,
                    prediction_ridge_residual=(recon+rhat)*sigma+mu,
                    target_mean=mu,target_scale=sigma,target_components=ypca.components_,
                    target_explained_variance_ratio=ypca.explained_variance_ratio_,
                    residual_prediction=rhat*sigma)
                if view.startswith('rank_pca'):
                    save['input_explained_variance_ratio']=xpca.explained_variance_ratio_
            else:save['prediction_native']=pred
            np.savez_compressed(complete,**save)
            for name,value in save.items():
                if name.startswith('prediction_'):
                    print('FOLD_METRIC '+json.dumps({'view':view,'variant':name,'fold':f,**assess(Y[te],value)}),flush=True)
            summarize(out,x,y,Y,fold)
    predictions=summarize(out,x,y,Y,fold)
    paired_uncertainty(out,Y,predictions)
    print('COMPLETE '+json.dumps([{'model':k,**assess(Y,v)} for k,v in predictions.items()]),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--views',nargs='*',choices=VIEWS)
    main(parser.parse_args())
