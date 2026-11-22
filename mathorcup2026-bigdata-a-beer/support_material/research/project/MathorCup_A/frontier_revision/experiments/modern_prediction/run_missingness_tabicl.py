"""Frozen TabICLv2 raw231/PCA16 missingness stress, identical legacy masks.

The only engineering choice resolved by a pilot is numerical equivalence of
batched queries and KV caching; the pilot computes no predictive score.
All targets, folds, seeds, corruption rules and comparisons are fixed here.
"""
from pathlib import Path
import os
THREADS=int(os.environ.get('BEER_MODERN_THREADS','6'))
for _k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_k]=str(THREADS)
import argparse
import hashlib
import importlib.metadata
import json
import resource
import time
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from tabicl import TabICLRegressor
from threadpoolctl import threadpool_limits
from run_tabicl_benchmark import data, assess, BASE, SEED
from audit_tabicl_query_batching import audit_preprocessing

HERE=Path(__file__).resolve().parent
OUT=HERE/'missingness'
SCENARIOS=('derived_only_missing','MCAR_10_strict','MCAR_30_strict','MCAR_50_strict')
DERIVED=('kcalperc','esters_sum','aroma_hops_sum','acids_sum','sulfur_sum')
CHECKPOINT=HERE/'models/tabicl-regressor-v2-20260212.ckpt'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    temp.replace(path)


def make_cases(Xtest, fields, fold):
    derived=[fields.index(f) for f in DERIVED]
    cases={'derived_only_missing':Xtest.copy()}
    cases['derived_only_missing'][:,derived]=np.nan
    for level in (10,30,50):
        a=Xtest.copy()
        rng=np.random.default_rng(SEED+1000*fold+level)
        a[rng.random(a.shape)<level/100]=np.nan
        a[:,derived]=np.nan
        cases[f'MCAR_{level}_strict']=a
    return cases


def encoding(Ytrain):
    mean=Ytrain.mean(0)
    scale=np.maximum(Ytrain.std(0,ddof=1),1e-12)
    pca=PCA(n_components=16,svd_solver='full').fit((Ytrain-mean)/scale)
    return mean,scale,pca,pca.transform((Ytrain-mean)/scale)


def model(cache=False):
    return TabICLRegressor(n_estimators=8,n_jobs=THREADS,device='cpu',use_amp=False,
        model_path=CHECKPOINT,allow_auto_download=False,random_state=SEED,kv_cache=cache)


def numerical_probe(x,X,Y,fold,preprocessing_audit):
    """Only compares equivalent executions, never held-out prediction quality."""
    path=OUT/'batch_cache_probe.json'
    if path.exists():
        return json.loads(path.read_text())
    tr=np.flatnonzero(fold!=0);te=np.flatnonzero(fold==0)
    cases=make_cases(X[te],x.columns[3:].tolist(),0)
    mean,scale,pca,target=encoding(Y[tr])
    single=model(False)
    start=time.perf_counter();single.fit(X[tr],target[:,0]);uncached_fit=time.perf_counter()-start
    reference=[];separate_times=[]
    for s in SCENARIOS:
        start=time.perf_counter();reference.append(single.predict(cases[s]));separate_times.append(time.perf_counter()-start)
    reference=np.stack(reference)
    joined=np.concatenate([cases[s] for s in SCENARIOS])
    start=time.perf_counter();joined_prediction=single.predict(joined).reshape(4,len(te));joined_seconds=time.perf_counter()-start
    # Original 50-row benchmark reproduction verifies the same statistical model.
    start=time.perf_counter();original=single.predict(X[te]);original_seconds=time.perf_counter()-start
    original_file=HERE/'checkpoints/raw231_y16_fold0_complete.npz'
    baseline_latent=np.load(original_file)['latent_prediction'][:,0]
    original_error=float(np.max(np.abs(original-baseline_latent)))
    if not np.allclose(original,baseline_latent,rtol=2e-5,atol=2e-5):
        raise AssertionError('Uncached probe does not reproduce the frozen raw231/PCA16 benchmark.')
    concat_error=float(np.max(np.abs(reference-joined_prediction)))
    concat_ok=bool(preprocessing_audit['all_folds_equal'] and
                   np.allclose(reference,joined_prediction,rtol=2e-5,atol=2e-5))
    del single
    cached=model(True)
    start=time.perf_counter();cached.fit(X[tr],target[:,0]);cached_fit=time.perf_counter()-start
    start=time.perf_counter()
    if concat_ok:
        cached_prediction=cached.predict(joined).reshape(4,len(te))
    else:
        cached_prediction=np.stack([cached.predict(cases[s]) for s in SCENARIOS])
    cached_seconds=time.perf_counter()-start
    cache_error=float(np.max(np.abs(reference-cached_prediction)))
    cache_ok=bool(np.allclose(reference,cached_prediction,rtol=2e-5,atol=2e-5))
    del cached
    resolved=dict(concatenate_scenarios=concat_ok,kv_cache=cache_ok)
    protocol=dict(fold=0,latent_target=0,heldout_scores_computed=False,
        comparison='Four independent 50-row scenario calls versus one 200-row query, and cached versus uncached execution',
        tolerance=dict(rtol=2e-5,atol=2e-5),concat_max_abs_difference=concat_error,
        all_folds_preprocessing_exact_equal=preprocessing_audit['all_folds_equal'],
        cache_max_abs_difference=cache_error,original_benchmark_max_abs_difference=original_error,
        uncached_fit_seconds=uncached_fit,uncached_separate_predict_seconds=separate_times,
        uncached_joined_predict_seconds=joined_seconds,original_predict_seconds=original_seconds,
        cached_fit_seconds=cached_fit,cached_predict_seconds=cached_seconds,resolved=resolved,
        source_semantics='Installed model defaults embed_with_test=False; test row queries attend to training context. Numerical check gates batching/cache use.',
        effective_seconds_per_16target_fold_estimate=16*((cached_fit+cached_seconds) if cache_ok else
            (uncached_fit+joined_seconds if concat_ok else uncached_fit+sum(separate_times))))
    atomic_json(path,protocol)
    np.savez_compressed(OUT/'batch_cache_probe_arrays.npz',uncached_separate=reference,
                        uncached_joined=joined_prediction,cached=cached_prediction,
                        original=original,baseline_latent=baseline_latent)
    print('PROBE '+json.dumps(protocol),flush=True)
    return protocol


def grades(values,reference):
    cuts=np.quantile(reference,np.arange(1,7)/7,axis=0)
    return np.stack([np.searchsorted(cuts[:,j],values[:,j],side='right') for j in range(50)],axis=1)


def summarize(x,y,Y,fold,final=False):
    paths=[OUT/'checkpoints'/f'fold{f}_complete.npz' for f in range(5)]
    foldmetrics=[]
    for f,path in enumerate(paths):
        if path.exists():
            a=np.load(path);te=a['test_indices'];tr=a['train_indices']
            truth=grades(Y[te],Y[tr])
            for s in SCENARIOS:
                pred=a[s];err=np.abs(grades(pred,Y[tr])-truth)
                foldmetrics.append(dict(fold=f,scenario=s,**assess(Y[te],pred),
                    core30_grade_mae=float(err[:,:30].mean()),core30_grade_exact=float((err[:,:30]==0).mean()),
                    core30_grade_within_one=float((err[:,:30]<=1).mean())))
    pd.DataFrame(foldmetrics).to_csv(OUT/'fold_metrics.csv',index=False)
    state=dict(completed_folds=sum(p.exists() for p in paths),total_folds=5,
               timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    if not all(path.exists() for path in paths):
        atomic_json(OUT/'status.json',state)
        return
    tabicl={s:np.empty_like(Y) for s in SCENARIOS}
    for path in paths:
        a=np.load(path)
        for s in SCENARIOS:tabicl[s][a['test_indices']]=a[s]
    main_oof=np.load(HERE/'oof_arrays.npz')
    tabicl['original']=main_oof['raw231_y16_no_residual']
    baseline_path=BASE/'A_work/audit/model_validation'
    baselines={name:np.load(baseline_path/file) for name,file in
        [('old_knn','strict_knn_missingness_oof.npz'),('old_median','strict_missingness_oof.npz')]}
    for a in baselines.values():
        assert np.allclose(Y,a['Y']) and np.array_equal(x.beer,a['beer']) and np.array_equal(fold,a['fold'])
    predictions={'tabicl_raw231_y16':tabicl}
    for name,a in baselines.items():predictions[name]={s:a[s] for s in ('original',)+SCENARIOS}
    truth=np.empty_like(Y,dtype=int)
    for f in range(5):truth[fold==f]=grades(Y[fold==f],Y[fold!=f])
    metrics=[];dimensions=[];cis=[]
    rng=np.random.default_rng(SEED+79011)
    boot=rng.integers(0,len(Y),size=(2000,len(Y)))
    for name,scenarios in predictions.items():
        for s,pred in scenarios.items():
            g=np.empty_like(Y,dtype=int)
            for f in range(5):g[fold==f]=grades(pred[fold==f],Y[fold!=f])
            err=np.abs(g-truth);r=r2_score(Y,pred,multioutput='raw_values')
            for group,ix in [('all50',np.arange(50)),('core30',np.arange(30))]:
                metrics.append(dict(model=name,scenario=s,targets=group,r2=float(r[ix].mean()),
                    grade_mae=float(err[:,ix].mean()),grade_exact=float((err[:,ix]==0).mean()),
                    grade_within_one=float((err[:,ix]<=1).mean())))
            for j,t in enumerate(y.columns[3:]):
                dimensions.append(dict(model=name,scenario=s,target=t,core30=j<30,r2=float(r[j]),
                    grade_mae=float(err[:,j].mean()),grade_exact=float((err[:,j]==0).mean()),
                    grade_within_one=float((err[:,j]<=1).mean())))
            if name=='tabicl_raw231_y16':
                pd.concat([pd.DataFrame(dict(beer=x.beer,fold=fold)),pd.DataFrame(g,columns=y.columns[3:])],axis=1).to_csv(
                    OUT/f'{s}_relative_grades_0_6.csv',index=False)
                ref=predictions['old_knn'][s]
                refg=np.empty_like(Y,dtype=int)
                for f in range(5):refg[fold==f]=grades(ref[fold==f],Y[fold!=f])
                refe=np.abs(refg-truth)
                rdiff=r-r2_score(Y,ref,multioutput='raw_values')
                rboot=[]
                for ids in boot:
                    yy=Y[ids];den=((yy-yy.mean(0))**2).sum(0)
                    rr=(((yy-ref[ids])**2).sum(0)-((yy-pred[ids])**2).sum(0))/den
                    rboot.append([rr.mean(),rr[:30].mean()])
                rboot=np.asarray(rboot)
                for gidx,(group,ix) in enumerate([('all50',np.arange(50)),('core30',np.arange(30))]):
                    for metric,delta in [('grade_mae',err[:,ix].mean(1)-refe[:,ix].mean(1)),
                         ('grade_exact',(err[:,ix]==0).mean(1)-(refe[:,ix]==0).mean(1)),
                         ('grade_within_one',(err[:,ix]<=1).mean(1)-(refe[:,ix]<=1).mean(1))]:
                        samples=delta[boot].mean(1)
                        cis.append(dict(scenario=s,targets=group,metric=metric,delta=float(delta.mean()),
                            ci025=float(np.quantile(samples,.025)),ci975=float(np.quantile(samples,.975)),reference='old_knn'))
                    cis.append(dict(scenario=s,targets=group,metric='r2',delta=float(rdiff[ix].mean()),
                        ci025=float(np.quantile(rboot[:,gidx],.025)),ci975=float(np.quantile(rboot[:,gidx],.975)),reference='old_knn'))
    pd.DataFrame(metrics).to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(dimensions).to_csv(OUT/'dimension_metrics.csv',index=False)
    pd.DataFrame(cis).to_csv(OUT/'paired_ci_vs_knn.csv',index=False)
    np.savez_compressed(OUT/'oof_predictions.npz',Y=Y,beer=x.beer.to_numpy(),fold=fold,targets=y.columns[3:].to_numpy(str),
                        **{f'{m}__{s}':pred for m,scenarios in predictions.items() for s,pred in scenarios.items()})
    state['metrics']=metrics
    state['bootstrap_note']='2000 paired beer resamples of fixed OOF predictions; no refit or multiplicity correction'
    state['complete']=bool(final)
    atomic_json(OUT/'status.json',state)


def main(args):
    OUT.mkdir(exist_ok=True);(OUT/'checkpoints').mkdir(exist_ok=True)
    torch.set_num_threads(THREADS);torch.manual_seed(SEED)
    x,y,X,Y,fold=data()
    source=HERE.parent/'missingness_augmentation/run_mask_augmentation.py'
    files={path:sha(path) for path in [source,BASE/'A_work/audit/model_validation/strict_knn_missingness.py',
        BASE/'A_work/audit/model_validation/strict_missingness.py']}
    spec=dict(model='raw231_y16_no_residual',parameters=dict(n_estimators=8,n_jobs=THREADS,
        use_amp=False,device='cpu',random_state=SEED,allow_auto_download=False),
        scenarios=list(SCENARIOS),derived_hidden=list(DERIVED),training_corruption=False,
        test_mask_seed='20261023 + 1000*fold + level',target_pca_rank=16,
        target_standardization='Train-fold mean and sample sd; PCA train-fold only',
        mask_reference_sha256={str(k):v for k,v in files.items()},
        data_sha256={n:sha(BASE/'data/A'/n) for n in ('1.csv','4.csv')},
        folds_sha256=sha(BASE/'A_work/q1/fold_assignments.csv'),checkpoint_sha256=sha(CHECKPOINT),
        versions={n:importlib.metadata.version(n) for n in ('tabicl','torch','numpy','scikit-learn')},
        batch_cache_resolution='All-fold preprocessing equality audit plus no-score numerical model probe with fixed rtol=atol=2e-5; fallback separate/uncached on mismatch',
        implementation_sha256={name:sha(HERE/name) for name in ['run_missingness_tabicl.py','audit_tabicl_query_batching.py']},
        previous_complete_input_selection='Candidate chosen for missingness follow-up after internal complete-input comparison; disclosed adaptive research')
    specpath=OUT/'experiment_spec.json'
    if specpath.exists():
        assert json.loads(specpath.read_text())==spec,'Frozen missingness protocol changed.'
    else:atomic_json(specpath,spec)
    # Save exact mask tensors and per-scenario hashes before any predictions.
    maskrecords=[]
    for f in range(5):
        tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f)
        cases=make_cases(X[te],x.columns[3:].tolist(),f)
        masks={s:np.isnan(cases[s]) for s in SCENARIOS}
        np.savez_compressed(OUT/'checkpoints'/f'fold{f}_masks.npz',train_indices=tr,test_indices=te,**masks)
        for s in SCENARIOS:
            maskrecords.append(dict(fold=f,scenario=s,missing_fraction=float(masks[s].mean()),
                                    mask_sha256=hashlib.sha256(masks[s].tobytes()).hexdigest()))
    pd.DataFrame(maskrecords).to_csv(OUT/'mask_audit.csv',index=False)
    with threadpool_limits(limits=THREADS):
        preprocessing_audit=audit_preprocessing(X,x.columns[3:].tolist(),fold,OUT/'preprocessing_batch_audit.json')
        probe=numerical_probe(x,X,Y,fold,preprocessing_audit)
        if args.probe_only:return
        cache=probe['resolved']['kv_cache'];join=probe['resolved']['concatenate_scenarios']
        for f in range(5):
            complete=OUT/'checkpoints'/f'fold{f}_complete.npz'
            if complete.exists():continue
            tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f)
            cases=make_cases(X[te],x.columns[3:].tolist(),f)
            joined=np.concatenate([cases[s] for s in SCENARIOS])
            mean,scale,pca,target=encoding(Y[tr])
            original=np.load(HERE/'checkpoints'/f'raw231_y16_fold{f}_complete.npz')
            assert np.allclose(pca.components_,original['target_components'],atol=1e-12,rtol=1e-12)
            latent=np.full((4,len(te),16),np.nan);timing=[]
            partial=OUT/'checkpoints'/f'fold{f}_partial.npz'
            if partial.exists():
                prev=np.load(partial);latent=prev['latent_prediction'];timing=prev['timing'].tolist()
                assert np.array_equal(prev['test_indices'],te)
            for j in range(16):
                if np.isfinite(latent[:,:,j]).all():continue
                estimator=model(cache)
                start=time.perf_counter();estimator.fit(X[tr],target[:,j]);fit=time.perf_counter()-start
                start=time.perf_counter()
                if join:latent[:,:,j]=estimator.predict(joined).reshape(4,len(te))
                else:latent[:,:,j]=np.stack([estimator.predict(cases[s]) for s in SCENARIOS])
                elapsed=time.perf_counter()-start
                assert np.isfinite(latent[:,:,j]).all()
                timing.append([j,fit,elapsed])
                np.savez_compressed(partial,latent_prediction=latent,timing=np.asarray(timing),
                                    train_indices=tr,test_indices=te)
                print(json.dumps(dict(fold=f,target_done=j+1,target_total=16,scenarios=4,
                    fit_seconds=fit,predict_seconds=elapsed,cache=cache,concatenate=join,
                    peak_rss_MB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
                del estimator
            save=dict(latent_prediction=latent,timing=np.asarray(timing),train_indices=tr,test_indices=te,
                      target_mean=mean,target_scale=scale,target_components=pca.components_,target_pca_mean=pca.mean_)
            for i,s in enumerate(SCENARIOS):
                pred=pca.inverse_transform(latent[i])*scale+mean;save[s]=pred
                print('FOLD_METRIC '+json.dumps(dict(fold=f,scenario=s,**assess(Y[te],pred))),flush=True)
            np.savez_compressed(complete,**save)
            summarize(x,y,Y,fold)
    summarize(x,y,Y,fold,final=True)
    print('COMPLETE '+json.dumps(dict(completed_folds=5,scenarios=list(SCENARIOS))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--probe-only',action='store_true')
    main(parser.parse_args())
