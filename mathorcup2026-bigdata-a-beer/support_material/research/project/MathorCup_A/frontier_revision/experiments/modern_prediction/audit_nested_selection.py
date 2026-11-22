#!/usr/bin/env python3
"""Independently audit a completed 5x3 selection experiment and summarize it.

This program never trains or calls a predictive model. It does not import the
experiment runner, scikit-learn, PyTorch, or TabICL. All selection scores are
recomputed from inner predictions and outer-training labels. Saved transforms
are checked by arithmetic and eigenspace identities, not by refitting models.

Incomplete/inconsistent experiments produce no statistical result files.
Only after all gates pass are results published into a new output directory.
Paired bootstrap intervals condition on the existing OOF predictions; neither
models nor configuration choices are refitted in bootstrap replicates.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

for _thread_name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_thread_name]='1'

import numpy as np
import pandas as pd
from scipy.special import ndtri

HERE=Path(__file__).resolve().parent
SEED=20261023
BOOTSTRAP_SEED=SEED+2026
NESTED='tabicl_nested_selection'
SIX='old_nested_ensemble'
FIXED='raw231_y16_no_residual'
NATIVE='raw231_y50_native'
VIEWS=('rank_pca64_y16','raw231_y16','raw231_y50')
VIEW_KEYS={
    'rank_pca64_y16':('prediction_no_residual','prediction_ridge_residual'),
    'raw231_y16':('prediction_no_residual','prediction_ridge_residual'),
    'raw231_y50':('prediction_native',),
}
CANDIDATES=tuple(view+'_'+key.removeprefix('prediction_')
                 for view in VIEWS for key in VIEW_KEYS[view])
TARGETS={'all50':np.arange(50),'core30':np.arange(30),'other20':np.arange(30,50)}
ALIASES={SIX:'六模型集成',FIXED:'固定Y16配置',NESTED:'训练内选模流程',
         'rank_pca64_y16_no_residual':'秩/PCA64→Y16',
         'rank_pca64_y16_ridge_residual':'秩/PCA64→Y16＋残差',
         'raw231_y16_no_residual':'固定Y16配置',
         'raw231_y16_ridge_residual':'原始231→Y16＋残差',
         'raw231_y50_native':'原始231→50输出'}
INPUT_HASHES={
    '1.csv':'419849227a02e0c792cd405e6865035b7f58af1a92efa362dbb68a7ca2882f69',
    '4.csv':'43a7b38682ed81ce64d97ebc1874c09dffac023ade4877f34e3c83ef7d42dc1c',
    'fold_assignments.csv':'19a765548313818741b4425a3d5f0745d807b9f1c1abbdd73d1c0f1f52c90f98',
}
HISTORICAL_POINTS={
    SIX:{'all50':0.26371907467338696,'core30':0.22929121976146177},
    FIXED:{'all50':0.2798519385904179,'core30':0.24277503695049749},
    NATIVE:{'all50':0.2863038138321149,'core30':0.2366463254294722},
}


class AuditFailure(RuntimeError):
    pass


class IncompleteExperiment(AuditFailure):
    pass


def require(condition,label):
    if not condition:
        raise AuditFailure(label)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def exact(actual,expected,label):
    actual,expected=np.asarray(actual),np.asarray(expected)
    if actual.dtype.kind in 'OUS' or expected.dtype.kind in 'OUS':
        same=np.array_equal(actual,expected)
    else:
        same=np.array_equal(actual,expected,equal_nan=True)
    require(same,'Exact identity mismatch: '+label)


def close(actual,expected,label,atol=1e-11,rtol=1e-11):
    actual,expected=np.asarray(actual),np.asarray(expected)
    require(actual.shape==expected.shape,'Shape mismatch: '+label)
    require(np.isfinite(actual).all() and np.isfinite(expected).all(),
            'Nonfinite value: '+label)
    error=float(np.max(np.abs(actual-expected))) if actual.size else 0.0
    require(np.allclose(actual,expected,atol=atol,rtol=rtol),
            f'Numerical identity mismatch: {label}; max_abs={error:.17g}')
    return error


def finite_shape(array,shape,label):
    require(np.asarray(array).shape==shape and np.isfinite(array).all(),
            f'Incomplete/nonfinite array {label}; expected{shape}')


def scalar(saved,key):
    value=np.asarray(saved[key])
    require(value.size==1,f'{key} is not scalar')
    return value.item()


class Inputs:
    def __init__(self):
        self.records={}

    def record(self,path):
        path=Path(path).resolve()
        require(path.is_file(),'Missing input: '+str(path))
        key=str(path)
        if key not in self.records:
            self.records[key]={'sha256':sha(path),'bytes':path.stat().st_size}
        return self.records[key]['sha256']

    def json(self,path):
        self.record(path)
        return json.loads(Path(path).read_text())

    def npz(self,path):
        self.record(path)
        with np.load(path,allow_pickle=False) as archive:
            return {key:archive[key] for key in archive.files}

    def csv(self,path):
        self.record(path)
        return pd.read_csv(path)

    def recheck(self):
        for path,record in self.records.items():
            require(sha(path)==record['sha256'],'Input changed during audit: '+path)


def kfold_indices(n,splits,seed):
    """Independent NumPy reconstruction of shuffled sklearn KFold index sets."""
    shuffled=np.arange(n)
    np.random.RandomState(seed).shuffle(shuffled)
    sizes=np.full(splits,n//splits,dtype=int)
    sizes[:n%splits]+=1
    cursor=0
    result=[]
    for size in sizes:
        validation=np.zeros(n,dtype=bool)
        validation[shuffled[cursor:cursor+size]]=True
        result.append((np.flatnonzero(~validation),np.flatnonzero(validation)))
        cursor+=size
    return result


def r2_by_target(Y,prediction):
    finite_shape(prediction,Y.shape,'prediction for score')
    sst=np.square(Y-Y.mean(axis=0)).sum(axis=0)
    require(np.all(sst>0),'Constant target in R2 evaluation')
    return 1-np.square(Y-prediction).sum(axis=0)/sst


def assess(Y,prediction):
    values=r2_by_target(Y,prediction)
    return {'core30_r2':float(values[:30].mean()),'macro_r2':float(values.mean()),
            'aroma15_r2':float(values[:15].mean()),
            'flavour15_r2':float(values[15:30].mean()),
            'other20_r2':float(values[30:].mean())}


def score_close(record,expected,label):
    require(set(expected)<=set(record),'Missing score fields: '+label)
    return max(close(float(record[k]),np.asarray(v),label+'/'+k,atol=1e-12,rtol=1e-12)
               for k,v in expected.items())


def required_complete_paths(nested):
    paths=[nested/'experiment_spec.json',nested/'results.json',
           nested/'nested_oof.npz',nested/'inner_candidate_scores.csv']
    for outer in range(5):
        directory=nested/f'outer_{outer}'
        paths.extend(directory/name for name in ('selection.json','inner_oof.npz','selected_outer.npz'))
        for inner in range(3):
            split=directory/f'inner_{inner}'
            paths.append(split/'split_complete.json')
            paths.extend(split/f'{view}_complete.npz' for view in VIEWS)
    return paths


def completeness_gate(nested):
    missing=[str(path) for path in required_complete_paths(nested) if not path.is_file()]
    if missing:
        raise IncompleteExperiment(f'{len(missing)} required completed outputs absent; first: '+', '.join(missing[:4]))
    results=json.loads((nested/'results.json').read_text())
    if results.get('complete') is not True:
        raise IncompleteExperiment('results.json does not declare complete=true')
    actual=set(nested.glob('outer_*/inner_*/*_complete.npz'))
    expected={nested/f'outer_{o}/inner_{i}/{v}_complete.npz'
              for o in range(5) for i in range(3) for v in VIEWS}
    require(actual==expected,'Complete-file set differs from the required45 views')
    require(len(list(nested.glob('outer_*/inner_*/split_complete.json')))==15,
            'Exactly15 split completion markers required')
    require(len(list(nested.glob('outer_*/selection.json')))==5,
            'Exactly5 outer selections required')


def load_reference(project,modern,inputs):
    x=inputs.csv(project/'data/A/1.csv').sort_values('beer').reset_index(drop=True)
    y=inputs.csv(project/'data/A/4.csv')
    assignment=inputs.csv(project/'A_work/q1/fold_assignments.csv')
    for frame,label in ((x,'chemistry'),(y,'sensory'),(assignment,'folds')):
        require(len(frame)==250 and frame.beer.notna().all() and frame.beer.is_unique,
                'Invalid beer identities in '+label)
        require(set(frame.beer)==set(x.beer),'Beer set mismatch in '+label)
    y=y.set_index('beer').loc[x.beer].reset_index()
    folds=assignment.set_index('beer').loc[x.beer,'fold'].to_numpy()
    exact(x.tasting_category_fine.to_numpy(),y.tasting_category_fine.to_numpy(),
          'style alignment by beer')
    X,Y=x.iloc[:,3:].to_numpy(float),y.iloc[:,3:].to_numpy(float)
    finite_shape(Y,(250,50),'observed Y')
    require(X.shape==(250,231) and not np.isinf(X).any(),'Invalid observed X')
    require(not np.all(np.isnan(X),axis=1).any(),'Fully missing observed chemistry row')
    require(set(folds)==set(range(5)),'Invalid outer fold labels')
    for f,(_,te) in enumerate(kfold_indices(250,5,SEED)):
        exact(np.flatnonzero(folds==f),te,f'original outer split{f}')
    for key,path in [('1.csv',project/'data/A/1.csv'),('4.csv',project/'data/A/4.csv'),
                     ('fold_assignments.csv',project/'A_work/q1/fold_assignments.csv')]:
        require(inputs.record(path)==INPUT_HASHES[key],'Official input hash mismatch: '+key)
    targets=y.columns[3:].to_numpy(str)
    styles=x.tasting_category_fine.astype(str).to_numpy()
    require(len(np.unique(styles))==22 and x.tasting_category_fine.notna().all(),
            'Expected22 original fine styles')
    archive=inputs.npz(modern/'oof_arrays.npz')
    for key,value in [('Y',Y),('beer',x.beer.to_numpy()),('fold',folds),('targets',targets)]:
        exact(archive[key],value,'original modern OOF '+key)
    for model in (*CANDIDATES,SIX):
        finite_shape(archive[model],Y.shape,'original '+model)
    legacy=inputs.npz(project/'A_work/q1/oof_arrays.npz')
    exact(legacy['Y'],Y,'legacy original Y')
    exact(legacy['nested_ensemble'],archive[SIX],'six-model copies')
    metric_table=inputs.csv(modern/'model_comparison.csv')
    require(not metric_table.model.duplicated().any(),'Duplicate fixed-model metric rows')
    historical_checks=[]
    for model,expected in HISTORICAL_POINTS.items():
        values=r2_by_target(Y,archive[model])
        metric_row=metric_table.loc[metric_table.model==model]
        require(len(metric_row)==1,'Missing fixed-model metric row: '+model)
        score_close(metric_row.iloc[0].to_dict(),assess(Y,archive[model]),
                    'original fixed-model metrics versus independently scored OOF '+model)
        for group,score in expected.items():
            actual=float(values[TARGETS[group]].mean())
            # The independent score must match its actual archive's metric CSV
            # to1e-12 above. Historical printed scores are a separate guard;
            # permit sub-rounding repeat-run floating-point differences only.
            close(actual,np.asarray(score),'historical fixed-reference point '+model+'/'+group,
                  atol=5e-6,rtol=0)
            historical_checks.append(dict(model=model,targets=group,actual=actual,
                historical_value=score,absolute_difference=abs(actual-score),
                historical_tolerance=5e-6,actual_archive_metric_tolerance=1e-12))
    return dict(X=X,Y=Y,beer=x.beer.to_numpy(),fold=folds,targets=targets,
                styles=styles,modern=archive,historical_reference_checks=historical_checks)


def rank_training_matrix(X):
    """Arithmetic reconstruction of the existing median/midrank scaling rule."""
    filled=X.copy()
    for j in range(X.shape[1]):
        observed=~np.isnan(filled[:,j])
        median=float(np.median(filled[observed,j])) if observed.any() else 0.0
        filled[~observed,j]=median
    scores=np.empty_like(filled)
    for j in range(X.shape[1]):
        ordered=np.sort(filled[:,j])
        lo=np.searchsorted(ordered,filled[:,j],side='left')
        hi=np.searchsorted(ordered,filled[:,j],side='right')
        scores[:,j]=ndtri((lo+hi+1.)/(2*(len(filled)+1.)))
    mean=scores.mean(axis=0)
    variance=scores.var(axis=0,ddof=0)
    error_bound=len(scores)*np.finfo(float).eps*variance+(len(scores)*mean*np.finfo(float).eps)**2
    scale=np.sqrt(variance)
    scale[variance<=error_bound]=1.0
    return (scores-mean)/scale


def verify_saved_pca(training,components,center,ratios,label):
    """Check given PCA parameters against training-only covariance identities."""
    k,d=components.shape
    require(training.shape[1]==d,'PCA dimension mismatch: '+label)
    errors={}
    errors['center']=close(center,training.mean(axis=0),label+' center',atol=2e-11)
    errors['orthogonality']=close(components@components.T,np.eye(k),label+' orthogonality',atol=2e-10)
    centered=training-training.mean(axis=0)
    covariance=centered.T@centered/(len(training)-1)
    projected=components@covariance@components.T
    eigenvalues=np.linalg.eigvalsh(covariance)[::-1]
    errors['eigen_equation']=close(covariance@components.T,
        components.T*eigenvalues[:k][None,:],label+' covariance eigen-equation',atol=2e-9,rtol=2e-9)
    errors['top_variance']=close(np.diag(projected),eigenvalues[:k],label+' leading variances',atol=2e-9)
    errors['variance_ratio']=close(ratios,eigenvalues[:k]/np.trace(covariance),label+' explained ratios',atol=2e-10)
    return errors


def expected_inner(data,outer,inner):
    outer_train=np.flatnonzero(data['fold']!=outer)
    outer_test=np.flatnonzero(data['fold']==outer)
    trpos,tepos=kfold_indices(200,3,SEED+1000+outer)[inner]
    train,test=outer_train[trpos],outer_train[tepos]
    require(not np.intersect1d(train,test).size,'Inner training/validation overlap')
    require(not np.intersect1d(outer_test,np.r_[train,test]).size,'Outer-test leakage into inner split')
    exact(np.sort(np.r_[train,test]),outer_train,'inner split union')
    return dict(train_indices=train,test_indices=test,outer_train_indices=outer_train,
        outer_test_indices=outer_test,inner_train_positions=trpos,inner_test_positions=tepos)


def audit_complete(args):
    completeness_gate(args.nested_dir)
    inputs=Inputs()
    spec=inputs.json(args.nested_dir/'experiment_spec.json')
    result=inputs.json(args.nested_dir/'results.json')
    spec_id=json_digest(spec)
    protocol=spec['protocol']
    for key,expected in [('outer_folds',5),('outer_seed',SEED),('inner_folds',3),
        ('outer_training_rows',200),('total_inner_splits',15),
        ('scalar_components_per_inner_split',82),('total_scalar_components',1230),
        ('input_pca_rank',64),('target_pca_rank',16),('residual_ridge_alpha',100.0)]:
        require(protocol.get(key)==expected,'Protocol mismatch: '+key)
    require(protocol['inner_seed_rule']=='20261023 + 1000 + outer_fold (0..4)',
            'Unregistered inner seed rule')
    require(protocol['outer_labels_used_in_candidate_selection'] is False,
            'Protocol permits outer labels in selection')
    require(protocol['fixed_deployment_y16_is_changed'] is False,
            'Protocol changes fixed deployment Y16')
    require(protocol['candidates']==list(CANDIDATES) and protocol['base_views']==list(VIEWS),
            'Candidate/view family differs from the five registered choices')
    require(protocol['shuffle_inner'] is True,'Inner shuffling differs')
    for key,value in [('n_estimators',8),('random_state',SEED),('device','cpu')]:
        require(protocol['tabicl'][key]==value,'TabICL protocol mismatch '+key)
    for key in ['use_amp','allow_auto_download','kv_cache']:
        require(protocol['tabicl'][key] is False,'TabICL protocol mismatch '+key)
    data=load_reference(args.project_dir,args.modern_dir,inputs)
    original_spec=inputs.json(args.modern_dir/'experiment_spec.json')
    provenance={
        '1.csv':args.project_dir/'data/A/1.csv','4.csv':args.project_dir/'data/A/4.csv',
        'fold_assignments.csv':args.project_dir/'A_work/q1/fold_assignments.csv',
        'original_oof_arrays.npz':args.modern_dir/'oof_arrays.npz',
        'original_experiment_spec.json':args.modern_dir/'experiment_spec.json',
        'public_checkpoint':args.modern_dir/'models/tabicl-regressor-v2-20260212.ckpt'}
    for key,path in provenance.items():
        require(inputs.record(path)==spec['inputs_sha256'][key],'Frozen input hash differs: '+key)
    require(spec['inputs_sha256']['public_checkpoint']==original_spec['checkpoint_sha256'],
            'Nested checkpoint differs from original fixed candidates')
    for key,path in [('this_script',args.runner_script),
        ('ChemicalPreprocessor_source',args.project_dir/'A_work/q1/sensory_model.py'),
        ('original_run_tabicl_benchmark.py',args.modern_dir/'run_tabicl_benchmark.py')]:
        require(inputs.record(path)==spec['implementation_sha256'][key],
                'Registered source hash differs: '+key)
    for key,value in [('complete',True),('spec_id',spec_id),('outer_folds',5),
        ('inner_splits_completed',15),('scalar_components_completed',1230),
        ('fixed_raw231_y16_deployment_changed',False)]:
        require(result.get(key)==value,'Final result metadata mismatch: '+key)
    require(len(result['selections'])==5,'Final result requires5 selections')
    result_selections={r['outer_fold']:r for r in result['selections']}
    require(set(result_selections)==set(range(5)),'Duplicate/missing final outer selections')
    original_outer={}
    for outer in range(5):
        tr,te=np.flatnonzero(data['fold']!=outer),np.flatnonzero(data['fold']==outer)
        for view in VIEWS:
            rel=f'checkpoints/{view}_fold{outer}_complete.npz'
            path=args.modern_dir/rel
            saved=inputs.npz(path)
            require(inputs.record(path)==spec['inputs_sha256']['original_outer_complete_files'][rel],
                    'Original outer cache hash mismatch: '+rel)
            exact(saved['train_indices'],tr,rel+' train');exact(saved['test_indices'],te,rel+' test')
            for key in VIEW_KEYS[view]:
                candidate=view+'_'+key.removeprefix('prediction_')
                finite_shape(saved[key],(50,50),rel+'/'+key)
                exact(saved[key],data['modern'][candidate][te],rel+' versus original OOF')
                original_outer[outer,candidate]=saved[key]
    assembled=np.full((250,50),np.nan)
    outer_coverage=np.zeros(250,dtype=int)
    selected_rows=np.full(250,'',dtype='U80')
    selected_records=[];inner_scores=[];split_records=[];transform_checks=[]
    maximum_score_difference=0.0
    scalar_count=0
    for outer in range(5):
        tr,te=np.flatnonzero(data['fold']!=outer),np.flatnonzero(data['fold']==outer)
        pool={candidate:np.full((200,50),np.nan) for candidate in CANDIDATES}
        coverage=np.zeros(200,dtype=int);hashes={}
        for inner in range(3):
            indices=expected_inner(data,outer,inner)
            directory=args.nested_dir/f'outer_{outer}/inner_{inner}'
            marker=inputs.json(directory/'split_complete.json')
            require(marker.get('complete') is True and marker.get('scalar_components')==82,
                    f'Invalid split completion marker {outer}/{inner}')
            require(marker['outer_fold']==outer and marker['inner_fold']==inner and marker['spec_id']==spec_id,
                    'Split marker identity mismatch')
            positions=indices['inner_test_positions'];coverage[positions]+=1
            saved_latent=[]
            for view in VIEWS:
                path=directory/f'{view}_complete.npz'
                saved=inputs.npz(path)
                for key,value in indices.items():exact(saved[key],value,f'{outer}/{inner}/{view} {key}')
                require(str(scalar(saved,'spec_id'))==spec_id,'Complete cache specification mismatch')
                require(str(scalar(saved,'view'))==view and int(scalar(saved,'outer_fold'))==outer
                        and int(scalar(saved,'inner_fold'))==inner,'Complete cache split/view mismatch')
                file_hash=inputs.record(path)
                require(marker['files_sha256'][path.name]==file_hash,'Split-marker file hash mismatch')
                hashes[str(path.relative_to(args.nested_dir))]=file_hash
                width=50 if view.endswith('y50') else 16
                finite_shape(saved['latent_prediction'],(len(positions),width),'complete latent prediction')
                finite_shape(saved['timing'],(width,3),'complete scalar timing')
                exact(np.sort(saved['timing'][:,0]),np.arange(width,dtype=float),'scalar target timing coverage')
                require(np.all(saved['timing'][:,1:]>=0),'Negative timing')
                scalar_count+=width
                for key in VIEW_KEYS[view]:
                    finite_shape(saved[key],(len(positions),50),'complete candidate '+key)
                    name=view+'_'+key.removeprefix('prediction_')
                    pool[name][positions]=saved[key]
                if width==16:
                    train_y=data['Y'][indices['train_indices']]
                    mean=train_y.mean(axis=0);scale=np.maximum(train_y.std(axis=0,ddof=1),1e-12)
                    close(saved['target_mean'],mean,'inner-training target mean',atol=1e-12)
                    close(saved['target_scale'],scale,'inner-training target sample SD',atol=1e-12)
                    finite_shape(saved['target_components'],(16,50),'target PCA components')
                    finite_shape(saved['target_explained_variance_ratio'],(16,),'target PCA ratios')
                    training=(train_y-mean)/scale
                    errors=verify_saved_pca(training,saved['target_components'],saved['target_pca_mean'],
                        saved['target_explained_variance_ratio'],f'target PCA {outer}/{inner}/{view}')
                    reconstructed=(saved['latent_prediction']@saved['target_components']+saved['target_pca_mean'])*scale+mean
                    close(reconstructed,saved['prediction_no_residual'],'target inverse identity',atol=2e-11)
                    close(reconstructed+saved['residual_prediction'],saved['prediction_ridge_residual'],
                          'orthogonal residual addition',atol=2e-11)
                    close((saved['residual_prediction']/scale)@saved['target_components'].T,
                          np.zeros((len(positions),16)),'residual target orthogonality',atol=2e-9)
                    transform_checks.append(dict(outer_fold=outer,inner_fold=inner,view=view,
                                                 target_training_only_identities=errors))
                    saved_latent.append(saved)
                    if view.startswith('rank_pca'):
                        rank_training=rank_training_matrix(data['X'][indices['train_indices']])
                        finite_shape(saved['input_components'],(64,231),'input PCA components')
                        errors=verify_saved_pca(rank_training,saved['input_components'],saved['input_mean'],
                            saved['input_explained_variance_ratio'],f'input PCA {outer}/{inner}')
                        transform_checks[-1]['input_training_only_identities']=errors
                else:
                    exact(saved['prediction_native'],saved['latent_prediction'],'native50 identity')
            for key in ('target_mean','target_scale','target_components','target_pca_mean','residual_prediction'):
                exact(saved_latent[0][key],saved_latent[1][key],'Y16 views share training-only '+key)
            split_records.append(dict(outer_fold=outer,inner_fold=inner,
                inner_training_n=len(indices['train_indices']),inner_validation_n=len(indices['test_indices']),
                outer_test_n=len(te),train_validation_intersection=0,outer_test_inner_intersection=0,
                complete_views=3,completed_scalar_components=82))
        exact(coverage,np.ones(200,dtype=int),'each outer-training beer has exactly one inner prediction')
        directory=args.nested_dir/f'outer_{outer}'
        stored_pool=inputs.npz(directory/'inner_oof.npz')
        exact(stored_pool['Y'],data['Y'][tr],'pooled inner Y is outer-training only')
        exact(stored_pool['beer'],data['beer'][tr],'pooled inner beer order')
        exact(stored_pool['train_indices'],tr,'pooled inner global indices')
        exact(stored_pool['targets'],data['targets'],'pooled inner target order')
        require(str(scalar(stored_pool,'spec_id'))==spec_id,'Pooled inner spec mismatch')
        scores=[]
        for candidate in CANDIDATES:
            finite_shape(pool[candidate],(200,50),'pooled '+candidate)
            exact(stored_pool[candidate],pool[candidate],'independently assembled inner prediction')
            score={'model':candidate,**assess(data['Y'][tr],pool[candidate])}
            scores.append(score);inner_scores.append({'outer_fold':outer,**score})
        winner=sorted(scores,key=lambda r:(-r['core30_r2'],-r['macro_r2'],r['model']))[0]['model']
        selection=inputs.json(directory/'selection.json')
        require(selection['outer_fold']==outer and selection['spec_id']==spec_id,'Selection identity mismatch')
        require(selection.get('inner_complete') is True and selection.get('training_beers')==200,
                'Selection does not use complete200 inner validation rows')
        require(selection['selected_model']==winner,'Recorded selected model differs from inner-only recomputation')
        require(selection['inner_files_sha256']==hashes,'Selection input-file hash list mismatch')
        recorded={r['model']:r for r in selection['candidate_scores']}
        require(len(selection['candidate_scores'])==5 and set(recorded)==set(CANDIDATES),
                'Incomplete/duplicate candidate score records')
        for score in scores:
            expected={k:v for k,v in score.items() if k!='model'}
            maximum_score_difference=max(maximum_score_difference,
                score_close(recorded[score['model']],expected,'inner candidate score'))
        result_record=result_selections[outer]
        for key,value in selection.items():
            require(result_record.get(key)==value,'Final selection record differs from selection.json: '+key)
        selected=inputs.npz(directory/'selected_outer.npz')
        exact(selected['train_indices'],tr,'selected outer training indices')
        exact(selected['test_indices'],te,'selected outer test indices')
        require(str(scalar(selected,'selected_model'))==winner and str(scalar(selected,'spec_id'))==spec_id,
                'Selected outer identity differs')
        exact(selected['prediction'],original_outer[outer,winner],'selected outer versus historical complete cache')
        exact(selected['prediction'],data['modern'][winner][te],'selected outer versus historical OOF')
        maximum_score_difference=max(maximum_score_difference,
            score_close(result_record['outer_metrics'],assess(data['Y'][te],selected['prediction']),
                        'recorded outer evaluation'))
        assembled[te]=selected['prediction'];selected_rows[te]=winner;outer_coverage[te]+=1
        selected_records.append(dict(outer_fold=outer,training_n=200,test_n=50,selected_model=winner,
            selected_inner_metrics=next(s for s in scores if s['model']==winner),
            outer_metrics=assess(data['Y'][te],selected['prediction'])))
    require(scalar_count==1230,'Scalar completion total differs from1230')
    exact(outer_coverage,np.ones(250,dtype=int),'each original beer predicted once externally')
    nested=inputs.npz(args.nested_dir/'nested_oof.npz')
    for key,value in [('Y',data['Y']),('beer',data['beer']),('fold',data['fold']),('targets',data['targets']),
                      ('prediction',assembled),('selected_candidate',selected_rows)]:
        exact(nested[key],value,'final nested archive '+key)
    require(str(scalar(nested,'spec_id'))==spec_id,'Final nested archive spec mismatch')
    require(result['nested_oof_sha256']==inputs.record(args.nested_dir/'nested_oof.npz'),
            'Final nested archive hash mismatch')
    maximum_score_difference=max(maximum_score_difference,
        score_close(result['metrics'],assess(data['Y'],assembled),'final pooled metrics'))
    csv_scores=inputs.csv(args.nested_dir/'inner_candidate_scores.csv')
    require(len(csv_scores)==25 and not csv_scores.duplicated(['outer_fold','model']).any(),
            'Expected25 unique candidate-score CSV rows')
    for row in inner_scores:
        match=csv_scores[(csv_scores.outer_fold==row['outer_fold'])&(csv_scores.model==row['model'])]
        require(len(match)==1,'Missing candidate-score CSV row')
        score_close(match.iloc[0].to_dict(),{k:v for k,v in row.items() if k not in ('outer_fold','model')},
                    'candidate-score CSV')
    data['predictions']={SIX:data['modern'][SIX],FIXED:data['modern'][FIXED],NESTED:assembled}
    audit=dict(status='PASS',spec_id=spec_id,inner_splits_complete=15,complete_view_files=45,
        scalar_components_complete=1230,outer_selections_complete=5,outer_oof_shape=[250,50],
        all_inner_and_outer_intersections_zero=True,inner_validation_coverage_each_outer_training_row=1,
        outer_validation_coverage_each_original_row=1,inner_only_selection_recomputed=True,
        recorded_selection_agrees=True,original_outer_caches_exactly_reused=True,
        fixed_raw231_y16_deployment_changed=False,maximum_recorded_score_difference=maximum_score_difference,
        selections=selected_records,split_records=split_records,transform_checks=transform_checks,
        fixed_reference_score_checks=data['historical_reference_checks'],
        selection_audit_scope='Recomputed from saved inner predictions and only Y[outer_train]; source hashes, split identities, and training-only transform identities checked. Predictive models are not rerun.',
        predictive_models_fitted=0,predictive_models_invoked=0)
    return data,audit,inputs


def relative_grades(Y,fold,predictions):
    thresholds=np.stack([np.quantile(Y[fold!=f],np.arange(1,7)/7,axis=0,method='linear') for f in range(5)])
    truth=np.empty(Y.shape,dtype=np.int64)
    grades={name:np.empty(Y.shape,dtype=np.int64) for name in predictions}
    def transform(values,cuts):
        return np.stack([np.searchsorted(cuts[:,j],values[:,j],side='right')
                         for j in range(values.shape[1])],axis=1)
    for f in range(5):
        mask=fold==f
        require(mask.sum()==50 and (~mask).sum()==200,'Relative-grade fold sizes differ')
        truth[mask]=transform(Y[mask],thresholds[f])
        for name,prediction in predictions.items():grades[name][mask]=transform(prediction[mask],thresholds[f])
    return thresholds,truth,grades


def verify_reference_grades(data,grades,args,inputs):
    sources={SIX:args.project_dir/'A_work/q1/oof_relative_grades_0_6.csv',
             FIXED:args.modern_dir/'diagnostics/raw231_y16_no_residual_oof_relative_grades_0_6.csv'}
    records=[]
    for name,path in sources.items():
        frame=inputs.csv(path)
        require(len(frame)==250 and frame.beer.is_unique,'Invalid saved grade IDs')
        exact(frame.set_index('beer').loc[data['beer'],data['targets']].to_numpy(),
              grades[name],'independent reproduction of saved '+name+' grades')
        records.append(dict(model=name,source=str(path),maximum_absolute_difference=0))
    inputs.record(args.project_dir/'A_work/q1/sensory_model.py')
    inputs.record(args.modern_dir/'postprocess_modern_results.py')
    return records


def summarize(data,truth,grades):
    Y=data['Y'];variance=Y.var(axis=0,ddof=0)
    require(np.all(variance>0),'Nonpositive fixed evaluation variance')
    summaries=[];groups=[];per_beer=[];tails=[];dimensions=[];losses={}
    unique_styles=sorted(np.unique(data['styles']).tolist())
    for model,prediction in data['predictions'].items():
        r2=r2_by_target(Y,prediction);error=np.abs(grades[model]-truth)
        for j,name in enumerate(data['targets']):
            dimensions.append(dict(model=model,target=str(name),r2=float(r2[j]),
                rmse=float(np.sqrt(np.mean((Y[:,j]-prediction[:,j])**2))),
                grade_mae=float(error[:,j].mean()),grade_exact=float((error[:,j]==0).mean()),
                grade_within_one=float((error[:,j]<=1).mean())))
        for group,columns in TARGETS.items():
            loss=np.mean((Y[:,columns]-prediction[:,columns])**2/variance[columns],axis=1)
            losses[model,group]=loss
            close(np.mean(loss),np.asarray(1-r2[columns].mean()),'NMSE versus macro R2',atol=2e-12)
            group_records=[]
            for style in unique_styles:
                mask=data['styles']==style
                row=dict(model=model,targets=group,style=style,n=int(mask.sum()),
                         small_n=bool(mask.sum()<10),nmse=float(loss[mask].mean()))
                groups.append(row);group_records.append(row)
            own=np.argsort(-loss,kind='stable')[:25]
            total=float(loss.sum());tail_sum=float(loss[own].sum())
            tail=dict(model=model,targets=group,own_worst25_n=25,
                own_worst25_mean_nmse=float(loss[own].mean()),
                own_worst25_sum_nmse=tail_sum,total_sum_nmse=total,
                own_worst25_error_share=tail_sum/total if total>0 else 0.,
                own_worst25_to_all_mean_ratio=float(loss[own].mean()/loss.mean()) if total>0 else 0.,
                own_worst25_beer=data['beer'][own].tolist())
            tails.append(tail)
            row=dict(model=model,targets=group,n=250,macro_r2=float(r2[columns].mean()),
                nmse=float(loss.mean()),style_equal_nmse=float(np.mean([r['nmse'] for r in group_records])),
                grade_mae=float(error[:,columns].mean()),
                grade_exact=float((error[:,columns]==0).mean()),
                grade_within_one=float((error[:,columns]<=1).mean()),
                own_worst25_mean_nmse=tail['own_worst25_mean_nmse'],
                own_worst25_error_share=tail['own_worst25_error_share'])
            summaries.append(row)
            own_set=set(own.tolist())
            for i,value in enumerate(loss):
                per_beer.append(dict(model=model,targets=group,beer=data['beer'][i],fold=int(data['fold'][i]),
                    style=str(data['styles'][i]),nmse=float(value),in_own_worst25=i in own_set))
    counts=[];comparisons=[]
    group_index={(r['model'],r['targets'],r['style']):r for r in groups}
    for reference in (SIX,FIXED):
        for target_set in TARGETS:
            current=[]
            for style in unique_styles:
                a=group_index[NESTED,target_set,style];b=group_index[reference,target_set,style]
                delta=a['nmse']-b['nmse'];tie=abs(delta)<=1e-12
                row=dict(model=NESTED,reference=reference,targets=target_set,style=style,n=a['n'],
                    small_n=a['small_n'],nested_nmse=a['nmse'],reference_nmse=b['nmse'],
                    delta_nmse=delta,outcome='tie' if tie else('lower_error' if delta<0 else'higher_error'))
                comparisons.append(row);current.append(row)
            for stratum,predicate in [('all',lambda r:True),('n_ge_10',lambda r:r['n']>=10),
                                     ('n_lt_10',lambda r:r['n']<10)]:
                subset=[r for r in current if predicate(r)]
                counts.append(dict(model=NESTED,reference=reference,targets=target_set,stratum=stratum,
                    groups=len(subset),lower_error=sum(r['outcome']=='lower_error' for r in subset),
                    higher_error=sum(r['outcome']=='higher_error' for r in subset),
                    ties=sum(r['outcome']=='tie' for r in subset),absolute_tie_tolerance=1e-12))
    coverage=[]
    for style in unique_styles:
        for f in range(5):
            mask=data['styles']==style
            test_n=int(np.sum(mask&(data['fold']==f)))
            train_n=int(np.sum(mask&(data['fold']!=f)))
            coverage.append(dict(style=style,outer_fold=f,training_n=train_n,test_n=test_n,
                                 unseen_style_evaluated=bool(test_n and not train_n)))
    return dict(model_metrics=summaries,style_metrics=groups,style_comparisons=comparisons,
                style_win_counts=counts,per_beer_errors=per_beer,own_worst25=tails,
                dimension_metrics=dimensions,style_fold_coverage=coverage),losses


def paired_bootstrap(Y,predictions,truth,grades,repetitions=2000,seed=BOOTSTRAP_SEED):
    """One common beer resampling scheme; recompute R2 denominator per draw."""
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(Y),size=(repetitions,len(Y)),dtype=np.int64)
    counts=np.stack([np.bincount(row,minlength=len(Y)) for row in indices]).astype(float)
    centered=Y-Y.mean(axis=0)
    denominator=counts@(centered**2)-(counts@centered)**2/len(Y)
    require(np.all(denominator>0),'A bootstrap draw has a constant target; no undefined draws are silently discarded')
    metric_names=('delta_macro_r2','delta_grade_mae','delta_grade_exact','delta_grade_within_one')
    rows=[];samples={}
    nested_error=np.abs(grades[NESTED]-truth)
    squared_nested=(Y-predictions[NESTED])**2
    for reference in (SIX,FIXED):
        squared_reference=(Y-predictions[reference])**2
        target_delta=(counts@(squared_reference-squared_nested))/denominator
        ref_error=np.abs(grades[reference]-truth)
        r2delta=r2_by_target(Y,predictions[NESTED])-r2_by_target(Y,predictions[reference])
        for target_set,columns in TARGETS.items():
            per_beer={
                'delta_grade_mae':nested_error[:,columns].mean(axis=1)-ref_error[:,columns].mean(axis=1),
                'delta_grade_exact':(nested_error[:,columns]==0).mean(axis=1)-(ref_error[:,columns]==0).mean(axis=1),
                'delta_grade_within_one':(nested_error[:,columns]<=1).mean(axis=1)-(ref_error[:,columns]<=1).mean(axis=1)}
            for metric in metric_names:
                draws=(target_delta[:,columns].mean(axis=1) if metric=='delta_macro_r2'
                       else counts@per_beer[metric]/len(Y))
                point=(float(r2delta[columns].mean()) if metric=='delta_macro_r2'
                       else float(per_beer[metric].mean()))
                quantiles=np.quantile(draws,[.025,.975],method='linear')
                rows.append(dict(model=NESTED,reference=reference,targets=target_set,metric=metric,
                    delta=point,ci025=float(quantiles[0]),ci975=float(quantiles[1]),
                    repetitions=repetitions,seed=seed,
                    unit=('R2' if metric=='delta_macro_r2' else'grade_units' if metric=='delta_grade_mae' else'fraction'),
                    zero_in_interval=bool(quantiles[0]<=0<=quantiles[1]),
                    interpretation='Paired beer bootstrap conditional on existing OOF; no model refit, no re-selection, no multiplicity adjustment.'))
                samples[f'{reference}__{target_set}__{metric}']=draws
    return rows,dict(beer_resample_indices=indices,**samples)


def render_text(data,audit,summary,intervals):
    model={(r['model'],r['targets']):r for r in summary['model_metrics']}
    boot={(r['reference'],r['targets'],r['metric']):r for r in intervals}
    all50=model[NESTED,'all50'];core=model[NESTED,'core30'];other=model[NESTED,'other20']
    delta=boot[SIX,'core30','delta_macro_r2'];fixed_delta=boot[FIXED,'core30','delta_macro_r2']
    def number(value):return f'{value:.5f}'
    def interval(row):return f"[{row['ci025']:.5f},{row['ci975']:.5f}]"
    counts=Counter(r['selected_model'] for r in audit['selections'])
    choices='、'.join(ALIASES[name]+f'（{n}折）' for name,n in sorted(counts.items()))
    table2='| TabICL：训练内选模流程 | '+ ' | '.join(number(r['macro_r2']) for r in [all50,core,other])+' |'
    abstract=(f"五外折、三内折评价的训练内选模流程获得全部50维宏R²={all50['macro_r2']:.5f}、"
        f"核心30维R²={core['macro_r2']:.5f}；相对六模型的核心差值为{delta['delta']:+.5f}，"
        f"95%配对区间为{interval(delta)}。固定Y16部署配置与该选择流程分别评价。")
    p35=(f"五个外折依据内层留出预测选择{choices}。合并250款外层留出预测后，全部50维、核心30维和其余20维宏R²分别为"
        f"{all50['macro_r2']:.5f}、{core['macro_r2']:.5f}和{other['macro_r2']:.5f}。相对六模型集成，核心30维差值为"
        f"{delta['delta']:+.5f}，95%配对区间为{interval(delta)}；相对固定Y16配置，差值为{fixed_delta['delta']:+.5f}，"
        f"区间为{interval(fixed_delta)}。区间由既有折外预测的2000次啤酒配对重采样构造，条件于这些预测，"
        "不包含重新训练和重新选模的不确定性。")
    grade=(f"使用各外折200款训练样本确定分位切点，训练内选模流程在核心30维的相对等级MAE为{core['grade_mae']:.5f}，"
        f"同档率为{100*core['grade_exact']:.2f}%，±1档率为{100*core['grade_within_one']:.2f}%。"
        "这些等级表示相对于训练参考样本的强度位置。")
    wc={(r['reference'],r['targets'],r['stratum']):r for r in summary['style_win_counts']}
    supported=wc[SIX,'core30','n_ge_10'];small=wc[SIX,'core30','n_lt_10']
    group=(f"训练内选模流程的样本加权核心NMSE为{core['nmse']:.5f}，22种风格等权NMSE为{core['style_equal_nmse']:.5f}。"
        f"在样本数不少于10的{supported['groups']}种风格中，相对六模型有{supported['lower_error']}种误差较低、"
        f"{supported['higher_error']}种较高、{supported['ties']}种相同；其余{small['groups']}种小样本风格相应为"
        f"{small['lower_error']}、{small['higher_error']}和{small['ties']}种，完整数值见表A2。")
    tail_rows={(r['model'],r['targets']):r for r in summary['own_worst25']}
    tail=tail_rows[NESTED,'core30'];a=tail_rows[SIX,'core30'];b=tail_rows[FIXED,'core30']
    tail_text=(f"训练内选模流程自身误差最大的25款啤酒贡献核心30维总归一化平方误差的{100*tail['own_worst25_error_share']:.2f}%，"
        f"六模型和固定Y16各自对应的比例为{100*a['own_worst25_error_share']:.2f}%和{100*b['own_worst25_error_share']:.2f}%。"
        "各模型的25款样本按自身误差分别选取，三个集合可能不同。")
    unseen=sum(r['unseen_style_evaluated'] for r in summary['style_fold_coverage'])
    boundary=('各外折测试样本所属风格均在训练侧出现，以上结果评价已见风格分布中的随机留出泛化；'
              '未见风格及独立测试数据上的表现仍需单独评价。') if unseen==0 else(
              f'共有{unseen}个风格—外折评价单元在训练侧没有同风格样本；分组结果仍不构成专门的留风格外评价。')
    conclusion=(f"感官预测中，分离配置选择与外层评价的流程获得核心30维宏R²={core['macro_r2']:.5f}，"
        f"相对六模型差值为{delta['delta']:+.5f}，95%条件配对区间为{interval(delta)}。"
        "该结果属于训练内选模流程；原目标误差分解、缺测与跨预测器分析仍对应固定Y16配置。")
    style_index={(r['model'],r['targets'],r['style']):r for r in summary['style_metrics']}
    lines=['表A2　核心30维折外误差的风格分组诊断','',
        '| 风格 | 样本数 | 六模型NMSE | 固定Y16 NMSE | 训练内选模NMSE |',
        '| --- | ---: | ---: | ---: | ---: |']
    for style in sorted(np.unique(data['styles'])):
        r=style_index[NESTED,'core30',style]
        values=[style_index[name,'core30',style]['nmse'] for name in (SIX,FIXED,NESTED)]
        label=style.replace('|','\\|')+('∗' if r['n']<10 else'')
        lines.append(f"| {label} | {r['n']} | "+' | '.join(f'{v:.6f}' for v in values)+' |')
    for label,key,count in [('按样本数加权','nmse','250'),('风格等权','style_equal_nmse','22种')]:
        lines.append(f'| {label} | {count} | '+' | '.join(f"{model[name,'core30'][key]:.6f}" for name in (SIX,FIXED,NESTED))+' |')
    lines+=['','∗表示样本数少于10，仅作描述性诊断。所有模型使用全250款样本的相同逐目标评价方差；'
            '样本加权NMSE等于1减去相同250行的宏R²。固定Y16与训练内选模分别列示。']
    return dict(abstract_prediction_sentence=abstract,table2_nested_row=table2,
                section35_selection_and_r2=p35,section35_relative_grades=grade,
                section35_style_groups=group,optional_own_worst25_paragraph=tail_text,
                generalization_scope_sentence=boundary,conclusion_prediction_sentence=conclusion,
                appendix_a2_markdown='\n'.join(lines)+'\n')


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def publish(args,data,audit,inputs,summary,intervals,bootstrap_arrays,thresholds,truth,grades,texts,grade_checks):
    require(not args.output.exists(),'Output directory already exists; choose a new one')
    inputs.recheck()
    protocol=dict(completed_experiment_required=True,bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,bootstrap_unit='original beer; sample250 with replacement; common draws for all comparisons',
        bootstrap_r2='R2 denominator recomputed within every beer-resampled dataset',
        bootstrap_conditioning='Existing complete OOF predictions and selected configurations held fixed; no refit or re-selection',
        confidence_interval='95% empirical percentile interval, numpy quantile method=linear; no multiplicity adjustment',
        relative_grades='Each outer training fold only: quantiles1/7,...,6/7; method=linear; searchsorted side=right; truth and predictions use the same cuts',
        relative_grades_are_raw_panel_scores=False,relative_grade_reference_checks=grade_checks,
        group_nmse='Mean_j in target set[(prediction-truth)^2 / variance_all250_ddof0_j], then average within style',
        variance_used_for_training=False,small_group_threshold=10,
        own_worst25='Each model separately ranks its250 individual NMSE values, stable ties by canonical beer order; sum of its worst25 losses divided by its total loss',
        own_worst25_sets_need_not_match=True,independent_hidden_test=False,leave_style_out_experiment=False,
        fixed_y16_and_nested_selection_distinct=True,predictive_models_fitted=0,predictive_models_invoked=0,
        code_sha256=sha(Path(__file__)),versions={k:importlib.metadata.version(k) for k in ('numpy','pandas','scipy')})
    results=dict(status='PASS',complete=True,model=NESTED,audit=audit,protocol=protocol,
        model_metrics=summary['model_metrics'],paired_bootstrap=intervals,
        style_win_counts=summary['style_win_counts'],own_worst25=summary['own_worst25'],
        manuscript_text=texts,input_files=inputs.records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='nested_audit_staging_',dir=args.output.parent))
    try:
        write_json(temporary/'independent_audit.json',audit)
        write_json(temporary/'nested_postprocess_results.json',results)
        write_json(temporary/'statistical_protocol.json',protocol)
        write_json(temporary/'manuscript_text.json',texts)
        for name,rows in summary.items():pd.DataFrame(rows).to_csv(temporary/(name+'.csv'),index=False)
        pd.DataFrame(intervals).to_csv(temporary/'paired_bootstrap.csv',index=False)
        pd.DataFrame([{'outer_fold':r['outer_fold'],'selected_model':r['selected_model'],
                       **{k:v for k,v in r['selected_inner_metrics'].items() if k!='model'}}
                      for r in audit['selections']]).to_csv(temporary/'selected_configurations.csv',index=False)
        np.savez_compressed(temporary/'paired_bootstrap_samples.npz',**bootstrap_arrays)
        np.savez_compressed(temporary/'relative_grade_arrays.npz',truth=truth,thresholds=thresholds,
            beer=data['beer'],fold=data['fold'],targets=data['targets'],**grades)
        meta=pd.DataFrame(dict(beer=data['beer'],fold=data['fold']))
        pd.concat([meta,pd.DataFrame(grades[NESTED],columns=data['targets'])],axis=1).to_csv(
            temporary/'nested_oof_relative_grades_0_6.csv',index=False)
        a2=pd.DataFrame(summary['style_metrics'])
        a2=a2[a2.targets=='core30'].pivot(index=['style','n','small_n'],columns='model',values='nmse').reset_index()
        a2=a2.rename(columns={SIX:'six_model_nmse',FIXED:'fixed_y16_nmse',NESTED:'nested_selection_nmse'})
        a2[['style','n','small_n','six_model_nmse','fixed_y16_nmse','nested_selection_nmse']].to_csv(
            temporary/'table_A2_core30_nmse.csv',index=False)
        (temporary/'table_A2_core30_nmse.md').write_text(texts['appendix_a2_markdown'])
        headings={'abstract_prediction_sentence':'摘要预测句','table2_nested_row':'表2新增行',
            'section35_selection_and_r2':'3.5选择与连续预测','section35_relative_grades':'3.5相对等级',
            'section35_style_groups':'3.5风格分组','optional_own_worst25_paragraph':'可选误差集中段',
            'generalization_scope_sentence':'评价范围','conclusion_prediction_sentence':'结论预测句'}
        document=['# 已完成嵌套评价的论文替换文字','']
        for key,heading in headings.items():document+=['## '+heading,'',texts[key],'']
        document+=['## 附录A2','',texts['appendix_a2_markdown']]
        (temporary/'manuscript_replacements.md').write_text('\n'.join(document))
        file_index={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(temporary.iterdir()) if p.is_file()}
        write_json(temporary/'OUTPUT_MANIFEST.json',dict(complete=True,files=file_index))
        temporary.rename(args.output)
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True)
        raise


def locate_project():
    candidates=[HERE.parents[1]/'project/MathorCup_A']
    for ancestor in (HERE,*HERE.parents):
        candidates.extend((ancestor,ancestor/'project/MathorCup_A'))
    for candidate in candidates:
        if ((candidate/'data/A/1.csv').is_file()
                and (candidate/'A_work/q1/fold_assignments.csv').is_file()):
            return candidate.resolve()
    raise FileNotFoundError('No default project found; supply --project-dir')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir',type=Path)
    parser.add_argument('--modern-dir',type=Path)
    parser.add_argument('--nested-dir',type=Path,help='Default: HERE/nested_selection')
    parser.add_argument('--runner-script',type=Path,help='Default: sibling run_nested_selection.py')
    parser.add_argument('--output',type=Path,help='Default: nested_selection/statistics; new directory required')
    parser.add_argument('--bootstrap-repetitions',type=int,default=2000)
    parser.add_argument('--bootstrap-seed',type=int,default=BOOTSTRAP_SEED)
    args=parser.parse_args()
    if args.bootstrap_repetitions!=2000:parser.error('This registered postprocess requires exactly2000 paired beer resamples')
    try:
        args.project_dir=(args.project_dir or locate_project()).resolve()
        default_modern=HERE if (HERE/'oof_arrays.npz').is_file() else args.project_dir/'frontier_revision/experiments/modern_prediction'
        args.modern_dir=(args.modern_dir or default_modern).resolve()
        args.nested_dir=(args.nested_dir or HERE/'nested_selection').resolve()
        args.runner_script=(args.runner_script or HERE/'run_nested_selection.py').resolve()
        args.output=(args.output or args.nested_dir/'statistics').resolve()
        require(not args.output.exists(),'Output directory already exists; choose a new one')
        data,audit,inputs=audit_complete(args)
        thresholds,truth,grades=relative_grades(data['Y'],data['fold'],data['predictions'])
        grade_checks=verify_reference_grades(data,grades,args,inputs)
        summary,_=summarize(data,truth,grades)
        intervals,samples=paired_bootstrap(data['Y'],data['predictions'],truth,grades,
                                          args.bootstrap_repetitions,args.bootstrap_seed)
        texts=render_text(data,audit,summary,intervals)
        publish(args,data,audit,inputs,summary,intervals,samples,thresholds,truth,grades,texts,grade_checks)
    except IncompleteExperiment as exc:
        print(json.dumps({'status':'INCOMPLETE','reason':str(exc),'metrics_emitted':False,
                          'result_directory_created':False},ensure_ascii=False),file=sys.stderr)
        return 3
    except (AuditFailure,KeyError,ValueError,FileNotFoundError) as exc:
        print(json.dumps({'status':'FAIL','reason':str(exc),'metrics_emitted':False,
                          'result_directory_created':False},ensure_ascii=False),file=sys.stderr)
        return 2
    print(json.dumps({'status':'PASS','output':str(args.output),'complete_inner_views':45,
        'outer_selections':5,'oof_rows':250,'paired_bootstrap_repetitions':2000,
        'predictive_models_fitted':0,'predictive_models_invoked':0},ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
