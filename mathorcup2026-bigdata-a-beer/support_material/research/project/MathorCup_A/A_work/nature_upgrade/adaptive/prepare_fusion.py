# Team review and independent work must be documented truthfully.
"""Cache old nested ensemble OOF and fit a fixed shrinkage stacking rule.

Primary rule is declared before the new external-fold scores are examined:
one old ensemble, four new target-adaptive families, and the two GBDT
selection rules; per-target simplex weights shrink to global weights at .20.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k] = '1'
import sys, json, argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(WORK/'q1'))
from project_paths import resolve_data_dir
from sensory_model import SensoryEstimator, learn_weights
from run_q1 import tune, FAMILIES
from adaptive_models import AdaptiveChemicalRegressor, specifications, MODEL_VERSION
SEED = 20261023
NEW_FAMILIES = ['sparse_ridge','sparse_rbf','sparse_svr','auxiliary_rrr']


def dataset():
    data = resolve_data_dir()
    x = pd.read_csv(data/'1.csv').sort_values('beer').reset_index(drop=True)
    y = pd.read_csv(data/'4.csv').set_index('beer').loc[x.beer]
    w = pd.read_csv(data/'7.csv').set_index('beer').loc[x.beer]
    return x, x.iloc[:,3:].to_numpy(float), y.iloc[:,2:].to_numpy(float), w.iloc[:,2:].to_numpy(float), y.columns[2:].tolist()


def cache_old():
    x, X, Y, W, targets = dataset()
    previous = json.loads((WORK/'q1/nested_selection.json').read_text())
    old_oof = np.load(WORK/'q1/oof_arrays.npz')['nested_ensemble']
    for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(X)):
        path = HERE/f'outer_{f}_old_cache.npz'
        if path.exists():
            continue
        p = previous[f]
        inner = np.zeros_like(Y[tr])
        for a, b in KFold(3, shuffle=True, random_state=SEED+f).split(tr):
            for family in FAMILIES:
                model = SensoryEstimator(p['specs'][family]).fit(X[tr[a]], Y[tr[a]])
                inner[b] += p['weights'][family] * model.predict(X[tr[b]])
        np.savez_compressed(path, train_indices=tr, test_indices=te,
                            inner_predictions=inner, test_predictions=old_oof[te])
        print('old cache', f, flush=True)
    path = HERE/'full_data_old_cache.npz'
    if not path.exists():
        cand, weights, tuning = tune(X, Y, SEED+99)
        pred = sum(w*cand[f][2] for w, f in zip(weights, FAMILIES))
        np.savez_compressed(path, Y=Y, inner_predictions=pred)
        (HERE/'full_data_old_selection.json').write_text(json.dumps(dict(weights=weights.tolist(),
            families=FAMILIES, specs={f:cand[f][1] for f in FAMILIES}), indent=2))
        print('old full cache', flush=True)
    path = HERE/'full_data_candidates.npz'
    valid_cache = path.exists() and 'algorithm_version' in np.load(path).files and str(np.load(path)['algorithm_version']) == MODEL_VERSION
    if not valid_cache:
        specs = specifications(); pred = np.empty((len(specs), len(X), Y.shape[1]))
        for c, spec in enumerate(specs):
            for tr, te in KFold(3, shuffle=True, random_state=SEED+99).split(X):
                m = AdaptiveChemicalRegressor(spec).fit(X[tr], Y[tr], W[tr])
                pred[c,te] = m.predict(X[te])
        np.savez_compressed(path, algorithm_version=MODEL_VERSION, inner_predictions=pred)
        print('adaptive full cache', flush=True)


def shrink_weights(P, Y, eta=.20):
    global_w = learn_weights(list(P), Y)
    scale = np.maximum(Y.std(axis=0, ddof=1), 1e-8)
    nmodel, _, nout = P.shape
    weights = np.empty((nmodel, nout))
    for j in range(nout):
        z = P[:,:,j].T / scale[j]
        target = Y[:,j] / scale[j]
        def loss(w):
            return np.mean((z@w-target)**2) + eta*np.sum((w-global_w)**2)
        def grad(w):
            return 2*z.T@(z@w-target)/len(z) + 2*eta*(w-global_w)
        r = minimize(loss, global_w, jac=grad, method='SLSQP',
                     bounds=[(0.,1.)]*nmodel, constraints={'type':'eq','fun':lambda w:w.sum()-1},
                     options={'ftol':1e-11,'maxiter':300})
        if not r.success:
            raise RuntimeError(r.message)
        weights[:,j] = r.x
    return global_w, weights


def choose_adaptive(inner, test, y):
    specs = specifications()
    loss = np.mean(((inner-y) / np.maximum(y.std(axis=0, ddof=1),1e-8))**2, axis=1)
    inp=[]; out=[]; selected={}
    for family in NEW_FAMILIES:
        ix=np.array([i for i,s in enumerate(specs) if s['family']==family])
        sel=ix[np.argmin(loss[ix],axis=0)]
        selected[family]=sel.tolist()
        inp.append(np.column_stack([inner[k,:,j] for j,k in enumerate(sel)]))
        if test is not None:
            out.append(np.column_stack([test[k,:,j] for j,k in enumerate(sel)]))
    return inp,out,selected


def metric(y,p):
    rr=r2_score(y,p,multioutput='raw_values')
    return dict(macro_r2=float(rr.mean()),core30_r2=float(rr[:30].mean()),
                macro_rmse=float(np.sqrt(np.mean((y-p)**2,axis=0)).mean()),
                positive_r2=int((rr>0).sum()))


def available_fusion(ignore_gbr=False):
    x,X,Y,W,targets=dataset()
    output=HERE/'fusion'; output.mkdir(exist_ok=True)
    spec=dict(primary_rule='per-target simplex shrinkage eta=.20 to global weights',
              bases=['old_ensemble']+NEW_FAMILIES+['gbr_global','gbr_target'],
              primary_external_score='pooled five-outer-fold macro R2',
              secondary='same authors 175/75 holdout; no independent external validation claim',
              model_selection='training inner OOF only',
              declared_before_inspection='2026-10-23; before adaptive/GBr complete results inspected')
    specfile=output/'design.json'
    if not specfile.exists():specfile.write_text(json.dumps(spec,indent=2))
    combined={}; all_rows=[]
    for f in range(5):
        name=f'outer_{f}'
        old=np.load(HERE/f'{name}_old_cache.npz')
        aa=np.load(HERE/f'{name}_candidates.npz')
        tr=aa['train_indices'];te=aa['test_indices']
        assert np.array_equal(tr,old['train_indices']) and np.array_equal(te,old['test_indices'])
        inner=[old['inner_predictions']];test=[old['test_predictions']]
        ai,ap,sel=choose_adaptive(aa['inner_predictions'],aa['test_predictions'],Y[tr])
        inner+=ai;test+=ap
        base_names=['old_ensemble']+NEW_FAMILIES
        gb=HERE.parent/'benchmark'/f'{name}_candidates.npz'
        if gb.exists() and not ignore_gbr:
            g=np.load(gb)
            # Verified below against the benchmark's saved global indices.
            assert np.array_equal(tr,g['train_indices']) and np.array_equal(te,g['test_indices'])
            ip=g['inner_predictions'];tp=g['test_predictions']
            loss=np.mean(((ip-Y[tr])/np.maximum(Y[tr].std(axis=0,ddof=1),1e-8))**2,axis=1)
            gi=int(np.argmin(loss.mean(axis=1))); ti=np.argmin(loss,axis=0)
            inner += [ip[gi],np.column_stack([ip[k,:,j] for j,k in enumerate(ti)])]
            test += [tp[gi],np.column_stack([tp[k,:,j] for j,k in enumerate(ti)])]
            base_names += ['gbr_global','gbr_target']
        P=np.stack(inner);T=np.stack(test)
        gw,tw=shrink_weights(P,Y[tr])
        gp=np.einsum('m,mij->ij',gw,T)
        tp=np.einsum('mj,mij->ij',tw,T)
        variant='full' if len(base_names)==7 else 'without_gbr'
        np.savez_compressed(output/f'{name}_{variant}.npz',inner_predictions=P,test_predictions=T,
                            test_indices=te,train_indices=tr,global_weights=gw,target_weights=tw,
                            global_prediction=gp,target_prediction=tp)
        (output/f'{name}_{variant}.json').write_text(json.dumps(dict(bases=base_names,adaptive_selection=sel,
            global_metrics=metric(Y[te],gp),target_metrics=metric(Y[te],tp)),indent=2))
        combined.setdefault(variant,[]).append((te,gp,tp))
    for variant,folds in combined.items():
        if len(folds)!=5:continue
        gp=np.empty_like(Y);tp=np.empty_like(Y)
        for idx,p,q in folds:gp[idx]=p;tp[idx]=q
        np.savez_compressed(output/f'{variant}_oof.npz',Y=Y,global_prediction=gp,target_prediction=tp)
        rows=[dict(variant=variant,model='global',**metric(Y,gp)),dict(variant=variant,model='target_shrink',**metric(Y,tp))]
        all_rows+=rows
    if all_rows:
        pd.DataFrame(all_rows).to_csv(output/'model_comparison.csv',index=False)
        print(json.dumps(all_rows,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache-only',action='store_true');a=p.parse_args()
    cache_old()
    if not a.cache_only:
        available_fusion(ignore_gbr=True)
        available_fusion()
