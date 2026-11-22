"""Read-only OOF diagnostics: same training-fold relative grades and paired CIs.

No fitted model or hyperparameter is changed. Rerun after additional complete
OOF candidates arrive; every included candidate must cover all 250 test beers.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]/'project'/'MathorCup_A'
SEED=20261023


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grade_with_cuts(values,cuts):
    return np.stack([np.searchsorted(cuts[:,j],values[:,j],side='right')
                     for j in range(values.shape[1])],axis=1)


def main():
    out=HERE/'diagnostics'
    out.mkdir(exist_ok=True)
    a=np.load(HERE/'oof_arrays.npz',allow_pickle=False)
    Y=a['Y'];beers=a['beer'];fold=a['fold'];targets=a['targets']
    preds={k:a[k] for k in a.files if k not in ('Y','beer','fold','targets')}
    tabdpt=HERE/'tabdpt/oof_arrays.npz'
    if tabdpt.exists():
        d=np.load(tabdpt,allow_pickle=False)
        assert np.allclose(d['Y'],Y) and np.array_equal(d['beer'],beers) and np.array_equal(d['fold'],fold)
        for k in d.files:
            if k not in ('Y','beer','fold','targets') and not k.startswith('old_'):
                if d[k].shape==Y.shape and np.isfinite(d[k]).all():
                    preds['tabdpt13_raw231_pca128_y50' if k=='prediction' else k]=d[k]
    thresholds=np.stack([np.quantile(Y[fold!=f],np.arange(1,7)/7,axis=0) for f in range(5)])
    truth=np.empty_like(Y,dtype=int)
    grades={name:np.empty_like(Y,dtype=int) for name in preds}
    for f in range(5):
        mask=fold==f
        assert mask.sum()==50 and (~mask).sum()==200
        truth[mask]=grade_with_cuts(Y[mask],thresholds[f])
        for name,p in preds.items():
            assert p.shape==Y.shape and np.isfinite(p).all()
            grades[name][mask]=grade_with_cuts(p[mask],thresholds[f])
    # Exact reproduction of the legacy relative-grade evaluation.
    old=pd.read_csv(BASE/'A_work/q1/oof_relative_grades_0_6.csv').set_index('beer').loc[beers]
    legacy_error=int(np.max(np.abs(old[targets].to_numpy()-grades['old_nested_ensemble'])))
    if legacy_error!=0:
        raise AssertionError('Legacy fold-local relative-grade reproduction failed.')
    np.savez_compressed(out/'relative_grade_arrays.npz',truth=truth,beer=beers,fold=fold,targets=targets,
                        thresholds=thresholds,**grades)
    meta=pd.DataFrame(dict(beer=beers,fold=fold))
    pd.concat([meta,pd.DataFrame(truth,columns=targets)],axis=1).to_csv(out/'oof_truth_relative_grades_0_6.csv',index=False)
    metrics=[];dimension=[];differences=[];ci=[]
    reference='old_nested_ensemble'
    ref_r=r2_score(Y,preds[reference],multioutput='raw_values')
    ref_e=np.abs(grades[reference]-truth)
    rng=np.random.default_rng(SEED+971)
    boot=rng.integers(0,len(Y),size=(2000,len(Y)))
    for name,p in preds.items():
        g=grades[name];err=np.abs(g-truth)
        r=r2_score(Y,p,multioutput='raw_values')
        if not name.startswith('old_') or name==reference:
            pd.concat([meta,pd.DataFrame(g,columns=targets)],axis=1).to_csv(out/f'{name}_oof_relative_grades_0_6.csv',index=False)
        for group,ix in [('all50',np.arange(50)),('core30',np.arange(30))]:
            e=err[:,ix]
            metrics.append(dict(model=name,targets=group,macro_r2=float(r[ix].mean()),
                grade_mae=float(e.mean()),grade_exact=float((e==0).mean()),grade_within_one=float((e<=1).mean())))
            if name!=reference:
                perbeer={'grade_mae':e.mean(1)-ref_e[:,ix].mean(1),
                         'grade_exact':(e==0).mean(1)-(ref_e[:,ix]==0).mean(1),
                         'grade_within_one':(e<=1).mean(1)-(ref_e[:,ix]<=1).mean(1)}
                for metric,delta in perbeer.items():
                    samples=delta[boot].mean(1)
                    ci.append(dict(model=name,reference=reference,targets=group,metric=metric,
                        delta=float(delta.mean()),ci025=float(np.quantile(samples,.025)),
                        ci975=float(np.quantile(samples,.975)),repetitions=2000,
                        interpretation='Paired beer bootstrap of fixed OOF grades; no refit or multiplicity correction'))
        for j,t in enumerate(targets):
            dimension.append(dict(model=name,target=str(t),core30=j<30,r2=float(r[j]),
                grade_mae=float(err[:,j].mean()),grade_exact=float((err[:,j]==0).mean()),
                grade_within_one=float((err[:,j]<=1).mean())))
            if name!=reference:
                differences.append(dict(model=name,reference=reference,target=str(t),core30=j<30,
                    delta_r2=float(r[j]-ref_r[j]),delta_grade_mae=float(err[:,j].mean()-ref_e[:,j].mean()),
                    delta_grade_exact=float((err[:,j]==0).mean()-(ref_e[:,j]==0).mean()),
                    delta_grade_within_one=float((err[:,j]<=1).mean()-(ref_e[:,j]<=1).mean())))
    pd.DataFrame(metrics).to_csv(out/'relative_grade_metrics.csv',index=False)
    pd.DataFrame(dimension).to_csv(out/'relative_grade_dimension_metrics.csv',index=False)
    pd.DataFrame(differences).to_csv(out/'paired_dimension_differences.csv',index=False)
    pd.DataFrame(ci).to_csv(out/'relative_grade_paired_ci.csv',index=False)
    (out/'relative_grade_protocol.json').write_text(json.dumps(dict(
        model_count=len(preds),models=list(preds),outer_seed=SEED,train_per_fold=200,test_per_fold=50,
        quantiles=(np.arange(1,7)/7).tolist(),side='right',numpy_quantile_method='linear',
        legacy_grade_reproduction_max_error=legacy_error,
        interpretation='Relative within-study intensity grades; not recovered raw 0-6 panel scores.',
        thresholds_fit='Each outer training fold only; truth and predictions use identical fold cuts.',
        source_oof_sha256=digest(HERE/'oof_arrays.npz'),source_old_grade_sha256=digest(BASE/'A_work/q1/oof_relative_grades_0_6.csv')
    ),ensure_ascii=False,indent=2))
    from diagnose_target_subspace import diagnose
    subspace = diagnose(Y, fold, preds, grades, truth, out)
    print('SUBSPACE_DIAGNOSTIC '+json.dumps(subspace))
    print(pd.DataFrame(metrics).query("model in ['old_nested_ensemble','raw231_y16_no_residual','raw231_y50_native']").to_string(index=False))


if __name__=='__main__':
    main()
