"""Outer-holdout evaluation of a training-only semantic correction to baseline.

All scalar mixing weights are fitted from the enclosing training OOF arrays.
The shrinkage coefficient .02 is fixed before this comparison and matches the
legacy ensemble's order of regularization. No outer labels select weights.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from run_semantic_rrr import HERE,AW,SPECS,paired_ci

def main():
    dat=np.load(HERE/'oof_predictions.npz');Y=dat['Y'];core=np.arange(30)
    selections=json.loads((HERE/'nested_selection.json').read_text())
    index={tuple(s.values()):i for i,s in enumerate(SPECS)}
    baseline=np.load(AW/'q1/oof_arrays.npz')['nested_ensemble']
    names=['no_aux_all50','aligned_aux_all50','permuted_aux_all50','no_aux_core30','aligned_aux_core30','permuted_aux_core30']
    pred={name:np.empty_like(Y) for name in names};rows=[]
    for fold in range(5):
        old=np.load(AW/f'nature_upgrade/adaptive/outer_{fold}_old_cache.npz')
        train,test=old['train_indices'],old['test_indices']
        assert np.array_equal(dat['fold'][test],np.full(len(test),fold))
        np.testing.assert_allclose(old['test_predictions'],baseline[test],atol=1e-13,rtol=0)
        for name in names:
            spec=next(x['spec'] for x in selections if x['fold']==fold and x['family']==name)
            kind='permuted' if name.startswith('permuted') else 'aligned'
            inp=np.load(HERE/f'fold_{fold}_inner.npz')[kind][index[tuple(spec.values())]]
            idx=core if name.endswith('core30') else np.arange(50)
            sd=np.maximum(Y[train].std(0,ddof=1),1e-10)
            residual=(old['inner_predictions'][:,idx]-Y[train][:,idx])/sd[idx]
            change=(inp[:,idx]-old['inner_predictions'][:,idx])/sd[idx]
            weight=float(np.clip(-np.mean(residual*change)/(np.mean(change**2)+.02),0,1))
            pred[name][test]=(1-weight)*old['test_predictions']+weight*dat[name][test]
            rows.append(dict(fold=fold,model=name,auxiliary_weight=weight))
    scores=[]
    for name,p in pred.items():
        r=r2_score(Y,p,multioutput='raw_values')
        scores.append(dict(model='nested_blend_'+name,all50_r2=float(r.mean()),core30_r2=float(r[:30].mean()),
            delta_vs_frozen_all50=float(r.mean()-r2_score(Y,baseline)),ci_vs_frozen_all50=paired_ci(Y,p,baseline,np.arange(50)),
            ci_vs_frozen_core30=paired_ci(Y,p,baseline,core)))
    pd.DataFrame(scores).to_csv(HERE/'blend_comparison.csv',index=False)
    pd.DataFrame(rows).to_csv(HERE/'blend_weights.csv',index=False)
    np.savez_compressed(HERE/'blend_oof.npz',Y=Y,beer=dat['beer'],fold=dat['fold'],**pred)
    print(pd.DataFrame(scores).to_string(index=False))

if __name__=='__main__':main()
