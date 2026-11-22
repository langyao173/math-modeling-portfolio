"""No-score audit of test-batch invariance in pinned official preprocessing.

The installed power-transform fallback may clip a whole query batch if one
entry raises ValueError. Therefore attention independence alone is not enough:
check all five actual folds and all four exact missingness query batches.
No model weights are loaded and no validation prediction is scored.
"""
import os
from pathlib import Path
import hashlib
import inspect
import json
import numpy as np
from tabicl._sklearn.preprocessing import TransformToNumerical, EnsembleGenerator
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
SEED=20261023
DERIVED=('kcalperc','esters_sum','aroma_hops_sum','acids_sum','sulfur_sum')


def audit_preprocessing(X,fields,fold,output=None):
    records=[]
    for f in range(5):
        tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f)
        columns=[fields.index(name) for name in DERIVED]
        cases=[]
        original=X[te].copy();original[:,columns]=np.nan;cases.append(original)
        for level in (10,30,50):
            query=X[te].copy()
            rng=np.random.default_rng(SEED+1000*f+level)
            query[rng.random(query.shape)<level/100]=np.nan
            query[:,columns]=np.nan;cases.append(query)
        encoder=TransformToNumerical()
        train=encoder.fit_transform(X[tr])
        # Regression feature preprocessing and feature shuffles do not depend
        # on the target values. A constant placeholder avoids reading test Y.
        generator=EnsembleGenerator(classification=False,n_estimators=8,
            norm_methods=['none','power'],feat_shuffle_method='latin',
            outlier_threshold=4.,random_state=SEED)
        generator.fit(train,np.zeros(len(tr)))
        separate=[generator.transform(encoder.transform(q),mode='test') for q in cases]
        joined=generator.transform(encoder.transform(np.concatenate(cases)),mode='test')
        for method,(values,) in joined.items():
            reference=np.concatenate([a[method][0] for a in separate],axis=1)
            error=float(np.max(abs(values-reference),initial=0.))
            records.append(dict(fold=f,normalization=method,
                maximum_absolute_difference=error,
                exact_equal=bool(np.array_equal(values,reference,equal_nan=True)),
                shape=list(values.shape)))
    module=Path(inspect.getfile(EnsembleGenerator)).resolve()
    package=module.parents[1]
    source_paths=['_sklearn/preprocessing.py','_sklearn/regressor.py',
                  '_model/embedding.py','_model/interaction.py','_model/learning.py',
                  '_model/layers.py','_model/attention.py']
    sources={name:hashlib.sha256((package/name).read_bytes()).hexdigest() for name in source_paths}
    result=dict(all_folds_equal=all(r['exact_equal'] for r in records),
        heldout_scores_computed=False,model_weights_loaded=False,
        reason='Whole-query exception fallback in the power transform warrants checking all actual folds, even though attention uses train-only keys.',
        compared='Four separate 50-row queries versus one concatenated 200-row query after every fixed official preprocessing and feature-shuffle branch',
        target_dependence='Feature transforms and regression ensemble permutations do not depend on response values; placeholder train target is used.',
        package='tabicl==2.2.0',official_source='https://github.com/soda-inria/tabicl',
        source_sha256=sources,records=records)
    if output:
        Path(output).write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    from run_tabicl_benchmark import data
    x,_,X,_,fold=data()
    output=HERE/'missingness/preprocessing_batch_audit.json'
    output.parent.mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        result=audit_preprocessing(X,x.columns[3:].tolist(),fold,output)
    print(json.dumps(result,indent=2))
