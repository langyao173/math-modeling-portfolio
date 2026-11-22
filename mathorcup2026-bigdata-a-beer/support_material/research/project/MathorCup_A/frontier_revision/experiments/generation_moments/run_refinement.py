"""Fixed same-fold generation refinement; no outer-score parameter tuning."""
import os
for _k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_k]='1'
from pathlib import Path
import sys,json,time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from conditional_moments import ConditionalMomentCopula,IntervalCopula,HERE,AW
from control_problem import PROJECT,file_sha
from conditional_model import DERIVED,TECHNICAL
from evaluate_extensions import evaluate
from run_q3 import paired_ci,SEED

def main():
    df=pd.read_csv(PROJECT/'data/A/1.csv')
    features=[c for c in df.columns[3:] if c not in DERIVED+TECHNICAL]
    X=df[features].to_numpy(float);style=df.tasting_category_fine.to_numpy(str)
    params=dict(kappa=2.,nu=20.,iterations=8,burn_sweeps=8,retained_sweeps=8)
    protocol=dict(date='2026-10-24',n=250,p=225,outer_folds=3,split_seed=SEED+300,
        parameters=params,primary_seed=20261024,models=['interval_reference','conditional_moments','conditional_moments_missing'],
        selection='Three fixed candidates; no outer selection or tuning. Diagnostic refinement/internal validation.',
        covariance='Conditional first and full second moments; fixed initial shrinkage; Monte Carlo draws not independent beers.',
        missing_ablation='Reference and primary retain fold median replacement; missing-aware candidate uses observed CDF and unconstrained latent coordinates.',
        X_sha256=file_sha(PROJECT/'data/A/1.csv'))
    (HERE/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];scores=[];t0=time.perf_counter()
    for fold,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEED+300).split(X,style)):
        for name in protocol['models']:
            file=HERE/f'fold_{fold}_{name}.joblib'
            if file.exists():model=joblib.load(file)
            else:
                if name=='interval_reference':model=IntervalCopula(kind='interval_gaussian',kappa=2,nu=20).fit(X[a],style[a])
                else:model=ConditionalMomentCopula(**params,missing_intervals=name.endswith('_missing'),seed=20261024+fold).fit(X[a],style[a])
                joblib.dump(model,file)
            record,es,vs=evaluate(model,X[b],style[b],SEED+500+fold)
            rows.append(dict(fold=fold,model=name,n_validation=len(b),**record))
            for j,idx in enumerate(b):scores.append(dict(fold=fold,model=name,beer=int(df.beer.iloc[idx]),energy=es[j],variogram=vs[j]))
            if hasattr(model,'refinement_trace_'):
                pd.DataFrame(model.refinement_trace_).to_csv(HERE/f'fold_{fold}_{name}_trace.csv',index=False)
            pd.DataFrame(rows).to_csv(HERE/'outer_metrics.csv',index=False)
            pd.DataFrame(scores).to_csv(HERE/'scores_by_beer.csv',index=False)
            print(json.dumps(dict(fold=fold,model=name,elapsed_seconds=time.perf_counter()-t0,**record)),flush=True)
    out=pd.DataFrame(rows);bs=pd.DataFrame(scores)
    summary=out.groupby('model').mean(numeric_only=True)
    for name in summary.index:
        this=bs[bs.model==name].sort_values('beer');ref=bs[bs.model=='interval_reference'].sort_values('beer')
        d,l,u=paired_ci(this.energy.to_numpy(),ref.energy.to_numpy())
        summary.loc[name,'beer_weighted_energy']=this.energy.mean()
        summary.loc[name,'paired_energy_delta']=d;summary.loc[name,'energy_delta_ci_lower']=l;summary.loc[name,'energy_delta_ci_upper']=u
    summary.to_csv(HERE/'summary.csv')
    # Exact reproduction of reference diagnostic is an important comparability gate.
    legacy=pd.read_csv(AW/'q3/extension_outer_comparison.csv')
    for row in out[out.model=='interval_reference'].itertuples():
        ref=legacy[(legacy.model=='interval_gaussian')&(legacy.outer_fold==row.fold)].iloc[0]
        for col in ['energy','variogram','c2st_auc','nn_ratio']:
            np.testing.assert_allclose(getattr(row,col),ref[col],atol=1e-12,rtol=1e-12)
    (HERE/'run_summary.json').write_text(json.dumps(dict(completed=True,reference_reproduction=True,elapsed_seconds=time.perf_counter()-t0),indent=2))
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
