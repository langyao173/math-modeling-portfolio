"""Fixed budget and higher Monte Carlo budget challenges, on the same 3 folds."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from penalized_interval_em import PenalizedIntervalEM
from conditional_moments import HERE, AW
from control_problem import PROJECT, file_sha
from conditional_model import DERIVED, TECHNICAL
from evaluate_extensions import evaluate
from run_q3 import paired_ci, SEED

OUT = HERE/'penalized_em'
CONFIGS = {
    'penalized_em_8x8': dict(iterations=8, burn_sweeps=8, retained_sweeps=8),
    'penalized_em_24x32': dict(iterations=24, burn_sweeps=8, retained_sweeps=32),
}


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(PROJECT/'data/A/1.csv')
    features = [c for c in df.columns[3:] if c not in DERIVED+TECHNICAL]
    x, style = df[features].to_numpy(float), df.tasting_category_fine.to_numpy(str)
    protocol = dict(date='2026-10-24', seed=2026102409,
        trigger='Independent audit found the first conditional-moment refinement lacked the mean-shrinkage scatter correction of an exact M step; additionally its finite Monte Carlo budget did not establish convergence.',
        parameters=dict(kappa=2., nu=20.), configs=CONFIGS,
        fixed_prior='Training-only initial IntervalCopula grand latent mean and pooled covariance, frozen through all EM updates.',
        objective='One half [(n_s+nu) logdet Sigma_s + trace(K_s(mu_s) Sigma_s^-1)], K_s=expected residual sum of squares + nu Sigma_0 + kappa(mu_s-mu_0)(mu_s-mu_0)^T.',
        M_step='Exact joint minimizer of this fixed-E penalized criterion. The n*kappa/(n+kappa) mean-shift scatter term is retained.',
        E_step='Finite sequential truncated Gaussian Gibbs samples; not independent beer observations.',
        convergence='Fixed compute budget comparison, with all iteration traces. No observed-likelihood or MCEM convergence certification.',
        outer_selection='None. Two fixed challenges after earlier diagnostic results; exploratory internal validation.',
        folds=3, split_seed=SEED+300, evaluation_seed=SEED+500,
        source_sha256=file_sha(PROJECT/'data/A/1.csv'), features=features)
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows, scores = [], []
    start = time.perf_counter()
    for fold, (a, b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED+300).split(x, style)):
        for name, config in CONFIGS.items():
            checkpoint = OUT/f'fold_{fold}_{name}.joblib'
            if checkpoint.exists():
                model = joblib.load(checkpoint)
            else:
                model = PenalizedIntervalEM(kappa=2., nu=20., seed=2026102409+fold, **config).fit(x[a], style[a])
                joblib.dump(model, checkpoint, compress=3)
            metric, es, vs = evaluate(model, x[b], style[b], SEED+500+fold)
            rows.append(dict(model=name, fold=fold, n_validation=len(b), **metric))
            for j, idx in enumerate(b):
                scores.append(dict(model=name, fold=fold, beer=int(df.beer.iloc[idx]), energy=es[j], variogram=vs[j]))
            pd.DataFrame(model.refinement_trace_).to_csv(OUT/f'fold_{fold}_{name}_trace.csv', index=False)
            pd.DataFrame(rows).to_csv(OUT/'outer_metrics.csv', index=False)
            pd.DataFrame(scores).to_csv(OUT/'scores_by_beer.csv', index=False)
            print(json.dumps(dict(model=name, fold=fold, seconds=time.perf_counter()-start, **metric)), flush=True)
    table = pd.DataFrame(rows).groupby('model').mean(numeric_only=True)
    reference = pd.read_csv(HERE/'scores_by_beer.csv')
    reference = reference[reference.model == 'interval_reference'].sort_values('beer')
    ss = pd.DataFrame(scores)
    for name in table.index:
        this = ss[ss.model == name].sort_values('beer')
        assert np.array_equal(reference.beer, this.beer)
        d, low, high = paired_ci(this.energy.to_numpy(), reference.energy.to_numpy())
        table.loc[name, 'beer_weighted_energy'] = this.energy.mean()
        table.loc[name, 'paired_energy_delta'] = d
        table.loc[name, 'energy_delta_ci_lower'] = low
        table.loc[name, 'energy_delta_ci_upper'] = high
    table.to_csv(OUT/'summary.csv')
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(completed=True, seconds=time.perf_counter()-start), indent=2))
    print(table.to_string(), flush=True)


if __name__ == '__main__':
    main()
