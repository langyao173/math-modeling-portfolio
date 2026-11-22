"""Harmonize already-frozen auxiliary results without fitting or method selection."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    base = args.root.resolve()
    seed = 2026102527
    random = np.random.default_rng(seed)
    ids = random.integers(0, 250, size=(2000, 250))
    groups = {'core30': np.arange(30), 'all50': np.arange(50), 'other20': np.arange(30, 50)}
    metrics, comparisons, sources = [], [], {}
    for phase, folder in [('standard', 'results'), ('normal_rank', 'results_rank')]:
        path = base/folder/'outer_predictions.npz'
        sources[folder] = sha(path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        y, fold = arrays['Y'], arrays['fold']
        variance = y.var(0)
        for scene in ('original', 'derived_hidden', 'MCAR10', 'MCAR30', 'MCAR50'):
            baseline = arrays['tau0p0_'+scene]
            loss0 = (baseline-y)**2/variance
            for tau in (0., .5, 1.):
                key = 'tau'+str(tau).replace('.', 'p')+'_'+scene
                prediction = arrays[key]
                loss = (prediction-y)**2/variance
                for group, js in groups.items():
                    metrics.append(dict(phase=phase, tau=tau, scenario=scene, targets=group,
                                        macro_r2=float(1-loss[:, js].mean())))
                    if tau > 0:
                        diff = (loss0-loss)[:, js].mean(1)
                        values = diff[ids].mean(1)
                        comparisons.append(dict(phase=phase, tau=tau, reference_tau=0., scenario=scene,
                                                targets=group, delta_macro_r2=float(diff.mean()),
                                                ci025=float(np.quantile(values, .025)), ci975=float(np.quantile(values, .975)),
                                                bootstrap_replications=2000, bootstrap_seed=seed,
                                                refit=False, row_unit='beer', variance='frozen pooled250'))
    m = pd.DataFrame(metrics)
    c = pd.DataFrame(comparisons)
    m.to_csv(base/'all_phases_metrics.csv', index=False, float_format='%.17g')
    c.to_csv(base/'all_phases_paired_differences.csv', index=False, float_format='%.17g')
    # Descriptive input diagnostic for the first-stage failure. It never alters
    # predictions, held-out membership, feature transforms, or reported scores.
    with np.load(args.input, allow_pickle=False) as archive:
        source = {key: archive[key] for key in archive.files}
    i = int(np.flatnonzero(source['beer'] == 156)[0])
    training = source['fold'] != source['fold'][i]
    x = source['X']
    median = np.nanmedian(x[training], axis=0)
    filled = np.where(np.isfinite(x[training]), x[training], median)
    scale = filled.std(0)
    scale[scale < 1e-12] = 1.
    standardized = (np.where(np.isfinite(x[i]), x[i], median)-filled.mean(0))/scale
    strongest = np.argsort(np.abs(standardized))[-5:][::-1]
    extreme = dict(beer=156, outer_fold=int(source['fold'][i]), diagnostic_only=True,
                   largest_feature_extrapolations=[dict(feature=str(source['features'][j]),
                     test_value=float(x[i, j]), training_min=float(np.nanmin(x[training, j])),
                     training_max=float(np.nanmax(x[training, j])),
                     training_standardized_value=float(standardized[j])) for j in strongest])
    (base/'chemical_extreme_diagnostic.json').write_text(json.dumps(extreme, ensure_ascii=False, indent=2)+'\n')
    # The experiment remains separate from the old fixed-half neural baseline.
    record = dict(status='PASS', no_model_fits=True, no_parameter_or_method_selection=True,
                  phase_order='standard prespecified, then normal_rank development amendment after first-stage failure diagnosis',
                  bootstrap_seed=seed, bootstrap_replications=2000, sources=sources,
                  first_stage_original_intervals_retained='results/paired_differences.csv (seed2026102507)',
                  harmonized_intervals='all_phases_paired_differences.csv (seed2026102527)',
                  target='official standardized Z; raw means supply auxiliary training responses only',
                  primary_rows=c[(c.tau == 1)&(c.scenario == 'original')&(c.targets == 'core30')].to_dict('records'),
                  manuscript_decision='Supplementary diagnostic only: no convincing incremental gain and all methods below strong fixed-half baseline.',
                  script_sha256=sha(Path(__file__)))
    (base/'SUMMARY.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(c[(c.scenario == 'original')&(c.tau == 1)].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parent.parent/'models/inputs/inputs.npz')
    main(parser.parse_args())
