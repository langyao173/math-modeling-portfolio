"""Read-only independent auxiliary-array and constrained-CDF checks."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def enumerate_projection(vector):
    """Enumerate all 32 contiguous constant-block partitions of six values."""
    best, best_cost = None, np.inf
    for mask in range(32):
        cuts = [0]+[index+1 for index in range(5) if mask & (1 << index)]+[6]
        values = np.concatenate([np.full(stop-start, np.clip(vector[start:stop].mean(), 0, 1))
                                 for start, stop in zip(cuts[:-1], cuts[1:])])
        if np.any(np.diff(values) < -1e-14):
            continue
        cost = float(np.sum((values-vector)**2))
        if cost < best_cost:
            best, best_cost = values, cost
    assert best is not None
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--auxiliary', type=Path, required=True)
    parser.add_argument('--model-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = load(args.inputs)
    files = [args.inputs, args.model_source, args.auxiliary/'all_phases_metrics.csv',
             args.auxiliary/'all_phases_paired_differences.csv']
    for phase in ('results', 'results_rank'):
        files += [args.auxiliary/phase/name for name in
                  ('outer_predictions.npz', 'training_identity_trace.json',
                   'inner_selection_grid.csv', 'selected_parameters.csv')]
    before = {str(path): sha(path) for path in files}
    # Only the six-value projection function is executed for this component
    # check; learner construction, model fitting and study main() are absent.
    module = ast.parse(args.model_source.read_text())
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'pava')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(args.model_source), 'exec'), namespace)
    projection = namespace['pava']
    rng = np.random.default_rng(2026102531)
    cases = np.vstack([rng.normal(.5, .7, size=(100, 6)),
                       [.4, .3, .2, .6, .1, 1.2], [-.2, -.1, .3, .2, 1.5, 1.4],
                       np.zeros(6), np.ones(6), np.arange(6)/5])
    actual = projection(cases)
    expected = np.stack([enumerate_projection(row) for row in cases])
    projection_error = float(np.max(np.abs(actual-expected)))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13)
    legitimate = np.cumsum(rng.dirichlet(np.ones(7), size=len(cases)), axis=1)[:, :-1]
    margin = np.sum((cases-legitimate)**2, axis=1)-np.sum((actual-legitimate)**2, axis=1)-np.sum((cases-actual)**2, axis=1)
    assert margin.min() > -1e-12
    # Recompute both stages with the common paired beer blocks. Source data,
    # splits and selected candidates remain immutable.
    indices = np.random.default_rng(2026102527).integers(0, 250, size=(2000, 250))
    groups = {'core30': np.arange(30), 'all50': np.arange(50), 'other20': np.arange(30, 50)}
    metric_rows, contrast_rows = [], []
    traces_verified, choices_verified = 0, 0
    for phase, folder in [('standard', 'results'), ('normal_rank', 'results_rank')]:
        base = args.auxiliary/folder
        arrays = load(base/'outer_predictions.npz')
        for field, source in [('beer', 'beer'), ('fold', 'fold'), ('targets', 'targets'),
                              ('features', 'features'), ('raw_mean', 'raw_mean'), ('Y', 'Yz')]:
            np.testing.assert_array_equal(arrays[field], data[source])
        trace = json.loads((base/'training_identity_trace.json').read_text())
        assert len(trace) == 15
        for row in trace:
            outer, inner = row['outer_fold'], row['inner_fold']
            test = np.flatnonzero(data['fold'] == outer)
            validation = np.flatnonzero(data['inner_fold'][outer] == inner)
            train = np.flatnonzero((data['fold'] != outer) & (data['inner_fold'][outer] != inner))
            np.testing.assert_array_equal(row['training_beers'], data['beer'][train])
            np.testing.assert_array_equal(row['validation_beers'], data['beer'][validation])
            np.testing.assert_array_equal(row['outer_test_beers'], data['beer'][test])
            traces_verified += 1
        grid = pd.read_csv(base/'inner_selection_grid.csv', float_precision='round_trip')
        choices = pd.read_csv(base/'selected_parameters.csv', float_precision='round_trip')
        for choice in choices.to_dict('records'):
            group = grid[(grid.outer_fold == choice['outer_fold']) & (grid.tau == choice['tau'])]
            assert len(group) == 16
            assert abs(group.inner_loss.min()-choice['inner_loss']) < 1e-12
            selected = group[(group.alpha == choice['alpha']) & (group['rank'] == choice['rank'])]
            assert len(selected) == 1 and abs(selected.iloc[0].inner_loss-choice['inner_loss']) < 1e-12
            choices_verified += 1
        y = data['Yz']
        variance = y.var(axis=0)
        for scene in ('original', 'derived_hidden', 'MCAR10', 'MCAR30', 'MCAR50'):
            base_prediction = arrays['tau0p0_'+scene]
            for tau in (0., .5, 1.):
                prediction = arrays['tau'+str(tau).replace('.', 'p')+'_'+scene]
                assert np.isfinite(prediction).all()
                for group, js in groups.items():
                    square = (prediction[:, js]-y[:, js])**2
                    r2 = 1-np.mean(square/variance[js])
                    metric_rows.append(dict(phase=phase, tau=tau, scenario=scene, targets=group, macro_r2=r2))
                    if tau:
                        change = (((base_prediction[:, js]-y[:, js])**2-square)/variance[js]).mean(axis=1)
                        boot = change[indices].mean(axis=1)
                        lower, upper = np.quantile(boot, [.025, .975])
                        contrast_rows.append(dict(phase=phase, tau=tau, reference_tau=0., scenario=scene,
                                                  targets=group, delta_macro_r2=change.mean(), ci025=lower, ci975=upper))
    metrics = pd.DataFrame(metric_rows)
    contrasts = pd.DataFrame(contrast_rows)
    stored_metrics = pd.read_csv(args.auxiliary/'all_phases_metrics.csv', float_precision='round_trip')
    stored_contrasts = pd.read_csv(args.auxiliary/'all_phases_paired_differences.csv', float_precision='round_trip')
    keys = ['phase', 'tau', 'scenario', 'targets']
    joined = metrics.merge(stored_metrics, on=keys, suffixes=('_audit', '_stored'), validate='one_to_one')
    assert len(joined) == len(metrics) == len(stored_metrics) == 90
    metric_error = float(np.max(np.abs(joined.macro_r2_audit-joined.macro_r2_stored)))
    assert metric_error < 1e-12
    joined = contrasts.merge(stored_contrasts, on=keys, suffixes=('_audit', '_stored'), validate='one_to_one')
    assert len(joined) == len(contrasts) == len(stored_contrasts) == 60
    contrast_error = max(float(np.max(np.abs(joined[name+'_audit']-joined[name+'_stored'])))
                         for name in ('delta_macro_r2', 'ci025', 'ci975'))
    assert contrast_error < 1e-12
    metrics.to_csv(args.output/'auxiliary_metrics_independent.csv', index=False)
    contrasts.to_csv(args.output/'auxiliary_contrasts_independent.csv', index=False)
    assert all(sha(path) == before[str(path)] for path in files)
    report = dict(status='PASS', pava_cases=len(cases), pava_global_partition_checks_per_case=32,
                  pava_global_optimum_max_error=projection_error, projection_inequality_min_margin=float(margin.min()),
                  auxiliary_metric_rows=len(metrics), auxiliary_metric_max_error=metric_error,
                  auxiliary_contrast_rows=len(contrasts), auxiliary_contrast_max_error=contrast_error,
                  training_split_traces_verified=traces_verified, selected_grid_rows_verified=choices_verified,
                  auxiliary_phase_order='standard first; normal-rank second-stage development amendment retained',
                  bootstrap_seed=2026102527, bootstrap_resamples=2000,
                  new_predictive_training_fits=0, new_parameter_searches=0,
                  component_scope='Synthetic six-coordinate projection verification only; not a predictive performance experiment.',
                  input_sha256=before, sources_unchanged=True, audit_script_sha256=sha(Path(__file__)))
    (args.output/'COMPONENT_AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'input_sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
