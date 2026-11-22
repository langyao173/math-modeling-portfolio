"""Require complete registered experiments before building a deliverable paper.

This checks completion and coverage; it does not assign a scientific quality
score. Numerical, scientific, and visual audits remain separate requirements.
"""
from pathlib import Path
import argparse
import json
import csv
import hashlib

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / 'experiments'


def completion_status():
    records = []

    def check(name, relative, predicate):
        path = ROOT / relative if relative.startswith('audit/') else EX / relative
        result = dict(name=name, path=str(path.relative_to(ROOT)), complete=False)
        try:
            value = json.loads(path.read_text())
            result['complete'] = bool(predicate(value))
            if not result['complete']:
                result['reason'] = 'Present, but required completion or coverage is not satisfied.'
        except FileNotFoundError:
            result['reason'] = 'Completion record not yet present.'
        except (ValueError, KeyError, TypeError) as exc:
            result['reason'] = f'Invalid completion record: {exc}'
        records.append(result)

    check('All five registered TabICL candidates', 'modern_prediction/status.json',
          lambda d: d['completed_model_folds'] == d['total_model_folds'] == 25
          and len(d['completed_models']) == 5)
    check('TabICL deployable training context', 'modern_prediction/final_artifact/manifest.json',
          lambda d: d['task']['context_rows'] == 250 and d['task']['outputs'] == 50
          and (EX / 'modern_prediction/final_artifact' / d['context_file']).is_file())
    check('Second modern predictor', 'modern_prediction/tabdpt/status.json',
          lambda d: d['completed_folds'] == d['required_folds'] == 5 and 'metrics' in d)
    check('Modern predictor on all four missingness scenarios', 'modern_prediction/missingness/status.json',
          lambda d: d['complete'] and d['completed_folds'] == 5)
    check('Deployed predictor cross-check of final transfers', 'modern_prediction/deployment_transfer_challenge/COMPLETE.json',
          lambda d: d['complete'] and d['all50'] and d['source_count'] == 11)
    check('Independent deployed chemistry and output join', 'audit/deployment_transfer_independent_audit.json',
          lambda d: d['completed'] and d['passed'] and d['cases'] == 88
          and d['actual_csv_input_rows'] == 69 and d['sensory_targets'] == 50
          and d['unique_source_candidate_pairs'] == 58 and d['original_pool_unchanged'])
    check('Final candidate pool and perspective coverage', 'sparse_transfer/final_summary/COMPLETE.json',
          lambda d: d['complete'] and d['all_perspective'] and d['sdp_finished']
          and d['selections'] == 88 and d['feasible_selections'] == 88)
    check('Post-aggregation evidence and candidate identity', 'sparse_transfer/post_aggregation_integrity/verification.json',
          lambda d: d['complete'] and d['passed'] and len(d['directories']) == 2
          and d['final_summary_sha256'] == hashlib.sha256((EX/'sparse_transfer/final_summary/COMPLETE.json').read_bytes()).hexdigest()
          and d['selection_cases_sha256'] == d['source_pool_sha256']
              == hashlib.sha256((EX/'sparse_transfer/final_summary/selection_cases.csv').read_bytes()).hexdigest())
    check('SDP pilot and predeclared conditional expansion', 'sparse_transfer/perspective_sdp/DONE.json',
          lambda d: d['cases_completed'] == (22 if d['expanded'] else 1))
    for variant in ('perspective', 'perspective_sdp'):
        check(f'Independent {variant} witness and bound audit', f'sparse_transfer/{variant}/INTEGRITY_DONE.json',
              lambda d, variant=variant: d['state'] == 'complete_independently_verified'
              and d['completed_cases'] == (22 if variant == 'perspective'
                  or json.loads((EX / 'sparse_transfer/perspective_sdp/DONE.json').read_text())['expanded'] else 1)
              and d['all_full_witnesses_pass'])
    check('Independent broader-problem certificates', 'sparse_transfer/certificate_without_proxy/COMPLETE.json',
          lambda d: d['complete'] and d['cases'] == 22 and d['primary_candidate_pool_unchanged'])
    check('Independent broader-certificate reconstruction', 'audit/certificate_without_proxy_independent_audit.json',
          lambda d: d['completed'] and d['cases'] == 22 and d['only_linear_stale_row_omitted']
          and d['retained_classification_rows_per_case'] == 21 and d['all_22_warm_witnesses_pass']
          and d['primary_selection_unchanged'])
    check('Quantified-concentration subspace challenge', 'sparse_transfer/concentration_space/COMPLETE.json',
          lambda d: d['complete'] and d['source_policy_cases'] == 22 and d['margin_cases'] == 132
          and d['primary_candidate_pool_unchanged'])
    check('Full bootstrap candidate assessment', 'transfer_robustness/run_summary.json',
          lambda d: d['completed'] and d['replicates'] == 200 and d['full_sensory_refits'] == 1200
          and d['targets_per_fit'] == 50 and d['selection_cases'] == 88)
    check('Independent bootstrap arrays and cache replay', 'transfer_robustness/independent_checks.json',
          lambda d: d['completed'] and d['replicates'] == 200 and d['selection_cases'] == 88
          and d['sensory_fits'] == 1200 and d['exact_stratified_indices_reproduced']
          and d['source_hashes_unchanged'] and d['paired_old80_rho_eligibility_checked'])
    check('Cross-policy lower-bound set inclusion', 'sparse_transfer/cross_policy_bound_transfer/independent_audit.json',
          lambda d: d['completed'] and d['passed'] and d['source_cases'] == 11
          and d['coordinate_boxes_checked'] == 2475 and d['raw_defect_caps_checked'] == 44
          and d['frozen_files_unchanged'] and d['frozen_models_unchanged'])
    check('Both exact-M-step Monte Carlo challenges', 'generation_moments/penalized_em/COMPLETE.json',
          lambda d: d['completed'])
    check('Style-preserving and style-only semantic controls', 'semantic_auxiliary/style_controls/COMPLETE.json',
          lambda d: d['complete'] and d['variants'] == 6 and d['permutation_replicates'] == 5
          and d['fitted_outer_models'] == 60 and d['final_selected_predictors_unchanged'])
    check('Independent saved MCEM updates', 'audit/penalized_em_saved_model_audit.json',
          lambda d: d['completed'] and d['model_checkpoints'] == 6 and d['style_mstep_checks'] == 132
          and d['all_fixed_prior_checks_pass'] and d['all_last_M_steps_replayed'])
    check('Independent concentration certificates', 'audit/concentration_space_independent_audit.json',
          lambda d: d['completed'] and d['margin_cases'] == 132 and d['source_policy_cases'] == 22
          and d['maximum_rebuilt_matrix_error'] < 1e-10 and d['maximum_independent_dual_error'] < 1e-8)
    check('Independent semantic control replay', 'audit/semantic_style_controls_independent_audit.json',
          lambda d: d['completed'] and d['outer_models_replayed'] == 60
          and d['local_style_membership_and_validation_exclusion']
          and d['paired_CI_independent_count_weighted_reconstruction'])
    check('Independent task-weighted error decomposition', 'audit/target_metric_budget_independent_audit.json',
          lambda d: d['completed'] and d['models'] == 14 and d['fold_target_model_cells'] == 140
          and d['training_fold_projectors_reconstructed'] == 5 and d['source_hashes_match']
          and d['maximum_direct_macro_r2_error'] < 1e-10)
    expected = [(f'fold_{fold}_{name}.joblib', f'fold_{fold}_{name}_trace.csv')
                for fold in range(3) for name in ('penalized_em_8x8', 'penalized_em_24x32')]
    present = all((EX / 'generation_moments/penalized_em' / name).is_file()
                  for pair in expected for name in pair)
    records.append(dict(name='All six saved penalized MCEM models and traces', complete=present))
    for relative, rows in [('generation_moments/penalized_em/outer_metrics.csv', 6),
                           ('sparse_transfer/final_summary/selection_cases.csv', 88)]:
        path = EX / relative
        try:
            with path.open(newline='') as stream:
                actual = sum(1 for _ in csv.DictReader(stream))
        except FileNotFoundError:
            actual = 0
        records.append(dict(name=f'Exact coverage: {relative}', complete=actual == rows,
                            expected=rows, observed=actual))
    return dict(complete=all(r['complete'] for r in records), checks=records)


def require_complete():
    state = completion_status()
    failed = [r['name'] for r in state['checks'] if not r['complete']]
    if failed:
        raise RuntimeError('Final build blocked by unfinished registered work:\n' + '\n'.join(failed))
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require', action='store_true')
    args = parser.parse_args()
    report = completion_status()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require and not report['complete']:
        raise SystemExit(2)
