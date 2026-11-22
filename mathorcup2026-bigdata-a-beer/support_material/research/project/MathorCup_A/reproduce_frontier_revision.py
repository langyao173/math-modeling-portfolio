#!/usr/bin/env python3
"""Portable entry point for the 2026-10-24 research revision.

Run --list for the dependency-ordered workflow. Without --recompute, verify
the delivered completion records, source inputs, and frozen model identities.
Separate interpreter arguments preserve the isolated modern-model environments.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone

PROJECT = Path(__file__).resolve().parent
FRONTIER = PROJECT / 'frontier_revision'
if not FRONTIER.exists():
    development = PROJECT.parents[1]
    if (development / 'experiments/sparse_transfer').is_dir():
        FRONTIER = development


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def stages():
    e = 'experiments/'
    return [
        ('semantic', 'base', e+'semantic_auxiliary/run_semantic_rrr.py', []),
        ('semantic_blend', 'base', e+'semantic_auxiliary/blend_nested.py', []),
        ('semantic_deploy', 'base', e+'semantic_auxiliary/fit_final.py', []),
        ('semantic_style_controls', 'base', e+'semantic_auxiliary/style_controls.py', []),
        ('semantic_style_report', 'base', e+'semantic_auxiliary/summarize_style_controls.py', []),
        ('semantic_style_audit', 'base', 'audit/audit_semantic_style_controls.py', []),
        ('missing_augmentation', 'base', e+'missingness_augmentation/run_mask_augmentation.py', []),
        ('within_style', 'base', e+'within_style_explanation/run_within_style.py', ['--bootstrap', '400']),
        ('within_fixed_alpha', 'base', e+'within_style_explanation/fixed_alpha_sensitivity.py', []),
        ('conditional_moments', 'base', e+'generation_moments/run_refinement.py', []),
        ('generation_table', 'base', 'audit/build_generation_main_table.py', []),
        ('generation_audit', 'base', 'audit/audit_generation_moments.py', []),
        ('penalized_mcem', 'base', e+'generation_moments/run_penalized_em.py', []),
        ('audit_mcem', 'base', 'audit/audit_saved_penalized_em.py', []),
        ('public_weights', 'base', e+'modern_prediction/fetch_public_weights.py', ['--verify-only']),
        ('tabicl', 'modern', e+'modern_prediction/run_tabicl_benchmark.py', []),
        ('tabicl_grades', 'modern', e+'modern_prediction/postprocess_modern_results.py', []),
        ('tabicl_audit', 'modern', e+'modern_prediction/audit_tabicl_outputs.py', []),
        ('target_subspace_audit', 'modern', e+'modern_prediction/diagnose_target_subspace.py', []),
        ('tabicl_deploy', 'modern', e+'modern_prediction/export_tabicl_context.py', []),
        ('tabdpt', 'tabdpt', e+'modern_prediction/tabdpt/run_tabdpt_benchmark.py', []),
        ('tabicl_missing', 'modern', e+'modern_prediction/run_missingness_tabicl.py', []),
        ('missingness_report', 'modern', e+'modern_prediction/write_missingness_summary.py', []),
        ('modern_finalization', 'modern', e+'modern_prediction/finalize_after_missingness.py', []),
        ('target_metric_budget', 'modern', e+'modern_prediction/diagnose_target_metric_budget.py', []),
        ('target_metric_budget_audit', 'base', 'audit/audit_target_metric_budget.py', []),
        ('modern_readonly_audit', 'base', e+'modern_prediction/audit_completed_modern_arrays.py', ['--phase', 'all', '--require-final-ready']),
        ('geometry_certificate', 'solver', e+'sparse_transfer/geometry_certificate.py', []),
        ('sparse_strict', 'solver', e+'sparse_transfer/run_joint_search.py', ['--policy', 'strict', '--scip-seconds', '180']),
        ('sparse_exceedance', 'solver', e+'sparse_transfer/run_joint_search.py', ['--policy', 'exceedance', '--scip-seconds', '180']),
        ('matched_reference', 'solver', e+'sparse_transfer/run_grid_reference.py', []),
        ('perspective', 'solver', e+'sparse_transfer/reproduce_perspective.py', ['--variant', 'lambda']),
        ('diagonal_sdp', 'solver', e+'sparse_transfer/diagonal_sdp_admm.py', []),
        ('perspective_sdp', 'solver', e+'sparse_transfer/reproduce_perspective.py', ['--variant', 'sdp']),
        ('perspective_strength_report', 'base', e+'sparse_transfer/summarize_perspective_strength.py', []),
        ('broader_certificate', 'solver', e+'sparse_transfer/certificate_without_proxy.py', ['--seconds', '30']),
        ('broader_certificate_audit', 'base', 'audit/audit_certificate_without_proxy.py', []),
        ('concentration_space', 'solver', e+'sparse_transfer/concentration_space_audit.py', []),
        ('concentration_audit', 'base', 'audit/audit_concentration_space.py', []),
        ('perspective_evidence_snapshot', 'base', e+'sparse_transfer/audit_perspective_snapshot.py', ['--stage', 'snapshot']),
        ('aggregate_sparse', 'solver', e+'sparse_transfer/aggregate_results.py', []),
        ('perspective_postaggregation_audit', 'base', e+'sparse_transfer/audit_perspective_snapshot.py', ['--stage', 'verify']),
        ('deployed_transfer_check', 'modern', e+'modern_prediction/run_deployment_transfer_challenge.py', []),
        ('deployed_transfer_report', 'modern', e+'modern_prediction/write_deployment_transfer_summary.py', []),
        ('deployed_transfer_array_audit', 'base', e+'modern_prediction/audit_deployment_transfer_outputs.py', []),
        ('deployed_transfer_join_audit', 'base', 'audit/audit_deployment_transfer_results.py', []),
        ('bootstrap_prefit', 'base', e+'transfer_robustness/run_transfer_robustness.py', ['--prefit']),
        ('bootstrap_stability', 'base', e+'transfer_robustness/run_transfer_robustness.py', []),
        ('bootstrap_report', 'base', e+'transfer_robustness/summarize_transfer_robustness.py', []),
        ('bootstrap_audit', 'base', e+'transfer_robustness/audit_completed_results.py', []),
        ('cross_policy_bound_audit', 'base', e+'sparse_transfer/cross_policy_bound_transfer/audit_cross_policy_bound_transfer.py', []),
        ('final_figures', 'base', 'visuals/rebuild_all_figures.py', ['--final']),
        ('paper', 'base', 'paper/build_manuscript.py', []),
        ('paper_render', 'base', 'paper/render_manuscript.py', ['--standard']),
        ('paper_toc', 'base', 'paper/update_toc.py', ['--max-main-pages', '30']),
        ('paper_final', 'base', 'paper/build_manuscript.py', []),
        ('paper_render_final', 'base', 'paper/render_manuscript.py', ['--standard']),
        ('paper_toc_check', 'base', 'paper/update_toc.py', ['--check', '--max-main-pages', '30']),
        ('ai_disclosure', 'base', 'paper/build_ai_disclosure.py', []),
    ]


def verify_sources():
    original = module(PROJECT / 'reproduce_competition_revision.py', 'original_identity_protocol')
    paths = {PROJECT / 'data/A' / n: h for n, h in original.INPUT_SHA256.items()}
    paths.update({PROJECT / 'A_work' / n: h for n, h in original.FROZEN_SHA256.items()})
    paths[PROJECT / 'A_work/competition_revision/style_feasibility/frozen_style_classifier.joblib'] = original.STYLE_SNAPSHOT_SHA256
    records = []
    for path, expected in paths.items():
        if not path.exists():
            raise FileNotFoundError(f'Missing official input or frozen model: {path}. The competition support ZIP omits official data; place the exact eight attachments in data/A.')
        actual = digest(path)
        if actual != expected:
            raise ValueError(f'Input/model identity changed: {path.relative_to(PROJECT)}')
        records.append(dict(path=str(path.relative_to(PROJECT)), sha256=actual))
    return records


def main(args):
    global PROJECT, FRONTIER
    plan = stages()
    if args.list:
        for name, runtime, path, extra in plan:
            print(f'{name:24s} [{runtime:6s}] {path} {shlex.join(extra)}')
        return
    source_records = verify_sources()
    if not args.recompute and not args.prepare_only:
        completion = module(FRONTIER / 'paper/completion_gate.py', 'revision_completion_gate').require_complete()
        report = dict(verified=True, scope='Input identity and complete registered experimental coverage; numerical and visual audit reports remain separately available.',
                      project=str(PROJECT), checks=completion['checks'], source_sha256=source_records)
        manifest = PROJECT / 'frontier_delivery_manifest.json'
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            for item in saved['files']:
                path = PROJECT / item['path']
                if digest(path) != item['sha256']:
                    raise ValueError(f'Delivered file changed: {item["path"]}')
            report['delivered_files_verified'] = len(saved['files'])
        readonly = FRONTIER / 'experiments/modern_prediction/audit_completed_modern_arrays.py'
        audit = subprocess.run([args.python, str(readonly), '--phase', 'all', '--require-final-ready'],
                               check=True, capture_output=True, text=True)
        report['modern_array_audit'] = json.loads(audit.stdout)
        deployment = FRONTIER / 'experiments/modern_prediction/audit_deployment_transfer_outputs.py'
        audit = subprocess.run([args.python, str(deployment)], check=True, capture_output=True, text=True)
        report['deployed_transfer_array_audit'] = json.loads(audit.stdout)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    selected = set(args.stages.split(',')) if args.stages != 'all' else {p[0] for p in plan}
    unknown = selected - {p[0] for p in plan}
    if unknown:
        raise ValueError('Unknown stages: ' + ', '.join(sorted(unknown)))
    interpreters = dict(base=args.python, modern=args.modern_python or args.python,
                        tabdpt=args.tabdpt_python or args.python, solver=args.solver_python or args.python)
    if args.work_dir is None:
        raise ValueError('--recompute/--prepare-only requires --work-dir: a separate empty replay directory. Delivered protocols and results are never overwritten.')
    replay = module(PROJECT / 'prepare_frontier_replay.py', 'frontier_replay')
    PROJECT, FRONTIER, created = replay.prepare(PROJECT, args.work_dir)
    aliases = [replay.interpreter_alias(FRONTIER / folder, interpreters[runtime])
               for folder, runtime in [('experiments/modern_prediction/venv', 'modern'),
                                       ('experiments/modern_prediction/venv_tabdpt', 'tabdpt'),
                                       ('experiments/sparse_transfer/venv', 'solver')]]
    print(json.dumps(dict(replay_workspace=str(FRONTIER), created=created,
                          historical_results_changed=False, interpreter_aliases=aliases), indent=2), flush=True)
    if args.prepare_only:
        return
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       BEER_MODERN_THREADS=str(args.modern_threads), BEER_TABDPT_THREADS=str(args.modern_threads))
    for name, runtime, path, extra in plan:
        if name not in selected:
            continue
        target = FRONTIER / path
        if not target.exists():
            raise FileNotFoundError(target)
        command = [interpreters[runtime], str(target), *extra]
        print(f'[{datetime.now(timezone.utc).isoformat()}] {name}: {shlex.join(command)}', flush=True)
        subprocess.run(command, cwd=PROJECT, env=environment, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--verify-only', action='store_true', help='Default mode; input/hash/completion checks without fitting.')
    parser.add_argument('--recompute', action='store_true', help='Execute stages in a separate development-shaped replay workspace.')
    parser.add_argument('--prepare-only', action='store_true', help='Create an empty-result replay workspace without fitting.')
    parser.add_argument('--work-dir', type=Path, help='Separate empty directory, or a previously prepared workspace to resume.')
    parser.add_argument('--stages', default='all', help='Comma-separated stage names; dependencies must already exist or be selected.')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--modern-python')
    parser.add_argument('--tabdpt-python')
    parser.add_argument('--solver-python')
    parser.add_argument('--modern-threads', type=int, default=6)
    parsed = parser.parse_args()
    if sum([parsed.verify_only, parsed.recompute, parsed.prepare_only]) > 1:
        parser.error('--verify-only, --recompute and --prepare-only are mutually exclusive.')
    main(parsed)
