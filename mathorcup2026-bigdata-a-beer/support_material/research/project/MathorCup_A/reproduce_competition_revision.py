#!/usr/bin/env python3
"""Reproduce the competition revision using saved models and official data.

Default: recompute revision diagnostics and optimizations, without retraining
the saved Q1/Q3 models. --verify-only audits the already saved results.
All package paths and subprocess working directories derive from this file.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'A_work'
REVISION = WORK / 'competition_revision'
STYLE = REVISION / 'style_feasibility'

# Hashes of the exact supplied inputs; caches in this package belong to these
# measurements. Different data requires a new fitting/evaluation workflow.
INPUT_SHA256 = {
    '1.csv': '419849227a02e0c792cd405e6865035b7f58af1a92efa362dbb68a7ca2882f69',
    '2.csv': '415709ac10f2f730d31f249bd6ac108250bdfbfd62e0055f4d8db9b41ec9cc4c',
    '3.xlsx': 'fe4dcabd48431e6d677c095962ea9c447e10e509c4f3f8dd511a48c5541ab587',
    '4.csv': '43a7b38682ed81ce64d97ebc1874c09dffac023ade4877f34e3c83ef7d42dc1c',
    '5.csv': 'c512eed108de6f98e0c6ed66497ed2319745d30ca7f6f4acac4f175a4bdcde9a',
    '6.csv': 'a55004fb0390299a5c8089f0c37d3bfc006e7049963c84d7d9652affabe614ae',
    '7.csv': '2083138df46640e20b46f7e76ec32cf9b380fd34e019269afd819edeb546ff5e',
    '8.xlsx': 'f2316766792111d98281bd35c8403682be8c8069d166a7a7d5887ffbb227700a',
}
FROZEN_SHA256 = {
    'q1/sensory_ensemble.joblib': '7dc738a32c08226de4a0c6ace36c703c32bfe77cf2beaac23cb8b4ea18cc3667',
    'q3/transfer_and_augmentation_models.joblib': 'b2a569a81f26f6c7470bd92c2963f7401bd7ad39e212e95a27fd3fe211a32bbb',
}
STYLE_SNAPSHOT_SHA256 = '69b25690c4e90df3ef173986b2e794c01768dc287ee9ef1542c161c8291dbb32'


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def dump(value, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def resolve_data(value: str | None) -> Path:
    configured = value or os.environ.get('DATA_DIR') or os.environ.get('MATHORCUP_DATA_DIR')
    data = Path(configured).expanduser() if configured else ROOT / 'data/A'
    # Relative user overrides are package-relative, independent of the caller's cwd.
    return (data if data.is_absolute() else ROOT / data).resolve()


def plan(verify_only: bool, build_report: bool, data: Path):
    if verify_only:
        stages = [
            ('audit_style', STYLE / 'audit_style_feasibility.py', []),
            ('verify_budget', REVISION / 'verification/verify_budget_transfer.py', []),
        ]
    else:
        stages = [
            ('analyze_saved_outputs', REVISION / 'analyze_task_outputs.py', []),
            ('budget_transfer', REVISION / 'budget_transfer/run_budget_transfer.py', ['--data', str(data)]),
            ('style_feasibility', STYLE / 'run_style_feasibility.py', []),
            # Joint selection requires this run's audit_selected_candidates.csv.
            ('audit_style', STYLE / 'audit_style_feasibility.py', []),
            ('select_joint_candidates', STYLE / 'select_joint_candidates.py', []),
            ('validate_fixed_classifier', STYLE / 'validate_fixed_classifier.py', []),
            ('make_diagnostic_artifacts', STYLE / 'make_diagnostic_artifacts.py', []),
            ('make_revision_figures', REVISION / 'make_revision_figures.py', []),
            ('verify_budget', REVISION / 'verification/verify_budget_transfer.py', []),
        ]
    if build_report:
        stages.append(('build_docx', REVISION / 'build_competition_report.py', []))
    return stages


def preflight(data: Path, stages, verify_only: bool, build_report: bool):
    missing = [str(data / name) for name in INPUT_SHA256 if not (data / name).is_file()]
    missing += [str(path) for _, path, _ in stages if not path.is_file()]
    missing += [str(WORK / name) for name in FROZEN_SHA256 if not (WORK / name).is_file()]
    if missing:
        raise FileNotFoundError('Required files are missing:\n' + '\n'.join(missing) +
            '\nPlace the official eight attachments in data/A or provide --data-dir / DATA_DIR.')
    input_hashes = {name: sha256(data / name) for name in INPUT_SHA256}
    wrong = [name for name in INPUT_SHA256 if input_hashes[name] != INPUT_SHA256[name]]
    if wrong:
        raise ValueError('Official input hashes do not match the saved-model study: ' + ', '.join(wrong) +
            '. This entry point does not refit the old models for altered inputs.')
    model_hashes = {name: sha256(WORK / name) for name in FROZEN_SHA256}
    wrong_models = [name for name in FROZEN_SHA256 if model_hashes[name] != FROZEN_SHA256[name]]
    if wrong_models:
        raise ValueError('Frozen model hashes differ: ' + ', '.join(wrong_models))
    if verify_only:
        classifier = STYLE / 'frozen_style_classifier.joblib'
        if not classifier.is_file() or sha256(classifier) != STYLE_SNAPSHOT_SHA256:
            raise ValueError('The saved fixed classifier is missing or does not match its recorded SHA-256.')
    if build_report and shutil.which('pandoc') is None:
        raise FileNotFoundError('--build-report requires the external pandoc command in PATH.')
    return dict(input_sha256=input_hashes, frozen_model_sha256=model_hashes)


def audit_saved_summaries(data: Path):
    """Recalculate existing joint selection and CV accounting without fitting.

    The class-only profiles are audited by audit_style_feasibility first. Here
    each joint profile must equal its already audited persisted origin or the
    unchanged measured source, and each selected objective must be the minimum
    among eligible saved paths. CV metrics are reconstructed from saved OOF.
    """
    import numpy as np
    import pandas as pd

    def read(name):
        return pd.read_csv(STYLE / name, float_precision='round_trip')

    def require(condition, message):
        if not bool(condition):
            raise AssertionError(message)

    def numbers_equal(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))

    base = pd.read_csv(data / '1.csv').set_index('beer')
    sources = set(base.index[base.tasting_category_fine == 'Pils/Lager'])
    candidates = read('candidate_checks.csv')
    joint = read('joint_selected_diagnostics.csv')
    profiles = read('joint_selected_profiles.csv')
    parent_profiles = read('selected_profiles.csv')
    parent_diag = read('selected_diagnostics.csv')
    parent_audit = read('audit_selected_candidates.csv')
    supports = read('support_definitions.csv')
    parent_sensory = read('selected_sensory_predictions.csv')
    joint_sensory = read('joint_selected_sensory_predictions.csv')
    require(parent_audit.pass_audit.all(), 'An origin candidate failed the independent style audit.')
    require(joint[['beer','method']].equals(profiles[['beer','method']]), 'Joint profiles and diagnostics are misaligned.')
    require(joint[['beer','method']].equals(joint_sensory[['beer','method']]), 'Joint sensory rows are misaligned.')
    require(set(map(tuple, joint[['beer','method']].to_numpy())) == set(map(tuple, supports[['beer','method']].to_numpy())),
            'Joint results do not retain every source/support pair.')
    require(len(joint)==121 and not joint.duplicated(['beer','method']).any(), 'Expected 121 distinct joint results.')
    numeric = base.select_dtypes(include='number').columns.tolist()
    # beer_id may be numeric or a label; retain every numerical chemical column.
    numeric = [c for c in numeric if c in profiles and c in parent_profiles]
    sensory_cols = [c for c in joint_sensory if c not in ['beer','method']]
    origin = {int(r.candidate_index):i for i,r in parent_diag.iterrows() if r.accepted}
    fallback_origin = {int(b):int(g[g.source_fallback].index[0]) for b,g in parent_diag.groupby('beer')}
    predicate = candidates.accepted & (candidates.target_distance_reduction > 1e-8)
    for i, row in joint.iterrows():
        pool = candidates[(candidates.beer==row.beer)&(candidates.allowed_coordinates<=row.allowed_coordinates)&predicate]
        if len(pool):
            chosen = pool.sort_values(['quadratic_objective','raw_direct_fields_changed','margin','alpha'],kind='stable').iloc[0]
            require(row.joint_accepted and not row.source_fallback and int(row.candidate_index)==int(chosen.candidate_index),
                    f'Joint selection mismatch: beer {row.beer}, {row.method}')
            idx = origin[int(row.candidate_index)]
            require(abs(row.quadratic_objective-chosen.quadratic_objective)<1e-10, 'Joint objective differs from its saved minimum.')
            expected = parent_profiles.loc[idx,numeric]
        else:
            require(row.source_fallback and not row.joint_accepted and row.target_distance_reduction==0., 'Invalid joint fallback.')
            idx = fallback_origin[int(row.beer)]
            expected = base.loc[int(row.beer),numeric]
        require(numbers_equal(profiles.loc[i,numeric],expected), 'Joint profile differs from its audited origin/source.')
        require(numbers_equal(joint_sensory.loc[i,sensory_cols],parent_sensory.loc[idx,sensory_cols]),
                'Joint sensory output differs from its audited origin/source.')
    summary = read('joint_budget_summary.csv').set_index('budget')
    for k, group in joint.groupby('allowed_coordinates'):
        require(len(group)==11 and set(group.beer)==sources, f'Budget {k} omitted a source.')
        accepted = group[group.joint_accepted]
        saved = summary.loc[int(k)]
        require(int(saved.joint_accepted)==len(accepted) and int(saved.source_fallbacks)==int(group.source_fallback.sum()),
                f'Joint count mismatch at budget {k}.')
        require(abs(saved.all_11_mean_distance_reduction-group.target_distance_reduction.mean())<1e-12,
                f'All-source mean mismatch at budget {k}.')
        require((pd.isna(saved.accepted_conditional_mean_distance_reduction) and not len(accepted)) or
                abs(saved.accepted_conditional_mean_distance_reduction-accepted.target_distance_reduction.mean())<1e-12,
                f'Conditional mean mismatch at budget {k}.')
    # OOF classification validation is audited from saved predictions, not rerun.
    oof = read('classifier_cv_oof_predictions.csv')
    cv = json.loads((STYLE/'classifier_cv_summary.json').read_text())
    protocol = json.loads((STYLE/'classifier_cv_protocol.json').read_text())
    splits = read('classifier_cv_splits.csv')
    require(len(oof)==250 and oof.beer.nunique()==250 and set(oof.beer)==set(base.index), 'OOF coverage is incomplete.')
    require(np.array_equal(oof.style_observed.to_numpy(),base.loc[oof.beer,'tasting_category_fine'].to_numpy()), 'OOF labels mismatch.')
    classes = sorted(oof.style_observed.unique())
    require(len(classes)==22, 'Expected 22 observed classes.')
    cm = pd.crosstab(oof.style_observed,oof.style_oof_predicted).reindex(index=classes,columns=classes,fill_value=0)
    saved_cm = pd.read_csv(STYLE/'classifier_cv_confusion.csv',index_col=0).loc[classes,classes]
    require(np.array_equal(cm.to_numpy(),saved_cm.to_numpy()), 'OOF confusion matrix does not match saved predictions.')
    accuracy = float(np.trace(cm.to_numpy())/len(oof))
    balanced = float(np.mean(np.diag(cm.to_numpy())/cm.sum(axis=1).to_numpy()))
    tp = int(cm.loc['Tripel','Tripel']);fp=int(cm['Tripel'].sum()-tp);fn=int(cm.loc['Tripel'].sum()-tp)
    require(abs(accuracy-cv['accuracy'])<1e-12 and abs(balanced-cv['balanced_accuracy'])<1e-12, 'CV summary mismatch.')
    require((tp,fp,fn)==(cv['Tripel_TP'],cv['Tripel_FP'],cv['Tripel_FN']), 'Tripel confusion accounting mismatch.')
    require(abs(tp/(tp+fp)-cv['Tripel_precision'])<1e-12 and abs(tp/(tp+fn)-cv['Tripel_recall'])<1e-12,
            'Tripel precision/recall mismatch.')
    for fold, rows in splits.groupby('fold'):
        training = set(rows.loc[rows.split=='train','beer'])
        validation = set(rows.loc[rows.split=='validation','beer'])
        require(not training & validation and training | validation == set(base.index), 'CV split overlaps or omits samples.')
        require(validation==set(oof.loc[oof.fold==fold,'beer']), 'CV split and OOF fold IDs disagree.')
    prob = oof[['p_'+c for c in classes]].to_numpy(float)
    require(np.isfinite(prob).all() and np.allclose(prob.sum(axis=1),1.,rtol=0.,atol=1e-12), 'Invalid saved CV probabilities.')
    require(np.array_equal(np.asarray(classes)[prob.argmax(axis=1)],oof.style_oof_predicted), 'OOF argmax mismatch.')
    require(sha256(STYLE/'frozen_style_classifier.joblib')==protocol['frozen_classifier_sha256'], 'The classifier changed after CV.')
    verification = json.loads((REVISION/'verification/review.json').read_text())
    require(verification['n_failed']==0, 'The budget/grading/module verification reported failed checks.')
    return dict(joint_profiles_checked=len(joint),joint_saved_candidate_minima_checked=len(joint),
                joint_budget_summaries_checked=len(summary),classifier_oof_rows=len(oof),
                classifier_confusion_cells=len(classes)**2,classifier_accuracy=accuracy,
                classifier_balanced_accuracy=balanced,Tripel_TP=tp,Tripel_FP=fp,Tripel_FN=fn,
                budget_grading_module_checks=int(verification['n_checks']),all_passed=True)


def run_stage(label: str, path: Path, extra, env, folder: Path):
    command = [sys.executable, str(path), *extra]
    log = folder / (label + '.log')
    print(f'\nRUN {label}: {shlex.join(command)}', flush=True)
    started = time.perf_counter()
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                   text=True,encoding='utf-8',errors='replace',bufsize=1)
        for line in process.stdout:
            print(line,end='',flush=True)
            stream.write(line)
        returncode = process.wait()
    return dict(stage=label,script=str(path.relative_to(ROOT)),arguments=extra,
                seconds=time.perf_counter()-started,returncode=returncode,log=str(log.relative_to(ROOT)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir','--data',dest='data_dir',help='Official attachments directory; relative paths are resolved from the package root.')
    parser.add_argument('--verify-only',action='store_true',help='Audit saved results only; skip candidate search, figure regeneration and classifier CV fitting.')
    parser.add_argument('--build-report',action='store_true',help='Also build the editable DOCX report using Pandoc; no PDF appendix assembly.')
    parser.add_argument('--dry-run',action='store_true',help='Print the exact stage order and commands without running or writing files.')
    args = parser.parse_args()
    data = resolve_data(args.data_dir)
    stages = plan(args.verify_only,args.build_report,data)
    print('Package root:',ROOT,flush=True)
    print('Official data:',data,flush=True)
    print('Mode:','verify existing artifacts' if args.verify_only else 'recompute revision with saved Q1/Q3 models',flush=True)
    print('Child working directory:',ROOT,flush=True)
    if args.dry_run:
        print('Environment: DATA_DIR = MATHORCUP_DATA_DIR =',data)
        for i,(label,path,extra) in enumerate(stages,1):
            print(f'{i:02d}. {label}: {shlex.join([sys.executable,str(path),*extra])}')
        print('Final: audit saved joint selections and classifier OOF accounting; check Q1/Q3 hashes unchanged.')
        return 0
    try:
        initial = preflight(data,stages,args.verify_only,args.build_report)
    except (OSError,ValueError) as exc:
        print('Preflight failed:',exc,file=sys.stderr)
        return 2
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder = REVISION / 'reproduction' / (('verify_' if args.verify_only else 'recompute_')+tag)
    folder.mkdir(parents=True,exist_ok=False)
    env = os.environ.copy()
    env.update(DATA_DIR=str(data),MATHORCUP_DATA_DIR=str(data),MATHORCUP_Q1_DIR=str(WORK/'q1'),
               MPLCONFIGDIR=str(REVISION/'reproduction/matplotlib_cache'),PYTHONUNBUFFERED='1',
               PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
               OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    font = ROOT/'assets/fonts/NotoSansCJKsc-Regular.otf'
    if font.exists():
        env['MATHORCUP_CJK_FONT']=str(font)
    manifest = dict(mode='verify_only' if args.verify_only else 'recompute_revision',build_report=args.build_report,
                    root=str(ROOT),data_dir=str(data),python=sys.version,created_utc=tag,
                    entry_script_sha256=sha256(Path(__file__)),initial=initial,stages=[],status='running')
    logpath = folder/'execution.json';dump(manifest,logpath)
    started = time.perf_counter()
    try:
        for label,path,extra in stages:
            result = run_stage(label,path,extra,env,folder)
            manifest['stages'].append(result);dump(manifest,logpath)
            if result['returncode']:
                raise RuntimeError(f"{label} exited with status {result['returncode']}; see {result['log']}")
            if label == 'verify_budget':
                check_report = json.loads((REVISION/'verification/review.json').read_text())
                if check_report['n_failed']:
                    raise RuntimeError('The budget verification contains failed checks; stop before building a report.')
        manifest['saved_result_audit'] = audit_saved_summaries(data)
        final = {name:sha256(WORK/name) for name in FROZEN_SHA256}
        if final != initial['frozen_model_sha256']:
            raise RuntimeError('A frozen Q1/Q3 model changed during execution.')
        manifest['final_frozen_model_sha256']=final
        manifest['final_style_classifier_sha256']=sha256(STYLE/'frozen_style_classifier.joblib')
        manifest['status']='passed'
    except (OSError,ValueError,RuntimeError,AssertionError,KeyError) as exc:
        manifest['status']='failed';manifest['error']=str(exc)
        print('Execution failed:',exc,file=sys.stderr)
    finally:
        manifest['seconds']=time.perf_counter()-started;dump(manifest,logpath)
    print('\nExecution record:',logpath,flush=True)
    if manifest['status']!='passed':
        return 1
    print('Completed:',len(stages),'scripts; saved joint/CV accounting passed; frozen Q1/Q3 hashes unchanged.',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
