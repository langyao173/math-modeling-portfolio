#!/usr/bin/env python3
"""Prepare a clean workspace and replay the supplied scientific source stages.

The code package intentionally contains no input data, saved predictions or
trained model binaries. This launcher reconstructs inputs from the original
eight contest attachments in a new directory. It does not rewrite historical
hashes or declare newly fitted models identical to the published frozen ones.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'project/MathorCup_A'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def official_inputs(data):
    original=module(SOURCE/'reproduce_competition_revision.py','official_input_definition')
    rows=[]
    for name,expected in original.INPUT_SHA256.items():
        p=data/name
        if not p.is_file(): raise FileNotFoundError(f'Missing contest attachment: {p}')
        actual=digest(p)
        if actual!=expected: raise ValueError(f'Attachment differs from the analyzed input: {name}')
        rows.append(dict(file=name,sha256=actual,bytes=p.stat().st_size))
    return rows

def commands(work):
    project=work/'project/MathorCup_A'
    data=project/'data/A'
    base=module(SOURCE/'reproduce_all.py','base_stage_definitions')
    # jobs() contains absolute --output arguments assembled from module WORK.
    # Rebind these only in the new launcher's imported stage definitions.
    base.ROOT=project
    base.WORK=project/'A_work'
    plan=[]
    for name,jobs in base.jobs(data).items():
        if name=='figures':continue
        plan.append((f'baseline_{name}','base',[(project/'A_work'/p,a) for p,a in jobs]))
    up=project/'A_work/nature_upgrade'
    # These controlled comparisons use the same contest folds; the source-
    # matching and external-validation research programs remain inspectable but
    # are not prerequisites for contest-data-only cold recomputation.
    plan.extend([
        ('comparison_data_claims','base',[(up/'data_claim_audit/audit_claims.py',['--data',str(data)])]),
        ('comparison_holdout_baseline','base',[(ROOT/'run_submission.py',[
            '--build-holdout-cache','--data-dir',str(data),'--work-dir',str(work)])]),
        ('comparison_adaptive','base',[(up/'adaptive/run_adaptive.py',['--data',str(data)])]),
        ('comparison_gbr','base',[(up/'benchmark/run_gbr_challenger.py',['--data',str(data)]),(up/'benchmark/fit_final_gbr.py',['--data',str(data)])]),
        ('comparison_catboost','base',[(up/'catboost/run_catboost.py',['--data',str(data)])]),
        ('comparison_fusion','base',[(up/'adaptive/prepare_fusion.py',['--cache-only']),(up/'fusion/run_fusion.py',['--variants','5,7,8','--fit-final'])]),
    ])
    old=module(SOURCE/'reproduce_competition_revision.py','competition_stage_definitions')
    for name,path,args in old.plan(False,False,data):
        relative=path.relative_to(SOURCE)
        plan.append((f'competition_{name}','base',[(project/relative,args)]))
    frontier=module(SOURCE/'reproduce_frontier_revision.py','frontier_stage_definitions')
    for name,runtime,path,args in frontier.stages():
        if name=='final_figures' or name.startswith('paper') or name=='ai_disclosure':continue
        if name=='public_weights':args=[]
        # Fresh-run audits use this run's verified identities and recomputed
        # counts. Historical snapshot auditors remain in the source appendix.
        fresh_auditors = {
            'broader_certificate_audit': 'audit/audit_certificate_without_proxy_fresh.py',
            'deployed_transfer_join_audit': 'audit/audit_deployment_transfer_results_fresh.py',
            'cross_policy_bound_audit': 'experiments/sparse_transfer/cross_policy_bound_transfer/audit_cross_policy_bound_transfer_fresh.py',
            'perspective_strength_report': 'experiments/sparse_transfer/summarize_perspective_strength_fresh.py',
            'perspective_evidence_snapshot': 'experiments/sparse_transfer/audit_perspective_snapshot_fresh.py',
            'perspective_postaggregation_audit': 'experiments/sparse_transfer/audit_perspective_snapshot_fresh.py',
        }
        path = fresh_auditors.get(name, path)
        plan.append((name,runtime,[(work/path,args)]))
        if name=='tabicl':
            plan.append(('tabicl_nested_selection','modern',[(
                work/'experiments/modern_prediction/run_nested_selection.py',
                ['--threads','2','--workers','4'])]))
    # Group diagnostics read only completed, aligned OOF predictions. They run
    # after the nested selector, semantic blend and missingness predictions.
    plan.append(('generalization_groups','modern',[(
        work/'experiments/modern_prediction/analyze_existing_oof.py',[
            '--project-dir',str(project),
            '--experiment-dir',str(work/'experiments'),
            '--extra-oof',str(work/'experiments/modern_prediction/nested_selection/nested_oof.npz'),
            '--output',str(work/'experiments/modern_prediction/generalization_groups')])]))
    return plan

def build_holdout_cache(work):
    """Rebuild the two fusion inputs using the unchanged six-model functions.

    The historical runner also compares with an external workbook after these
    arrays are written. Fusion only needs its train/test arrays, so reuse the
    exact tuning and estimator functions without invoking that reporting tail.
    No pre-existing predictions or fitted models are inserted into this cache.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    project=work/'project/MathorCup_A'
    up=project/'A_work/nature_upgrade'
    sys.path.insert(0,str(up))
    old=module(up/'run_author_split_baseline.py','holdout_baseline_functions')
    data=project/'data/A'
    x=pd.read_csv(data/'1.csv')
    y=pd.read_csv(data/'4.csv').set_index('beer').loc[x.beer].reset_index()
    assert np.array_equal(x.beer.to_numpy(),y.beer.to_numpy())
    X=x.iloc[:,3:].to_numpy(float);Y=y.iloc[:,3:].to_numpy(float)
    tr,te=train_test_split(np.arange(len(x)),test_size=.30,random_state=0,
        shuffle=True,stratify=x.tasting_category_fine)
    candidates,weights,_,inner=old.tune_five(X[tr],Y[tr],20261023)
    models=[old.SensoryEstimator(candidates[f][1]).fit(X[tr],Y[tr]) for f in old.FAMILIES]
    predictions={f:m.predict(X[te]) for f,m in zip(old.FAMILIES,models)}
    predictions['ensemble']=sum(weights[i]*predictions[f] for i,f in enumerate(old.FAMILIES))
    predictions['training_mean']=np.broadcast_to(Y[tr].mean(axis=0),Y[te].shape).copy()
    predictions['style_mean']=np.array([Y[tr][
        x.iloc[tr].tasting_category_fine.to_numpy()==x.iloc[i].tasting_category_fine].mean(axis=0) for i in te])
    inner_predictions={f:candidates[f][2] for f in old.FAMILIES}
    inner_predictions['ensemble']=sum(w*inner_predictions[f] for w,f in zip(weights,old.FAMILIES))
    inner_id=np.full(len(tr),-1)
    for k,(_,va) in enumerate(inner):inner_id[va]=k
    out=up/'author_protocol_baseline';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'holdout_arrays.npz',Y=Y[te],beer=x.iloc[te].beer.to_numpy(),
        train_rows=tr,test_rows=te,**predictions)
    np.savez_compressed(out/'inner_selection_arrays.npz',Y=Y[tr],beer=x.iloc[tr].beer.to_numpy(),
        inner_fold=inner_id,**inner_predictions)
    record=dict(purpose='Dependency arrays for the historical fusion comparison; fresh training with unchanged functions.',
        source_code_sha256=digest(up/'run_author_split_baseline.py'),
        source_input_sha256={n:digest(data/n) for n in ('1.csv','4.csv')},
        split=dict(train_n=len(tr),test_n=len(te),source_row_order=True,seed=0,stratified=True),
        inner=dict(n_splits=5,shuffle=True,seed=20261023),
        reporting_scope='No external-workbook comparison is performed.',
        outputs={n:digest(out/n) for n in ('holdout_arrays.npz','inner_selection_arrays.npz')})
    (out/'cold_dependency_record.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2),flush=True)

def prepare(work,data):
    inputs=official_inputs(data)
    marker=work/'COLD_RECOMPUTATION.json'
    if work==ROOT or ROOT in work.parents or work in ROOT.parents:
        raise ValueError('Choose a separate workspace outside the source package.')
    if work.exists() and any(work.iterdir()):
        if not marker.is_file():raise FileExistsError('Existing work directory has no preparation marker; choose an empty directory.')
        saved=json.loads(marker.read_text())
        if saved['inputs']!=inputs:raise ValueError('Existing workspace belongs to different official inputs.')
        if saved['source_manifest_sha256']!=digest(ROOT/'SOURCE_MANIFEST.json'):
            raise ValueError('Source package changed; use a new workspace.')
        for item in inputs:
            if digest(work/'project/MathorCup_A/data/A'/item['file'])!=item['sha256']:
                raise ValueError('Prepared input was modified: '+item['file'])
        return False
    work.mkdir(parents=True,exist_ok=True)
    for p in sorted(SOURCE.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(SOURCE)
        target=work/Path(*rel.parts[1:]) if rel.parts[0]=='frontier_revision' else work/'project/MathorCup_A'/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
    target=work/'project/MathorCup_A/data/A'
    target.mkdir(parents=True,exist_ok=True)
    for item in inputs:shutil.copy2(data/item['file'],target/item['file'])
    marker.write_text(json.dumps(dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        inputs=inputs,source_manifest_sha256=digest(ROOT/'SOURCE_MANIFEST.json'),
        policy='Fresh cold recomputation from original contest attachments and source code. No historical fitted model identity is asserted; historical source and acceptance gates remain unchanged.'),ensure_ascii=False,indent=2))
    return True

def main(args):
    work=args.work_dir.expanduser().resolve() if args.work_dir else Path('/path/to/fresh_run')
    plan=commands(work)
    if args.list:
        for name,runtime,jobs in plan:
            print(f'{name} [{runtime}]')
            for path,extra in jobs: print('  '+shlex.join([runtime+'_python',str(path),*extra]))
        return
    if args.data_dir is None:raise ValueError('--data-dir must point to the original eight attachments.')
    data=args.data_dir.expanduser().resolve()
    if args.check_input:
        print(json.dumps({'all_eight_inputs_match':True,'inputs':official_inputs(data)},ensure_ascii=False,indent=2))
        return
    if args.work_dir is None:raise ValueError('--work-dir must be a separate empty directory, or the prepared directory to resume.')
    if args.build_holdout_cache:
        if not (work/'COLD_RECOMPUTATION.json').is_file():
            raise FileNotFoundError('Prepare the cold workspace before building the fusion holdout dependency.')
        official_inputs(data)
        build_holdout_cache(work)
        return
    created=prepare(work,data)
    runtimes=dict(base=args.python,solver=args.solver_python or args.python,
                  modern=args.modern_python or args.python,tabdpt=args.tabdpt_python or args.python)
    replay=module(SOURCE/'prepare_frontier_replay.py','interpreter_aliases')
    for directory,runtime in [('experiments/sparse_transfer/venv','solver'),
        ('experiments/modern_prediction/venv','modern'),('experiments/modern_prediction/venv_tabdpt','tabdpt')]:
        replay.interpreter_alias(work/directory,runtimes[runtime])
    print(json.dumps({'work_dir':str(work),'created':created,'prepared':True},ensure_ascii=False),flush=True)
    if args.prepare_only:return
    names={x[0] for x in plan}
    selected=names if args.stages=='all' else set(args.stages.split(','))
    if selected-names:raise ValueError('Unknown stages: '+','.join(sorted(selected-names)))
    project=work/'project/MathorCup_A'
    env=os.environ.copy()
    env.update(MATHORCUP_DATA_DIR=str(project/'data/A'),MATHORCUP_PROJECT_DIR=str(project),
        MATHORCUP_Q1_DIR=str(project/'A_work/q1'),OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(work/'mpl_cache'),
        BEER_MODERN_THREADS=str(args.threads),BEER_TABDPT_THREADS=str(args.threads))
    logs=work/'execution_logs';logs.mkdir(exist_ok=True)
    for name,runtime,jobs in plan:
        if name not in selected:continue
        with (logs/(name+'.log')).open('a',encoding='utf-8') as handle:
            for path,extra in jobs:
                if not path.is_file():raise FileNotFoundError(path)
                command=[runtimes[runtime],str(path),*extra]
                print(name+': '+shlex.join(command),flush=True)
                handle.write('\n'+datetime.now(timezone.utc).isoformat()+' '+shlex.join(command)+'\n');handle.flush()
                process=subprocess.Popen(command,cwd=project,env=env,stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
                for line in process.stdout:
                    print(line,end='',flush=True);handle.write(line);handle.flush()
                code=process.wait()
                if code:raise subprocess.CalledProcessError(code,command)
        record=dict(stage=name,finished_utc=datetime.now(timezone.utc).isoformat(),
                    result='stage completed in fresh workspace',source_manifest_sha256=digest(ROOT/'SOURCE_MANIFEST.json'))
        with (logs/'completed_stages.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list',action='store_true',help='Print dependency-ordered stage commands without data or training.')
    p.add_argument('--check-input',action='store_true',help='Verify only the eight original attachment hashes.')
    p.add_argument('--prepare-only',action='store_true',help='Create an empty-result workspace, without training or downloading.')
    p.add_argument('--build-holdout-cache',action='store_true',help='Recompute the training/holdout arrays required by the fusion comparison, using the existing six-model functions.')
    p.add_argument('--data-dir',type=Path)
    p.add_argument('--work-dir',type=Path)
    p.add_argument('--stages',default='all',help='all or comma-separated stage names; earlier dependencies must already be complete.')
    p.add_argument('--python',default=sys.executable)
    p.add_argument('--solver-python')
    p.add_argument('--modern-python')
    p.add_argument('--tabdpt-python')
    p.add_argument('--threads',type=int,default=6)
    main(p.parse_args())
