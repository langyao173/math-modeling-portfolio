"""Challenge the actual CSV deployment on the frozen transfer candidates.

The final chemistry pool is fixed before any f+ prediction. Its selection and
the original f-star gate are never changed. All 50 outputs are inferred for
the exact-chemistry-deduplicated candidate rows and 11 source rows together.
This is cross-predictor sensitivity analysis, not independent wet-lab truth.
"""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name] = '1'
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE/'deployment_transfer_challenge'
SPARSE = HERE.parent/'sparse_transfer'
SUMMARY = SPARSE/'final_summary'
TARGET = 'stale_hops'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return {}


def status(state,**kwargs):
    value = dict(state=state,timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kwargs)
    write_json(OUT/'status.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def ready():
    modern = read_json(HERE/'FINAL_READY.json')
    missing = read_json(HERE/'missingness/status.json')
    summary = read_json(SUMMARY/'COMPLETE.json')
    return (modern.get('state') == 'all_registered_modern_prediction_work_complete'
            and missing.get('complete') is True
            and all(summary.get(k) is True for k in ('complete','all_perspective','sdp_finished'))
            and (SPARSE/'perspective/INTEGRITY_DONE.json').exists()
            and (SPARSE/'perspective_sdp/INTEGRITY_DONE.json').exists())


def true_column(series):
    return series.astype(str).str.lower().eq('true')


def profile_hash(values):
    v = np.asarray(values,dtype='<f8').copy()
    mask = np.isnan(v)
    v[mask] = 0.
    v[v == 0.] = 0.  # canonicalize signed zero; preserve all other finite bits
    return hashlib.sha256(mask.tobytes()+v.tobytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    lock = (OUT/'run.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'COMPLETE.json').exists():
        done = read_json(OUT/'COMPLETE.json')
        assert done.get('complete') is True
        assert done['candidate_pool_sha256'] == sha(SUMMARY/'selection_cases.csv')
        print('ALREADY_COMPLETE '+json.dumps(done),flush=True)
        return
    status('waiting_for_modern_completion_and_frozen_transfer_pool')
    start = time.monotonic()
    while not ready():
        if time.monotonic()-start > 8*3600:
            raise TimeoutError('Deployment or audited frozen transfer pool did not finish within eight hours.')
        time.sleep(15)

    frozen = ['COMPLETE.json','selection_cases.csv','selected_profiles.csv','selected_full_checks.csv']
    source_hashes = {name:sha(SUMMARY/name) for name in frozen}
    gate = read_json(SUMMARY/'COMPLETE.json')
    assert gate['source_pool_sha256'] == source_hashes['selection_cases.csv']
    for name in frozen:
        shutil.copy2(SUMMARY/name,OUT/('frozen_'+name))
        assert sha(OUT/('frozen_'+name)) == source_hashes[name]

    manifest = read_json(HERE/'final_artifact/manifest.json')
    context = np.load(HERE/'final_artifact/model_context.npz',allow_pickle=False)
    assert sha(HERE/'final_artifact/model_context.npz') == manifest['context_sha256']
    features = context['features'].tolist()
    targets = context['targets'].tolist()
    assert len(features) == 231 and len(targets) == 50 and TARGET in targets
    assert sha(HERE/'predict_tabicl_csv.py') == manifest['inference_script_sha256']
    selected = pd.read_csv(OUT/'frozen_selection_cases.csv',float_precision='round_trip')
    profiles = pd.read_csv(OUT/'frozen_selected_profiles.csv',float_precision='round_trip')
    checks = pd.read_csv(OUT/'frozen_selected_full_checks.csv',float_precision='round_trip')
    assert len(selected) == 88 and selected.groupby(['policy','rho']).size().eq(11).all()
    assert true_column(selected.candidate_exists).all() and true_column(selected.accepted).all()
    assert true_column(checks.accepted).all()
    assert (selected.ensemble_stale_delta <= 1e-8).all()
    assert not profiles.duplicated(['evaluation_policy','candidate_uid']).any()
    assert not checks.duplicated(['evaluation_policy','candidate_uid']).any()
    check_index = checks.set_index(['evaluation_policy','candidate_uid'])
    profile_index = profiles.set_index(['evaluation_policy','candidate_uid'])
    source_ids = sorted(selected.beer.astype(int).unique().tolist())
    assert len(source_ids) == 11
    context_ids = context['beer'].astype(int)
    source_positions = {beer:int(np.flatnonzero(context_ids == beer).item()) for beer in source_ids}

    unique = {}
    mappings = []
    for row in selected.itertuples():
        key = (row.policy,row.candidate_uid)
        p, old_check = profile_index.loc[key], check_index.loc[key]
        assert int(p.beer) == int(row.beer) == int(old_check.beer)
        assert abs(float(old_check.ensemble_stale_delta)-float(row.ensemble_stale_delta)) < 1e-12
        values = p[features].to_numpy(float)
        assert not np.isinf(values).any()
        digest = profile_hash(values)
        if digest in unique:
            assert np.array_equal(values,unique[digest],equal_nan=True)
        else:
            unique[digest] = values
        mappings.append(dict(policy=row.policy,beer=int(row.beer),rho=float(row.rho),
            candidate_uid=row.candidate_uid,profile_sha256=digest,
            source_input_id=f'source_{int(row.beer)}',
            fstar_stale_delta=float(row.ensemble_stale_delta),
            fstar_full_accepted=True,raw_fields_changed=int(row.integer_upper_bound)))
    input_rows = []
    identity_rows = []
    for beer in source_ids:
        identifier = f'source_{beer}'
        values = context['X'][source_positions[beer]]
        input_rows.append(dict(beer=identifier,**dict(zip(features,values))))
        identity_rows.append(dict(input_id=identifier,kind='source',source_beer=beer,
                                  profile_sha256=profile_hash(values)))
    identifiers = {}
    for number,digest in enumerate(sorted(unique)):
        identifier = f'candidate_{number:03d}'
        identifiers[digest] = identifier
        input_rows.append(dict(beer=identifier,**dict(zip(features,unique[digest]))))
        identity_rows.append(dict(input_id=identifier,kind='candidate',source_beer=None,
                                  profile_sha256=digest))
    mapping = pd.DataFrame(mappings)
    mapping['candidate_input_id'] = mapping.profile_sha256.map(identifiers)
    mapping.to_csv(OUT/'case_input_mapping.csv',index=False)
    pd.DataFrame(identity_rows).to_csv(OUT/'input_identity.csv',index=False)
    chemistry = OUT/'source_and_candidate_chemistry.csv'
    pd.DataFrame(input_rows,columns=['beer',*features]).to_csv(chemistry,index=False)
    protocol = dict(challenge='Actual final CSV deployment on frozen source/candidate chemical rows',
        model=manifest['chosen_cv_candidate'],all50=True,source_count=11,
        unique_candidate_profiles=len(unique),input_rows=len(input_rows),selection_cases=len(mapping),
        candidate_pool_sha256=source_hashes['selection_cases.csv'],source_summary_sha256=source_hashes,
        artifact_manifest_sha256=sha(HERE/'final_artifact/manifest.json'),
        csv_interface_sha256=sha(HERE/'predict_tabicl_csv.py'),
        challenge_source_sha256=sha(__file__),input_sha256=sha(chemistry),
        target=TARGET,stale_delta='f_plus(candidate)-f_plus(source) within the same combined inference batch',
        exact_nonincrease_threshold=0.,numerical_nonincrease_threshold=1e-8,
        inference_threads=6, candidate_selection_feedback=False,
        scope='Cross-predictor sensitivity on internally fitted surrogate models. The all-data context includes the source beers. It is not independent experimental validation, and no optimized candidate or f-star acceptance rule is changed.',
        batch='All unique candidate chemistry rows and all 11 source rows in one CSV invocation; input identities are metadata, never model features.')
    write_json(OUT/'protocol.json',protocol)
    status('running_full_50_output_csv_challenge',model=manifest['chosen_cv_candidate'],
           rows=len(input_rows),unique_candidate_profiles=len(unique),threads=6)
    predictions_path = OUT/'all50_sensory_predictions.csv'
    with (OUT/'inference.log').open('w') as log:
        result = subprocess.run([str(HERE/'venv/bin/python'),str(HERE/'predict_tabicl_csv.py'),
            str(chemistry),str(predictions_path),'--threads','6'],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'CSV deployment challenge failed with return code {result.returncode}.')
    predictions = pd.read_csv(predictions_path,float_precision='round_trip').set_index('beer')
    assert predictions.shape == (len(input_rows),50)
    assert predictions.index.is_unique and np.isfinite(predictions.to_numpy()).all()
    assert list(predictions.columns) == targets
    mapping['fplus_source_stale'] = mapping.source_input_id.map(predictions[TARGET])
    mapping['fplus_candidate_stale'] = mapping.candidate_input_id.map(predictions[TARGET])
    mapping['fplus_stale_delta'] = mapping.fplus_candidate_stale-mapping.fplus_source_stale
    assert np.isfinite(mapping.fplus_stale_delta).all()
    mapping['fplus_nonincrease_exact'] = mapping.fplus_stale_delta <= 0.
    mapping['fplus_nonincrease_at_1e_8'] = mapping.fplus_stale_delta <= 1e-8
    mapping['event_relation'] = np.where(mapping.fplus_nonincrease_exact,'same_nonincrease_event','reversed_to_increase')
    mapping.to_csv(OUT/'cross_model_case_checks.csv',index=False)
    unique_checks = mapping.drop_duplicates(['beer','profile_sha256']).copy()
    unique_checks.to_csv(OUT/'cross_model_unique_checks.csv',index=False)
    summaries = []
    for (policy,rho),group in mapping.groupby(['policy','rho'],sort=False):
        summaries.append(dict(policy=policy,rho=float(rho),cases=len(group),
            fstar_full_accepted=int(group.fstar_full_accepted.sum()),
            fplus_nonincrease_exact=int(group.fplus_nonincrease_exact.sum()),
            fplus_nonincrease_at_1e_8=int(group.fplus_nonincrease_at_1e_8.sum()),
            fplus_increase=int((~group.fplus_nonincrease_exact).sum()),
            fplus_stale_delta_mean=float(group.fplus_stale_delta.mean()),
            fplus_stale_delta_min=float(group.fplus_stale_delta.min()),
            fplus_stale_delta_max=float(group.fplus_stale_delta.max())))
    pd.DataFrame(summaries).to_csv(OUT/'policy_rho_summary.csv',index=False)
    dimension = pd.read_csv(HERE/'dimension_metrics.csv')
    target_score = dimension[(dimension.model == manifest['chosen_cv_candidate']) & (dimension.target == TARGET)]
    assert len(target_score) == 1
    # The main experiment records the value under the column r2.
    target_r2 = float(target_score.iloc[0]['r2'])
    assert {name:sha(SUMMARY/name) for name in frozen} == source_hashes
    report = dict(protocol,complete=True,selected_model_stale_oof_r2=target_r2,
        unique_source_candidate_pairs=len(unique_checks),
        unique_pair_nonincrease_exact=int(unique_checks.fplus_nonincrease_exact.sum()),
        unique_pair_increase=int((~unique_checks.fplus_nonincrease_exact).sum()),
        original_fstar_pool_unchanged=True,policy_rho_summary=summaries,
        result_sha256={name:sha(OUT/name) for name in ['all50_sensory_predictions.csv',
            'all50_sensory_predictions.relative_grades_0_6.csv','cross_model_case_checks.csv',
            'cross_model_unique_checks.csv','policy_rho_summary.csv']},
        timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write_json(OUT/'COMPLETE.json',report)
    status('all_done',gate='COMPLETE.json',source_count=11,all50=True,
           unique_candidate_profiles=len(unique),unique_pair_nonincrease_exact=report['unique_pair_nonincrease_exact'],
           unique_source_candidate_pairs=len(unique_checks),selected_model_stale_oof_r2=target_r2)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        OUT.mkdir(exist_ok=True)
        write_json(OUT/'FAILED.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise
