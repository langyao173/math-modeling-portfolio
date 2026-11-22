"""Finalize saved results and one-row deployment smoke after all heavy jobs.

This queue starts no extra inference before the registered missingness job
finishes. It does not refit or predict all 250 training examples.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import hashlib
import json
import subprocess
import time

HERE=Path(__file__).resolve().parent
PYTHON=str(HERE/'venv/bin/python')
BASE=HERE.parents[1]/'project/MathorCup_A'


def status(state,**extra):
    value=dict(state=state,timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra)
    (HERE/'finalization_status.json').write_text(json.dumps(value,indent=2))
    print(json.dumps(value),flush=True)


def run(script,*args,log):
    with (HERE/log).open('w') as output:
        result=subprocess.run([PYTHON,str(HERE/script),*map(str,args)],stdout=output,stderr=subprocess.STDOUT)
    if result.returncode:
        status('failed_finalization_step',script=script,returncode=result.returncode,log=log)
        raise RuntimeError(f'Finalization failed: {script}; inspect {log}')


status('waiting_for_all_modern_experiments')
start=time.monotonic()
while True:
    path=HERE/'missingness/status.json'
    if path.exists() and json.loads(path.read_text()).get('complete'):break
    if time.monotonic()-start>8*3600:raise TimeoutError('Modern experiments did not complete in eight hours.')
    time.sleep(20)
status('rechecking_complete_saved_predictions')
run('audit_tabicl_outputs.py',log='final_reconstruction_audit.log')
run('postprocess_modern_results.py',log='final_postprocessing.log')
if not (HERE/'final_artifact/manifest.json').exists():
    run('export_tabicl_context.py',log='final_context_export.log')
import numpy as np
import pandas as pd
frame=pd.read_csv(BASE/'data/A/1.csv').sort_values('beer').head(1)
input_path=HERE/'final_artifact/example_chemistry.csv'
frame.to_csv(input_path,index=False)
output=HERE/'final_artifact/example_sensory_predictions.csv'
status('one_row_csv_inference_smoke',rows=1,threads=6,
       meaning='Operational schema/finite-output/grade check only; no predictive score on this training-context row.')
run('predict_tabicl_csv.py',input_path,output,'--threads','6',log='final_inference_smoke.log')
pred=pd.read_csv(output)
grades=pd.read_csv(output.with_name(output.stem+'.relative_grades_0_6.csv'))
assert pred.shape==(1,51) and grades.shape==(1,51)
assert np.isfinite(pred.iloc[:,1:].to_numpy(float)).all()
assert grades.iloc[:,1:].to_numpy().min()>=0 and grades.iloc[:,1:].to_numpy().max()<=6
smoke=dict(passed=True,rows=1,outputs=50,grade_range=[0,6],
           observed_grade_min=int(grades.iloc[:,1:].to_numpy().min()),
           observed_grade_max=int(grades.iloc[:,1:].to_numpy().max()),
           test_scope='Operational chemistry CSV inference and relative-grade output only; this training-context row is not an accuracy test.')
(HERE/'final_artifact/smoke_test.json').write_text(json.dumps(smoke,indent=2))
run('write_completion_report.py',log='completion_report.log')
files=['model_comparison.csv','paired_bootstrap.csv','oof_arrays.npz','reconstruction_audit.json',
       'tabdpt/model_comparison.csv','tabdpt/paired_bootstrap.csv','tabdpt/oof_arrays.npz',
       'diagnostics/relative_grade_metrics.csv','diagnostics/relative_grade_paired_ci.csv',
       'missingness/comparison.csv','missingness/paired_ci_vs_knn.csv','missingness/oof_predictions.npz',
       'missingness/batch_cache_probe.json','missingness/preprocessing_batch_audit.json',
       'final_artifact/manifest.json','final_artifact/model_context.npz','final_artifact/smoke_test.json',
       'predict_tabicl_csv.py','RESULTS.md']
hashes={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in files}
manifest=json.loads((HERE/'final_artifact/manifest.json').read_text())
ready=dict(state='all_registered_modern_prediction_work_complete',tabicl_candidates=5,
           outer_folds=5,tabdpt_targets=50,missingness_scenarios=4,
           chosen_tabicl_deployment=manifest['chosen_cv_candidate'],csv_inference_smoke_passed=True,
           result_sha256=hashes,timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
temp=HERE/'FINAL_READY.tmp.json';temp.write_text(json.dumps(ready,indent=2));temp.replace(HERE/'FINAL_READY.json')
status('all_done',gate='FINAL_READY.json',chosen_tabicl_deployment=manifest['chosen_cv_candidate'])
