"""Validate deployment CSV I/O by exact replay of real saved outer predictions.

This does not test a new neural forward pass. The cache contains real prior
out-of-fold branch predictions and the original chemical rows only.
"""
from pathlib import Path
import argparse,json,subprocess,sys
import numpy as np
import pandas as pd
from fusion_model import relative_grades
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'deployment_validation_recheck')
OUT=parser.parse_args().output
if OUT.exists() and any(OUT.iterdir()):raise FileExistsError('Validation output exists.')
OUT.mkdir(exist_ok=True)
with np.load(ROOT/'deployment_fixed_half/model_context.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
with np.load(ROOT/'fusion_replay_v1/outer_predictions.npz',allow_pickle=False) as z:oof={k:z[k] for k in z.files}
frame=pd.DataFrame(a['X'],columns=a['features']);frame.insert(0,'beer',a['beer']);frame.to_csv(OUT/'query_all250.csv',index=False)
# Pandas CSV round-trip default float parsing may round the final bit. The
# cache requires exact numeric equality, so retain the original official CSV.
source=ROOT/'validation_inputs/chemistry_250.csv'
results=[]
for mode,key in [('fixed_half','fixed_half_y16_native50'),('groupwise','inner_groupwise_y16_native50')]:
    artifact=ROOT/('deployment_'+mode);output=OUT/(mode+'.csv')
    command=[sys.executable,str(ROOT/'predict_fusion_csv.py'),str(source),str(output),'--artifact',str(artifact),'--replay-cache',str(artifact/'outer_cv_validation_cache.npz'),'--write-branches']
    p=subprocess.run(command,text=True,capture_output=True)
    if p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
    got=pd.read_csv(output)
    # The official CSV is not sorted. Preserve its exact input row order.
    original_ids=pd.read_csv(source).beer.tolist()
    assert got.beer.tolist()==original_ids
    positions={int(v):i for i,v in enumerate(a['beer'])}
    row_index=np.asarray([positions[int(v)] for v in original_ids])
    pred=got[a['targets']].to_numpy();expected=oof[key][row_index]
    maxerr=float(np.max(np.abs(pred-expected)));assert maxerr<1e-14
    grade=pd.read_csv(output.with_name(output.stem+'.relative_grades_0_6.csv'))[a['targets']].to_numpy()
    true_grades=np.empty(grade.shape,int)
    expected_grade=np.empty(grade.shape,int)
    for f in range(5):
        ix=oof['fold'][row_index]==f;cuts=np.quantile(oof['Y'][oof['fold']!=f],np.arange(1,7)/7,axis=0)
        expected_grade[ix]=relative_grades(expected[ix],cuts);true_grades[ix]=relative_grades(oof['Y'][row_index][ix],cuts)
    np.testing.assert_array_equal(grade,expected_grade)
    audit=json.loads(output.with_suffix(output.suffix+'.audit.json').read_text());assert audit['neural_inference_executed'] is False
    r=1-((pred-oof['Y'][row_index])**2).sum(0)/((oof['Y']-oof['Y'].mean(0))**2).sum(0)
    results.append({'mode':mode,'query_rows':250,'max_csv_prediction_error':maxerr,'all_relative_grades_exact':True,'all50_r2':float(r.mean()),'core30_r2':float(r[:30].mean()),'core30_grade_mae':float(np.abs(grade[:,:30]-true_grades[:,:30]).mean()),'new_neural_forward_pass':False})
# Reordering input rows and columns must preserve the row identity contract.
reordered=pd.read_csv(source).iloc[[103,7,201,0,249]].copy();reordered=reordered[['beer']+list(reversed(a['features'].tolist()))];reordered.to_csv(OUT/'query_reordered.csv',index=False)
# Re-read original values through the same parser to avoid CSV numeric-rounding
# differences in exact cache mode: use the original text rows/headers instead.
import csv
with source.open(newline='',encoding='utf-8-sig') as h:raw=list(csv.reader(h))
header=raw[0];idx=[header.index('beer')]+[header.index(f) for f in reversed(a['features'].tolist())]
with (OUT/'query_reordered.csv').open('w',newline='',encoding='utf-8') as h:
    w=csv.writer(h);w.writerow([header[j] for j in idx]);w.writerows([[raw[i+1][j] for j in idx] for i in [103,7,201,0,249]])
p=subprocess.run([sys.executable,str(ROOT/'predict_fusion_csv.py'),str(OUT/'query_reordered.csv'),str(OUT/'reordered.csv'),'--artifact',str(ROOT/'deployment_fixed_half'),'--replay-cache',str(ROOT/'deployment_fixed_half/outer_cv_validation_cache.npz')],text=True,capture_output=True)
if p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
r=pd.read_csv(OUT/'reordered.csv');assert r.beer.tolist()==reordered.beer.tolist()
positions={int(v):i for i,v in enumerate(a['beer'])}
query_index=[positions[int(v)] for v in r.beer]
np.testing.assert_allclose(r[a['targets']].to_numpy(),oof['fixed_half_y16_native50'][query_index],rtol=0,atol=1e-14)
# Alter a chemistry value: replay must reject and produce no fake prediction.
changed=reordered.copy();changed.loc[changed.index[0],a['features'][0]]+=1.;changed.to_csv(OUT/'changed_query.csv',index=False)
p=subprocess.run([sys.executable,str(ROOT/'predict_fusion_csv.py'),str(OUT/'changed_query.csv'),str(OUT/'must_not_exist.csv'),'--artifact',str(ROOT/'deployment_fixed_half'),'--replay-cache',str(ROOT/'deployment_fixed_half/outer_cv_validation_cache.npz')],text=True,capture_output=True)
assert p.returncode!=0 and not (OUT/'must_not_exist.csv').exists()
report={'complete':True,'real_saved_CV_cache_replays':results,'reordered_rows_and_columns_preserved':True,'modified_chemistry_rejected':True,'new_250_context_neural_inference_executed':False,'environment_limit':'No torch/tabicl packages or pretrained checkpoint were installed for this validation. A fresh neural prediction requires the pinned dependencies and checkpoint.','deployment_250_context_export_validated':True}
(OUT/'validation_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
