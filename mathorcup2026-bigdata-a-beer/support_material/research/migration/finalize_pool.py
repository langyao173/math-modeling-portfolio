from pathlib import Path
import sys,json
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import run_scenario as m
import pandas as pd,numpy as np,joblib
from threadpoolctl import threadpool_limits
out=P/'all11';extra=P/'audit_pipeline/recovered_candidates.csv'
assert extra.exists(), 'Independent duplicate recovery has not completed'
problem=m.FixedProblem('strict');W,bb=m.scenario_planes(joblib.load(out/'development_classifiers.joblib'))
a=m.read_csv(out/'all_candidate_profiles.csv');e=m.read_csv(extra)
a=pd.concat([a,e],ignore_index=True);assert a.candidate_id.is_unique
# Recompute after exact float parsing and independently restored cross-method duplicates.
z=problem.control.map_.transform(a[problem.features].to_numpy(float));a['development_min_style_margin']=(z@W.T+bb).min(axis=1)
with threadpool_limits(limits=1):c,s=problem.check_rows(a.to_dict('records'))
c=pd.concat([a[['candidate_id','strategy','scope','development_min_style_margin']].reset_index(drop=True),c],axis=1)
a.to_csv(out/'all_candidate_profiles.csv',index=False);c.to_csv(out/'all_candidate_checks.csv',index=False);s.to_csv(out/'all_candidate_sensory.csv',index=False)
a.loc[c.accepted.to_numpy(bool)].reset_index(drop=True).to_csv(out/'accepted_profiles.csv',index=False);c.loc[c.accepted.to_numpy(bool)].reset_index(drop=True).to_csv(out/'accepted_checks.csv',index=False)
assert c[c.strategy.eq('original_min_edit')].raw_fields_changed.sum()==245
m.dump(out/'pool_freeze.json',dict(accepted=int(c.accepted.sum()),proposed=len(a),cross_method_duplicates_restored=len(e),raw_profiles_sha256=m.sha(out/'all_candidate_profiles.csv'),accepted_profiles_sha256=m.sha(out/'accepted_profiles.csv'),repair_proxy_grid=[.05,.1,.25,.5],repair_lambdas=[0,.1,1.,3.],round_trip_float_parsing=True))
print(c.groupby(['beer','strategy']).accepted.sum().to_string(),flush=True)
