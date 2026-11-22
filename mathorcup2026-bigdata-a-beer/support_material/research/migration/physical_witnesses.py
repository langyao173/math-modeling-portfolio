from pathlib import Path
import sys,hashlib,time,json
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import run_scenario as m
import pandas as pd,numpy as np,joblib
from threadpoolctl import threadpool_limits
out=P/'all11';problem=m.FixedProblem('strict');Wdev,bdev=m.scenario_planes(joblib.load(out/'development_classifiers.joblib'))
u=pd.read_csv(m.PROJECT/'A_work/nature_upgrade/data_claim_audit/paired_70_field_encodings.csv');names=set(u.loc[u.unit.isin(['mg/L','ug/L','g/L','v/v']),'field']);allowed=np.array([j for j,f in enumerate(problem.features) if f in names]);assert len(allowed)==61
rows=[];records=[];started=time.perf_counter()
with threadpool_limits(limits=1):
 for beer in sorted(problem.contexts):
  for name,eps,scenario in m.METHODS:
   W,bb=(Wdev,bdev) if scenario else(problem.w,problem.b)
   for pm in [0.,.05,.1,.25,.5]:
    solver=m.QPSolver(problem,beer,W,bb,eps,allowed=allowed,l1=False,proxy_margin=pm);d,inf=solver.solve();r=dict(beer=beer,strategy=name,proxy_optimization_margin=pm,**inf)
    if d is None or inf['violation']>2e-5:r['nominal_accepted']=False;records.append(r);continue
    d[np.abs(d)<1e-8]=0;row,z=problem.decode(beer,d,allowed);row.update(required_rho=.1,required_proxy_margin=0.)
    c,_=problem.check_rows([row]);r.update(c.iloc[0].to_dict());r['nominal_accepted']=bool(c.iloc[0].accepted);records.append(r)
    if c.iloc[0].accepted:
     uid=hashlib.sha256(('physical61'+name+str(beer)+np.round(z,9).tobytes().hex()).encode()).hexdigest()[:18]
     row.update(candidate_id=uid,strategy='physical61_'+name,scope='calibrated61',proxy_optimization_margin=pm,style_epsilon_optimization=eps,development_min_style_margin=float(np.min(Wdev@z+bdev)));rows.append(row);break
 pd.DataFrame(records).to_csv(out/'physical61_witness_checks.csv',index=False)
 pd.DataFrame(rows).to_csv(out/'physical61_frozen_profiles.csv',index=False)
 m.dump(out/'physical61_protocol.json',dict(allowed_fields=[problem.features[j] for j in allowed],methods=m.METHODS,proxy_grid=[0,.05,.1,.25,.5],selection='first proxy in listed order passing complete nominal checks; no bootstrap sensory score used',nominal_feasible=len(rows),subspace_infeasibility_not_claimed=True,code_sha256=m.sha(Path(__file__))))
 print('PHYSICAL61',pd.DataFrame(rows).groupby('strategy').beer.agg(['count',list]).to_string(),'seconds',time.perf_counter()-started,flush=True)
