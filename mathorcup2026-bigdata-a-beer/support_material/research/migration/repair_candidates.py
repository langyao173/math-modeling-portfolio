from pathlib import Path
import sys,hashlib,json,time
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import run_scenario as m
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
out=P/'all11';problem=m.FixedProblem('strict');models=joblib.load(out/'development_classifiers.joblib');Wdev,bdev=m.scenario_planes(models)
frame=pd.read_csv(out/'all_candidate_profiles.csv');checks=pd.read_csv(out/'all_candidate_checks.csv');rows=frame.to_dict('records');seen=set(frame.candidate_id.astype(str));records=[];start=time.perf_counter()
fail=[int(b) for b,g in checks[~checks.strategy.eq('original_min_edit')].groupby('beer') if not g.accepted.any()]
print('repair_beers',fail,flush=True)
with threadpool_limits(limits=1):
 for beer in fail:
  for name,eps,is_scenario in m.METHODS:
   W,bb=(Wdev,bdev) if is_scenario else (problem.w,problem.b)
   for pm in [.05,.1,.25,.5]:
    solver=m.QPSolver(problem,beer,W,bb,eps,l1=True,proxy_margin=pm);dense,info=solver.solve()
    if dense is None:continue
    for lam in [0.,.1,1.,3.]:
     d=dense.copy()
     for it in range(1 if lam==0 else 2):
      if lam:
       dd,info=solver.solve(lam,1/(np.abs(d)+.1),d)
       if dd is None:break
       d=dd
      support=np.union1d(np.flatnonzero(np.abs(d)>2e-5),problem.contexts[beer]['mandatory']).astype(int)
      polish=m.QPSolver(problem,beer,W,bb,eps,allowed=support,l1=False,proxy_margin=pm);dd,info=polish.solve(initial=d)
      records.append(dict(beer=beer,strategy=name,proxy_optimization_margin=pm,lam=lam,iteration=it,**info))
      if dd is None or info['violation']>2e-5:continue
      dd[np.abs(dd)<1e-8]=0;row,z=problem.decode(beer,dd,support);uid=hashlib.sha256((str(beer)+np.round(z,9).tobytes().hex()).encode()).hexdigest()[:18]
      if uid in seen:continue
      seen.add(uid);row.update(candidate_id=uid,strategy=name,scope='all225',style_epsilon_optimization=eps,required_rho=.1,required_proxy_margin=0.,proxy_optimization_margin=pm,lam=lam,iteration=it,development_min_style_margin=float(np.min(Wdev@z+bdev)));rows.append(row)
   print(json.dumps(dict(beer=beer,strategy=name,pool=len(rows),seconds=time.perf_counter()-start)),flush=True)
 frame=pd.DataFrame(rows);ck,sensory=problem.check_rows(frame.to_dict('records'));ck=pd.concat([frame[['candidate_id','strategy','scope','development_min_style_margin']].reset_index(drop=True),ck],axis=1)
 frame.to_csv(out/'all_candidate_profiles.csv',index=False);ck.to_csv(out/'all_candidate_checks.csv',index=False);sensory.to_csv(out/'all_candidate_sensory.csv',index=False);pd.DataFrame(records).to_csv(out/'repair_optimization_records.csv',index=False)
 valid=ck.accepted.to_numpy(bool);frame.loc[valid].reset_index(drop=True).to_csv(out/'accepted_profiles.csv',index=False);ck.loc[valid].reset_index(drop=True).to_csv(out/'accepted_checks.csv',index=False)
 print(ck.groupby(['beer','strategy']).accepted.sum().to_string(),flush=True)
