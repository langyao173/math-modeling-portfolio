from pathlib import Path
import sys,hashlib,json
HERE=Path(__file__).resolve().parent;P=HERE.parent;sys.path.insert(0,str(P));import run_scenario as m
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
out=P/'all11';problem=m.FixedProblem('strict');W,bb=m.scenario_planes(joblib.load(out/'development_classifiers.joblib'));name='scenario32_margin025';beer=28;eps=.25
rows=[];audit=[]
with threadpool_limits(limits=1):
 solver=m.QPSolver(problem,beer,W,bb,eps,l1=True);dense,_=solver.solve()
 for lam,last in [(0.,0),(.01,2),(.03,1)]:
  d=dense.copy()
  for it in range(last+1):
   if lam:d,_=solver.solve(lam,1/(np.abs(d)+.1),d)
   support=np.union1d(np.flatnonzero(np.abs(d)>2e-5),problem.contexts[beer]['mandatory']).astype(int)
   polish=m.QPSolver(problem,beer,W,bb,eps,allowed=support,l1=False);dd,info=polish.solve(initial=d);assert dd is not None and info['violation']<=2e-5
   dd[np.abs(dd)<1e-8]=0;row,z=problem.decode(beer,dd,support);uid=hashlib.sha256((str(beer)+np.round(z,9).tobytes().hex()).encode()).hexdigest()[:18]
   row.update(candidate_id=name+'__'+uid,strategy=name,scope='all225',style_epsilon_optimization=eps,required_rho=.1,required_proxy_margin=0.,lam=lam,iteration=it,development_min_style_margin=float(np.min(W@z+bb)),recovered_physical_id=uid,recovery_reason='Pure scenario-method optimization replay; original global cross-method dedup dropped this point.')
   rows.append(row)
 frame=pd.DataFrame(rows);checks,sensory=problem.check_rows(frame.to_dict('records'))
 assert len(frame)==6 and checks.accepted.all()
 frame.to_csv(HERE/'recovered_candidates.csv',index=False);checks.to_csv(HERE/'recovered_checks.csv',index=False);sensory.to_csv(HERE/'recovered_sensory.csv',index=False)
 report=dict(cross_method_dropped=6,within_method_dedup=26,affected_beer=28,affected_method=name,actual_edits=checks.raw_fields_changed.tolist(),affected_prespecified_budgets=[225],unaffected_prespecified_budgets=[20,30,40,60,100],all_complete_nominal_checks_pass=True,all_scenario_margins_ge025=bool((frame.development_min_style_margin>=.25-2e-5).all()))
 (HERE/'recovery_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
