"""Matched-budget low-margin challenge; existing v1 controls remain frozen."""
from pathlib import Path
import sys,hashlib,json,time,os,shutil
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;sys.path.insert(0,str(BASE));import run_scenario as m
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
BEERS=[28,81,90,96,119,124,139,185,209,218,230];SEED=202610251503;METHOD='low_margin_same_qp'

def freeze_protocol():
 p=HERE/'protocol.json';r=json.loads(p.read_text());r.update(locked_at_utc=datetime.now(timezone.utc).isoformat(),v1_candidate_freeze_sha256=m.sha(BASE/'all11/CANDIDATES_FROZEN.json'),v1_primary_lock_sha256=m.sha(BASE/'all11/PRIMARY_ANALYSIS_LOCK.json'),script_sha256=m.sha(Path(__file__)),qp_implementation_sha256=m.sha(BASE/'run_scenario.py'),new_validation_results_examined=False)
 m.dump(HERE/'PROTOCOL_LOCK.json',r)

def generate(problem):
 rows=[];records=[];seen=set();begin=time.perf_counter()
 def search(beer,pm,lams,steps):
  solver=m.QPSolver(problem,beer,problem.w,problem.b,1e-5,l1=True,proxy_margin=pm);dense,inf=solver.solve()
  if dense is None:records.append(dict(beer=beer,proxy_margin=pm,stage='dense',**inf));return
  for lam in lams:
   d=dense.copy()
   for it in range(1 if lam==0 else steps):
    if lam:
     dd,inf=solver.solve(lam,1/(np.abs(d)+.1),d)
     if dd is None:break
     d=dd
    support=np.union1d(np.flatnonzero(np.abs(d)>2e-5),problem.contexts[beer]['mandatory']).astype(int)
    polish=m.QPSolver(problem,beer,problem.w,problem.b,1e-5,allowed=support,l1=False,proxy_margin=pm);dd,inf=polish.solve(initial=d)
    records.append(dict(beer=beer,proxy_optimization_margin=pm,lam=lam,iteration=it,support_size=len(support),**inf))
    if dd is None or inf['violation']>2e-5:continue
    dd[np.abs(dd)<1e-8]=0;row,z=problem.decode(beer,dd,support);uid=hashlib.sha256((METHOD+str(beer)+np.round(z,9).tobytes().hex()).encode()).hexdigest()[:18]
    if uid in seen:continue
    seen.add(uid);row.update(candidate_id=uid,strategy=METHOD,scope='all225',style_epsilon_optimization=1e-5,required_rho=.1,required_proxy_margin=0.,proxy_optimization_margin=pm,lam=lam,iteration=it);rows.append(row)
 for beer in BEERS:
  search(beer,0.,[0.]+m.LAMS,3);print(json.dumps(dict(stage='low_margin_base',beer=beer,rows=len(rows),seconds=time.perf_counter()-begin)),flush=True)
 frame=pd.DataFrame(rows);checks,_=problem.check_rows(frame.to_dict('records'));failed=[b for b in BEERS if not checks[checks.beer.eq(b)].accepted.any()]
 for beer in failed:
  for pm in [.05,.1,.25,.5]:search(beer,pm,[0.,.1,1.,3.],2)
  print(json.dumps(dict(stage='low_margin_repair',beer=beer,rows=len(rows),seconds=time.perf_counter()-begin)),flush=True)
 frame=pd.DataFrame(rows);checks,sensory=problem.check_rows(frame.to_dict('records'));checks=pd.concat([frame[['candidate_id','strategy','scope']].reset_index(drop=True),checks],axis=1)
 frame.to_csv(HERE/'all_proposed_profiles.csv',index=False);checks.to_csv(HERE/'all_proposed_checks.csv',index=False);pd.DataFrame(records).to_csv(HERE/'optimization_records.csv',index=False)
 frame=frame.loc[checks.accepted.to_numpy(bool)].reset_index(drop=True);checks=checks.loc[checks.accepted].reset_index(drop=True)
 frame.to_csv(HERE/'accepted_profiles.csv',index=False);checks.to_csv(HERE/'accepted_checks.csv',index=False);return frame,checks

def select_and_freeze(problem,frame,checks,dev):
 meta=checks[['candidate_id','beer','raw_fields_changed','quadratic_objective','target_distance_reduction']].merge(dev,on=['candidate_id','beer'])
 frontier=[]
 for beer in BEERS:
  for B in [60,100]:
   g=meta[meta.beer.eq(beer)&meta.raw_fields_changed.le(B)]
   if not len(g):frontier.append(dict(beer=beer,strategy=METHOD,budget=B,exists=False));continue
   r=g.sort_values(['joint_frequency','style_frequency','raw_fields_changed','quadratic_objective','candidate_id'],ascending=[False,False,True,True,True]).iloc[0]
   frontier.append(dict(beer=beer,strategy=METHOD,budget=B,exists=True,candidate_id=r.candidate_id,raw_fields_changed=r.raw_fields_changed,target_distance_reduction=r.target_distance_reduction,development_joint_frequency=r.joint_frequency))
 oldf=m.read_csv(BASE/'all11/development_frontier.csv');keep=((oldf.strategy.eq('original_min_edit')&oldf.budget.isin([60,100]))|(oldf.strategy.eq('fixed_margin_1')&oldf.budget.eq(60))|(oldf.strategy.eq('fixed_margin_2')&oldf.budget.isin([60,100])))
 old=oldf[keep].copy();old=old.rename(columns={'joint_frequency':'development_joint_frequency'});front=pd.concat([pd.DataFrame(frontier),old],ignore_index=True);front.to_csv(HERE/'frozen_frontier.csv',index=False)
 previous=m.read_csv(BASE/'all11/frozen_validation_profiles.csv');combined=pd.concat([frame,previous],ignore_index=True);selected=set(front.loc[front.exists,'candidate_id']);frozen=combined[combined.candidate_id.isin(selected)].drop_duplicates('candidate_id').reset_index(drop=True);frozen.to_csv(HERE/'frozen_profiles.csv',index=False)
 m.dump(HERE/'CANDIDATES_FROZEN.json',dict(locked_at_utc=datetime.now(timezone.utc).isoformat(),profiles_sha256=m.sha(HERE/'frozen_profiles.csv'),frontier_sha256=m.sha(HERE/'frozen_frontier.csv'),protocol_sha256=m.sha(HERE/'PROTOCOL_LOCK.json'),seed=SEED,scenarios=100,validation_results_examined=False))
 return frozen

def main():
 freeze_protocol();problem=m.FixedProblem('strict')
 if (HERE/'accepted_profiles.csv').exists():frame=m.read_csv(HERE/'accepted_profiles.csv');checks=m.read_csv(HERE/'accepted_checks.csv')
 else:frame,checks=generate(problem)
 devix=np.load(BASE/'all11/development_indices.npz')['indices'];print('ACCEPTED',len(frame),flush=True)
 if (HERE/'development_stability.csv').exists():dev=m.read_csv(HERE/'development_stability.csv')
 else:dev,_=m.replay(HERE,problem,frame,devix,'development',2)
 frozen=select_and_freeze(problem,frame,checks,dev)
 val=m.bootstrap(problem.df.tasting_category_fine.to_numpy(str),100,SEED);old=np.load(BASE/'all11/validation_indices.npz')['indices'];assert not any(np.array_equal(x,y) for x in old for y in val)
 np.savez_compressed(HERE/'validation_indices.npz',indices=val,beer=problem.df.beer.to_numpy())
 result,_=m.replay(HERE,problem,frozen,val,'validation',2);m.dump(HERE/'COMPLETE.json',dict(complete=True,frozen_profiles=len(frozen),validation_scenarios=100,seed=SEED,all_results_reported=True));print('COMPLETE',len(frozen),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
