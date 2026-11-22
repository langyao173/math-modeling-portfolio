"""Sparse style-margin and finite-scenario migration with fresh conditional replay.

The empirical model draw distribution is fixed by style-stratified resampling
of the same 250 beers. Results are model-refit stability, not intervention success.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,argparse,time,json,hashlib,warnings
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
import numpy as np,pandas as pd,joblib
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from joblib import Parallel,delayed
from threadpoolctl import threadpool_limits
import osqp

PROJECT=Path(os.environ.get('MATHORCUP_PROJECT_DIR',str(HERE.parent/'project/MathorCup_A'))).resolve()
os.environ['MATHORCUP_PROJECT_DIR']=str(PROJECT)
sys.path[:0]=[str(PROJECT/'frontier_revision/experiments/sparse_transfer'),str(PROJECT/'A_work/q1'),str(PROJECT/'A_work/q3')]
from control_problem import FixedProblem
from sensory_model import SensoryEstimator

RHO=.1
DEV_COUNT=32
FRESH_SEED=202610250731
BUDGETS=[20,30,40,60,100,225]
LAMS=[.01,.03,.1,.3,1.,3.,10.]
METHODS=[('fixed_margin_1',1.,False),('fixed_margin_2',2.,False),('scenario32_margin025',.25,True)]

def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=True))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_csv(path):return pd.read_csv(path,float_precision='round_trip')
def bootstrap(styles,count,seed):
 rng=np.random.default_rng(seed);groups=[np.flatnonzero(styles==s) for s in np.unique(styles)]
 return np.array([np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]) for _ in range(count)])
def style_fit(z,styles,ix):
 with warnings.catch_warnings():
  warnings.simplefilter('ignore');return LogisticRegression(C=.1,class_weight='balanced',max_iter=3000,solver='lbfgs',tol=.0001).fit(z[ix],styles[ix])

def scenario_planes(models):
 ws=[];bs=[]
 for m in models:
  ti=list(m.classes_).index('Tripel');oi=np.flatnonzero(np.arange(len(m.classes_))!=ti)
  ws.append(m.coef_[ti]-m.coef_[oi]);bs.append(m.intercept_[ti]-m.intercept_[oi])
 return np.concatenate(ws),np.concatenate(bs)

class QPSolver:
 def __init__(self,problem,beer,W,intercept,epsilon,allowed=None,l1=True,proxy_margin=0.):
  self.problem=problem;self.beer=beer;self.p=problem.p;c=problem.contexts[beer];p=self.p
  # A small proxy interior buffer only prevents numerical false acceptance.
  self.A=np.vstack([problem.a,-problem.w,-W])
  self.rhs=np.r_[-float(proxy_margin)-1e-5,c['base_margin']-1e-5,W@c['z0']+intercept-epsilon]
  self.lo=c['lo'].copy();self.hi=c['hi'].copy()
  if allowed is not None:
   fixed=np.ones(p,bool);fixed[np.asarray(allowed,int)]=False;self.lo[fixed]=0;self.hi[fixed]=0
  self.l1=l1
  if l1:
   I=sparse.eye(p,format='csc');Z=sparse.csc_matrix((p,p));P=sparse.block_diag([sparse.csc_matrix(problem.H),Z],format='csc')
   A=sparse.vstack([sparse.hstack([sparse.csc_matrix(self.A),sparse.csc_matrix((len(self.A),p))]),sparse.hstack([I,Z]),sparse.hstack([I,-I]),sparse.hstack([-I,-I])],format='csc')
   low=np.r_[np.full(len(self.A),-np.inf),self.lo,np.full(2*p,-np.inf)];high=np.r_[self.rhs,self.hi,np.zeros(2*p)]
   q=np.r_[c['gradient'],np.zeros(p)]
  else:
   P=sparse.csc_matrix(problem.H);A=sparse.vstack([sparse.csc_matrix(self.A),sparse.eye(p)],format='csc')
   low=np.r_[np.full(len(self.A),-np.inf),self.lo];high=np.r_[self.rhs,self.hi];q=c['gradient']
  self.solver=osqp.OSQP();self.solver.setup(P=sparse.triu(P,format='csc'),q=q,A=A,l=low,u=high,verbose=False,eps_abs=2e-6,eps_rel=2e-6,max_iter=40000,polishing=True,adaptive_rho=True)
 def solve(self,penalty=0.,weights=None,initial=None):
  c=self.problem.contexts[self.beer]
  if self.l1:self.solver.update(q=np.r_[c['gradient'],penalty*(np.ones(self.p) if weights is None else weights)])
  if initial is not None:self.solver.warm_start(x=np.r_[initial,np.abs(initial)] if self.l1 else initial)
  t=time.perf_counter();r=self.solver.solve(raise_error=False)
  if r.info.status_val not in (1,2):return None,dict(status=r.info.status,seconds=time.perf_counter()-t)
  d=r.x[:self.p];violation=max(float(np.max(self.A@d-self.rhs)),float(np.max(self.lo-d)),float(np.max(d-self.hi)))
  return d,dict(status=r.info.status,violation=violation,iterations=r.info.iter,seconds=time.perf_counter()-t)

def make_candidates(out,beers,reuse=None):
 problem=FixedProblem('strict');styles=problem.df.tasting_category_fine.to_numpy(str)
 ix=np.load(PROJECT/'frontier_revision/experiments/transfer_robustness/bootstrap_indices.npz')['indices'][:DEV_COUNT]
 np.savez_compressed(out/'development_indices.npz',indices=ix,beer=problem.df.beer.to_numpy())
 cp=out/'development_classifiers.joblib'
 if cp.exists():models=joblib.load(cp)
 else:
  models=Parallel(n_jobs=2)(delayed(style_fit)(problem.z,styles,i) for i in ix);joblib.dump(models,cp,compress=3)
 Wdev,bdev=scenario_planes(models)
 units=pd.read_csv(PROJECT/'A_work/nature_upgrade/data_claim_audit/paired_70_field_encodings.csv')
 known=set(units.loc[units.unit.isin(['mg/L','ug/L','g/L','v/v']),'field']);physical=np.array([j for j,n in enumerate(problem.features) if n in known])
 records=[];rows=[];seen=set();started=time.perf_counter()
 # Existing minimum-edit full-model solutions are the comparator on the same beers.
 pool=PROJECT/'frontier_revision/experiments/sparse_transfer/final_summary'
 cases=read_csv(pool/'selection_cases.csv');profiles=read_csv(pool/'selected_profiles.csv')
 for beer in beers:
  case=cases[cases.policy.eq('strict') & cases.beer.eq(beer)&np.isclose(cases.rho,RHO)].iloc[0]
  row=profiles[profiles.evaluation_policy.eq('strict')&profiles.candidate_uid.eq(case.candidate_uid)].iloc[0].to_dict()
  row.update(candidate_id=f'original_b{beer}',strategy='original_min_edit',scope='all225',required_rho=RHO,required_proxy_margin=0.);rows.append(row)
 reused=set()
 if reuse is not None:
  previous=read_csv(HERE/reuse/'all_candidate_profiles.csv')
  previous=previous[previous.beer.isin(beers)&~previous.strategy.eq('original_min_edit')]
  rows.extend(previous.to_dict('records'));reused=set(previous.beer.astype(int));seen.update(previous.candidate_id.astype(str))
  records.extend(pd.read_csv(HERE/reuse/'optimization_records.csv').to_dict('records'))
 for beer in beers:
  if beer in reused:continue
  for name,eps,is_scenario in METHODS:
   W,bb=(Wdev,bdev) if is_scenario else (problem.w,problem.b)
   solver=QPSolver(problem,beer,W,bb,eps,l1=True);dense,info=solver.solve()
   if dense is None:
    records.append(dict(beer=beer,strategy=name,stage='dense',**info));continue
   for lam in [0.]+LAMS:
    d=dense.copy()
    for it in range(1 if lam==0 else 3):
     if lam:
      d1,info=solver.solve(lam,1/(np.abs(d)+.1),d)
      if d1 is None:break
      d=d1
     support=np.union1d(np.flatnonzero(np.abs(d)>2e-5),problem.contexts[beer]['mandatory']).astype(int)
     # Reoptimize only the proposed support; all class and defect guards retained.
     polish=QPSolver(problem,beer,W,bb,eps,allowed=support,l1=False);dd,inf=polish.solve(initial=d)
     rec=dict(beer=beer,strategy=name,lam=lam,iteration=it,support_size=len(support),**inf);records.append(rec)
     if dd is None or inf['violation']>2e-5:continue
     dd[np.abs(dd)<1e-8]=0
     row,z=problem.decode(beer,dd,support);uid=hashlib.sha256((name+str(beer)+np.round(z,9).tobytes().hex()).encode()).hexdigest()[:18]
     if uid in seen:continue
     seen.add(uid);row.update(candidate_id=uid,strategy=name,scope='all225',style_epsilon_optimization=eps,required_rho=RHO,required_proxy_margin=0.,lam=lam,iteration=it)
     row['development_min_style_margin']=float(np.min(Wdev@z+bdev))
     rows.append(row)
   print(json.dumps(dict(beer=beer,strategy=name,pool=len(rows),seconds=time.perf_counter()-started)),flush=True)
 frame=pd.DataFrame(rows);checks,sensory=problem.check_rows(frame.to_dict('records'))
 checks=pd.concat([frame[['candidate_id','strategy','scope','development_min_style_margin']].reset_index(drop=True),checks],axis=1)
 frame.to_csv(out/'all_candidate_profiles.csv',index=False);checks.to_csv(out/'all_candidate_checks.csv',index=False);sensory.to_csv(out/'all_candidate_sensory.csv',index=False);pd.DataFrame(records).to_csv(out/'optimization_records.csv',index=False)
 # Retain complete accepted rows only; failed candidates remain in the audit files.
 valid=checks.accepted.to_numpy(bool);frame=frame.loc[valid].reset_index(drop=True);checks=checks.loc[valid].reset_index(drop=True)
 frame.to_csv(out/'accepted_profiles.csv',index=False);checks.to_csv(out/'accepted_checks.csv',index=False)
 return problem,frame,checks,ix

def replay_one(ix,x,y,z,style,specs,weights,query,zquery,source_positions,stale_index):
 begin=time.perf_counter();clf=style_fit(z,style,ix);ti=list(clf.classes_).index('Tripel');logit=clf.decision_function(zquery);other=logit.copy();other[:,ti]=-np.inf
 margin=logit[:,ti]-other.max(axis=1);component=[];predicted=[]
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  for spec in specs:
   m=SensoryEstimator(spec).fit(x[ix],y[ix]);pred=m.predict(query);component.append(pred[:,stale_index]-pred[source_positions,stale_index]);predicted.append(pred-pred[source_positions]);del m
 delta=np.einsum('f,fnk->nk',weights,np.stack(predicted));stale_component=np.stack(component)
 return dict(margin=margin,stale=delta[:,stale_index],delta=delta,component_stale=stale_component,seconds=time.perf_counter()-begin)

def replay(out,problem,frame,indices,tag,workers=2):
 sources=[problem.contexts[b]['src'].to_dict() for b in sorted(frame.beer.unique())];sourcebeers=[int(x['beer']) for x in sources];allframe=pd.concat([pd.DataFrame(sources),frame],ignore_index=True);nsource=len(sources)
 xpos=np.array([sourcebeers.index(int(b)) for b in allframe.beer])
 x=problem.df[problem.bundle['features']].to_numpy(float);y=pd.read_csv(PROJECT/'data/A/4.csv').set_index('beer').loc[problem.df.beer,problem.bundle['targets']].to_numpy(float)
 query=allframe[problem.bundle['features']].to_numpy(float);zquery=problem.control.map_.transform(allframe[problem.features].to_numpy(float));style=problem.df.tasting_category_fine.to_numpy(str);stale=list(problem.bundle['targets']).index('stale_hops')
 identity=hashlib.sha256(np.ascontiguousarray(query).tobytes()+np.ascontiguousarray(zquery).tobytes()+xpos.tobytes()+indices.tobytes()+json.dumps(problem.bundle['specs'],sort_keys=True).encode()+np.asarray(problem.bundle['weights']).tobytes()).hexdigest()
 checkpoints=out/(tag+'_replay');checkpoints.mkdir(exist_ok=True);results=[]
 for start in range(0,len(indices),workers):
  jobs=[]
  for k in range(start,min(start+workers,len(indices))):
   path=checkpoints/f'draw_{k:03d}.npz'
   if path.exists():
    with np.load(path,allow_pickle=False) as cached:assert str(cached['input_identity'])==identity,'Replay cache has a different query or resampling plan'
    continue
   jobs.append((k,indices[k]))
  rr=Parallel(n_jobs=workers)(delayed(replay_one)(ix,x,y,problem.z,style,problem.bundle['specs'],np.asarray(problem.bundle['weights']),query,zquery,xpos,stale) for k,ix in jobs)
  for (k,ix),result in zip(jobs,rr):np.savez_compressed(checkpoints/f'draw_{k:03d}.npz',input_identity=np.array(identity),**result)
  if (start+workers)%10==0 or start+workers>=len(indices):print(json.dumps(dict(stage=tag,draws=min(start+workers,len(indices)),total=len(indices),profiles=len(frame))),flush=True)
 results=[dict(np.load(checkpoints/f'draw_{k:03d}.npz',allow_pickle=False)) for k in range(len(indices))]
 arrays={key:np.stack([r[key] for r in results])[:,nsource:] for key in ['margin','stale','delta']}
 arrays['component_stale']=np.stack([r['component_stale'] for r in results])[:,:,nsource:]
 arrays['indices']=indices;arrays['candidate_id']=frame.candidate_id.to_numpy(str);arrays['beer']=frame.beer.to_numpy(int);arrays['fit_seconds']=np.array([r['seconds'] for r in results])
 arrays['input_identity']=np.array(identity)
 np.savez_compressed(out/(tag+'_draws.npz'),**arrays)
 rows=[]
 for j,meta in frame.iterrows():
  joint=(arrays['margin'][:,j]>=0)&(arrays['stale'][:,j]<=1e-8)
  rows.append(dict(candidate_id=meta.candidate_id,beer=int(meta.beer),strategy=meta.strategy,scope=meta.scope,style_frequency=float((arrays['margin'][:,j]>=0).mean()),stale_frequency=float((arrays['stale'][:,j]<=1e-8).mean()),joint_frequency=float(joint.mean()),stale_mean=float(arrays['stale'][:,j].mean()),stale_q975=float(np.quantile(arrays['stale'][:,j],.975)),class_margin_mean=float(arrays['margin'][:,j].mean())))
 summary=pd.DataFrame(rows);summary.to_csv(out/(tag+'_stability.csv'),index=False);return summary,arrays

def select_frontier(out,frame,checks,dev):
 meta=checks[['candidate_id','beer','raw_fields_changed','target_distance_reduction','ensemble_stale_delta','quadratic_objective']].merge(dev,on=['candidate_id','beer'])
 select=[]
 for beer in sorted(frame.beer.unique()):
  for strategy in meta.strategy.unique():
   g=meta[(meta.beer==beer)&meta.strategy.eq(strategy)]
   for budget in BUDGETS:
    eligible=g[g.raw_fields_changed<=budget]
    if not len(eligible):
     select.append(dict(beer=int(beer),strategy=strategy,budget=budget,exists=False));continue
    c=eligible.sort_values(['joint_frequency','style_frequency','raw_fields_changed','quadratic_objective','candidate_id'],ascending=[False,False,True,True,True]).iloc[0]
    select.append(dict(beer=int(beer),strategy=strategy,budget=budget,exists=True,**{k:c[k] for k in ['candidate_id','raw_fields_changed','target_distance_reduction','joint_frequency','style_frequency','stale_frequency']}))
 selected=pd.DataFrame(select);selected.to_csv(out/'development_frontier.csv',index=False);uids=set(selected.loc[selected.exists,'candidate_id']);frozen=frame[frame.candidate_id.isin(uids)].copy().reset_index(drop=True);frozen.to_csv(out/'frozen_validation_profiles.csv',index=False)
 dump(out/'CANDIDATES_FROZEN.json',dict(profile_sha256=sha(out/'frozen_validation_profiles.csv'),frontier_sha256=sha(out/'development_frontier.csv'),validation_seed=FRESH_SEED,validation_replicates=200,rho=RHO,selection='By development paired joint model-refit frequency under each prespecified field budget; ties by style frequency, lower edits, objective, stable id.',validation_unused=True))
 return selected,frozen

def main(args):
 out=HERE/args.run;out.mkdir(exist_ok=True)
 beers=[int(x) for x in args.beers.split(',')]
 dump(out/'protocol.json',dict(rho=RHO,beers=beers,policy='strict',development_replicates=DEV_COUNT,development_draws='first32 pre-existing 2026102407 style-stratified bootstrap draws, used only for development',validation_seed=FRESH_SEED,validation_replicates=args.validation_draws,budgets=BUDGETS,methods=METHODS,lambdas=LAMS,reweight_steps=3,weights='frozen original six-model weights; each replicate refits preprocessing and all50 targets',inference='Conditional empirical model-refit stability; not independent new beer validation or physical success probability.'))
 if not (out/'accepted_profiles.csv').exists():problem,frame,checks,devix=make_candidates(out,beers,args.reuse)
 else:
  problem=FixedProblem('strict');frame=read_csv(out/'accepted_profiles.csv');checks=read_csv(out/'accepted_checks.csv');devix=np.load(out/'development_indices.npz')['indices']
 print(json.dumps(dict(stage='candidate_pool',accepted=len(frame),per_strategy=frame.strategy.value_counts().to_dict())),flush=True)
 if args.candidates_only:return
 if not (out/'development_stability.csv').exists():dev,_=replay(out,problem,frame,devix,'development',args.workers)
 else:
  prior=np.load(out/'development_draws.npz',allow_pickle=False);assert np.array_equal(prior['candidate_id'],frame.candidate_id.to_numpy(str));assert np.array_equal(prior['indices'],devix)
  dev=read_csv(out/'development_stability.csv')
 frontier,frozen=select_frontier(out,frame,checks,dev)
 # Only now instantiate fresh validation randomness; the frozen candidates cannot see it.
 valix=bootstrap(problem.df.tasting_category_fine.to_numpy(str),args.validation_draws,FRESH_SEED);np.savez_compressed(out/'validation_indices.npz',indices=valix,beer=problem.df.beer.to_numpy())
 summary,arrays=replay(out,problem,frozen,valix,'validation',args.workers)
 merged=frontier.merge(summary,on=['candidate_id','beer','strategy'],how='left',suffixes=('_development','_validation'));merged.to_csv(out/'validated_frontier.csv',index=False)
 print(merged.groupby(['strategy','budget']).agg(covered=('exists','sum'),joint=('joint_frequency_validation','mean'),edits=('raw_fields_changed','mean')).reset_index().to_json(orient='records'),flush=True)
 dump(out/'COMPLETE.json',dict(complete=True,validation_replicates=args.validation_draws,profiles=len(frozen),sources=len(beers),accepted_pool=len(frame),validation_indices_sha256=sha(out/'validation_indices.npz'),frozen_profiles_sha256=sha(out/'frozen_validation_profiles.csv')))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run',default='pilot3');ap.add_argument('--beers',default='28,81,124');ap.add_argument('--workers',type=int,default=2);ap.add_argument('--validation-draws',type=int,default=200);ap.add_argument('--candidates-only',action='store_true');ap.add_argument('--reuse')
 with threadpool_limits(limits=1):main(ap.parse_args())
