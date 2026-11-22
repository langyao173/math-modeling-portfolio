"""Complete, paired scenario-block summaries of the held-out matched-budget test."""
from pathlib import Path
import sys,json,os,hashlib
import numpy as np,pandas as pd
from scipy.stats import beta
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent));import run_scenario as m

def atomic_csv(path,df):
 temp=path.with_suffix('.writing');df.to_csv(temp,index=False)
 with temp.open('rb') as f:os.fsync(f.fileno())
 os.replace(temp,path);assert len(m.read_csv(path))==len(df)

def main():
 complete=json.loads((H/'COMPLETE.json').read_text());assert complete['complete'] and complete['validation_scenarios']==100
 freeze=json.loads((H/'CANDIDATES_FROZEN.json').read_text());assert m.sha(H/'frozen_profiles.csv')==freeze['profiles_sha256'];assert m.sha(H/'frozen_frontier.csv')==freeze['frontier_sha256'];assert m.sha(H/'PROTOCOL_LOCK.json')==freeze['protocol_sha256']
 p=m.read_csv(H/'frozen_profiles.csv');front=m.read_csv(H/'frozen_frontier.csv');a=np.load(H/'validation_draws.npz',allow_pickle=False);assert p.candidate_id.to_list()==a['candidate_id'].tolist()
 ids=p.candidate_id.tolist();lookup={x:i for i,x in enumerate(ids)};N=100;sourceids=sorted(front.beer.unique());assert len(sourceids)==11
 problem=m.FixedProblem('strict');check,_=problem.check_rows(p.to_dict('records'));assert check.accepted.all()
 old=m.read_csv(H.parent/'all11/frozen_validation_profiles.csv').set_index('candidate_id');cols=problem.bundle['features'];oldcontrol=p[~p.strategy.eq('low_margin_same_qp')]
 for _,r in oldcontrol.iterrows():
  x=r[cols].to_numpy(float);y=old.loc[r.candidate_id,cols].to_numpy(float);assert ((x==y)|(np.isnan(x)&np.isnan(y))).all()
 expected=m.bootstrap(problem.df.tasting_category_fine.to_numpy(str),100,202610251503);assert np.array_equal(expected,a['indices'])
 v1=np.load(H.parent/'all11/validation_draws.npz',allow_pickle=False)['indices'];assert not any(np.array_equal(x,y) for x in v1 for y in expected)
 style=a['margin']>=0;stale=a['stale']<=1e-8;joint=style&stale;assert joint.shape==(N,len(p))
 rng=np.random.default_rng(202610251504);picks=rng.integers(0,N,(20000,N));blocks={};summary=[];cases=[]
 for (strategy,B),g in front.groupby(['strategy','budget']):
  assert len(g)==11;arr={k:np.zeros((N,11)) for k in ['joint','style','stale']}
  for ii,b in enumerate(sourceids):
   r=g[g.beer.eq(b)].iloc[0];row=r.to_dict()
   if r.exists:
    j=lookup[r.candidate_id]
    for k,v in [('joint',joint),('style',style),('stale',stale)]:arr[k][:,ii]=v[:,j]
    success=int(joint[:,j].sum());row.update(joint_successes=success,trials=N,joint_rate=success/N,style_successes=int(style[:,j].sum()),stale_successes=int(stale[:,j].sum()),joint_cp_lower_95=0. if not success else float(beta.ppf(.05,success,N-success+1)),joint_cp_lower_family_95=0. if not success else float(beta.ppf(.05/len(p),success,N-success+1)))
   else:row.update(joint_successes=0,trials=N,joint_rate=0.,style_successes=0,stale_successes=0,joint_cp_lower_95=0.,joint_cp_lower_family_95=0.)
   cases.append(row)
  blocks[(strategy,int(B))]=arr;covered=int(g.exists.sum());v=g[g.exists];mean=arr['joint'].mean(axis=1);ci=np.quantile(mean[picks].mean(axis=1),[.025,.975])
  summary.append(dict(strategy=strategy,budget=int(B),candidate_sources=covered,total_sources=11,joint_all11=float(arr['joint'].mean()),style_all11=float(arr['style'].mean()),stale_all11=float(arr['stale'].mean()),joint_conditional=float(arr['joint'].sum()/N/covered) if covered else np.nan,mean_edits_available=float(v.raw_fields_changed.mean()),total_edits_available=float(v.raw_fields_changed.sum()),mean_distance_reduction_available=float(v.target_distance_reduction.mean()),joint_mc_95_lower=float(ci[0]),joint_mc_95_upper=float(ci[1])))
 comparisons=[('fixed_margin_1','low_margin_same_qp',60),('fixed_margin_2','low_margin_same_qp',60),('fixed_margin_2','low_margin_same_qp',100),('low_margin_same_qp','original_min_edit',60),('low_margin_same_qp','original_min_edit',100),('fixed_margin_1','original_min_edit',60),('fixed_margin_2','original_min_edit',60),('fixed_margin_2','original_min_edit',100)]
 paired=[]
 for new,ref,B in comparisons:
  x=blocks[(new,B)]['joint'].mean(axis=1);y=blocks[(ref,B)]['joint'].mean(axis=1);d=x-y;ci=np.quantile(d[picks].mean(axis=1),[.025,.975]);gn=front[front.strategy.eq(new)&front.budget.eq(B)].set_index('beer');go=front[front.strategy.eq(ref)&front.budget.eq(B)].set_index('beer');common=(gn.exists&go.exists).to_numpy(bool)
  common_delta=(blocks[(new,B)]['joint'][:,common]-blocks[(ref,B)]['joint'][:,common]).mean(axis=1)
  paired.append(dict(new=new,comparator=ref,budget=B,new_candidate_sources=int(gn.exists.sum()),comparator_candidate_sources=int(go.exists.sum()),new_joint_all11=float(x.mean()),comparator_joint_all11=float(y.mean()),paired_difference_all11=float(d.mean()),paired_mc_standard_error=float(d.std(ddof=1)/np.sqrt(N)),paired_block_mc_95_lower=float(ci[0]),paired_block_mc_95_upper=float(ci[1]),paired_hoeffding_family95_lower=float(d.mean()-np.sqrt(2*np.log(len(comparisons)/.05)/N)),common_sources=int(common.sum()),paired_difference_common=float(common_delta.mean()),new_mean_edits=float(gn.raw_fields_changed.mean()),comparator_mean_edits=float(go.raw_fields_changed.mean()),new_mean_distance_reduction=float(gn.target_distance_reduction.mean()),comparator_mean_distance_reduction=float(go.target_distance_reduction.mean())))
 summary=pd.DataFrame(summary);paired=pd.DataFrame(paired);cases=pd.DataFrame(cases)
 atomic_csv(H/'matched_summary.csv',summary);atomic_csv(H/'matched_paired_comparisons.csv',paired);atomic_csv(H/'matched_per_source.csv',cases)
 m.dump(H/'analysis.json',dict(complete=True,validation_seed=202610251503,scenarios=100,frozen_profiles=len(p),all_sources=11,old_controls_bitwise_identical=True,new_indices_disjoint_from_v1=True,nominal_full_checks_pass=True,summary=summary.to_dict('records'),paired=paired.to_dict('records'),scope='All intervals are conditional Monte Carlo scenario-block intervals on the same observed250 beers; not external generalization or physical success.'))
 # Atomic readback of every completed top-level numerical artifact.
 manifest={}
 for path in sorted(H.iterdir()):
  if not path.is_file() or path.suffix not in ['.csv','.json','.npz']:continue
  original_stat=path.stat();data=path.read_bytes();tmp=path.with_suffix(path.suffix+'.writing')
  with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  os.utime(tmp,ns=(original_stat.st_atime_ns,original_stat.st_mtime_ns));os.replace(tmp,path);assert path.read_bytes()==data
  if path.suffix=='.npz':
   with np.load(path,allow_pickle=False) as z:
    for k in z.files:assert z[k].dtype!=object
  elif path.suffix=='.json':json.loads(path.read_text())
  else:m.read_csv(path)
  manifest[path.name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 m.dump(H/'FINAL_READBACK.json',dict(validated_files=len(manifest),files=manifest))
 print(summary.to_string(index=False),flush=True);print(paired.to_string(index=False),flush=True);print('MATCHED_AUDIT_PASSED',flush=True)
if __name__=='__main__':main()
