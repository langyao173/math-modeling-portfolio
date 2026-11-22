"""Audit and summarize frozen-candidate conditional Monte Carlo validation."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from scipy.stats import beta
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import run_scenario as m
OUT=HERE/'all11';ALPHA=.05

def cp(k,n,alpha=.05):return 0. if k==0 else float(beta.ppf(alpha,k,n-k+1))
def ci(x,rng):
 v=np.asarray(x,float);pick=rng.integers(0,len(v),(5000,len(v)));return np.quantile(v[pick].mean(axis=1),[.025,.975]).tolist()

def main():
 complete=json.loads((OUT/'COMPLETE.json').read_text());assert complete['complete'] and complete['validation_replicates']==200
 freeze=json.loads((OUT/'CANDIDATES_FROZEN.json').read_text());assert m.sha(OUT/'frozen_validation_profiles.csv')==freeze['profile_sha256'];assert m.sha(OUT/'development_frontier.csv')==freeze['frontier_sha256']
 lock=json.loads((OUT/'PRIMARY_ANALYSIS_LOCK.json').read_text());assert lock['validation_outcomes_examined'] is False
 frame=m.read_csv(OUT/'frozen_validation_profiles.csv');front=m.read_csv(OUT/'development_frontier.csv');a=np.load(OUT/'validation_draws.npz',allow_pickle=False);ids=a['candidate_id'].tolist();assert frame.candidate_id.to_list()==ids
 n=a['margin'].shape[0];assert n==200 and a['stale'].shape==(n,len(frame));assert a['delta'].shape==(n,len(frame),50)
 problem=m.FixedProblem('strict');check,_=problem.check_rows(frame.to_dict('records'));assert check.accepted.all()
 style=(a['margin']>=0);stale=(a['stale']<=1e-8);joint=style&stale
 expected=m.bootstrap(problem.df.tasting_category_fine.to_numpy(str),n,m.FRESH_SEED);assert np.array_equal(expected,a['indices'])
 old=np.load(OUT/'development_indices.npz')['indices'];assert not any(np.array_equal(x,y) for x in old for y in expected)
 targets=list(problem.bundle['targets']);st=targets.index('stale_hops');assert np.max(np.abs(a['delta'][:,:,st]-a['stale']))<1e-12
 lookup={s:i for i,s in enumerate(ids)};sourceids=sorted(problem.contexts);rng=np.random.default_rng(202610250732);curves=[];outcomes={};individual=[]
 for j,r in frame.iterrows():
  k=int(joint[:,j].sum());individual.append(dict(candidate_id=r.candidate_id,beer=int(r.beer),strategy=r.strategy,successes=k,trials=n,joint_rate=k/n,joint_cp_lower_95=cp(k,n),joint_cp_lower_family_95=cp(k,n,ALPHA/len(frame)),style_rate=float(style[:,j].mean()),stale_rate=float(stale[:,j].mean())))
 pd.DataFrame(individual).to_csv(OUT/'individual_validation_bounds.csv',index=False)
 for (strategy,budget),g in front.groupby(['strategy','budget']):
  assert len(g)==len(sourceids)==11;arr={x:np.zeros((n,11),float) for x in ['joint','style','stale']}
  for ii,beer in enumerate(sourceids):
   row=g[g.beer.eq(beer)].iloc[0]
   if bool(row.exists):
    j=lookup[row.candidate_id];arr['joint'][:,ii]=joint[:,j];arr['style'][:,ii]=style[:,j];arr['stale'][:,ii]=stale[:,j]
  outcomes[(strategy,int(budget))]=arr;covered=int(g.exists.sum());valid=g[g.exists]
  mean_by_draw=arr['joint'].mean(axis=1);lo,hi=ci(mean_by_draw,rng)
  curves.append(dict(strategy=strategy,budget=int(budget),total_sources=11,candidate_sources=covered,coverage=covered/11,
   mean_edits_available=float(valid.raw_fields_changed.mean()) if covered else np.nan,total_edits_available=float(valid.raw_fields_changed.sum()),max_edits_available=float(valid.raw_fields_changed.max()) if covered else np.nan,
   mean_distance_reduction_available=float(valid.target_distance_reduction.mean()) if covered else np.nan,
   style_all11=float(arr['style'].mean()),stale_all11=float(arr['stale'].mean()),joint_all11=float(mean_by_draw.mean()),
   style_conditional=float(arr['style'].sum()/n/covered) if covered else np.nan,stale_conditional=float(arr['stale'].sum()/n/covered) if covered else np.nan,joint_conditional=float(arr['joint'].sum()/n/covered) if covered else np.nan,
   joint_mc_standard_error=float(mean_by_draw.std(ddof=1)/np.sqrt(n)),joint_mc_bootstrap_95_lower=float(lo),joint_mc_bootstrap_95_upper=float(hi),
   joint_hoeffding_95_lower=max(0.,float(mean_by_draw.mean()-np.sqrt(np.log(1/ALPHA)/(2*n))))))
 summary=pd.DataFrame(curves);summary.to_csv(OUT/'validated_frontier_summary.csv',index=False)
 primary=front[front.budget.eq(lock['primary_budget'])].merge(pd.DataFrame(individual),on=['candidate_id','beer','strategy'],how='left');primary.to_csv(OUT/'primary_budget_cases.csv',index=False)
 chosen=lock['selected_strategy'];B=lock['primary_budget'];pairs=[]
 for compare in summary.strategy.unique():
  if compare==chosen:continue
  x=outcomes[(chosen,B)]['joint'].mean(axis=1);y=outcomes[(compare,B)]['joint'].mean(axis=1);d=x-y;lo,hi=ci(d,rng)
  cnew=front[front.strategy.eq(chosen)&front.budget.eq(B)].set_index('beer');cold=front[front.strategy.eq(compare)&front.budget.eq(B)].set_index('beer');common=(cnew.exists&cold.exists).to_numpy(bool)
  dc=(outcomes[(chosen,B)]['joint'][:,common]-outcomes[(compare,B)]['joint'][:,common]).mean(axis=1)
  pairs.append(dict(new_strategy=chosen,comparator=compare,budget=B,total_sources=11,new_available_sources=int(cnew.exists.sum()),comparator_available_sources=int(cold.exists.sum()),common_sources=int(common.sum()),
   mean_joint_difference_all11=float(d.mean()),paired_mc_standard_error=float(d.std(ddof=1)/np.sqrt(n)),paired_mc_bootstrap_95_lower=float(lo),paired_mc_bootstrap_95_upper=float(hi),paired_hoeffding_95_lower=float(d.mean()-np.sqrt(2*np.log(1/ALPHA)/n)),paired_hoeffding_three_comparisons_95_lower=float(d.mean()-np.sqrt(2*np.log(3/ALPHA)/n)),mean_joint_difference_common=float(dc.mean())))
 pd.DataFrame(pairs).to_csv(OUT/'primary_paired_comparisons.csv',index=False)
 details=dict(validation_scope='Conditional style-stratified empirical model-refit law on the same250 observed beers. Not independent new-beer accuracy and not physical intervention success probability.',
  fresh_seed=m.FRESH_SEED,development_draws=32,validation_draws=n,source_beers=11,frozen_profiles=len(frame),weights=np.asarray(problem.bundle['weights']).tolist(),families=problem.bundle['families'],all_targets_refit=50,
  primary_analysis_lock=lock,main_results=summary[summary.budget.eq(B)].to_dict('records'),primary_comparisons=pairs,
  interval_scope='CP is exact per frozen profile under iid model scenarios; family lower bounds Bonferroni across all98 frozen profiles. Mean and paired bootstrap intervals resample200 model draws, preserving the11-source dependence; they estimate Monte Carlo uncertainty only. Hoeffding lower bounds retain the known outcome ranges and give conservative finite-sample statements.',
  no_candidate_handling='Unavailable budget cases count0 in all11 metrics; conditional means reported separately. No fallback source profile is passed off as a successful transfer.',
  validation_indices_sha256=m.sha(OUT/'validation_indices.npz'),frozen_profiles_sha256=m.sha(OUT/'frozen_validation_profiles.csv'),all_nominal_checks_pass=True)
 m.dump(OUT/'validated_analysis.json',details)
 print(summary[summary.budget.eq(B)].to_string(index=False),flush=True);print(pd.DataFrame(pairs).to_string(index=False),flush=True)
 print('VALIDATION_AUDIT_PASSED',flush=True)
if __name__=='__main__':main()
