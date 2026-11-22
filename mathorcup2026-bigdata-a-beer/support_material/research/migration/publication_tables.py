from pathlib import Path
import sys,json,os,hashlib
import numpy as np,pandas as pd
from scipy.stats import beta
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import run_scenario as m
O=HERE/'all11';a=np.load(O/'validation_draws.npz',allow_pickle=False);f=m.read_csv(O/'development_frontier.csv');c=m.read_csv(O/'accepted_checks.csv');p=m.read_csv(O/'frozen_validation_profiles.csv')
ids=a['candidate_id'].tolist();index={x:i for i,x in enumerate(ids)};sources=sorted(f.beer.unique());n=200;joint=(a['margin']>=0)&(a['stale']<=1e-8);style=(a['margin']>=0);stale=(a['stale']<=1e-8);rng=np.random.default_rng(202610250733)
def block(method,budget,kind='joint'):
 ans=np.zeros((n,len(sources)))
 for ii,b in enumerate(sources):
  r=f[f.strategy.eq(method)&f.budget.eq(budget)&f.beer.eq(b)].iloc[0]
  if r.exists:ans[:,ii]={'joint':joint,'style':style,'stale':stale}[kind][:,index[r.candidate_id]]
 return ans
pairs=[('fixed_margin_1','original_min_edit',60),('fixed_margin_2','original_min_edit',60),('fixed_margin_2','original_min_edit',100),('scenario32_margin025','original_min_edit',60),('scenario32_margin025','fixed_margin_2',60),('scenario32_margin025','fixed_margin_2',100)]
rows=[]
for new,old,B in pairs:
 x=block(new,B).mean(axis=1);y=block(old,B).mean(axis=1);d=x-y;pick=rng.integers(0,n,(20000,n));ci=np.quantile(d[pick].mean(axis=1),[.025,.975]);gn=f[f.strategy.eq(new)&f.budget.eq(B)];go=f[f.strategy.eq(old)&f.budget.eq(B)]
 rows.append(dict(new=new,comparator=old,budget=B,new_sources=int(gn.exists.sum()),comparator_sources=int(go.exists.sum()),new_joint_all11=float(x.mean()),comparator_joint_all11=float(y.mean()),paired_difference_all11=float(d.mean()),paired_mc_standard_error=float(d.std(ddof=1)/np.sqrt(n)),paired_model_block_bootstrap_95_lower=float(ci[0]),paired_model_block_bootstrap_95_upper=float(ci[1]),paired_hoeffding_95_lower=float(d.mean()-np.sqrt(2*np.log(20)/n)),paired_hoeffding_six_comparisons_95_lower=float(d.mean()-np.sqrt(2*np.log(6/.05)/n)),new_mean_edits=float(gn.raw_fields_changed.mean()),comparator_mean_edits=float(go.raw_fields_changed.mean()),new_total_edits=float(gn.raw_fields_changed.sum()),comparator_total_edits=float(go.raw_fields_changed.sum())))
r=pd.DataFrame(rows)
case=[]
for B in [60,100]:
 for _,x in f[f.budget.eq(B)].iterrows():
  v=x.to_dict()
  if x.exists:
   j=index[x.candidate_id];k=int(joint[:,j].sum());v.update(trials=200,joint_successes=k,joint_frequency=k/200,style_successes=int(style[:,j].sum()),style_frequency=float(style[:,j].mean()),stale_successes=int(stale[:,j].sum()),stale_frequency=float(stale[:,j].mean()),joint_cp_lower_95=0 if k==0 else float(beta.ppf(.05,k,201-k)),joint_cp_lower_95_all98=0 if k==0 else float(beta.ppf(.05/len(ids),k,201-k)))
   nominal=c[c.candidate_id.eq(x.candidate_id)].iloc[0];v.update(nominal_ensemble_stale_delta=float(nominal.ensemble_stale_delta),nominal_min_style_margin=float(nominal.min_style_margin),chemical_violation=float(nominal.chemical_violation),box_violation=float(nominal.box_violation))
  else:v.update(trials=200,joint_successes=0,joint_frequency=0.,style_successes=0,style_frequency=0.,stale_successes=0,stale_frequency=0.,joint_cp_lower_95=0.,joint_cp_lower_95_all98=0.)
  case.append(v)
case=pd.DataFrame(case)
for name,frame in [('publication_paired_comparisons.csv',r),('publication_per_source.csv',case)]:
 path=O/name;tmp=path.with_suffix('.writing');frame.to_csv(tmp,index=False)
 with tmp.open('rb') as stream:os.fsync(stream.fileno())
 os.replace(tmp,path);reread=m.read_csv(path);assert len(reread)==len(frame)
manifest={name:dict(bytes=(O/name).stat().st_size,sha256=hashlib.sha256((O/name).read_bytes()).hexdigest()) for name in ['publication_paired_comparisons.csv','publication_per_source.csv','validated_frontier_summary.csv','validation_draws.npz','frozen_validation_profiles.csv','PRIMARY_ANALYSIS_LOCK.json']}
(O/'publication_tables_manifest.json').write_text(json.dumps(manifest,indent=2))
print(r.to_string(index=False));print(case[case.strategy.eq('fixed_margin_2')&case.budget.eq(100)][['beer','raw_fields_changed','joint_successes','style_successes','stale_successes','joint_cp_lower_95','joint_cp_lower_95_all98']].to_string(index=False))
