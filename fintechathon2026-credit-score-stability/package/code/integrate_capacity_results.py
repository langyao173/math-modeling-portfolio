"""Align the updated score with the maintained manuscript's statistical figures."""
from pathlib import Path
import json,math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit
from sklearn.metrics import roc_curve

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'evidence/capacity_update/historical'
OUT=ROOT/'evidence/current';(OUT/'joint').mkdir(parents=True,exist_ok=True);(OUT/'operating').mkdir(exist_ok=True)
METHODS=['incumbent_R','strong_S','previous_tail','previous_FX','previous_E','base_reference','old_bound','guard10','guard_six']
RIVALS=['incumbent_R','strong_S','previous_tail','previous_FX','previous_E','base_reference','old_bound']
GROUPS=['stable_05','repayment_state','flow_structure']
def rank(p,ids):return np.lexsort((np.asarray(ids,str),-np.asarray(p,float)))
def main():
 p=pd.read_parquet(SOURCE/'predictions.parquet');e=pd.read_parquet(ROOT/'capacity_inputs/historical/edges.parquet');yy=pd.read_parquet(ROOT/'capacity_inputs/historical/outcomes.parquet')
 assert p.sample_id.equals(yy.sample_id)
 p['y']=yy.y.to_numpy();p.to_parquet(OUT/'joint/predictions.parquet',index=False);e.to_parquet(OUT/'joint/edges.parquet',index=False)
 for name in ['risk','tail','capacity','memberships']:pd.read_csv(SOURCE/(name+'.csv')).to_csv(OUT/'joint'/(name+'.csv'),index=False)
 # One customer weight is shared across all periods, groups and compared methods.
 ids=np.sort(e.customer_id.unique());mapping=pd.Series(np.arange(len(ids)),index=ids);B=2000
 weights=np.random.default_rng(43100417).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=B).astype(np.int16)
 rows=[];draws=[]
 for group in GROUPS:
  for phase in [0,1]:
   g=e[e[group]&e.phase_after.eq(phase)];ci=mapping.loc[g.customer_id].to_numpy();a=g.row_before.to_numpy(int);b=g.row_after.to_numpy(int);values={}
   for method in ['guard10']+RIVALS:
    v=100*abs(p[method].to_numpy()[b]-p[method].to_numpy()[a]);ix=np.argsort(-v,kind='stable');v=v[ix];ww=weights[:,ci[ix]]
    mass=.05*ww.sum(axis=1);used=np.clip(mass[:,None]-np.c_[np.zeros(B),np.cumsum(ww,axis=1)[:,:-1]],0,ww)
    bp=(used*v[None,:]).sum(axis=1)/mass
    m=.05*len(v);k=int(m);point=float((v[:k].sum()+(m-k)*v[k])/m)
    values[method]=(point,bp)
   own,ownb=values['guard10']
   for method in RIVALS:
    other,otherb=values[method];delta=own-other;z=ownb-otherb;draws.append(z)
    rows.append(dict(group=group,phase=phase,n=len(g),comparator=method,metric='cvar95_pp',own_cvar=own,other_cvar=other,difference=delta,reduction_percent=100*(1-own/other),ci_low=float(np.quantile(z,.025)),ci_high=float(np.quantile(z,.975))))
 boot=np.asarray(draws).T;point=np.asarray([r['difference'] for r in rows]);se=boot.std(axis=0,ddof=1)
 critical=float(np.quantile(np.max(abs((boot-point)/np.maximum(se,1e-12)),axis=1),.95))
 for j,r in enumerate(rows):r.update(sim_low=float(point[j]-critical*se[j]),sim_high=float(point[j]+critical*se[j]))
 allci=pd.DataFrame(rows);allci.to_csv(OUT/'tail_intervals.csv',index=False)
 allci[allci.group.eq('stable_05')].to_csv(OUT/'joint/paired_intervals.csv',index=False)
 (OUT/'bootstrap_definition.json').write_text(json.dumps(dict(repetitions=B,seed=43100417,family=len(rows),critical=critical,unit='customer',score_policy='frozen historical predictions',scope='descriptive intervals on reused historical cohort'),indent=2))
 # Risk diagnostics and equal-count calibration groups use exactly the final q.
 risk=pd.read_csv(OUT/'joint/risk.csv');metrics=[];reliability=[]
 for phase in [0,1]:
  g=p[p.phase.eq(phase)&p.y.notna()];y=g.y.to_numpy()
  for method in METHODS:
   q=g[method].to_numpy();fpr,tpr,_=roc_curve(y,q);z=logit(np.clip(q,1e-8,1-1e-8))
   fit=minimize(lambda b:np.mean(np.logaddexp(0,b[0]+b[1]*z)-y*(b[0]+b[1]*z)),[0,1],method='BFGS',tol=1e-10)
   r=risk[risk.candidate.eq(method)&risk.phase.eq(phase)].iloc[0].to_dict()
   metrics.append({**r,'gini':2*r['auc']-1,'ks':float(max(tpr-fpr)),'cal_intercept':float(fit.x[0]),'cal_slope':float(fit.x[1])})
   for j,ix in enumerate(np.array_split(np.argsort(q,kind='stable'),10)):
    n=len(ix);rate=y[ix].mean();den=1+1.96**2/n;mid=(rate+1.96**2/(2*n))/den;rad=1.96*np.sqrt(rate*(1-rate)/n+1.96**2/(4*n*n))/den
    reliability.append(dict(model=method,phase=phase,bin=j,n=n,mean_p=q[ix].mean(),event_rate=rate,ci_low=mid-rad,ci_high=mid+rad))
 pd.DataFrame(metrics).to_csv(OUT/'risk_extended.csv',index=False);pd.DataFrame(reliability).to_csv(OUT/'reliability.csv',index=False)
 # Keep both event windows and unique customers; count recurring cases explicitly.
 front=[];selected_sets={}
 for cap in [.05,.08,.10,.12,.15,.20]:
  for method in METHODS:
   selected=[]
   for phase in [0,1]:
    g=p[p.phase.eq(phase)];selected.append(g.iloc[rank(g[method],g.row_id)[:math.ceil(cap*len(g))]])
   a=pd.concat(selected);s0=set(selected[0].customer_id);s1=set(selected[1].customer_id)
   front.append(dict(model=method,capacity=cap,slots=len(a),unique_clients=a.customer_id.nunique(),unique_events=a[a.y.eq(1)].customer_id.nunique(),event_windows=int(a.y.sum()),repeat_clients=len(s0&s1),new_second=len(s1-s0)))
   if cap==.1:selected_sets[method]=a.groupby('customer_id').y.sum().reindex(ids,fill_value=0).to_numpy()
 pd.DataFrame(front).to_csv(OUT/'operating/joint_frontier.csv',index=False)
 coverage=[]
 for method in RIVALS:
  delta=selected_sets['guard10']-selected_sets[method];sample=weights@delta
  coverage.append(dict(comparator=method,difference=float(delta.sum()),ci_low=float(np.quantile(sample,.025)),ci_high=float(np.quantile(sample,.975))))
 pd.DataFrame(coverage).to_csv(OUT/'coverage_intervals.csv',index=False)
 print(allci[allci.comparator.eq('strong_S')].to_string(index=False));print(pd.DataFrame(front).query('capacity==.1').to_string(index=False))
if __name__=='__main__':main()
