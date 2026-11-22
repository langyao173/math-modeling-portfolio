"""Paired customer bootstrap, support diagnostics and frozen-model trade-offs.

Confidence intervals describe the reused historical sample; they do not repair
adaptive reuse or turn it into an independent prospective evaluation.
"""
from pathlib import Path
import json, argparse, hashlib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from prepare_inputs import ROOT,CAP,dump

SEED=430410

def weighted_cvar_sorted(x, w):
    """x is already descending; fractional boundary mass defines exact CVaR."""
    mass=.05*w.sum()
    if mass<=0:return np.nan
    used=np.clip(mass-np.r_[0,np.cumsum(w)[:-1]],0,w)
    return float(np.dot(used,x)/mass)

def risk_metrics():
    pred=pd.read_parquet(ROOT/'results/historical/predictions.parquet')
    y=pd.read_parquet(ROOT/'results/historical_outcomes.parquet').y.to_numpy(float)
    rows=[]
    for phase in [0,1]:
        k=pred.phase.eq(phase)&np.isfinite(y)
        for name in pred.columns[5:]:
            p=pred.loc[k,name].to_numpy();yy=y[k]
            fpr,tpr,_=roc_curve(yy,p)
            # Evaluation-only calibration diagnostic, never applied to scores.
            fit=LogisticRegression(C=1e8,max_iter=1000).fit(logit(np.clip(p,1e-8,1-1e-8))[:,None],yy)
            auc=roc_auc_score(yy,p)
            rows.append(dict(candidate=name,phase=phase,auc=auc,gini=2*auc-1,ks=float(max(tpr-fpr)),ap=average_precision_score(yy,p),calibration_intercept=float(fit.intercept_[0]),calibration_slope=float(fit.coef_[0,0])))
    pd.DataFrame(rows).to_csv(ROOT/'results/historical/risk_extended.csv',index=False)

def bootstrap(B):
    pred=pd.read_parquet(ROOT/'results/historical/predictions.parquet')
    edges=pd.read_parquet(ROOT/'results/historical_edges.parquet')
    ids=np.sort(edges.customer_id.unique());imap=pd.Series(np.arange(len(ids)),index=ids)
    rng=np.random.default_rng(SEED)
    # One resampling weight per customer is shared by both phases and all methods.
    weights=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=B).astype(np.int16)
    rows=[];bs=[]
    for group in ['repayment_state','flow_structure']:
        for phase in [0,1]:
            e=edges[edges[group]&edges.phase_after.eq(phase)]
            ci=imap.loc[e.customer_id].to_numpy();i=e.row_before.to_numpy();j=e.row_after.to_numpy()
            values={}
            for name in ['reference_bound','actual_static','previous_tail','reference_static']:
                v=100*abs(pred[name].to_numpy()[j]-pred[name].to_numpy()[i]);ix=np.argsort(-v,kind='stable');x=v[ix]
                ww=weights[:,ci[ix]];point=weighted_cvar_sorted(x,np.ones(len(x)))
                tail_mass=.05*ww.sum(axis=1)
                used=np.clip(tail_mass[:,None]-np.c_[np.zeros(B),np.cumsum(ww,axis=1)[:,:-1]],0,ww)
                val=(used*x[None,:]).sum(axis=1)/tail_mass
                values[name]=(point,val)
            own,ownb=values['reference_bound']
            for rival in ['actual_static','previous_tail','reference_static']:
                other,otherb=values[rival];delta=own-other;d=ownb-otherb
                rows.append(dict(group=group,phase=phase,n=len(e),comparator=rival,own_cvar=own,other_cvar=other,difference=delta,reduction_percent=100*(1-own/other),ci95_low=float(np.quantile(d,.025)),ci95_high=float(np.quantile(d,.975))))
                bs.append(d)
    boot=np.array(bs).T;point=np.array([r['difference'] for r in rows]);se=boot.std(axis=0,ddof=1)
    crit=float(np.quantile(np.max(abs((boot-point)/np.maximum(se,1e-12)),axis=1),.95))
    for j,row in enumerate(rows):row.update(sim95_low=float(point[j]-crit*se[j]),sim95_high=float(point[j]+crit*se[j]))
    pd.DataFrame(rows).to_csv(ROOT/'results/historical/tail_bootstrap.csv',index=False)
    dump(ROOT/'results/historical/bootstrap_definition.json',dict(repetitions=B,seed=SEED,cluster='customer',paired_phases=True,confidence='approximate centered max-standardized paired customer bootstrap',family=len(rows),critical_value=crit,scope='Conditional historical cohort reanalysis; no independent-validation claim.'))
    # Fixed observed selected sets: unlike a full model/threshold bootstrap, this
    # estimates variation in workload's observed event contribution, not retraining.
    mem=pd.read_csv(ROOT/'results/historical/memberships.csv');rr=[]
    for name in ['actual_static','previous_tail','reference_static','incumbent_R','reference_rank100']:
        contrib={}
        for cand in ['reference_bound',name]:
            a=mem[mem.candidate.eq(cand)].groupby('customer_id').y.sum().reindex(ids,fill_value=0).to_numpy(float)
            contrib[cand]=a
        a=contrib['reference_bound']-contrib[name];b=weights@a
        rr.append(dict(comparator=name,event_slot_difference=float(a.sum()),ci95_low=float(np.quantile(b,.025)),ci95_high=float(np.quantile(b,.975)),scope='fixed selected sets; repeated event windows are slots, not unique incidents'))
    pd.DataFrame(rr).to_csv(ROOT/'results/historical/coverage_contribution_bootstrap.csv',index=False)

def semantic_and_bound():
    pred=pd.read_parquet(ROOT/'results/historical/predictions.parquet');e=pd.read_parquet(ROOT/'results/historical_edges.parquet')
    y=pd.read_parquet(ROOT/'results/historical_outcomes.parquet').y.to_numpy(float)
    out=[]
    for phase in [0,1]:
        ee=e[e.phase_after.eq(phase)&e.new_payment_shortfall];i=ee.row_before.to_numpy();j=ee.row_after.to_numpy()
        for name in ['reference_bound','reference_stress_bypass','reference_static','actual_static']:
            p=pred[name].to_numpy();d=100*(p[j]-p[i])
            out.append(dict(phase=phase,candidate=name,n=len(ee),known=int(np.isfinite(y[j]).sum()),events=int(np.nansum(y[j])),mean_signed_pp=float(d.mean()),rising=int((d>1e-12).sum()),falling=int((d<-1e-12).sum()),unchanged=int((abs(d)<=1e-12).sum())))
    pd.DataFrame(out).to_csv(ROOT/'results/historical/payment_signal_response.csv',index=False)
    p=pred.reference_static.to_numpy();out=[]
    for phase in [0,1]:
        ii=np.flatnonzero(pred.phase.eq(phase));pos=pd.Series(np.arange(len(ii)),index=ii);order=np.argsort(p[ii],kind='stable')
        ee=e[e.phase_after.eq(phase)&e.seven_05];j=pos.loc[ee.row_after].to_numpy();old=p[ee.row_before]
        def feasible(radius):
            lo=p[ii].copy();hi=lo.copy();lo[j]=np.maximum(0,old-radius);hi[j]=np.minimum(1,old+radius)
            return bool(np.all(np.maximum.accumulate(lo[order])<=hi[order]+1e-12))
        low,high=0.,1.
        for _ in range(60):
            mid=(low+high)/2
            if feasible(mid):high=mid
            else:low=mid
        out.append(dict(phase=phase,nondecreasing_minimum_radius_pp=100*high,one_pp_feasible=feasible(.01),condition='prior static p; nongated current p fixed; same frozen model and cohort'))
    pd.DataFrame(out).to_csv(ROOT/'results/historical/minimum_rank_radius.csv',index=False)

def support():
    # Reconstruct the original frozen final fit rows. No historical outcomes are
    # used to establish the reference feature support interval.
    frames=[];receipt=[]
    for group in ['general','event_development']:
        m=pd.read_parquet(CAP/'data'/group/'meta.parquet');y=pd.read_parquet(CAP/'data'/group/'outcomes.parquet')
        folds=pd.read_parquet(CAP/'data/development_folds'/('general_rows.parquet' if group=='general' else 'event_rows.parquet'))
        assert np.array_equal(m.sample_id,folds.sample_id)
        k=folds.fold.ge(0)&m.reference_limit.gt(0)&y.y.notna()
        if group=='general':k &= ~m.is_target
        x=pd.read_parquet(CAP/'data'/group/'reference.parquet');extra=pd.read_parquet(CAP/'data'/group/'reference_extra.parquet')
        frames.append(pd.concat([x,extra],axis=1).loc[k]);receipt.append(dict(source=group,rows=int(k.sum()),customers=int(m.loc[k,'customer_id'].nunique())))
    train=pd.concat(frames,ignore_index=True).replace([np.inf,-np.inf],np.nan)
    bounds=train.quantile([.01,.99]);bounds.to_csv(ROOT/'results/support_reference_quantiles.csv')
    variable=train.nunique(dropna=True).gt(1)&train.notna().sum().ge(100)
    names=train.columns[variable];valid_names=list(names)
    dump(ROOT/'results/support_definition.json',dict(fit_rows=receipt,feature_count=len(train.columns),variable_features=len(names),interval=[.01,.99],unit='marginal feature interval',note='An interval excursion is a descriptive support warning, not proof of joint OOD or model failure.'))
    pred=pd.read_parquet(ROOT/'results/historical/predictions.parquet');obs=pd.read_parquet(ROOT/'results/historical_observations.parquet')
    x=pd.concat([pd.read_parquet(CAP/'data/event_evaluation/reference.parquet'),pd.read_parquet(CAP/'data/event_evaluation/reference_extra.parquet')],axis=1).replace([np.inf,-np.inf],np.nan)
    a=x[names];outside=(a.lt(bounds.loc[.01,names])|a.gt(bounds.loc[.99,names]))&a.notna()
    frac=outside.sum(axis=1)/a.notna().sum(axis=1)
    data=pred.iloc[:,:5].copy();data['outside_fraction']=frac;data['outside_count']=outside.sum(axis=1);data['observed_features']=a.notna().sum(axis=1)
    data['balance_over_reference']=obs.AMT_BALANCE/obs.reference_limit
    # Training, rather than historical outcomes, fixes the high-support-risk cut.
    at=train[names];tr=((at.lt(bounds.loc[.01,names])|at.gt(bounds.loc[.99,names]))&at.notna()).sum(axis=1)/at.notna().sum(axis=1)
    cutoff=float(tr.quantile(.95));data['high_excursion']=frac.gt(cutoff)
    data.to_parquet(ROOT/'results/historical/support_per_row.parquet',index=False)
    rows=[];y=pd.read_parquet(ROOT/'results/historical_outcomes.parquet').y.to_numpy(float)
    for phase in [-1,0,1]:
        m=data.phase.eq(phase)
        for group,k in [('all',m),('high_excursion',m&data.high_excursion),('other',m&~data.high_excursion),('balance_gt_ref',m&data.balance_over_reference.gt(1))]:
            known=k&np.isfinite(y);n=int(k.sum());row=dict(phase=phase,group=group,n=n,known=int(known.sum()),events=int(np.nansum(y[k])),balance_gt_ref=int((data.loc[k,'balance_over_reference']>1).sum()),any_excursion=int((data.loc[k,'outside_count']>0).sum()),mean_outside_fraction=float(data.loc[k,'outside_fraction'].mean()),high_count=int(data.loc[k,'high_excursion'].sum()),train_p95_outside_fraction=cutoff)
            for name in ['reference_bound','actual_static']:
                yy=y[known];pp=pred.loc[known,name]
                row[name+'_auc']=roc_auc_score(yy,pp) if len(np.unique(yy))==2 else np.nan
                row[name+'_ap']=average_precision_score(yy,pp) if yy.sum() else np.nan
            rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT/'results/historical/support_summary.csv',index=False)

def main(B,rebuild_support=False):
    for group,short in [('event_development','development'),('event_evaluation','historical')]:
        path=CAP/'data'/group/'outcomes.parquet' if short=='development' else CAP/'data/evaluation_outcomes/event_evaluation/outcomes.parquet'
        local=ROOT/'results'/f'{short}_outcomes.parquet'
        if not local.exists():pd.read_parquet(path).to_parquet(local,index=False)
    risk_metrics();bootstrap(B);semantic_and_bound()
    if rebuild_support or not (ROOT/'results/historical/support_per_row.parquet').exists():support()
    print('Diagnostics complete',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--bootstrap',type=int,default=2000);ap.add_argument('--rebuild-support',action='store_true');a=ap.parse_args();main(a.bootstrap,a.rebuild_support)
