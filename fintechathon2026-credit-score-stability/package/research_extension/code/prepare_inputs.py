"""Reconstruct pre-outcome behaviour groups and data-role coverage."""
from pathlib import Path
import json, hashlib, os
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('FINTECH_SOURCE_ROOT', ROOT.parent / 'research_recovery'))
V17 = SOURCE / 'FinTechathon_Task1_V17'
CAP = SOURCE / 'Capacity_Research/capacity_research'
JOINT = SOURCE / 'joint_advantage_research'

def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))

def main():
    out=ROOT/'results';out.mkdir(exist_ok=True)
    paths=sorted((V17/'baseline/data/raw').glob('part*.parquet'))
    raw=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
    assert len(raw)==3840312 and not raw.duplicated(['SK_ID_PREV','MONTHS_BALANCE']).any()
    role_sets={}
    for name,path in [('past',V17/'baseline/data/previously_used_customer_ids.csv.gz'),('extension',V17/'baseline/data/previous_extension_customers.csv.gz'),('v17_development',V17/'data/development_membership.csv.gz')]:
        role_sets[name]=set(pd.read_csv(path).customer_id)
    for group in ['general','event_development','event_evaluation']:
        role_sets[group]=set(pd.read_parquet(CAP/'data'/group/'meta.parquet',columns=['customer_id']).customer_id)
    used=set.union(*role_sets.values());unseen=set(raw.SK_ID_CURR)-used
    u=raw[raw.SK_ID_CURR.isin(unseen)].sort_values(['SK_ID_PREV','MONTHS_BALANCE']).copy()
    g=u.groupby('SK_ID_PREV',sort=False)
    u['history_n']=g.cumcount()+1
    u['old_limit']=g.AMT_CREDIT_LIMIT_ACTUAL.shift();u['old_month']=g.MONTHS_BALANCE.shift();u['old_dpd']=g.SK_DPD.shift()
    a=u[u.AMT_CREDIT_LIMIT_ACTUAL.gt(u.old_limit)&u.old_limit.gt(0)&u.MONTHS_BALANCE.eq(u.old_month+1)]
    eligible=a[a.SK_DPD.eq(0)&a.old_dpd.eq(0)&a.history_n.ge(7)]
    eligible[['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE','old_limit','history_n']].to_csv(out/'new_action_candidates_preoutcome.csv',index=False)
    dump(out/'data_roles.json',{'raw_rows':len(raw),'raw_customers':int(raw.SK_ID_CURR.nunique()),'past_role_customers':{k:len(v) for k,v in role_sets.items()},'unused_customers':len(unseen),'unused_rows':len(u),'unused_limit_increases':len(a),'eligible_new_events':len(eligible),'new_outcomes_generated':False,'raw_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})
    raw=raw.set_index(['SK_ID_PREV','MONTHS_BALANCE'])
    for group,short in [('event_development','development'),('event_evaluation','historical')]:
        d=CAP/'data'/group;meta=pd.read_parquet(d/'meta.parquet');edges=pd.read_parquet(d/'edges.parquet')
        key=pd.MultiIndex.from_arrays([meta.account_id,meta.anchor_month],names=raw.index.names)
        obs=raw.reindex(key).reset_index()
        assert np.array_equal(obs.SK_ID_CURR.to_numpy(),meta.customer_id.to_numpy())
        obs=pd.concat([meta[['sample_id','row_id','customer_id','phase','reference_limit']].reset_index(drop=True),obs.drop(columns=['SK_ID_CURR']).reset_index(drop=True)],axis=1)
        if short=='development':
            f=pd.read_parquet(CAP/'data/development_folds/event_rows.parquet');assert np.array_equal(f.sample_id,meta.sample_id)
            meta['fold']=f.fold.to_numpy();edges['fold']=edges.customer_id.map(meta.groupby('customer_id').fold.first())
        else:meta['fold']=0;edges['fold']=0
        i=edges.row_before.to_numpy(int);j=edges.row_after.to_numpy(int)
        cols=['AMT_RECEIVABLE_PRINCIPAL','AMT_TOTAL_RECEIVABLE','AMT_RECIVABLE']
        x=obs[cols].to_numpy(float)
        delta=np.max(abs(x[i]-x[j])/edges.reference_limit.to_numpy()[:,None],axis=1)
        edges['seven_05']=edges.stable_05&np.isfinite(delta)&(delta<=.05)
        edges['all_transitions']=True
        complete=['AMT_PAYMENT_TOTAL_CURRENT','AMT_INST_MIN_REGULARITY','CNT_DRAWINGS_CURRENT']
        a0=obs.iloc[i].reset_index(drop=True);a1=obs.iloc[j].reset_index(drop=True)
        common=(a0.NAME_CONTRACT_STATUS.eq('Active')&a1.NAME_CONTRACT_STATUS.eq('Active')&a0.SK_DPD.eq(0)&a1.SK_DPD.eq(0))
        common&=np.isfinite(a0[complete]).all(axis=1)&np.isfinite(a1[complete]).all(axis=1)&a0[complete].ge(0).all(axis=1)&a1[complete].ge(0).all(axis=1)
        def repayment(a):
            return np.where(a.AMT_INST_MIN_REGULARITY.eq(0),0,np.where(a.AMT_PAYMENT_TOTAL_CURRENT+1e-8<a.AMT_INST_MIN_REGULARITY,1,2))
        r0=repayment(a0);r1=repayment(a1)
        def count_band(a):return np.select([a.CNT_DRAWINGS_CURRENT.eq(0),a.CNT_DRAWINGS_CURRENT.eq(1),a.CNT_DRAWINGS_CURRENT.le(5)],[0,1,2],default=3)
        edges['repayment_state']=np.asarray(common)&(r0==r1)&(count_band(a0)==count_band(a1))
        p0=a0.AMT_PAYMENT_TOTAL_CURRENT/a0.AMT_INST_MIN_REGULARITY.replace(0,np.nan)
        p1=a1.AMT_PAYMENT_TOTAL_CURRENT/a1.AMT_INST_MIN_REGULARITY.replace(0,np.nan)
        def atm_share(a):
            return np.where(a.AMT_DRAWINGS_CURRENT.eq(0)&a.AMT_DRAWINGS_ATM_CURRENT.eq(0),0,a.AMT_DRAWINGS_ATM_CURRENT/a.AMT_DRAWINGS_CURRENT.replace(0,np.nan))
        s0=atm_share(a0);s1=atm_share(a1)
        count_change=2*abs(a1.CNT_DRAWINGS_CURRENT-a0.CNT_DRAWINGS_CURRENT)/(1+a0.CNT_DRAWINGS_CURRENT+a1.CNT_DRAWINGS_CURRENT)
        edges['flow_structure']=np.asarray(common)&a0.AMT_INST_MIN_REGULARITY.gt(0)&a1.AMT_INST_MIN_REGULARITY.gt(0)&np.isfinite(p0)&np.isfinite(p1)&abs(p1-p0).le(.25)&np.isfinite(s0)&np.isfinite(s1)&(abs(s1-s0)<=.10)&count_change.le(.25)
        edges['new_payment_shortfall']=np.asarray(common)&(r0==2)&(r1==1)
        edges['outside_seven']=~edges.seven_05
        edges['balance_above_reference']=a1.AMT_BALANCE.to_numpy()>edges.reference_limit.to_numpy()
        edges['payment_ratio_before']=p0;edges['payment_ratio_after']=p1
        edges['repayment_state_before']=r0;edges['repayment_state_after']=r1
        obs.to_parquet(out/f'{short}_observations.parquet',index=False)
        meta.to_parquet(out/f'{short}_meta.parquet',index=False)
        edges.to_parquet(out/f'{short}_edges.parquet',index=False)
    print((out/'data_roles.json').read_text(),flush=True)

if __name__=='__main__':main()
