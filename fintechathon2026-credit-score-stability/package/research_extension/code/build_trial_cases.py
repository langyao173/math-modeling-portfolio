"""Select fixed real historical cases without reading future outcomes."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
from prepare_inputs import ROOT,V17,dump

def native(obj):
    if isinstance(obj,dict):return {k:native(v) for k,v in obj.items()}
    if isinstance(obj,list):return [native(v) for v in obj]
    if obj is None or (isinstance(obj,(float,np.floating)) and not np.isfinite(obj)):return None
    if isinstance(obj,np.generic):return obj.item()
    return obj

def main():
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    out=ROOT/'product_trial';out.mkdir(exist_ok=True)
    obs=pd.read_parquet(ROOT/'results/historical_observations.parquet');e=pd.read_parquet(ROOT/'results/historical_edges.parquet')
    pred=pd.read_parquet(ROOT/'results/historical/predictions.parquet');support=pd.read_parquet(ROOT/'results/historical/support_per_row.parquet');support_cut=float(pd.read_csv(ROOT/'results/historical/support_summary.csv').iloc[0].train_p95_outside_fraction);pool=e[e.phase_after.eq(0)].copy()
    i=pool.row_before.to_numpy();j=pool.row_after.to_numpy();a=obs.iloc[i].reset_index(drop=True);b=obs.iloc[j].reset_index(drop=True)
    valid=np.ones(len(pool),bool)
    for name in ['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL','AMT_PAYMENT_TOTAL_CURRENT','AMT_INST_MIN_REGULARITY']:
        valid &= np.isfinite(a[name])&np.isfinite(b[name])&a[name].ge(0)&b[name].ge(0)
    valid &= a.AMT_CREDIT_LIMIT_ACTUAL.gt(0)&b.AMT_CREDIT_LIMIT_ACTUAL.gt(0)
    misleading=b.AMT_BALANCE.gt(a.AMT_BALANCE)&(b.AMT_BALANCE/b.AMT_CREDIT_LIMIT_ACTUAL).lt(a.AMT_BALANCE/a.AMT_CREDIT_LIMIT_ACTUAL)
    pool['stratum']=np.where(pool.new_payment_shortfall,'payment_check',np.where(misleading,'denominator_change','routine'))
    pool['seed_order']=pool.event_row_id.map(lambda x:hashlib.sha256(('043-trial-'+x).encode()).hexdigest())
    pool=pool[valid]
    selected=[];practice=[]
    for s in ['denominator_change','payment_check','routine']:
        g=pool[pool.stratum.eq(s)].sort_values('seed_order')
        assert len(g)>=7
        selected.extend(g.head(6).to_dict('records'));practice.append(g.iloc[6].to_dict())
    ids={int(r['account_id']) for r in selected+practice}
    fields=['SK_ID_PREV','MONTHS_BALANCE','AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL','AMT_PAYMENT_TOTAL_CURRENT','AMT_INST_MIN_REGULARITY','AMT_DRAWINGS_CURRENT','SK_DPD']
    raw=pd.concat([pd.read_parquet(p,columns=fields,filters=[('SK_ID_PREV','in',list(ids))],use_threads=False) for p in sorted((V17/'baseline/data/raw').glob('part*.parquet'))],ignore_index=True)
    cases=[];keys={};sources=[]
    for k,row in enumerate(selected+practice):
        before=obs.iloc[row['row_before']];after=obs.iloc[row['row_after']];ref=float(row['reference_limit']);caseid=('T' if k<18 else 'P')+f'{k+1:02d}'
        ledger=raw[raw.SK_ID_PREV.eq(int(row['account_id']))&raw.MONTHS_BALANCE.le(after.MONTHS_BALANCE)].sort_values('MONTHS_BALANCE').tail(6)
        hist=[]
        for ix,h in enumerate(ledger.to_dict('records')):
            hist.append(native(dict(source_id='L'+str(ix+1),month=int(h['MONTHS_BALANCE']),balance=h['AMT_BALANCE'],limit=h['AMT_CREDIT_LIMIT_ACTUAL'],payment=h['AMT_PAYMENT_TOTAL_CURRENT'],minimum=h['AMT_INST_MIN_REGULARITY'],drawings=h['AMT_DRAWINGS_CURRENT'],dpd=h['SK_DPD'])))
        assert len(hist)==6 and hist[-2]['month']+1==hist[-1]['month']
        cur,prev=hist[-1],hist[-2];rate=cur['balance']/cur['limit']*100;rater=cur['balance']/ref*100
        paystate='no_due' if cur['minimum']==0 else 'covered' if cur['payment']>=cur['minimum'] else 'short'
        key=dict(balance_delta=cur['balance']-prev['balance'],util_current=rate,util_reference=rater,payment_state=paystate,action_evidence='unknown',source_ids=['L5','L6','R1'])
        case=dict(id=caseid,customer='客户 '+caseid,account='账户 '+caseid+'-01',history=hist,reference_limit=ref,source_period=after.MONTHS_BALANCE,quality_previous=float(100*(1-pred.iloc[row['row_before']].reference_bound)),quality_current=float(100*(1-pred.iloc[row['row_after']].reference_bound)),action_evidence='未提供额度调整审批凭证',currency='原数据金额单位',practice=k>=18)
        sp=support.iloc[row['row_after']];case['input_support']=dict(outside_fraction=float(sp.outside_fraction),outside_features=int(sp.outside_count),observed_features=int(sp.observed_features),training_p95_fraction=support_cut,high=bool(sp.high_excursion))
        cases.append(native(case));keys[caseid]=native(key)
        sources.append(dict(case_id=caseid,stratum=row['stratum'],row_id=row['row_id_after'],practice=k>=18,selection='fixed hash order within pre-outcome strata'))
    dump(out/'cases.json',cases);dump(out/'answer_key.json',keys);pd.DataFrame(sources).to_csv(out/'case_sources.csv',index=False)
    dump(out/'case_receipt.json',dict(real_cases=18,practice_cases=3,source_rows=len(raw),future_outcomes_read=False,source_panel_sha256=json.loads((ROOT/'results/data_roles.json').read_text())['raw_files'],cases_sha256=hashlib.sha256((out/'cases.json').read_bytes()).hexdigest(),selection_rule='6 per denominator-change, payment-check, routine; hash fixed; one extra practice per stratum'))
    print('Prepared 18 real cases + 3 distinct practice cases')

if __name__=='__main__':main()
