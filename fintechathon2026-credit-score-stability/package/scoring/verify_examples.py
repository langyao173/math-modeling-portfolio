"""Replay 378 prior states and 189 recent snapshots using the packaged frozen scorer."""
from pathlib import Path
import json,warnings
import numpy as np,pandas as pd
from unified_scorer import UnifiedCreditScorer
from deploy_capacity_scorer import Observation,FINANCIAL
ROOT=Path(__file__).resolve().parent
def main():
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    scorer=UnifiedCreditScorer();e=ROOT/'examples';m=pd.read_parquet(e/'previous_metadata.parquet');x=pd.read_parquet(e/'previous_features.parquet');recent=pd.read_parquet(e/'recent_features.parquet');expected=pd.read_csv(e/'recent_expected.csv')
    scorer.raw_probabilities(pd.concat([x,recent[scorer.columns]],ignore_index=True),cache=True)
    states={};errors=[]
    for i,v in m.iterrows():
        o=Observation(str(v.customer_id),str(v.account_id),scorer.model_version,str(v.event_month),int(v.anchor_month),v.sample_id,float(v.reference_limit),x.iloc[i].to_dict(),{k:float(v[k]) for k in FINANCIAL},float(v.current_dpd),int(v.account_observations_to_date),bool(v.reference_supported))
        c=int(v.customer_id);states[c],r=scorer.initialize(o) if c not in states else scorer.advance(states[c],o);errors.append(abs(r['q']-v.expected_q))
    gates=[];new=[]
    for i,v in expected.iterrows():
        f=recent.iloc[i];previous=states[int(v.customer_id)]
        o=Observation(str(v.customer_id),str(v.account_id),scorer.model_version,previous.reference_period_id,int(v.anchor_month),v.sample_id,previous.reference_limit,f.to_dict(),{k:float(f['anchor__'+k]) for k in FINANCIAL},float(f.anchor__SK_DPD),int(f.account_observations_to_date),True)
        state,r=scorer.advance(previous,o);new.append(abs(r['q']-v.q));gates.append(r['gate_applied']==bool(v.gate))
    result={'previous_states':len(errors),'recent_states':len(new),'previous_max_error':max(errors),'recent_max_error':max(new),'gate_mismatches':int(len(gates)-sum(gates))}
    assert max(errors+new)<1e-12 and all(gates)
    (ROOT/'example_reproduction.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
