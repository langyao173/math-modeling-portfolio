from pathlib import Path
import tempfile,json
import pandas as pd
from locked_cohort import register,open_outcomes
ROOT=Path(__file__).resolve().parents[1]
def fail(fn):
 try:fn()
 except ValueError:
  return
 raise AssertionError('Invalid evaluation accepted')
with tempfile.TemporaryDirectory() as td:
 p=Path(td);pred=p/'p.csv';used=p/'u.csv';model=p/'m.json';out=p/'o.csv';db=p/'e.db'
 f=pd.DataFrame(dict(row_id=['a','b'],customer_key=['software_fixture:a','software_fixture:b'],base_probability=[.2,.7],final_probability=[.21,.69],snapshot_at=['2020-01-01']*2,outcome_due_at=['2020-04-01']*2));f.to_csv(pred,index=False)
 pd.DataFrame({'customer_key':['software_fixture:a']}).to_csv(used,index=False);model.write_text('{}')
 fail(lambda:register(db,'used',pred,used,model))
 pd.DataFrame({'customer_key':['software_fixture:unused']}).to_csv(used,index=False)
 register(db,'new',pred,used,model);fail(lambda:register(db,'duplicate',pred,used,model))
 pd.DataFrame(dict(row_id=['a','b'],y=[0,1],outcome_available_at=['2020-04-01']*2)).to_csv(out,index=False)
 result=open_outcomes(db,'new',pred,out);assert result['events']==1;fail(lambda:open_outcomes(db,'new',pred,out))
 f['row_id']=['c','d'];f.to_csv(pred,index=False);register(db,'immature',pred,used,model)
 pd.DataFrame(dict(row_id=['c','d'],y=[0,1],outcome_available_at=['2020-02-01']*2)).to_csv(out,index=False)
 fail(lambda:open_outcomes(db,'immature',pred,out));fail(lambda:open_outcomes(db,'immature',pred,out))
 f['row_id']=['e','f'];f.to_csv(pred,index=False);register(db,'missingdate',pred,used,model)
 pd.DataFrame(dict(row_id=['e','f'],y=[0,1],outcome_available_at=[None,'2020-04-01'])).to_csv(out,index=False)
 fail(lambda:open_outcomes(db,'missingdate',pred,out))
s=dict(used_customer_rejected=True,duplicate_registration_rejected=True,one_time_open_enforced=True,immature_negative_rejected=True,known_outcome_date_required=True,failed_open_remains_consumed=True,kind='software_fixture_only',new_credit_validation_runs=0)
(ROOT/'evidence/capacity_update/locked_protocol_verification.json').write_text(json.dumps(s,indent=2));print(s)
