"""Register a separately supplied cohort; open its local result exactly once.

This utility enforces local workflow discipline. Data-owner attestations and
institutional access logs establish whether people had previously seen labels.
It cannot retroactively turn a reused research cohort into a fresh test set.
"""
from pathlib import Path
import argparse,hashlib,json,sqlite3
import pandas as pd

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def register(database,cohort_id,predictions,used_customer_keys,model_receipt):
    f=pd.read_csv(predictions);required={'row_id','customer_key','base_probability','final_probability','snapshot_at','outcome_due_at'}
    if not required.issubset(f):raise ValueError('Missing registration columns')
    if any(c in f for c in ['y','y30','default','future_dpd']):raise ValueError('Outcome columns cannot enter registration')
    if f.row_id.duplicated().any() or f.empty or f[list(required)].isna().any().any():raise ValueError('Complete nonempty unique rows required')
    if not f.customer_key.astype(str).str.contains(':',regex=False).all():raise ValueError('Customer keys must include source namespace')
    used=set(pd.read_csv(used_customer_keys).customer_key.astype(str))
    if used&set(f.customer_key.astype(str)):raise ValueError('Customers already used in research cannot enter the sealed cohort')
    for c in ['base_probability','final_probability']:
        if f[c].isna().any() or not f[c].between(0,1).all():raise ValueError('Invalid probabilities')
    snapshot=pd.to_datetime(f.snapshot_at,utc=True,errors='raise');due=pd.to_datetime(f.outcome_due_at,utc=True,errors='raise')
    if snapshot.isna().any() or due.isna().any() or (due<=snapshot).any():raise ValueError('Outcome window must follow scoring time')
    spec=dict(cohort_id=cohort_id,registered_at=pd.Timestamp.now(tz='UTC').isoformat(),predictions_sha256=sha(predictions),model_receipt_sha256=sha(model_receipt),used_ids_sha256=sha(used_customer_keys),rows=len(f),customers=f.customer_key.nunique())
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE IF NOT EXISTS cohorts (id TEXT PRIMARY KEY,spec TEXT,opened INTEGER NOT NULL DEFAULT 0,result TEXT)')
        if any(json.loads(v[0])['predictions_sha256']==spec['predictions_sha256'] for v in db.execute('SELECT spec FROM cohorts')):raise ValueError('These predictions already have a registered cohort')
        db.execute('INSERT INTO cohorts(id,spec) VALUES (?,?)',(cohort_id,json.dumps(spec)))
    return spec

def open_outcomes(database,cohort_id,predictions,outcomes):
    with sqlite3.connect(database) as db:
        db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT spec,opened FROM cohorts WHERE id=?',(cohort_id,)).fetchone()
        if row is None or row[1]:raise ValueError('Cohort absent or already opened')
        spec=json.loads(row[0])
        if sha(predictions)!=spec['predictions_sha256']:raise ValueError('Predictions changed after registration')
        db.execute('UPDATE cohorts SET opened=1,result=? WHERE id=?',(json.dumps({'status':'opened_before_reading_outcomes'}),cohort_id));db.commit()
        p=pd.read_csv(predictions);y=pd.read_csv(outcomes)
        if y.row_id.duplicated().any() or set(y.row_id)!=set(p.row_id):raise ValueError('Outcome row set does not match registered cohort')
        if not y.loc[y.y.notna(),'y'].isin([0,1]).all():raise ValueError('Outcome must be 0, 1 or unknown')
        a=p.merge(y,on='row_id',validate='1:1');observed=pd.to_datetime(a.outcome_available_at,utc=True,errors='raise');due=pd.to_datetime(a.outcome_due_at,utc=True,errors='raise')
        if (a.y.notna()&observed.isna()).any():raise ValueError('Known outcomes require availability dates')
        if (observed>pd.Timestamp.now(tz='UTC')).any():raise ValueError('Outcome availability cannot lie in the future')
        if (observed<pd.to_datetime(a.snapshot_at,utc=True)).any():raise ValueError('Outcome precedes scoring')
        if (a.y.eq(0)&observed.lt(due)).any():raise ValueError('An immature negative cannot be scored as zero')
        from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
        known=a[a.y.notna()];result={'rows':len(a),'known':len(known),'events':int(known.y.sum()),'metrics':{}}
        if known.y.nunique()!=2:raise ValueError('Both outcome classes are required for this benchmark')
        for col in ['base_probability','final_probability']:
            result['metrics'][col]=dict(auc=roc_auc_score(known.y,known[col]),ap=average_precision_score(known.y,known[col]),brier=brier_score_loss(known.y,known[col]),logloss=log_loss(known.y,known[col]))
        db.execute('UPDATE cohorts SET opened=1,result=? WHERE id=?',(json.dumps(result),cohort_id));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['register','open']);p.add_argument('--database',required=True);p.add_argument('--cohort-id',required=True);p.add_argument('--predictions',required=True);p.add_argument('--used-customer-keys');p.add_argument('--model-receipt');p.add_argument('--outcomes');a=p.parse_args()
    f=register(a.database,a.cohort_id,a.predictions,a.used_customer_keys,a.model_receipt) if a.operation=='register' else open_outcomes(a.database,a.cohort_id,a.predictions,a.outcomes)
    print(json.dumps(f,ensure_ascii=False,indent=2))
