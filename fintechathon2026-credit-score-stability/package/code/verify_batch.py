from pathlib import Path
import sys,json,tempfile,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scoring'))
from capacity_guard import score_batch,protect_membership,order
from capacity_batch_scorer import ProtectedBatchScorer,Observation,BatchJournal
from deploy_capacity_scorer import FINANCIAL
def must_fail(fn):
    try:fn()
    except (ValueError,AssertionError):return
    raise AssertionError('Invalid input accepted')
def main():
    rng=np.random.default_rng(431004);checks=0
    for n in [2,7,101,1000]:
        for kind in ['random','ties','zero','one']:
            p=rng.random(n) if kind=='random' else rng.choice([.1,.5,.9],n) if kind=='ties' else np.zeros(n) if kind=='zero' else np.ones(n)
            old=rng.random(n);g=rng.random(n)>.3;ids=np.array([f'id{i:05}' for i in range(n)]);cuts=sorted(set([0,1,n//2,n-1,n]))
            q,r=score_batch(p,old,g,ids,quota=cuts)
            for k in cuts:assert set(order(q,ids)[:k])==set(order(p,ids)[:k])
            perm=rng.permutation(n);qq,_=score_batch(p[perm],old[perm],g[perm],ids[perm],quota=cuts)
            np.testing.assert_allclose(q[perm],qq,atol=0,rtol=0)
            np.testing.assert_allclose(q,np.clip(r['temporal_candidate'],r['lower'],r['upper']),atol=0,rtol=0)
            checks+=1
    for q in [-1,1.5,True,4]:
        must_fail(lambda:score_batch([.2,.3],[.2,.3],[True,False],['a','b'],quota=q))
    must_fail(lambda:score_batch([.2,.3],[.2,.3],[1,0],['a','b'],quota=1))
    must_fail(lambda:score_batch([.2,.3],[.2,.3],[True,False],['a','a'],quota=1))
    meta=pd.read_parquet(ROOT/'capacity_inputs/historical/meta.parquet')
    x=pd.read_parquet(ROOT/'scoring/examples/reference_features.parquet')
    target=pd.read_parquet(ROOT/'evidence/capacity_update/historical/predictions.parquet')
    scorer=ProtectedBatchScorer();scorer.raw_probabilities(x,cache=True)
    states={};q=np.empty(len(meta));p=np.empty(len(meta));g=np.zeros(len(meta),bool);timings=[]
    import time
    for phase in [-1,0,1]:
        idx=np.flatnonzero(meta.phase.eq(phase));observations=[];previous=[]
        for i in idx:
            m=meta.iloc[i];f=x.iloc[i].to_dict()
            observations.append(Observation(str(m.customer_id),str(m.account_id),scorer.model_version,str(m.event_row_id),int(m.anchor_month),str(m.sample_id),float(m.reference_limit),f,{c:f['anchor__'+c] for c in FINANCIAL},float(m.current_dpd),int(m.account_observations_to_date),bool(m.reference_supported)))
            previous.append(states.get(str(m.event_row_id)))
        start=time.perf_counter()
        ns,result=scorer.advance_batch(observations,previous,batch_id=f'historical-relative-phase-{phase}',expected_snapshot_ids=[o.snapshot_id for o in observations],quota=int(np.ceil(.1*len(idx))))
        timings.append(dict(phase=phase,rows=len(idx),cached_inference_validation_and_update_seconds=time.perf_counter()-start))
        for i,s,o in zip(idx,ns,result['rows']):
            states[str(meta.iloc[i].event_row_id)]=s;q[i]=o['q'];p[i]=o['current_probability'];g[i]=o['gate_applied']
        (ROOT/f'evidence/capacity_update/historical/batch_receipt_phase{phase}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        if phase==1:
            with tempfile.TemporaryDirectory() as td:
                j=BatchJournal(Path(td)/'b.sqlite');assert j.commit(result)==j.commit(result)
                changed=json.loads(json.dumps(result));changed['rows'][0]['q']+=.001;must_fail(lambda:j.commit(changed))
            must_fail(lambda:scorer.advance_batch(observations[:-1],previous[:-1],batch_id='missing',expected_snapshot_ids=[o.snapshot_id for o in observations],quota=268))
    np.testing.assert_allclose(p,target.base_reference,atol=2e-12,rtol=0)
    np.testing.assert_allclose(q,target.guard10,atol=2e-12,rtol=0)
    e=pd.read_parquet(ROOT/'capacity_inputs/historical/edges.parquet');assert np.array_equal(g[e.row_after],e.seven_05)
    summary=dict(vector_scenarios=checks,reproduced_rows=len(meta),updates=len(e),base_max_abs_error=float(abs(p-target.base_reference).max()),final_max_abs_error=float(abs(q-target.guard10).max()),gate_matches=True,batch_completeness_rejection=True,atomic_journal_replay=True,conflicting_replay_rejection=True,timing=timings)
    (ROOT/'evidence/capacity_update/verification.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
