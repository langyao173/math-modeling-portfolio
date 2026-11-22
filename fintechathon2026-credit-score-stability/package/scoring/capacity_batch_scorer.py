"""Frozen-model, contemporaneous-gate scoring with whole-batch capacity protection.

The caller commits eligible snapshot IDs and capacity before running inference.
All observations/states are validated by the existing raw-history adapter.
A batch either returns a complete set of new states or raises; inputs are immutable.
"""
from dataclasses import replace
from pathlib import Path
import hashlib,json,sqlite3,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from capacity_guard import score_batch,order
from unified_scorer import UnifiedCreditScorer
from deploy_capacity_scorer import Observation

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

class ProtectedBatchScorer(UnifiedCreditScorer):
    def __init__(self):
        super().__init__()
        self.model_version='reference-capacity-guard10-20261004'
        code=Path(__file__).resolve().parents[1]/'code/capacity_guard.py'
        self.model_fingerprint=hashlib.sha256((self.model_fingerprint+code.read_text()).encode()).hexdigest()

    def advance_batch(self,observations,states,*,batch_id,expected_snapshot_ids,quota):
        obs=list(observations);ids=[str(x.snapshot_id) for x in obs]
        expected=list(map(str,expected_snapshot_ids))
        if not batch_id or len(set(expected))!=len(expected) or len(set(ids))!=len(ids) or set(ids)!=set(expected):
            raise ValueError('Batch completeness / unique snapshot contract failed')
        if len(states)!=len(obs):raise ValueError('One previous state slot is required per observation')
        candidates=[];outputs=[];base=[];previous=[];gate=[]
        for x,state in zip(obs,states):
            new,out=self.initialize(x) if state is None else super().advance(state,x)
            if out['idempotent_replay']:raise ValueError('Replay must use the immutable batch receipt, not partial account replay')
            candidates.append(new);outputs.append(out);base.append(out['current_probability'])
            previous.append(state.filtered_probability if state else out['current_probability'])
            gate.append(out['gate_applied'])
        # row identity ordering follows the frozen evaluation protocol.
        row_ids=[f'{x.customer_id}:{x.account_id}:{x.timestamp}' for x in obs]
        q,rr=score_batch(base,previous,np.array(gate,bool),row_ids,quota=quota)
        ordinal=np.empty(len(q),int);ordinal[order(q,row_ids)]=np.arange(1,len(q)+1)
        spec=dict(batch_id=str(batch_id),snapshot_ids=sorted(expected),quota=int(quota),model_fingerprint=self.model_fingerprint,radius=.01)
        spec_id=digest(spec)
        new_states=[];rows=[]
        for j,(x,new,out) in enumerate(zip(obs,candidates,outputs)):
            receipt=dict(batch_id=str(batch_id),batch_spec_sha256=spec_id,batch_size=len(obs),quota=int(quota),
                         temporal_candidate=float(rr['temporal_candidate'][j]),guard_changed=bool(rr['guard_changed'][j]),
                         excess_pp=float(100*rr['unavoidable_excess'][j]),rank=int(ordinal[j]),selected=bool(ordinal[j]<=quota))
            out={**out,'candidate':'reference_capacity_guard10','q':float(q[j]),'quality_index':float(100*(1-q[j])),'capacity_receipt':receipt}
            receipts=dict(new.receipts);receipts[str(x.snapshot_id)]={**receipts[str(x.snapshot_id)],'output':out}
            new_states.append(replace(new,filtered_probability=float(q[j]),receipts=receipts));rows.append(out)
        return new_states,dict(schema='capacity-batch/1',spec=spec,spec_sha256=spec_id,thresholds=rr['thresholds'],rows=rows)

class BatchJournal:
    """Atomic, immutable batch receipts; same ID + different result is rejected."""
    def __init__(self,path):
        self.path=str(path)
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, hash TEXT NOT NULL, payload TEXT NOT NULL)')
    def commit(self,payload):
        ident=payload['spec']['batch_id'];hashed=digest(payload)
        with sqlite3.connect(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            old=db.execute('SELECT hash,payload FROM batches WHERE id=?',(ident,)).fetchone()
            if old:
                if old[0]!=hashed:raise ValueError('Batch ID already committed with different evidence or output')
                return json.loads(old[1])
            db.execute('INSERT INTO batches VALUES (?,?,?)',(ident,hashed,json.dumps(payload,ensure_ascii=False,allow_nan=False)))
        return payload
