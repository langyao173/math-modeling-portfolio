"""Live raw-history scoring adapter for the development-selected unified score.

Uses the prior package's validated complete-history feature engine and state
contract, and replaces its pre-calibration EWMA by calibrated bounded updates.
"""
import json,hashlib
from dataclasses import replace
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent
from deploy_capacity_scorer import CapacityChallengerScorer,Observation

class UnifiedCreditScorer(CapacityChallengerScorer):
    def __init__(self):
        super().__init__(ROOT,model_version='joint-reference-bound10-20261003')
        self.calibration=json.loads((ROOT/'calibration.json').read_text())
        self.model_fingerprint=hashlib.sha256((self.model_fingerprint+json.dumps(self.calibration,sort_keys=True)+'post-calibration-bound01').encode()).hexdigest()
    def _raw(self,vector):
        return self._calibrated(super()._raw(vector))
    def _result(self,obs,raw,filtered,applied,reason):
        return dict(candidate='old_reference_FX__bound10',customer_id=str(obs.customer_id),account_id=str(obs.account_id),
            snapshot_id=obs.snapshot_id,timestamp=int(obs.timestamp),model_version=self.model_version,model_fingerprint=self.model_fingerprint,
            reference_period_id=obs.reference_period_id,current_probability=float(raw),q=float(filtered),quality_index=100*(1-filtered),
            gate_applied=bool(applied),gate_reason=reason,idempotent_replay=False)
    def advance(self,state,obs):
        new,out=super().advance(state,obs)
        if out['idempotent_replay'] or not out['gate_applied']:return new,out
        q=float(np.clip(out['current_probability'],state.filtered_probability-.01,state.filtered_probability+.01))
        out={**out,'q':q,'quality_index':100*(1-q)}
        receipts=dict(new.receipts);receipts[str(obs.snapshot_id)]={**receipts[str(obs.snapshot_id)],'output':out}
        return replace(new,filtered_probability=q,receipts=receipts),out
