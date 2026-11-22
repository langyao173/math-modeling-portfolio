"""Legacy nested supports under the new common constraints and tolerances.

This comparison fixes the old forward-Schur supports, while using exactly the
new continuous objective, rho=.1, epsilon=1e-5, six proxy margins and decoding.
It isolates support search; it is not presented as identical CPU expenditure.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from control_problem import FixedProblem,AW
from joint_solvers import sparse_qp
HERE=Path(__file__).resolve().parent
MARGINS=[0.,.025,.05,.1,.15,.25]

def main():
    out=HERE/'matched_grid_reference';out.mkdir(exist_ok=True)
    supports=pd.read_csv(AW/'competition_revision/style_feasibility/support_definitions.csv')
    (out/'protocol.json').write_text(json.dumps(dict(rho=.1,style_epsilon=1e-5,margins=MARGINS,
        supports='Unchanged legacy nested forward-Schur supports; all 225 endpoint',
        objective='Same continuous F and full frozen nonlinear check as free-support search',
        budget='Same numerical tolerances, not a claim of matched CPU time'),indent=2))
    for policy in ['strict','exceedance']:
        p=FixedProblem(policy);rows=[];logs=[];t0=time.perf_counter()
        for beer in p.contexts:
            beer_rows=[]
            for srow in supports[supports.beer==beer].itertuples():
                s=np.array([p.features.index(f) for f in json.loads(srow.selected_fields)],int)
                if not set(p.contexts[beer]['mandatory']).issubset(set(s)):continue
                for margin in MARGINS:
                    d,info=sparse_qp(p,beer,s,margin=margin)
                    log=dict(policy=policy,beer=beer,support_method=srow.method,budget=int(srow.allowed_coordinates),margin=margin,**info)
                    if d is not None:
                        q,d,v=p.polish(beer,s,margin=margin,initial=d,rho=.1)
                        row,_=p.decode(beer,d,s)
                        row.update(policy=policy,method='legacy_support_common_polish',candidate_id=len(rows)+len(beer_rows),
                            support_method=srow.method,budget=int(srow.allowed_coordinates),required_proxy_margin=margin,required_rho=.1)
                        beer_rows.append(row);log.update(polish_success=bool(q.success),polish_violation=v)
                    logs.append(log)
            rows+=beer_rows
            print(json.dumps(dict(policy=policy,beer=beer,candidates=len(beer_rows),seconds=time.perf_counter()-t0)),flush=True)
        frame=pd.DataFrame(rows);checks,pred=p.check_rows(rows)
        checks=pd.concat([frame[['candidate_id','policy','method','support_method','budget']],checks],axis=1)
        frame.to_csv(out/f'{policy}_profiles.csv',index=False)
        checks.to_csv(out/f'{policy}_checks.csv',index=False)
        pred.to_csv(out/f'{policy}_sensory.csv',index=False)
        pd.DataFrame(logs).to_csv(out/f'{policy}_solver_attempts.csv',index=False)
        selected=checks[checks.accepted].sort_values(['raw_fields_changed','quadratic_objective']).groupby('beer',as_index=False).first()
        selected.to_csv(out/f'{policy}_minimum_found.csv',index=False)
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
