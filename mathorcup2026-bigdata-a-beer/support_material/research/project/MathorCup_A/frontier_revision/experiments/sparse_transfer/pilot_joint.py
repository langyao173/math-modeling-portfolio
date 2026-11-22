import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from control_problem import FixedProblem
from joint_solvers import reweighted_candidates,solve_cardinality,sparse_qp
HERE=Path(__file__).resolve().parent


def main():
    p=FixedProblem('strict');beer=124;allc=[];logs=[];rows=[]
    for margin in [0.,.05,.15]:
        cand,records=reweighted_candidates(p,beer,margin)
        logs+=records
        for cc in cand:
            row,z=p.decode(beer,cc['delta'],cc['support'])
            row.update(candidate_id=len(rows),method=cc['method'],margin=margin,lambda_=cc['lambda_'],iteration=cc['iteration'])
            rows.append(row);allc.append(cc)
        print('reweighted',margin,len(rows),flush=True)
    checks,preds=p.check_rows(rows)
    checks=pd.concat([pd.DataFrame(rows)[['candidate_id','method','margin','lambda_','iteration']],checks],axis=1)
    checks.to_csv(HERE/'pilot_joint_l1_checks.csv',index=False)
    pd.DataFrame(rows).to_csv(HERE/'pilot_joint_l1_profiles.csv',index=False)
    print(checks.loc[checks.accepted].sort_values('raw_fields_changed').head(15).to_string(index=False),flush=True)
    rho=.1
    feasible=checks[(checks.target_distance_reduction>=rho)&(checks.ensemble_stale_delta<=1e-8)]
    if len(feasible):
        ix=int(feasible.sort_values('raw_fields_changed').iloc[0].candidate_id);warm=allc[ix]['delta']
    else:warm=sparse_qp(p,beer)[0]
    cand,record=solve_cardinality(p,beer,rho=rho,time_limit=120,warm=warm,logfile=HERE/'pilot_scip.log')
    print(json.dumps(record),flush=True)
    (HERE/'pilot_scip_summary.json').write_text(json.dumps(record,indent=2))
    rr=[]
    for i,cc in enumerate(cand):
        q,d,v=p.polish(beer,cc['support'],initial=cc['delta'])
        row,z=p.decode(beer,d,cc['support']);row.update(candidate_id=i,support_size=len(cc['support']),qp_violation=v)
        rr.append(row)
    if rr:
        ck,pr=p.check_rows(rr);ck=pd.concat([pd.DataFrame(rr)[['candidate_id','support_size','qp_violation']],ck],axis=1)
        ck.to_csv(HERE/'pilot_scip_checks.csv',index=False);pd.DataFrame(rr).to_csv(HERE/'pilot_scip_profiles.csv',index=False)
        print(ck.to_string(index=False),flush=True)
    pd.DataFrame(logs).to_csv(HERE/'pilot_joint_qp_log.csv',index=False)


if __name__=='__main__':main()
