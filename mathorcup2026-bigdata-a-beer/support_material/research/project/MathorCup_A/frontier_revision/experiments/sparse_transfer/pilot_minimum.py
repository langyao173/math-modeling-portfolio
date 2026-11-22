"""Pilot of global linear minimum-cardinality bounds and continuous polishing."""
from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import time,json
import numpy as np
import pandas as pd
from control_problem import FixedProblem

HERE=Path(__file__).resolve().parent
p=FixedProblem();records=[];rows=[];details=[]
for beer,c in p.contexts.items():
    t=time.perf_counter();r=p.cardinality_milp(beer,time_limit=60)
    rec=dict(beer=beer,status=int(r.status),message=str(r.message),seconds=time.perf_counter()-t,
        objective=float(r.fun) if r.fun is not None else None,dual_bound=float(getattr(r,'mip_dual_bound',np.nan)),
        gap=float(getattr(r,'mip_gap',np.nan)),nodes=int(getattr(r,'mip_node_count',0)))
    if r.x is not None:
        s=np.flatnonzero(r.x[p.p:]>.5);rec['support']=[p.features[j] for j in s]
        rec['max_mip_linear_violation']=float(np.max(p.linear_constraints(beer)[0]@r.x[:p.p]-p.linear_constraints(beer)[1]))
        for m in [0.,.025,.05,.1,.2,.4,.8]:
            q,d,v=p.polish(beer,s,margin=m,initial=r.x[:p.p])
            info=dict(beer=beer,margin=m,support_size=len(s),qp_success=bool(q.success),violation=v)
            if q.success and v<=1e-8:
                row,z=p.decode(beer,d,s);row.update(candidate_id=len(rows),margin=m,support_size=len(s))
                rows.append(row);info['candidate_id']=len(rows)-1
            details.append(info)
    records.append(rec);print(json.dumps(rec,ensure_ascii=False),flush=True)
pd.DataFrame(records).to_json(HERE/'pilot_linear_bounds.json',orient='records',indent=2,force_ascii=False)
pd.DataFrame(details).to_csv(HERE/'pilot_qp.csv',index=False)
if rows:
    ch,pr=p.check_rows(rows);ch=pd.concat([pd.DataFrame(rows)[['candidate_id','margin','support_size']],ch],axis=1)
    ch.to_csv(HERE/'pilot_checks.csv',index=False)
    pd.DataFrame(rows).to_csv(HERE/'pilot_profiles.csv',index=False)
    print(ch.to_string(index=False),flush=True)
