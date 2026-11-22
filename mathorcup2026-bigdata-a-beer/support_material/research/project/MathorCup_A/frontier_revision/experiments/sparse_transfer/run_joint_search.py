"""Full 11-source, two-policy support search with saved feasible witnesses."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
from control_problem import FixedProblem,AW,file_sha
from joint_solvers import reweighted_candidates,prune_support,solve_cardinality
HERE=Path(__file__).resolve().parent
MARGINS=[0.,.025,.05,.1,.15,.25]
LAMBDAS=[.03,.1,.3,1.,2.,3.,5.,10.]
RHOS=[.001,.1,.25,.4]


def run(policy,scip_seconds):
    out=HERE/policy;out.mkdir(parents=True,exist_ok=True)
    p=FixedProblem(policy,style_epsilon=1e-5);allrows=[];allchecks=[];allbounds=[]
    protocol=dict(policy=policy,style_epsilon=p.style_epsilon,lambda_preservation=.2,
        reweighted_l1_margins=MARGINS,lambdas=LAMBDAS,steps=6,epsilon=.1,rhos=RHOS,
        backward_trials_per_step=12,scip_seconds_per_source=scip_seconds,
        scip_rho=.1,scip_proxy_margin=0.,nonlinear_sensory='Full frozen six-model ensemble, no increase in stale_hops',
        qp_frozen_sha256=file_sha(AW/'q3/transfer_and_augmentation_models.joblib'),
        sensory_sha256=file_sha(AW/'q1/sensory_ensemble.joblib'),
        classifier_sha256=file_sha(AW/'competition_revision/style_feasibility/frozen_style_classifier.joblib'),
        quantification='Observed raw field changes; derived 5 fields regenerated, technical internal standard fixed',
        lower_bound_scope='MIQCP relaxation at rho=.1, proxy margin=0. Not a bound for another rho/policy.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for beer in p.contexts:
        base=out/f'beer_{beer}';base.mkdir(exist_ok=True)
        if (base/'DONE.json').exists():
            allrows.append(pd.read_csv(base/'profiles.csv'));allchecks.append(pd.read_csv(base/'checks.csv'))
            if (base/'solver_bound.json').exists():allbounds.append(json.loads((base/'solver_bound.json').read_text()))
            continue
        t0=time.perf_counter();rows=[];qplog=[];candidates=[]
        for margin in MARGINS:
            cc,ll=reweighted_candidates(p,beer,margin,lambdas=LAMBDAS,steps=6)
            qplog+=ll
            for c in cc:
                row,z=p.decode(beer,c['delta'],c['support'])
                row.update(candidate_id=len(rows),method=c['method'],required_proxy_margin=margin,
                    lambda_=c['lambda_'],iteration=c['iteration'],required_rho=0.)
                rows.append(row);candidates.append(c)
        ck,_=p.check_rows(rows)
        ck=pd.concat([pd.DataFrame(rows)[['candidate_id','method','lambda_','iteration']],ck],axis=1)
        print(json.dumps(dict(stage='l1',policy=policy,beer=beer,candidates=len(rows),accepted=int(ck.accepted.sum()),
                              elapsed=time.perf_counter()-t0)),flush=True)
        # Three explicit target levels plus near-zero improvement, separately audited.
        for rho in RHOS:
            good=ck[ck.accepted&(ck.target_distance_reduction>=rho-1e-8)].sort_values(['raw_fields_changed','quadratic_objective'])
            if not len(good):continue
            ix=int(good.iloc[0].candidate_id);initial=candidates[ix]
            refined,log=prune_support(p,beer,initial['delta'],initial['support'],initial['margin'],rho)
            pd.DataFrame(log).to_csv(base/f'backward_rho_{rho:g}.csv',index=False)
            row,z=p.decode(beer,refined['delta'],refined['support'])
            row.update(candidate_id=len(rows),method=refined['method'],required_rho=rho,required_proxy_margin=refined['margin'])
            rows.append(row);candidates.append(refined)
        ck,_=p.check_rows(rows)
        ck=pd.concat([pd.DataFrame(rows)[['candidate_id','method']],ck],axis=1)
        # A relaxed margin=0 lower bound remains valid for every stronger-margin witness.
        if scip_seconds>0:
            good=ck[ck.accepted&(ck.target_distance_reduction>=.1-1e-8)].sort_values(['raw_fields_changed','quadratic_objective'])
            warm=None if not len(good) else candidates[int(good.iloc[0].candidate_id)]['delta']
            cc,bound=solve_cardinality(p,beer,rho=.1,margin=0.,time_limit=scip_seconds,warm=warm,logfile=base/'scip.log')
            (base/'solver_bound.json').write_text(json.dumps(bound,indent=2));allbounds.append(bound)
            # Retain unique supports from the solver pool; refine with the same ball.
            seen=set()
            for c in cc:
                skey=tuple(c['support'].tolist())
                if skey in seen:continue
                seen.add(skey)
                q,d,v=p.polish(beer,c['support'],margin=0.,initial=c['delta'],rho=.1)
                row,z=p.decode(beer,d,c['support'])
                row.update(candidate_id=len(rows),method='SCIP_MIQCP',required_rho=.1,required_proxy_margin=0.,qp_violation=v)
                rows.append(row);candidates.append(dict(c,delta=d))
        ck,pred=p.check_rows(rows)
        ck=pd.concat([pd.DataFrame(rows)[['candidate_id','method']],ck],axis=1)
        frame=pd.DataFrame(rows);frame.to_csv(base/'profiles.csv',index=False);ck.to_csv(base/'checks.csv',index=False)
        pred.to_csv(base/'predicted_sensory.csv',index=False);pd.DataFrame(qplog).to_csv(base/'qp_log.csv',index=False)
        np.savez_compressed(base/'rank_candidates.npz',delta=np.stack([c['delta'] for c in candidates]),
                            support=np.stack([np.isin(np.arange(p.p),c['support']) for c in candidates]))
        result=dict(beer=beer,policy=policy,seconds=time.perf_counter()-t0,candidates=len(rows),accepted=int(ck.accepted.sum()),
                    best_cardinality=int(ck[ck.accepted].raw_fields_changed.min()) if ck.accepted.any() else None)
        (base/'DONE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        allrows.append(frame);allchecks.append(ck)
    pd.concat(allrows,ignore_index=True).to_csv(out/'all_profiles.csv',index=False)
    pd.concat(allchecks,ignore_index=True).to_csv(out/'all_checks.csv',index=False)
    pd.DataFrame(allbounds).to_csv(out/'solver_bounds.csv',index=False)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--policy',choices=['strict','exceedance'],required=True);a.add_argument('--scip-seconds',type=float,default=120)
    args=a.parse_args();run(args.policy,args.scip_seconds)
