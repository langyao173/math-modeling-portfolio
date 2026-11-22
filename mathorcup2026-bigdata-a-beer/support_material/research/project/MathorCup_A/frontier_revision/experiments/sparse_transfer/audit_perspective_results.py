"""Independent saved-output audit; does not launch any optimization."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from control_problem import FixedProblem, AW

HERE=Path(__file__).resolve().parent
OUT=HERE/'perspective'


def main():
    frozen_hashes={key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in {
        'code_sha256':HERE/'perspective_solver.py',
        'control_code_sha256':HERE/'control_problem.py',
        'predictor_sha256':AW/'q1/sensory_ensemble.joblib',
        'control_sha256':AW/'q3/transfer_and_augmentation_models.joblib',
        'classifier_sha256':AW/'competition_revision/style_feasibility/frozen_style_classifier.joblib',
    }.items()}
    if (OUT/'diagonal_sdp.npz').exists():
        a=np.load(OUT/'diagonal_sdp.npz')
        saved=json.loads((OUT/'diagonal_sdp_audit.json').read_text())
        P,D,R,W=a['P'],a['diagonal'],a['R'],a['dual_matrix']
        primal=float(D.sum());dual=float(np.sum(P*W.T))
        minimum_R=float(np.linalg.eigvalsh(R).min());minimum_W=float(np.linalg.eigvalsh(W).min())
        assert D.min()>=0 and minimum_R>0 and minimum_W>=0 and np.diag(W).min()>=1
        assert np.max(abs(P-R-np.diag(D)))<1e-11
        assert abs(primal-saved['primal_trace'])<1e-10 and abs(dual-saved['dual_trace_upper_bound'])<1e-10
        weak_dual=float(np.sum(R*W.T)+D@(np.diag(W)-1))
        assert abs(weak_dual-(dual-primal))<1e-10 and weak_dual>=0
        (OUT/'independent_sdp_certificate.json').write_text(json.dumps(dict(primal_trace=primal,
            dual_trace_upper_bound=dual,absolute_gap=dual-primal,weak_dual_identity_error=abs(weak_dual-(dual-primal)),
            R_minimum_eigenvalue=minimum_R,W_minimum_eigenvalue=minimum_W,
            W_minimum_diagonal=float(np.diag(W).min()),passed=True,
            scope='Independent double-precision verification of saved feasible SDP primal/dual; not exact arithmetic.'),indent=2))
    records=[]
    for policy in ('strict','exceedance'):
        directories=sorted((OUT/policy).glob('beer_*'))
        directories=[d for d in directories if (d/'DONE.json').exists()]
        if not directories:continue
        problem=FixedProblem(policy,style_epsilon=1e-5)
        for directory in directories:
            done=json.loads((directory/'DONE.json').read_text())
            protocol=json.loads((directory/'protocol.json').read_text())
            bound=json.loads((directory/'solver_bound.json').read_text())
            params=json.loads((directory/'solver_parameters.json').read_text())
            warm=json.loads((directory/'warm_start.json').read_text())
            assert done['policy']==policy==protocol['policy']==bound['policy']
            assert int(done['beer'])==int(protocol['beer'])==int(bound['beer'])
            assert protocol['seconds']==180 and protocol['threads']==1
            assert protocol['rho']==.1 and protocol['proxy_margin']==0 and protocol['style_epsilon']==1e-5
            assert params['limits/time']==180 and params['parallel/maxnthreads']==1
            assert params['constraints/nonlinear/assumeconvex'] is False
            assert params['nlhdlr/soc/enabled'] is True
            assert params['numerics/feastol']==1e-8 and params['randomization/randomseedshift']==20261024
            assert bound['time_limit']==180 and bound['warm_added'] is True
            assert bound['status'] in ('timelimit','optimal')
            if bound['status']=='timelimit':assert bound['seconds']>=179.9
            assert abs(bound['dual_bound']-done['new_dual_bound'])<1e-12
            assert int(done['new_integer_lower_bound'])==math.ceil(bound['dual_bound']-1e-7)
            assert bound['solutions']==done['raw_pool_points']
            assert warm['full_check']['accepted'] is True
            local_warm=HERE/policy/f"beer_{done['beer']}"/'profiles.csv'
            warm_source=local_warm if local_warm.exists() else Path(warm['source'])
            assert hashlib.sha256(warm_source.read_bytes()).hexdigest()==warm['source_profiles_sha256']
            assert all(protocol[key]==value for key,value in frozen_hashes.items())
            if 'variant_wrapper_sha256' in protocol:
                assert protocol['variant_wrapper_sha256']==hashlib.sha256((HERE/'perspective_sdp_solver.py').read_bytes()).hexdigest()
                assert protocol['shared_constraint_core_sha256']==frozen_hashes['code_sha256']
                assert protocol['diagonal_sdp_sha256']==hashlib.sha256((OUT/'diagonal_sdp.npz').read_bytes()).hexdigest()
                assert protocol['diagonal_sdp_audit_sha256']==hashlib.sha256((OUT/'diagonal_sdp_audit.json').read_bytes()).hexdigest()
            beer=done['beer'];c=problem.contexts[beer]
            split=np.load(directory/'diagonal_split.npz')
            P=split['P'];D=split['diagonal'];R=split['R'];L=split['factor']
            eigmin=float(np.linalg.eigvalsh(R).min())
            assert eigmin>=0 and D.min()>=0
            assert np.max(abs(P-(problem.P+problem.P.T)/2))<1e-12
            assert np.max(abs(P-R-np.diag(D)))<1e-11
            assert np.max(abs(R-L.T@L))<1e-10
            raw=np.load(directory/'raw_pool.npz')
            dd=raw['delta'];bb=raw['binary'];tt=raw['epigraph']
            assert len(dd)==done['raw_pool_points']
            A,rhs=problem.linear_constraints(beer,margin=0.)
            offset=c['z0']-problem.mu
            constant=float(offset@P@offset)
            radius2=.9**2*constant
            original=np.einsum('ni,ij,nj->n',offset+dd,P,offset+dd)
            strengthened=np.einsum('ni,ij,nj->n',dd,R,dd)+tt@D+2*dd@(P@offset)+constant
            equivalence=np.einsum('ni,ij,nj->n',dd,R,dd)+(dd**2)@D+2*dd@(P@offset)+constant
            identity_error=float(np.max(np.abs(original-equivalence),initial=0.))
            assert identity_error<1e-9
            profiles=pd.read_csv(directory/'profiles.csv')
            checks=pd.read_csv(directory/'checks.csv')
            assert len(profiles)==1+2*len(dd) and len(checks)==len(profiles)
            raw_checks=pd.read_csv(directory/'raw_checks.csv')
            assert len(raw_checks)==1+len(dd)
            npz=np.load(directory/'rank_candidates.npz')
            assert len(npz['delta'])==len(profiles)
            accepted=checks.accepted.astype(str).str.lower().eq('true')
            actual=profiles.loc[accepted].to_dict(orient='records')
            fresh,_=problem.check_rows(actual)
            assert fresh.accepted.all()
            assert int(fresh.raw_fields_changed.min())==done['new_full_upper_bound']
            assert done['new_integer_lower_bound']<=done['new_full_upper_bound']
            rec=dict(beer=beer,policy=policy,raw_pool_points=len(dd),R_eigen_min=eigmin,
                matched_budget_seconds=bound['time_limit'],solver_seconds=bound['seconds'],
                wall_time_overrun_seconds=max(0.,bound['seconds']-bound['time_limit']),
                shared_cpu_overrun_above_5s=bool(bound['seconds']>bound['time_limit']+5),
                original_warm_resolved_path=str(warm_source),
                original_warm_source_hash_pass=True,constraint_source_hash_pass=True,
                frozen_predictor_classifier_control_hashes_pass=True,
                sdp_variant_hashes_pass=True if 'variant_wrapper_sha256' in protocol else None,
                decomposition_identity_error=identity_error,
                raw_binary_integrality_max=float(np.max(np.minimum(abs(bb),abs(bb-1)),initial=0.)),
                raw_rotated_cone_violation=float(np.max(dd**2-tt*bb,initial=0.)),
                raw_geometry_violation=float(np.max(original-radius2,initial=0.)),
                raw_perspective_geometry_violation=float(np.max(strengthened-radius2,initial=0.)),
                raw_linear_guard_violation=float(np.max(dd@A.T-rhs,initial=0.)),
                raw_lower_link_violation=float(np.max(c['lo']*bb-dd,initial=0.)),
                raw_upper_link_violation=float(np.max(dd-c['hi']*bb,initial=0.)),
                full_accepted_rechecked=len(fresh),full_upper_bound=done['new_full_upper_bound'],
                raw_integer_lb=done['new_integer_lower_bound'],all_full_witnesses_pass=True)
            records.append(rec)
    pd.DataFrame(records).to_csv(OUT/'independent_audit.csv',index=False)
    summary=dict(completed_cases=len(records),all_full_witnesses_pass=all(r['all_full_witnesses_pass'] for r in records),
        scope='Independent algebra, retained pool completeness and full decoded-witness rechecks; not an exact-arithmetic proof of solver dual bounds.',
        records=records)
    (OUT/'independent_audit.json').write_text(json.dumps(summary,indent=2))
    print(pd.DataFrame(records).to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,default=OUT,
                        help='Saved lambda-min or SDP perspective result directory; read-only audit.')
    args=parser.parse_args()
    OUT=args.directory.resolve()
    with threadpool_limits(limits=1):
        main()
