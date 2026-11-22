"""Independent dual certificates for classification-only sparse supports.

For any nonnegative multiplier nu, the strictly convex quadratic Lagrangian
gives a valid lower bound. Thus a positive certified distance worsening rules
out ALL continuous points on this fixed support, not just one failed point.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cho_factor,cho_solve
import osqp
from control_problem import FixedProblem
HERE=Path(__file__).resolve().parent

def main():
    p=FixedProblem('strict',style_epsilon=1e-5);out=[];payload={}
    for beer,c in p.contexts.items():
        mip=p.cardinality_milp(beer,time_limit=60)
        if mip.x is None:raise ValueError(f'No classification witness for {beer}')
        s=np.flatnonzero(mip.x[p.p:]>.5)
        A,rhs=p.linear_constraints(beer);G=np.vstack([A[:,s],np.eye(len(s)),-np.eye(len(s))])
        h=np.r_[rhs,c['hi'][s],-c['lo'][s]];P=p.P[np.ix_(s,s)];g=c['gradient'][s]
        qp=osqp.OSQP();qp.setup(P=sparse.csc_matrix(np.triu(P)),q=g,A=sparse.csc_matrix(G),
            l=np.full(len(h),-np.inf),u=h,verbose=False,eps_abs=1e-11,eps_rel=1e-11,max_iter=100000,polishing=True)
        q=qp.solve();nu=np.maximum(q.y,0);factor=cho_factor(P,lower=True)
        r=g+G.T@nu
        dual=float(c['F0']-nu@h-.5*r@cho_solve(factor,r))
        primal=float(c['F0']+g@q.x+.5*q.x@P@q.x)
        radius0=2*c['F0'];dstar=-cho_solve(factor,g);minimum=radius0+g@dstar
        room=max(radius0-minimum,0.)
        ww=p.w[:,s];Pinvw=cho_solve(factor,ww.T)
        max_margin=c['base_margin']+ww@dstar+np.sqrt(np.maximum(room*np.einsum('ij,ji->i',ww,Pinvw),0))
        row=dict(beer=beer,support_size=len(s),linear_mip_objective=float(mip.fun),linear_mip_dual_bound=float(mip.mip_dual_bound),
            linear_mip_gap=float(mip.mip_gap),style_epsilon=p.style_epsilon,qp_status=q.info.status,
            source_half_squared_distance=c['F0'],distance_objective_primal=primal,distance_objective_dual=dual,
            primal_dual_gap=primal-dual,primal_violation=float(np.max(G@q.x-h)),
            stationarity_residual=float(np.max(abs(P@q.x+g+G.T@nu))),
            unavoidable_distance_increase_lower_bound=float(np.sqrt(max(dual,0)/c['F0'])-1),
            all_continuous_points_fail_positive_improvement=bool(dual>c['F0']+1e-7),
            ellipsoid_single_halfspace_min_upper_margin=float(max_margin.min()),
            ellipsoid_single_halfspace_exclusion=bool(max_margin.min()<p.style_epsilon-1e-7))
        out.append(row)
        payload[str(beer)]=dict(support=s.tolist(),support_names=[p.features[i] for i in s],multipliers=nu.tolist(),
            primal=q.x.tolist(),dual_bound=dual,G=G.tolist(),h=h.tolist(),P=P.tolist(),g=g.tolist(),F0=c['F0'])
    pd.DataFrame(out).to_csv(HERE/'classification_geometry_certificates.csv',index=False)
    (HERE/'classification_geometry_witnesses.json').write_text(json.dumps(payload,indent=2))
    print(pd.DataFrame(out).to_string(index=False))

if __name__=='__main__':main()
